from Task import Task
import heapq

slowdown_factor = 0.5
speedup_factor = 4.0

class Scheduler:
    def __init__(self, slowdown_factor = 0.5, speedup_factor = 4):
        self.runqueue = []
        self.waitqueue = []
        self.system_time = 0.0
        self.avg_vruntime = 0.0
        self.curr_task = None
        self.curr_task_time_left = 0.0
        self.eventqueue = []
        self.slowdown_factor = slowdown_factor
        self.speedup_factor = speedup_factor
    
    def add(self, task):
        heapq.heappush(self.runqueue, (task.vdeadline, task))

    def pick_next(self):
        eligible = [(vd, t) for vd, t in self.runqueue if t.currency >= -1e-9] #floating point shenanigans
        if eligible:
            return min(eligible)[1]

        # Idea 1: if no eligible tasks, use earliest vdeadline (work conserving)
        # return self.runqueue[0][1] if self.runqueue else None
        # Idea 2: if no eligible task, wait until a task is eligible
        for vd, t in self.runqueue:
            print(t.currency)
        return None
    
    def remove_from_runqueue(self, task):
        # runqueue stores (vdeadline, task) tuples
        self.runqueue = [(vd, t) for (vd, t) in self.runqueue if t != task]
        heapq.heapify(self.runqueue)
    
    def remove_from_waitqueue(self, task):
        if task in self.waitqueue:
            self.waitqueue.remove(task)
    
    def compute_average_vruntime(self):
        numerator = 0.0
        denominator = 0.0
        for _, t in self.runqueue:
            numerator += (t.vruntime * t.weight)
            denominator += t.weight
        if denominator == 0.0:
            return self.avg_vruntime
        return numerator / denominator

    def elapse_time(self):
        self.system_time += 1.0
        self.curr_task_time_left = max(0.0, self.curr_task_time_left - 1.0)
        if self.curr_task:
            self.curr_task.increase_runtime(1.0)

        self.avg_vruntime = self.compute_average_vruntime()
        total_weight = sum(t.weight for _, t in self.runqueue)
        # Work-conserving accrual: scale entitlements so total accrual per tick is 1
        # denom = sum_i (entitlement_i * currency_rate_i)
        for _, t in self.runqueue:
            t.compute_lag(self.avg_vruntime)
        denom = 0.0
        if total_weight > 0:
            for _, t in self.runqueue:
                denom += (t.weight / total_weight) * t.currency_rate
        k = (1.0 / denom) if denom > 0.0 else 0.0
        s = 0
        for _, t in self.runqueue:
            if total_weight > 0:
                entitlement = t.weight / total_weight
                t.increase_currency(entitlement * k)
                s += entitlement * k * t.currency_rate
        print(s)
        if s != 1.0:
            print("what")

        if self.curr_task:
            if self.curr_task.remaining_time == 0:
                self.runqueue = [(vd, x) for vd, x in self.runqueue if x != self.curr_task]
                self.curr_task.reset_task()
                self.curr_task.compute_vdeadline()
                self.add(self.curr_task)
            else:
                self.curr_task.compute_vdeadline()

        heapq.heapify(self.runqueue)
    
    def handle_event(self, event):
        if event[1] == "sleep":
            t = self.curr_task
            if t:
                self.remove_from_runqueue(t)
                self.curr_task = None
                self.curr_task_time_left = 0.0
                self.waitqueue.append(t)
        elif event[1] == "wakeup":
            self.remove_from_waitqueue(event[0])
            self.add(event[0])
        elif event[1] == "slow":
            event[0].set_currency_rate(self.slowdown_factor)
        elif event[1] == "normal":
            event[0].set_currency_rate(self.speedup_factor)
    
    def handle_events(self):
        for event in self.eventqueue[:]:
            if self.curr_task == event[0] or event[1] == "wakeup":
                self.handle_event(event)
                self.eventqueue.remove(event)
                
    def process_events(self):
        for _, t in self.runqueue:
            event = t.process_events(self.system_time)
            if event:
                self.eventqueue.append((t, event))
        for t in self.waitqueue:
            event = t.process_events(self.system_time)
            if event:
                self.eventqueue.append((t, event))
        self.handle_events()
        heapq.heapify(self.runqueue)

    def tick(self):
        self.process_events()
        if self.curr_task_time_left == 0.0:
            self.curr_task = self.pick_next()
            self.curr_task_time_left = (
                min(self.curr_task.timeslice, self.curr_task.remaining_time) if self.curr_task else 0.0
            )
        self.elapse_time()

    def step(self):
        next = self.pick_next()
        self.elapse_time(next)
        heapq.heapify(self.runqueue)
