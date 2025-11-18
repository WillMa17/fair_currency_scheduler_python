from Task import Task
import heapq

slowdown_factor = 0.5
speedup_factor = 2.0

def special_event(event_name):
    return event_name == "wakeup" or event_name == "fork"

class Scheduler:
    def __init__(self):
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
        return self.runqueue[0][1] if self.runqueue else None
        # Idea 2: if no eligible task, wait until a task is eligible 
        # (no this is actually dumb)
        # for vd, t in self.runqueue:
        #     print(t.currency)
        # return None
    
    def redistribute_currency(self, currency, mode):
        print("redistributing")
        if mode == "weights": #we could also maybe consider rate weighing here
            total_weight = sum(t.weight for _, t in self.runqueue)
            for _, t in self.runqueue:
                t.currency += ((t.weight / total_weight) * currency)
        elif mode == "slowdown":
            total_weight = sum(t.slowdown_score for _, t in self.runqueue)
            for _, t in self.runqueue:
                t.currency += ((t.slowdown_score / total_weight) * currency)

    def task_fork(self, task):
        task.vruntime = self.avg_vruntime
        task.currency = -2 * task.timeslice #birth penalty
        self.redistribute_currency(2 * task.timeslice, "slowdown")
        self.add(task)

    def task_exit(self, task):
        self.remove_from_runqueue(task)
        self.avg_vruntime = self.compute_average_vruntime()
        self.redistribute_currency(task.currency, "weights")
    
    def remove_from_runqueue(self, task):
        for (vd, t) in self.runqueue:
            if t == task:
                self.runqueue.remove((vd, t))
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
     
    def accrue_currency(self):
        # Work-conserving accrual: scale entitlements so total accrual per tick is 1
        # denom = sum_i (entitlement_i * currency_rate_i)
        if len(self.runqueue) == 0: #currency shouldn't be accumulating if there are no tasks
            return
        total_weight = sum(t.weight for _, t in self.runqueue) + sum(t.weight for t in self.waitqueue)
        for _, t in self.runqueue:
            t.compute_lag(self.avg_vruntime)
        denom = 0.0
        if total_weight > 0:
            for _, t in self.runqueue:
                denom += (t.weight / total_weight) * t.currency_rate
            for t in self.waitqueue:
                if t.currency < t.timeslice:
                    denom += (t.weight / total_weight) * t.currency_rate
        k = (1.0 / denom) if denom > 0.0 else 0.0
        for _, t in self.runqueue:
            if total_weight > 0:
                entitlement = t.weight / total_weight
                t.increase_currency(entitlement * k)
        for t in self.waitqueue:
            if total_weight > 0 and t.currency < t.timeslice:
                entitlement = t.weight / total_weight
                t.increase_currency(entitlement * k)

    def elapse_time(self):
        self.system_time += 1.0
        self.curr_task_time_left = max(0.0, self.curr_task_time_left - 1.0)
        if self.curr_task:
            self.curr_task.increase_runtime(1.0)

        self.avg_vruntime = self.compute_average_vruntime()
        self.accrue_currency()
        if self.curr_task:
            if self.curr_task.remaining_time == 0.0:
                self.task_exit(self.curr_task)
            else:
                self.curr_task.compute_vdeadline()

        heapq.heapify(self.runqueue)
    
    def handle_event(self, event):
        t = event[0]
        if event[1] == "sleep":
            self.remove_from_runqueue(t)
            self.curr_task = None
            self.curr_task_time_left = 0.0
            t.reset_currency_rate()
            self.waitqueue.append(t)
        elif event[1] == "wakeup":
            self.remove_from_waitqueue(t)
            t.vruntime = max(t.vruntime, self.avg_vruntime)
            self.add(t)
            self.avg_vruntime = self.compute_average_vruntime()
        elif event[1] == "slow":
            t.set_currency_rate(self.slowdown_factor)
        elif event[1] == "normal":
            t.set_currency_rate(self.speedup_factor)
        elif event[1] == "exit":
            self.task_exit(t)
            self.curr_task = None
            self.curr_task_time_left = 0.0
    
    def handle_events(self):
        for event in self.eventqueue[:]:
            if self.curr_task == event[0] or special_event(event[1]):
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
