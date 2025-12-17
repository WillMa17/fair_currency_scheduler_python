from Task import Task
import heapq

slowdown_factor = 0.5
speedup_factor = 2.0
sleep_slowdown_factor = 0.9

def special_event(event_name):
    return event_name == "wakeup" or event_name == "fork"

class Scheduler:
    def __init__(self, use_currency: bool = True, wakeup_grace: float = 0.0):
        self.runqueue = []
        self.waitqueue = []
        self.system_time = 0.0
        self.avg_vruntime = 0.0
        self.max_vruntime = 0.0
        self.curr_task = None
        self.curr_task_time_left = 0.0
        self.eventqueue = []
        self.slowdown_factor = slowdown_factor
        self.speedup_factor = speedup_factor
        self.use_currency = use_currency
        self.wakeup_grace = wakeup_grace
    
    def add(self, task):
        heapq.heappush(self.runqueue, (task.vdeadline, task))
        self.update_max_vruntime()
    
    def _reheap_with_current_deadlines(self):
        # Rebuild runqueue tuples so updated vdeadlines are respected.
        self.runqueue = [(t.vdeadline, t) for _, t in self.runqueue]
        heapq.heapify(self.runqueue)

    def pick_next(self):
        if not self.runqueue:
            return None
        if not self.use_currency:
            return self.runqueue[0][1]
        eligible = [(vd, t) for vd, t in self.runqueue if t.currency >= -1e-9] #floating point shenanigans
        if eligible:
            return min(eligible)[1]

        # Idea: if no eligible tasks, use earliest vdeadline (work conserving)
        return self.runqueue[0][1]
    
    def redistribute_currency(self, currency, mode):
        if not self.use_currency:
            return
        recipients = [t for _, t in self.runqueue] + list(self.waitqueue)
        if not recipients:
            return
        if mode == "weights": #we could also maybe consider rate weighing here
            total_weight = sum(t.weight for t in recipients)
            for t in recipients:
                t.currency += ((t.weight / total_weight) * currency)
        elif mode == "slowdown":
            total_weight = sum(t.slowdown_score for t in recipients)
            for t in recipients:
                t.currency += ((t.slowdown_score / total_weight) * currency)

    def task_fork(self, task):
        task.vruntime = self.avg_vruntime
        if self.use_currency:
            task.currency = -2 * task.timeslice #birth penalty
            self.redistribute_currency(2 * task.timeslice, "slowdown")
        task.compute_vdeadline()
        self.add(task)

    def task_exit(self, task):
        self.remove_from_runqueue(task)
        self.avg_vruntime = self.compute_average_vruntime()
        if self.use_currency:
            self.redistribute_currency(task.currency, "weights")
    
    def remove_from_runqueue(self, task):
        for (vd, t) in self.runqueue:
            if t == task:
                self.runqueue.remove((vd, t))
        heapq.heapify(self.runqueue)
        self.update_max_vruntime()
    
    def remove_from_waitqueue(self, task):
        if task in self.waitqueue:
            self.waitqueue.remove(task)
        self.update_max_vruntime()
    
    def compute_average_vruntime(self):
        numerator = 0.0
        denominator = 0.0
        for _, t in self.runqueue:
            numerator += (t.vruntime * t.weight)
            denominator += t.weight
        if denominator == 0.0:
            return self.avg_vruntime
        return numerator / denominator

    def min_vruntime(self):
        min_runqueue_vr = min((t.vruntime for _, t in self.runqueue), default=None)
        if min_runqueue_vr is not None:
            return min_runqueue_vr
        return self.avg_vruntime
    
    def update_max_vruntime(self):
        max_runqueue_vr = max((t.vruntime for _, t in self.runqueue), default=None)
        max_waitqueue_vr = max((t.vruntime for t in self.waitqueue), default=None)
        if max_runqueue_vr is None and max_waitqueue_vr is None:
            self.max_vruntime = self.avg_vruntime
            return self.max_vruntime
        candidates = [vr for vr in (max_runqueue_vr, max_waitqueue_vr) if vr is not None]
        self.max_vruntime = max(candidates)
        return self.max_vruntime
     
    def accrue_currency(self):
        if not self.use_currency:
            return
        # Work-conserving accrual: scale entitlements so total accrual per tick is 1
        # denom = sum_i (entitlement_i * currency_rate_i)
        if len(self.runqueue) == 0: #currency shouldn't be accumulating if there are no tasks
            return
        total_weight = sum(t.weight for _, t in self.runqueue) 
        total_weight_including_sleep = total_weight + sum(t.weight for t in self.waitqueue)
        denom = 0.0
        denom_including_sleep = 0.0
        if total_weight > 0:
            for _, t in self.runqueue:
                denom += (t.weight / total_weight) * t.currency_rate
                denom_including_sleep += (t.weight / total_weight_including_sleep) * t.currency_rate
            for t in self.waitqueue:
                denom_including_sleep += (t.weight / total_weight_including_sleep) * t.currency_rate
        k = (1.0 / denom) if denom > 0.0 else 0.0
        k_sleep = (1.0 / denom_including_sleep) if denom_including_sleep > 0.0 else 0.0
        for _, t in self.runqueue:
            if total_weight > 0:
                entitlement = t.weight / total_weight
                t.increase_currency(entitlement * k)
        for t in self.waitqueue:
            if total_weight_including_sleep > 0:
                entitlement = (t.weight / total_weight_including_sleep) * sleep_slowdown_factor * t.currency_rate
                t.increase_vruntime(entitlement * k_sleep) #how much you would theoretically run if you weren't sleeping
        self.refresh_lags()

    def elapse_time(self):
        self.system_time += 1.0
        self.curr_task_time_left = max(0.0, self.curr_task_time_left - 1.0)
        if self.curr_task:
            if self.use_currency:
                self.curr_task.increase_runtime_currency(1.0)
            else:
                self.curr_task.increase_runtime_base(1.0)

        self.avg_vruntime = self.compute_average_vruntime()
        self.update_max_vruntime()
        self.refresh_lags()
        self.accrue_currency()
        if self.curr_task:
            if self.curr_task.remaining_time == 0.0:
                self.task_exit(self.curr_task)
            else:
                self.curr_task.compute_vdeadline()

        self._reheap_with_current_deadlines()

    def refresh_lags(self):
        vr_ref = self.avg_vruntime
        for _, t in self.runqueue:
            t.compute_lag(vr_ref, self.max_vruntime)
        for t in self.waitqueue:
            t.compute_lag(vr_ref, self.max_vruntime)
    
    def handle_event(self, event):
        t = event[0]
        if event[1] == "sleep":
            self.remove_from_runqueue(t)
            self.curr_task = None
            self.curr_task_time_left = 0.0
            if self.use_currency:
                t.set_currency_rate(self.slowdown_factor, self.system_time)
            self.waitqueue.append(t)
        elif event[1] == "wakeup":
            self.remove_from_waitqueue(t)
            if self.use_currency:
                t.set_currency_rate(self.speedup_factor, self.system_time)
                t.compute_vdeadline()
            else:
                self.place_on_wakeup(t)
            self.add(t)
            self.avg_vruntime = self.compute_average_vruntime()
        elif event[1] == "slow":
            if self.use_currency:
                t.set_currency_rate(self.slowdown_factor, self.system_time)
        elif event[1] == "normal":
            if self.use_currency:
                t.set_currency_rate(self.speedup_factor, self.system_time)
        elif event[1] == "exit":
            self.task_exit(t)
            self.curr_task = None
            self.curr_task_time_left = 0.0

    def place_on_wakeup(self, task):
        # Align wakeup vruntime to avoid gifting unfair negative lag.
        target = self.min_vruntime() - self.wakeup_grace
        if task.vruntime < target:
            task.vruntime = target
        task.compute_vdeadline()
    
    def handle_events(self):
        for event in self.eventqueue[:]:
            if self.curr_task == event[0] or special_event(event[1]):
                self.handle_event(event)
                self.eventqueue.remove(event)
                
    def process_events(self):
        for _, t in self.runqueue:
            events = t.process_events(self.system_time)
            for event in events:
                self.eventqueue.append((t, event))
        for t in self.waitqueue:
            events = t.process_events(self.system_time)
            for event in events:
                self.eventqueue.append((t, event))
        self.handle_events()
        self._reheap_with_current_deadlines()
        self.update_max_vruntime()

    def tick(self):
        self.process_events()
        if self.curr_task_time_left == 0.0:
            self.curr_task = self.pick_next()
            self.curr_task_time_left = (
                min(self.curr_task.timeslice, self.curr_task.remaining_time) if self.curr_task else 0.0
            )
        self.elapse_time()


class EEVDFScheduler(Scheduler):
    def __init__(self, wakeup_grace: float = 0.0):
        super().__init__(use_currency=False, wakeup_grace=wakeup_grace)
