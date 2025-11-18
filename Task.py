from dataclasses import dataclass, field
from typing import List, Tuple

@dataclass(order=True)
class Task:
    sort_key: float
    pid: int = field(compare=False)
    total_work: float = field(compare=False)
    fork_time: float = field(default=0.0, compare=False)
    remaining_time: float = field(default=0.0, compare=False)
    executed_time: float = field(default=0, compare=False)
    nice: int = field(default=0, compare=False)
    weight: float = field(default=1.0, compare=False)
    currency: float = field(default=0.0, compare=False)
    timeslice: float = field(default=0.0, compare=False)
    vruntime: float = field(default=0.0, compare=False)
    vdeadline: float = field(default=0.0, compare=False)
    currency_rate: float = field(default=1.0, compare=False)
    baseline_currency_rate: float = field(default=1.0, compare=False)
    lag: float = field(default=0.0, compare=False)
    #4 types of events: sleep, slow, normal, wakeup
    #special event: fork, exit
    state_changes: List[Tuple[float, str]] = field(default_factory=list, compare=False)
    slowdown_score: float = field(default=1.0, compare=False)

    def __post_init__(self):
        self.reset_task()
    
    def __eq__(self, other):
        if not isinstance(other, Task):
            return NotImplemented
        return self.pid == other.pid
    
    def process_events(self, time):
        if len(self.state_changes) == 0:
            return None
        next_event = self.state_changes[0]
        if next_event[0] == time:
            return self.state_changes.pop(0)[1]
        else:
            return None

    def reset_task(self):
        self.remaining_time = self.total_work
        self.currency_rate = self.baseline_currency_rate
        self.slowdown_score = 1.0
    
    def update_weight(self):
        self.weight = 1024 / (1.25 ** self.nice) #this is how the weight would generally look in eevdf

    def update_weight_simple(self):
        self.weight = 1.0 #we can use this one for simplicity

    def compute_vdeadline(self):
        self.vdeadline = self.vruntime + self.timeslice
    
    def reset_currency_rate(self):
        self.currency_rate = self.baseline_currency_rate

    def compute_lag(self, avg_vruntime):
        self.lag = self.weight * (avg_vruntime - self.vruntime)
        if self.lag < 0 and self.currency_rate > self.baseline_currency_rate:
            self.reset_currency_rate()
    
    def increase_currency(self, run):
        self.currency += run * self.currency_rate
        if abs(self.currency) < 1e-9:
            self.currency = 0.0
        if self.currency_rate < self.baseline_currency_rate:
            self.slowdown_score += 0.1
    
    def set_currency_rate(self, rate):
        if rate > 1 and self.currency_rate >= self.baseline_currency_rate:
            return
        self.currency_rate = rate
        print(rate)
    
    def increase_runtime(self, run):
        self.remaining_time = max(0.0, self.remaining_time - run)
        self.vruntime += run / self.weight
        self.currency -= run
