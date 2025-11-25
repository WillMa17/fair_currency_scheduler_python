from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

from Task import Task
from Scheduler import Scheduler


@dataclass
class ExperimentConfig:
    num_ticks: int = 150
    num_tasks: int = 3
    total_work: float = 1e9
    timeslice: float = 1.0
    starting_currency: float = 0.0


def experiment_base(cfg: ExperimentConfig) -> Tuple[Scheduler, List[Task], List[Task]]:
    sched = Scheduler()
    tasks: List[Task] = []
    for pid in range(cfg.num_tasks):
        t = Task(
            sort_key=0.0,
            pid=pid,
            total_work=cfg.total_work,
        )
        t.update_weight_simple()
        t.timeslice = cfg.timeslice
        t.currency = cfg.starting_currency
        t.compute_vdeadline()
        tasks.append(t)
        sched.add(t)
    return sched, tasks, []


def experiment_slowdown(cfg: ExperimentConfig) -> Tuple[Scheduler, List[Task], List[Task]]:
    sched, tasks, [] = experiment_base(cfg)
    if len(tasks) > 1:
        tasks[0].state_changes = [(0.0, "slow"), (50.0, "normal")]
        tasks[1].state_changes = [(30.0, "slow"), (70.0, "normal")]
        tasks[2].state_changes = [(40.0, "slow"), (90.0, "normal")]
    return sched, tasks, []


def experiment_sleep(cfg: ExperimentConfig) -> Tuple[Scheduler, List[Task], List[Task]]:
    sched, tasks, [] = experiment_base(cfg)
    if len(tasks) > 1:
        tasks[0].state_changes = [(10.0, "sleep"), (50.0, "wakeup")]
        tasks[1].state_changes = [(20.0, "slow"), (60.0, "normal")]
    return sched, tasks, []


def experiment_fork(cfg: ExperimentConfig) -> Tuple[Scheduler, List[Task], List[Task]]:
    sched, tasks, [] = experiment_base(cfg)
    incoming_tasks = []
    for pid in range(3, 6):
        t = Task(
            sort_key=0.0,
            pid=pid,
            total_work=cfg.total_work
        )
        t.fork_time = pid * 20.0
        t.update_weight_simple()
        t.timeslice = cfg.timeslice
        t.currency = cfg.starting_currency
        t.compute_vdeadline()
        incoming_tasks.append(t)
    incoming_tasks[0].state_changes = [(105.0, "sleep"), (140.0, "wakeup")]
    tasks[0].state_changes = [(30.0, "slow"), (120.0, "normal")]
    return sched, tasks, incoming_tasks


def run_experiment(setup: Callable[[ExperimentConfig], Tuple[Scheduler, List[Task], List[Task]]], cfg: ExperimentConfig) -> Dict:
    sched, starting_tasks, incoming_tasks = setup(cfg)
    tasks = starting_tasks + incoming_tasks
    
    time_axis: List[int] = []
    total_currency: List[float] = []
    vruntimes: List[List[float]] = [[] for _ in tasks]
    currency_rates: List[List[float]] = [[] for _ in tasks]
    currency_levels: List[List[float]] = [[] for _ in tasks]

    time_axis.append(0)
    total_currency.append(sum(t.currency for t in tasks))
    for i, t in enumerate(tasks):
        vruntimes[i].append(t.vruntime)
        currency_rates[i].append(t.currency_rate)
        currency_levels[i].append(t.currency)

    for tick in range(1, cfg.num_ticks + 1):
        for t in incoming_tasks:
            if t.fork_time <= tick:
                sched.task_fork(t)
                incoming_tasks.remove(t)

        sched.tick()
        time_axis.append(tick)
        total_currency.append(round(sum(t.currency for t in tasks), 9))
        for i, t in enumerate(tasks):
            vruntimes[i].append(t.vruntime)
            currency_rates[i].append(t.currency_rate)
            currency_levels[i].append(t.currency)

    return {
        "time": time_axis,
        "total_currency": total_currency,
        "vruntimes": vruntimes,
        "currency_rates": currency_rates,
        "currency_levels": currency_levels,
        "tasks": tasks,
        "scheduler": sched,
        "config": cfg,
    }


def plot_results(results: Dict, outfile: str = "model_output.png") -> None:
    import matplotlib.pyplot as plt

    time_axis = results["time"]
    total_currency = results["total_currency"]
    vruntimes = results["vruntimes"]
    currency_rates = results["currency_rates"]
    currency_levels = results["currency_levels"]

    fig, axes = plt.subplots(4, 1, figsize=(10, 12), sharex=True)

    # Total currency
    axes[0].plot(time_axis, total_currency, label="total_currency")
    axes[0].set_ylabel("Total Currency")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Per-task vruntime
    for i in range(len(vruntimes)):
        axes[1].plot(time_axis, vruntimes[i], label=f"task_{i} vruntime")
    axes[1].set_ylabel("vruntime")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Per-task currency_rate
    for i in range(len(currency_rates)):
        axes[2].plot(time_axis, currency_rates[i], label=f"task_{i} rate")
    axes[2].set_ylabel("currency_rate")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    # Per-task currency levels
    for i in range(len(currency_levels)):
        axes[3].plot(time_axis, currency_levels[i], label=f"task_{i} currency")
    axes[3].set_ylabel("currency")
    axes[3].set_xlabel("tick")
    axes[3].legend()
    axes[3].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(outfile, dpi=120)
    plt.close(fig)
    print(f"Saved plot to {outfile}")


def main():
    cfg = ExperimentConfig()
    results = run_experiment(experiment_fork, cfg)
    plot_results(results, outfile="model_output_fork.png")


if __name__ == "__main__":
    main()
