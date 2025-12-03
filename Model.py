from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple
import statistics
import random
import csv

from Task import Task
from Scheduler import Scheduler, EEVDFScheduler


@dataclass
class ExperimentConfig:
    num_ticks: int = 300
    num_tasks: int = 3
    total_work: float = 1e9
    timeslice: float = 1.0
    starting_currency: float = 0.0
    seed: int = 100
    max_incoming: int = 3
    timeslice_range: Tuple[float, float] = (0.5, 2.0)
    work_range: Tuple[float, float] = (5e8, 1e9)
    slow_prob: float = 0.4
    sleep_prob: float = 0.3
    slow_len_range: Tuple[int, int] = (10, 40)
    sleep_len_range: Tuple[int, int] = (10, 40)
    incoming_prob: float = 0.5
    base_range: Tuple[int, int] = (1, 4)
    incoming_range: Tuple[int, int] = (0, 4)
    incoming_latest_frac: float = 0.8
    max_phases: int = 4


def experiment_base(cfg: ExperimentConfig, scheduler_factory: Callable[[], Scheduler] = Scheduler) -> Tuple[Scheduler, List[Task], List[Task]]:
    sched = scheduler_factory()
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


def experiment_slowdown(cfg: ExperimentConfig, scheduler_factory: Callable[[], Scheduler] = Scheduler) -> Tuple[Scheduler, List[Task], List[Task]]:
    sched, tasks, [] = experiment_base(cfg, scheduler_factory)
    if len(tasks) > 1:
        tasks[0].state_changes = [(0.0, "slow"), (50.0, "normal")]
        tasks[1].state_changes = [(30.0, "slow"), (70.0, "normal")]
        tasks[2].state_changes = [(40.0, "slow"), (90.0, "normal")]
    return sched, tasks, []


def experiment_sleep(cfg: ExperimentConfig, scheduler_factory: Callable[[], Scheduler] = Scheduler) -> Tuple[Scheduler, List[Task], List[Task]]:
    sched, tasks, [] = experiment_base(cfg, scheduler_factory)
    if len(tasks) > 1:
        tasks[0].state_changes = [(10.0, "sleep"), (50.0, "wakeup")]
        tasks[1].state_changes = [(20.0, "slow"), (60.0, "normal")]
    return sched, tasks, []


def experiment_fork(cfg: ExperimentConfig, scheduler_factory: Callable[[], Scheduler] = Scheduler) -> Tuple[Scheduler, List[Task], List[Task]]:
    sched, tasks, [] = experiment_base(cfg, scheduler_factory)
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
    tasks[0].state_changes = [(30.0, "slow"), (120.0, "normal")] + [(200.0, "slow"), (201.0, "normal"), (240.0, "slow"), (241.0, "normal")]
    tasks[1].state_changes = [(15.0, "sleep"), (30.0, "wakeup"), (45.0, "sleep"), (70.0, "wakeup"),
                              (75.0, "slow"), (100.0, "normal"), (125.0, "sleep"), (160.0, "wakeup")] + [(200.0, "slow"), (201.0, "normal"), (240.0, "slow"), (241.0, "normal")]
    
    for task in incoming_tasks:
        task.state_changes += [(200.0, "slow"), (201.0, "normal"), (240.0, "slow"), (241.0, "normal")]
    return sched, tasks, incoming_tasks


def _random_window(rng: random.Random, max_tick: int, len_range: Tuple[int, int]) -> Optional[Tuple[float, float]]:
    if max_tick <= 1:
        return None
    min_len, max_len = len_range
    if min_len >= max_tick:
        return None
    start_upper = max_tick - min_len
    start = rng.randint(1, start_upper)
    dur_upper = min(max_len, max_tick - start)
    if dur_upper < min_len:
        return None
    dur = rng.randint(min_len, dur_upper)
    end = start + dur
    return float(start), float(end)


def _random_state_changes(rng: random.Random, cfg: ExperimentConfig) -> List[Tuple[float, str]]:
    events: List[Tuple[float, str]] = []
    phases = rng.randint(1, cfg.max_phases)
    for _ in range(phases):
        mode = rng.choice(["slow", "sleep", "slow_then_sleep", "sleep_then_slow"])
        if mode == "slow":
            win = _random_window(rng, cfg.num_ticks, cfg.slow_len_range) if rng.random() < cfg.slow_prob else None
            if win:
                events.extend([(win[0], "slow"), (win[1], "normal")])
        elif mode == "sleep":
            win = _random_window(rng, cfg.num_ticks, cfg.sleep_len_range) if rng.random() < cfg.sleep_prob else None
            if win:
                events.extend([(win[0], "sleep"), (win[1], "wakeup")])
        elif mode == "slow_then_sleep":
            slow_win = _random_window(rng, cfg.num_ticks, cfg.slow_len_range) if rng.random() < cfg.slow_prob else None
            sleep_win = _random_window(rng, cfg.num_ticks, cfg.sleep_len_range) if rng.random() < cfg.sleep_prob else None
            if slow_win:
                events.extend([(slow_win[0], "slow"), (slow_win[1], "normal")])
            if sleep_win:
                events.extend([(sleep_win[0], "sleep"), (sleep_win[1], "wakeup")])
        else:  # sleep_then_slow
            sleep_win = _random_window(rng, cfg.num_ticks, cfg.sleep_len_range) if rng.random() < cfg.sleep_prob else None
            slow_win = _random_window(rng, cfg.num_ticks, cfg.slow_len_range) if rng.random() < cfg.slow_prob else None
            if sleep_win:
                events.extend([(sleep_win[0], "sleep"), (sleep_win[1], "wakeup")])
            if slow_win:
                events.extend([(slow_win[0], "slow"), (slow_win[1], "normal")])
    events.sort(key=lambda x: x[0])
    return events


def _describe_task(prefix: str, t: Task) -> None:
    events = ", ".join([f"{int(ts)}:{name}" for ts, name in t.state_changes]) if t.state_changes else "none"
    fork = getattr(t, "fork_time", None)
    fork_txt = f" fork_at={fork}" if fork is not None else ""
    print(f"{prefix} pid={t.pid} work={t.total_work:.0f} tslice={t.timeslice:.2f}{fork_txt} events=[{events}]")


def experiment_random(cfg: ExperimentConfig, scheduler_factory: Callable[[], Scheduler] = Scheduler) -> Tuple[Scheduler, List[Task], List[Task]]:
    rng = random.Random(cfg.seed)
    sched = scheduler_factory()
    tasks: List[Task] = []
    incoming_tasks: List[Task] = []

    base_count = rng.randint(cfg.base_range[0], cfg.base_range[1])
    incoming_budget = rng.randint(cfg.incoming_range[0], cfg.incoming_range[1])

    print(f"[random] seed={cfg.seed} base_tasks={base_count} incoming_budget={incoming_budget}")

    for pid in range(base_count):
        t = Task(
            sort_key=0.0,
            pid=pid,
            total_work=rng.uniform(*cfg.work_range),
        )
        t.update_weight_simple()
        t.timeslice = rng.uniform(*cfg.timeslice_range)
        t.currency = cfg.starting_currency
        t.state_changes = _random_state_changes(rng, cfg)
        t.compute_vdeadline()
        tasks.append(t)
        sched.add(t)
        _describe_task("[task]", t)

    for pid in range(base_count, base_count + incoming_budget):
        t = Task(
            sort_key=0.0,
            pid=pid,
            total_work=rng.uniform(*cfg.work_range),
        )
        latest_fork = max(1, int(cfg.num_ticks * cfg.incoming_latest_frac))
        t.fork_time = rng.randint(1, latest_fork)
        t.update_weight_simple()
        t.timeslice = rng.uniform(*cfg.timeslice_range)
        t.currency = cfg.starting_currency
        t.state_changes = _random_state_changes(rng, cfg)
        t.compute_vdeadline()
        incoming_tasks.append(t)
        _describe_task("[incoming]", t)

    return sched, tasks, incoming_tasks


def run_experiment(setup: Callable[[ExperimentConfig, Callable[[], Scheduler]], Tuple[Scheduler, List[Task], List[Task]]], cfg: ExperimentConfig, scheduler_factory: Callable[[], Scheduler] = Scheduler) -> Dict:
    sched, starting_tasks, incoming_tasks = setup(cfg, scheduler_factory)
    tasks = starting_tasks + incoming_tasks
    
    time_axis: List[int] = []
    total_currency: List[float] = []
    vruntimes: List[List[float]] = [[] for _ in tasks]
    currency_rates: List[List[float]] = [[] for _ in tasks]
    currency_levels: List[List[float]] = [[] for _ in tasks]
    completion_times: List[Optional[int]] = [None for _ in tasks]

    time_axis.append(0)
    total_currency.append(sum(t.currency for t in tasks))
    for i, t in enumerate(tasks):
        vruntimes[i].append(t.vruntime)
        currency_rates[i].append(t.currency_rate)
        currency_levels[i].append(t.currency)

    for tick in range(1, cfg.num_ticks + 1):
        for t in list(incoming_tasks):
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
            if completion_times[i] is None and t.remaining_time <= 0.0:
                completion_times[i] = tick

    return {
        "time": time_axis,
        "total_currency": total_currency,
        "vruntimes": vruntimes,
        "currency_rates": currency_rates,
        "currency_levels": currency_levels,
        "tasks": tasks,
        "scheduler": sched,
        "config": cfg,
        "completion_times": completion_times,
    }


def run_comparison(setup: Callable[[ExperimentConfig, Callable[[], Scheduler]], Tuple[Scheduler, List[Task], List[Task]]], cfg: ExperimentConfig, wakeup_grace: float = 0.0) -> Tuple[Dict, Dict]:
    cur_sched = lambda: Scheduler(use_currency=True, wakeup_grace=wakeup_grace)
    eevdf_sched = lambda: EEVDFScheduler(wakeup_grace=wakeup_grace)
    results_currency = run_experiment(setup, cfg, scheduler_factory=cur_sched)
    results_eevdf = run_experiment(setup, cfg, scheduler_factory=eevdf_sched)
    return results_currency, results_eevdf


def plot_results(results: Dict, outfile: str = "model_output.png") -> None:
    def _plot_panel(axis_list, res: Dict, title: Optional[str] = None):
        time_axis = res["time"]
        vruntimes = res["vruntimes"]
        currency_rates = res["currency_rates"]
        currency_levels = res["currency_levels"]
        total_currency = res["total_currency"]

        if title:
            axis_list[0].set_title(title)

        axis_list[0].plot(time_axis, total_currency, label="total_currency")
        axis_list[0].set_ylabel("Total Currency")
        axis_list[0].legend()
        axis_list[0].grid(True, alpha=0.3)

        for i in range(len(vruntimes)):
            axis_list[1].plot(time_axis, vruntimes[i], label=f"task_{i} vruntime")
        axis_list[1].set_ylabel("vruntime")
        axis_list[1].legend()
        axis_list[1].grid(True, alpha=0.3)

        for i in range(len(currency_rates)):
            axis_list[2].plot(time_axis, currency_rates[i], label=f"task_{i} rate")
        axis_list[2].set_ylabel("currency_rate")
        axis_list[2].legend()
        axis_list[2].grid(True, alpha=0.3)

        for i in range(len(currency_levels)):
            axis_list[3].plot(time_axis, currency_levels[i], label=f"task_{i} currency")
        axis_list[3].set_ylabel("currency")
        axis_list[3].set_xlabel("tick")
        axis_list[3].legend()
        axis_list[3].grid(True, alpha=0.3)

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(4, 1, figsize=(10, 12), sharex=True)
    _plot_panel(axes, results)

    plt.tight_layout()
    plt.savefig(outfile, dpi=120)
    plt.close(fig)
    print(f"Saved plot to {outfile}")


def plot_side_by_side(results_left: Dict, results_right: Dict, labels: Tuple[str, str] = ("Currency", "EEVDF"), outfile: str = "model_output_comparison.png") -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(4, 2, figsize=(14, 12), sharex="row")

    def _panel(axis_list, res: Dict, title: str):
        time_axis = res["time"]
        vruntimes = res["vruntimes"]
        currency_rates = res["currency_rates"]
        currency_levels = res["currency_levels"]
        total_currency = res["total_currency"]

        axis_list[0].plot(time_axis, total_currency, label="total_currency")
        axis_list[0].set_ylabel("Total Currency")
        axis_list[0].legend()
        axis_list[0].grid(True, alpha=0.3)
        axis_list[0].set_title(title)

        for i in range(len(vruntimes)):
            axis_list[1].plot(time_axis, vruntimes[i], label=f"task_{i} vruntime")
        axis_list[1].set_ylabel("vruntime")
        axis_list[1].legend()
        axis_list[1].grid(True, alpha=0.3)

        for i in range(len(currency_rates)):
            axis_list[2].plot(time_axis, currency_rates[i], label=f"task_{i} rate")
        axis_list[2].set_ylabel("currency_rate")
        axis_list[2].legend()
        axis_list[2].grid(True, alpha=0.3)

        for i in range(len(currency_levels)):
            axis_list[3].plot(time_axis, currency_levels[i], label=f"task_{i} currency")
        axis_list[3].set_ylabel("currency")
        axis_list[3].set_xlabel("tick")
        axis_list[3].legend()
        axis_list[3].grid(True, alpha=0.3)

    _panel(axes[:, 0], results_left, labels[0])
    _panel(axes[:, 1], results_right, labels[1])

    plt.tight_layout()
    plt.savefig(outfile, dpi=120)
    plt.close(fig)
    print(f"Saved side-by-side plot to {outfile}")


def summarize_results(results: Dict, label: str, scheduler_name: str) -> List[Dict]:
    """Flatten per-task stats for CSV/console consumption."""
    cfg = results["config"]
    completion_times = results.get("completion_times", [])
    rows: List[Dict] = []
    for i, t in enumerate(results["tasks"]):
        curr_series = results["currency_levels"][i]
        vr_series = results["vruntimes"][i]
        rows.append({
            "label": label,
            "scheduler": scheduler_name,
            "pid": t.pid,
            "finished": t.remaining_time <= 0.0,
            "completion_tick": completion_times[i] if completion_times else None,
            "avg_currency": statistics.mean(curr_series),
            "min_currency": min(curr_series),
            "max_currency": max(curr_series),
            "final_currency": curr_series[-1],
            "final_vruntime": vr_series[-1],
            "num_ticks": cfg.num_ticks,
            "num_tasks": cfg.num_tasks,
            "timeslice": cfg.timeslice,
            "total_work": cfg.total_work,
        })
    return rows


def write_summary_csv(rows: List[Dict], outfile: str) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(outfile, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"Wrote summary CSV to {outfile}")


def run_suite(setup: Callable[[ExperimentConfig, Callable[[], Scheduler]], Tuple[Scheduler, List[Task], List[Task]]], cfgs: List[ExperimentConfig], wakeup_grace: float = 0.0, label: str = "") -> List[Dict]:
    all_rows: List[Dict] = []
    for idx, cfg in enumerate(cfgs):
        tag = label or f"{setup.__name__}_{idx}"
        res_cur, res_eevdf = run_comparison(setup, cfg, wakeup_grace=wakeup_grace)
        all_rows.extend(summarize_results(res_cur, tag, "currency"))
        all_rows.extend(summarize_results(res_eevdf, tag, "eevdf"))
    return all_rows


def main():
    cfg = ExperimentConfig()
    results_currency, results_eevdf = run_comparison(experiment_fork, cfg)
    plot_results(results_currency, outfile="model_output_fork_currency.png")
    plot_results(results_eevdf, outfile="model_output_fork_eevdf.png")
    plot_side_by_side(results_currency, results_eevdf, outfile="model_output_fork_comparison.png")

    # Example bulk run: generate CSV summaries without looking at plots
    suite_rows = run_suite(
        experiment_fork,
        [
            ExperimentConfig(num_ticks=150, num_tasks=3, timeslice=1.0),
            ExperimentConfig(num_ticks=200, num_tasks=4, timeslice=1.0),
        ],
        wakeup_grace=0.0,
        label="fork_suite",
    )
    write_summary_csv(suite_rows, outfile="model_output_fork_suite.csv")

    # Example random experiment; prints generated state changes and writes summaries
    rand_cfg = ExperimentConfig(num_ticks=200, num_tasks=5, max_incoming=2, seed=100)
    rand_cur, rand_eevdf = run_comparison(experiment_random, rand_cfg)
    plot_side_by_side(rand_cur, rand_eevdf, outfile="model_output_random_comparison.png")
    write_summary_csv(run_suite(experiment_random, [rand_cfg], label="random_123"), outfile="model_output_random_suite.csv")


if __name__ == "__main__":
    main()
