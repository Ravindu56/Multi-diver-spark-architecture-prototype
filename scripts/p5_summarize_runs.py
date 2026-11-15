"""Summarise P5 runner output (scripts/p5_run_cell.sh) per configuration.

Reads one or more runner directories ($HOME/p5-runs/<timestamp>-cell<X>) and, for each
(workload, cell, param) configuration, aggregates the measured repetitions: rep >= 1 with
rc 0. Rep 0 is the pilot and is excluded unless --include-pilot is given; failed runs are
counted and excluded from timing.

Per configuration it reports:
  - wall_s           runner wall time (mpirun + container exec), mean/std over reps
  - app_s            application total_time_s from rank 0's run JSON, mean/std
  - startup_s        wall_s - app_s (process startup outside the application)
  - setup_s          app_s - sum of iteration/epoch times (partition, Spark session, load)
  - per-iteration means over all ranks and iterations: spark, loss pass (LogReg only),
    sync and iteration time, and sync as a percentage of iteration time
  - iterations run, whether every rep converged, whether every rep ended with the same
    final model on all ranks, and the distinct final objective values
    (LogReg global_loss, K-Means global_wcss)

Usage:
    python scripts/p5_summarize_runs.py ~/p5-runs/20261115-192531-cellA ~/p5-runs/...-cellB
    python scripts/p5_summarize_runs.py RUN_DIR... --csv summary.csv

Exit status: 0 on success, 2 on bad input (no readable runlog.csv).
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import statistics as st
import sys
from dataclasses import dataclass, field

# Per-iteration metric files and their column names, by workload.
WORKLOADS = {
    "logreg": {
        "iter_glob": "logreg_rank*_epochs.csv",
        "run_json": "logreg_rank0_run.json",
        "iters_key": "epochs_run",
        "spark": "spark_time_s",
        "loss": "loss_time_s",
        "sync": "sync_time_s",
        "iter": "epoch_time_s",
        "final": ("global_loss", "weight_norm"),
        "objective": "global_loss",
    },
    "kmeans": {
        "iter_glob": "kmeans_metrics_rank*.csv",
        "run_json": "kmeans_metrics_rank0.json",
        "iters_key": "iterations_run",
        "spark": "spark_time_s",
        "loss": None,
        "sync": "sync_time_s",
        "iter": "iter_time_s",
        "final": ("centroid_shift", "global_wcss"),
        "objective": "global_wcss",
    },
}

CSV_FIELDS = [
    "workload",
    "cell",
    "param",
    "np",
    "reps",
    "failed",
    "wall_s_mean",
    "wall_s_std",
    "app_s_mean",
    "app_s_std",
    "startup_s_mean",
    "setup_s_mean",
    "spark_s",
    "loss_s",
    "sync_s",
    "iter_s",
    "sync_pct",
    "iterations",
    "all_converged",
    "same_model_all_ranks",
    "final_objective",
]


@dataclass
class Config:
    workload: str
    cell: str
    param: str
    np: str
    walls: list[float] = field(default_factory=list)
    apps: list[float] = field(default_factory=list)
    setups: list[float] = field(default_factory=list)
    iterations: list[int] = field(default_factory=list)
    converged: list[bool] = field(default_factory=list)
    same_model: list[bool] = field(default_factory=list)
    objectives: set[str] = field(default_factory=set)
    spark: list[float] = field(default_factory=list)
    loss: list[float] = field(default_factory=list)
    sync: list[float] = field(default_factory=list)
    iters: list[float] = field(default_factory=list)
    failed: int = 0


def _mean(v: list[float]) -> float | None:
    return st.mean(v) if v else None


def _std(v: list[float]) -> float | None:
    return st.stdev(v) if len(v) > 1 else (0.0 if v else None)


def _metrics_dir(run_dir: str, name: str) -> str:
    """The runner copies /data/results/p5-04/<name> into <run_dir>/<name>/<name>."""
    nested = os.path.join(run_dir, name, name)
    return nested if os.path.isdir(nested) else os.path.join(run_dir, name)


def _read_csv(path: str) -> list[dict[str, str]]:
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def add_run(cfg: Config, run_dir: str, row: dict[str, str]) -> None:
    """Add one successful run's metrics to its configuration."""
    spec = WORKLOADS[cfg.workload]
    mdir = _metrics_dir(run_dir, row["name"])
    with open(os.path.join(mdir, spec["run_json"])) as fh:
        run = json.load(fh)
    run = run.get("run", run)  # K-Means nests the summary under "run"

    rank_files = sorted(glob.glob(os.path.join(mdir, spec["iter_glob"])))
    if not rank_files:
        raise FileNotFoundError(f"no {spec['iter_glob']} in {mdir}")

    finals = set()
    iter_sum_rank0 = None
    for path in rank_files:
        rows = _read_csv(path)
        if not rows:
            raise ValueError(f"empty metrics file {path}")
        finals.add(tuple(rows[-1][k] for k in spec["final"]))
        cfg.objectives.add(rows[-1][spec["objective"]])
        for r in rows:
            cfg.spark.append(float(r[spec["spark"]]))
            cfg.sync.append(float(r[spec["sync"]]))
            cfg.iters.append(float(r[spec["iter"]]))
            if spec["loss"] and r.get(spec["loss"], "") != "":
                cfg.loss.append(float(r[spec["loss"]]))
        if iter_sum_rank0 is None:  # first file is rank 0 (sorted)
            iter_sum_rank0 = sum(float(r[spec["iter"]]) for r in rows)

    app = float(run["total_time_s"])
    cfg.walls.append(float(row["wall_s"]))
    cfg.apps.append(app)
    cfg.setups.append(app - iter_sum_rank0)
    cfg.iterations.append(int(run[spec["iters_key"]]))
    cfg.converged.append(bool(run["converged"]))
    cfg.same_model.append(len(finals) == 1)


def summarize(run_dirs: list[str], include_pilot: bool = False) -> list[Config]:
    configs: dict[tuple[str, str, str], Config] = {}
    for run_dir in run_dirs:
        log = os.path.join(run_dir, "runlog.csv")
        if not os.path.isfile(log):
            raise FileNotFoundError(f"no runlog.csv in {run_dir}")
        for row in _read_csv(log):
            if row["workload"] not in WORKLOADS:
                continue
            if int(row["rep"]) == 0 and not include_pilot:
                continue
            key = (row["workload"], row["cell"], row["param"])
            cfg = configs.setdefault(key, Config(*key, np=row["np"]))
            if int(row["rc"]) != 0:
                cfg.failed += 1
                continue
            add_run(cfg, run_dir, row)
    return sorted(configs.values(), key=lambda c: (c.workload, _param_key(c.param), c.cell))


def _param_key(param: str) -> tuple[int, str]:
    digits = "".join(ch for ch in param if ch.isdigit())
    return (int(digits) if digits else 0, param)


def to_row(c: Config) -> dict[str, object]:
    def r(x: float | None, nd: int) -> object:
        return "" if x is None else round(x, nd)

    iter_mean = _mean(c.iters)
    sync_mean = _mean(c.sync)
    wall, app = _mean(c.walls), _mean(c.apps)
    return {
        "workload": c.workload,
        "cell": c.cell,
        "param": c.param,
        "np": c.np,
        "reps": len(c.apps),
        "failed": c.failed,
        "wall_s_mean": r(wall, 1),
        "wall_s_std": r(_std(c.walls), 1),
        "app_s_mean": r(app, 2),
        "app_s_std": r(_std(c.apps), 2),
        "startup_s_mean": r(None if wall is None else wall - app, 2),
        "setup_s_mean": r(_mean(c.setups), 2),
        "spark_s": r(_mean(c.spark), 3),
        "loss_s": r(_mean(c.loss), 3),
        "sync_s": r(sync_mean, 3),
        "iter_s": r(iter_mean, 3),
        "sync_pct": r(None if not iter_mean else 100.0 * sync_mean / iter_mean, 1),
        "iterations": ",".join(map(str, c.iterations)),
        "all_converged": all(c.converged) if c.converged else "",
        "same_model_all_ranks": all(c.same_model) if c.same_model else "",
        "final_objective": ";".join(sorted(c.objectives)),
    }


def print_table(rows: list[dict[str, object]]) -> None:
    head = (
        f"{'workload':8} {'cell':4} {'param':7} {'reps':>4} {'fail':>4} {'wall_s':>13} "
        f"{'app_s':>15} {'startup':>7} {'setup':>7} | {'spark':>6} {'loss':>6} {'sync':>6} "
        f"{'iter':>6} {'sync%':>5} | {'iters':10} {'conv':5} {'same':5} objective"
    )
    print(head)
    for x in rows:
        print(
            f"{x['workload']:8} {x['cell']:4} {x['param']:7} {x['reps']:>4} {x['failed']:>4} "
            f"{str(x['wall_s_mean']) + '±' + str(x['wall_s_std']):>13} "
            f"{str(x['app_s_mean']) + '±' + str(x['app_s_std']):>15} "
            f"{x['startup_s_mean']!s:>7} {x['setup_s_mean']!s:>7} | "
            f"{x['spark_s']!s:>6} {x['loss_s']!s:>6} {x['sync_s']!s:>6} {x['iter_s']!s:>6} "
            f"{x['sync_pct']!s:>5} | {x['iterations']:10} {x['all_converged']!s:5} "
            f"{x['same_model_all_ranks']!s:5} {x['final_objective']}"
        )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("run_dirs", nargs="+", help="p5_run_cell.sh output directories")
    ap.add_argument("--csv", help="also write the summary to this CSV file")
    ap.add_argument(
        "--include-pilot", action="store_true", help="include rep 0 (the pilot) in the summary"
    )
    args = ap.parse_args(argv)
    try:
        configs = summarize(args.run_dirs, include_pilot=args.include_pilot)
    except (FileNotFoundError, ValueError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    rows = [to_row(c) for c in configs]
    print_table(rows)
    if args.csv:
        with open(args.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
            w.writeheader()
            w.writerows(rows)
        print(f"\nwrote {args.csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
