#!/usr/bin/env python3
"""
P4-08 / Issue #12: synchronization-overhead benchmark.

Compares:
- Single-driver baseline execution time
- MPI multi-driver timing metrics already written by K-Means and LogReg runs

Output:
/sync_overhead_benchmark.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

try:
    from mpi4py import MPI

    COMM = MPI.COMM_WORLD
    RANK = COMM.Get_rank()
    SIZE = COMM.Get_size()
except ImportError:
    COMM = None
    RANK = 0
    SIZE = 1

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CSV_FIELDS = [
    "timestamp_utc",
    "workload",
    "setup",
    "exec_time_s",
    "throughput_rows_per_s",
    "sync_time_mean_s",
    "sync_time_max_s",
    "sync_overhead_pct_mean",
    "iterations_run",
    "dataset_size_rows",
]


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def dataset_rows(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open(encoding="utf-8") as handle:
        row_count = sum(1 for _ in handle)
    return max(0, row_count - 1)


def load_latest_metrics(metrics_dir: Path, prefix: str) -> list[dict[str, str]]:
    candidates = sorted(
        metrics_dir.glob(f"{prefix}*.csv"),
        key=lambda candidate: candidate.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        return []
    with candidates[0].open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


# ---------------------------------------------------------------------------
# Single-driver baseline runners — call the real module functions directly
# ---------------------------------------------------------------------------


def baseline_kmeans(
    data_path: Path,
    k: int,
    max_iter: int,
    num_workers: int = 2,
    cores_override: int | None = None,
) -> dict:
    """Run the repository single-driver K-Means baseline."""

    from mpj_spark.applications.baseline_kmeans import run_baseline_kmeans  # noqa: PLC0415

    t0 = time.perf_counter()

    result = run_baseline_kmeans(
        input_file_path=str(data_path),
        num_workers=num_workers,
        cores_override=cores_override if cores_override else None,
        k=k,
        max_iter=max_iter,
    )

    exec_time = time.perf_counter() - t0
    nrows = dataset_rows(data_path)

    iterations_run = max_iter
    if isinstance(result, tuple) and result:
        candidate = result[-1]
        if isinstance(candidate, int):
            iterations_run = candidate

    return {
        "workload": "kmeans",
        "setup": "single_driver_baseline",
        "exec_time_s": round(exec_time, 6),
        "throughput_rows_per_s": (round(nrows / exec_time, 2) if exec_time > 0 else 0.0),
        "sync_time_mean_s": 0.0,
        "sync_time_max_s": 0.0,
        "sync_overhead_pct_mean": 0.0,
        "iterations_run": iterations_run,
        "dataset_size_rows": nrows,
    }


def baseline_logreg(
    data_path: Path,
    epochs: int,
    num_workers: int = 2,
    cores_override: int | None = None,
) -> dict:
    """Run the repository single-driver Logistic Regression baseline."""

    from mpj_spark.applications.baseline_logreg import run_baseline_logreg  # noqa: PLC0415

    t0 = time.perf_counter()

    run_baseline_logreg(
        input_file=str(data_path),
        num_workers=num_workers,
        cores_override=cores_override if cores_override else None,
        max_iter=epochs,
        reg_param=0.01,
        num_features=10,
    )

    exec_time = time.perf_counter() - t0
    nrows = dataset_rows(data_path)

    return {
        "workload": "logreg",
        "setup": "single_driver_baseline",
        "exec_time_s": round(exec_time, 6),
        "throughput_rows_per_s": (round(nrows / exec_time, 2) if exec_time > 0 else 0.0),
        "sync_time_mean_s": 0.0,
        "sync_time_max_s": 0.0,
        "sync_overhead_pct_mean": 0.0,
        "iterations_run": epochs,
        "dataset_size_rows": nrows,
    }


# ---------------------------------------------------------------------------
# MPI multi-driver measurements — read from metrics CSVs
# ---------------------------------------------------------------------------


def mpi_kmeans_from_metrics(
    metrics_dir: Path,
    data_path: Path,
) -> dict[str, Any] | None:
    rows = load_latest_metrics(metrics_dir, "kmeans")
    if not rows:
        return None

    sync_times = [safe_float(row.get("sync_time_s")) for row in rows]
    if not any(sync_times):
        sync_times = [safe_float(row.get("sync_time")) for row in rows]

    iter_times = [safe_float(row.get("iter_time_s")) for row in rows]
    if not any(iter_times):
        iter_times = [safe_float(row.get("iter_time")) for row in rows]

    if not any(iter_times):
        spark_times = [safe_float(row.get("spark_time_s")) for row in rows]
        if not any(spark_times):
            spark_times = [safe_float(row.get("spark_time")) for row in rows]
        iter_times = [spark + sync for spark, sync in zip(spark_times, sync_times)]

    total_exec = sum(iter_times)
    sync_pct = [
        100.0 * sync / iteration for sync, iteration in zip(sync_times, iter_times) if iteration > 0
    ]

    row_count = dataset_rows(data_path)

    return {
        "workload": "kmeans",
        "setup": "mpi_multi_driver",
        "exec_time_s": round(total_exec, 6),
        "throughput_rows_per_s": round(row_count / total_exec, 2) if total_exec > 0 else 0.0,
        "sync_time_mean_s": round(float(np.mean(sync_times)), 6) if sync_times else 0.0,
        "sync_time_max_s": round(float(np.max(sync_times)), 6) if sync_times else 0.0,
        "sync_overhead_pct_mean": round(float(np.mean(sync_pct)), 6) if sync_pct else 0.0,
        "iterations_run": len(rows),
        "dataset_size_rows": row_count,
    }


def mpi_logreg_from_metrics(
    metrics_dir: Path,
    data_path: Path,
) -> dict[str, Any] | None:
    rows = load_latest_metrics(metrics_dir, "logreg")
    if not rows:
        return None

    sync_times = [safe_float(row.get("sync_time_s")) for row in rows]
    if not any(sync_times):
        sync_times = [safe_float(row.get("sync_time")) for row in rows]

    epoch_times = [safe_float(row.get("epoch_time_s")) for row in rows]
    if not any(epoch_times):
        epoch_times = [safe_float(row.get("epoch_time")) for row in rows]
    if not any(epoch_times):
        epoch_times = [safe_float(row.get("iter_time_s")) for row in rows]

    if not any(epoch_times):
        spark_times = [safe_float(row.get("spark_time_s")) for row in rows]
        if not any(spark_times):
            spark_times = [safe_float(row.get("spark_time")) for row in rows]
        epoch_times = [spark + sync for spark, sync in zip(spark_times, sync_times)]

    total_exec = sum(epoch_times)
    sync_pct = [100.0 * sync / epoch for sync, epoch in zip(sync_times, epoch_times) if epoch > 0]

    row_count = dataset_rows(data_path)

    return {
        "workload": "logreg",
        "setup": "mpi_multi_driver",
        "exec_time_s": round(total_exec, 6),
        "throughput_rows_per_s": round(row_count / total_exec, 2) if total_exec > 0 else 0.0,
        "sync_time_mean_s": round(float(np.mean(sync_times)), 6) if sync_times else 0.0,
        "sync_time_max_s": round(float(np.max(sync_times)), 6) if sync_times else 0.0,
        "sync_overhead_pct_mean": round(float(np.mean(sync_pct)), 6) if sync_pct else 0.0,
        "iterations_run": len(rows),
        "dataset_size_rows": row_count,
    }


# ---------------------------------------------------------------------------
# Console reporter
# ---------------------------------------------------------------------------


def print_row(row: dict[str, Any]) -> None:
    print(
        f"{row['workload']:10}"
        f"{row['setup']:25}"
        f"{row['exec_time_s']:12.3f}"
        f"{row['throughput_rows_per_s']:18.1f}"
        f"{row['sync_time_mean_s']:14.6f}"
        f"{row['sync_overhead_pct_mean']:14.2f}"
    )


def print_comparison(rows: list[dict[str, Any]], workload: str) -> None:
    baseline = next(
        (
            row
            for row in rows
            if row["workload"] == workload and row["setup"] == "single_driver_baseline"
        ),
        None,
    )
    mpi_row = next(
        (row for row in rows if row["workload"] == workload and row["setup"] == "mpi_multi_driver"),
        None,
    )

    if baseline is None or mpi_row is None:
        return

    speedup = (
        baseline["exec_time_s"] / mpi_row["exec_time_s"] if mpi_row["exec_time_s"] > 0 else 0.0
    )

    print(
        f"{workload.upper()}: baseline/MPI execution-time ratio = {speedup:.3f}x; "
        f"mean MPI synchronization overhead = "
        f"{mpi_row['sync_overhead_pct_mean']:.2f}%"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="P4-08 synchronization-overhead benchmark.")
    parser.add_argument("--ranks", type=int, default=int(os.getenv("MPI_NUM_RANKS", "3")))
    parser.add_argument("--kmeans-k", type=int, default=3)
    parser.add_argument("--kmeans-iter", type=int, default=20)
    parser.add_argument("--logreg-epochs", type=int, default=14)
    parser.add_argument(
        "--metrics-dir", default=os.getenv("MPJ_METRICS_DIR", "/data/metrics/p4_08")
    )
    parser.add_argument(
        "--results-dir", default=os.getenv("MPJ_RESULTS_DIR", "/data/results/p4_08_sync")
    )
    parser.add_argument("--data-dir", default=os.getenv("MPJ_INPUT_DIR", "/data/input"))
    parser.add_argument("--skip-kmeans", action="store_true")
    parser.add_argument("--skip-logreg", action="store_true")
    parser.add_argument(
        "--baseline-workers",
        type=int,
        default=2,
        help="Logical workers used to calculate baseline Spark core budget.",
    )
    parser.add_argument(
        "--baseline-cores",
        type=int,
        default=0,
        help="Explicit Spark local[N] core override; 0 means module default.",
    )

    args = parser.parse_args()

    if SIZE != args.ranks:
        if RANK == 0:
            print(f"ERROR: launched with {SIZE} MPI ranks, " f"but --ranks={args.ranks}")
        if COMM is not None:
            COMM.Abort(2)
        raise SystemExit(2)

    if RANK != 0:
        if COMM is not None:
            COMM.Barrier()
        return

    metrics_dir = Path(args.metrics_dir)
    results_dir = Path(args.results_dir)
    data_dir = Path(args.data_dir)

    results_dir.mkdir(parents=True, exist_ok=True)

    kmeans_data = data_dir / "kmeans_data.csv"
    logreg_data = data_dir / "logreg_data.csv"
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    # Resolve cores_override: 0 means "let the module decide" → pass None
    cores_override: int | None = args.baseline_cores if args.baseline_cores > 0 else None

    rows: list[dict[str, Any]] = []

    print("=" * 78)
    print("P4-08 Synchronization-Overhead Benchmark")
    print(f"Timestamp: {timestamp}")
    print(f"MPI ranks: {SIZE}")
    print(f"Metrics directory: {metrics_dir}")
    print(f"Results directory: {results_dir}")
    print("=" * 78)

    if not args.skip_kmeans:
        if kmeans_data.exists():
            print("K-Means: running single-driver baseline")
            baseline = baseline_kmeans(
                kmeans_data,
                args.kmeans_k,
                args.kmeans_iter,
                num_workers=args.baseline_workers,
                cores_override=cores_override,
            )
            rows.append(baseline)
        else:
            print(f"WARNING: K-Means input not found: {kmeans_data}")

        print("K-Means: reading MPI metrics")
        mpi_metrics = mpi_kmeans_from_metrics(metrics_dir, kmeans_data)
        if mpi_metrics is not None:
            rows.append(mpi_metrics)
        else:
            print(f"WARNING: no K-Means metrics CSV found in {metrics_dir}")

    if not args.skip_logreg:
        if logreg_data.exists():
            print("LogReg: running single-driver baseline")
            baseline = baseline_logreg(
                logreg_data,
                args.logreg_epochs,
                num_workers=args.baseline_workers,
                cores_override=cores_override,
            )
            rows.append(baseline)
        else:
            print(f"WARNING: LogReg input not found: {logreg_data}")

        print("LogReg: reading MPI metrics")
        mpi_metrics = mpi_logreg_from_metrics(metrics_dir, logreg_data)
        if mpi_metrics is not None:
            rows.append(mpi_metrics)
        else:
            print(f"WARNING: no LogReg metrics CSV found in {metrics_dir}")

    for row in rows:
        row["timestamp_utc"] = timestamp

    output_path = results_dir / "sync_overhead_benchmark.csv"
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    print()
    print(f"Wrote {output_path} ({len(rows)} rows)")
    print()
    print(
        f"{'Workload':10}{'Setup':25}{'Exec(s)':12}"
        f"{'Throughput(rows/s)':18}{'Sync mean(s)':14}{'Sync % mean':14}"
    )
    print("-" * 93)
    for row in rows:
        print_row(row)

    print()
    print_comparison(rows, "kmeans")
    print_comparison(rows, "logreg")

    if COMM is not None:
        COMM.Barrier()


if __name__ == "__main__":
    main()
