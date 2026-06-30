#!/usr/bin/env python3
"""
scripts/sync_overhead_benchmark.py

Issue #12 — P3-06: Benchmark synchronisation overhead.

Compares MPI multi-driver (N=5) vs single-driver Spark baseline for:
  - K-Means  : centroid Allreduce overhead
  - LogReg   : gradient Allreduce overhead

Outputs
-------
  results/sync_overhead_benchmark.csv
    columns: workload, setup, exec_time_s, throughput_rows_per_s,
             sync_time_mean_s, sync_time_max_s, sync_overhead_pct_mean,
             iterations_run, dataset_size_rows

Usage
-----
  # Full benchmark (must be called from MPI environment, rank-0 only writes CSV)
  mpirun --oversubscribe -n 5 python scripts/sync_overhead_benchmark.py

  # Skip one workload
  mpirun --oversubscribe -n 5 python scripts/sync_overhead_benchmark.py --skip-logreg
  mpirun --oversubscribe -n 5 python scripts/sync_overhead_benchmark.py --skip-kmeans

  # Custom parameters
  mpirun --oversubscribe -n 5 python scripts/sync_overhead_benchmark.py \\
    --ranks 5 --kmeans-k 3 --kmeans-iter 20 --logreg-epochs 14
"""

import argparse
import csv
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# MPI bootstrap — rank 0 is coordinator, ranks 1..N are workers
# ---------------------------------------------------------------------------
try:
    from mpi4py import MPI
    COMM = MPI.COMM_WORLD
    RANK = COMM.Get_rank()
    SIZE = COMM.Get_size()
except ImportError:
    COMM = None
    RANK = 0
    SIZE = 1

# ---------------------------------------------------------------------------
# Project root on sys.path
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from mpj_spark.applications.baseline_kmeans import run_kmeans_baseline
from mpj_spark.applications.baseline_logreg import run_logreg_baseline


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_metrics_csv(metrics_dir: Path, prefix: str) -> list[dict]:
    """Return list of row-dicts from the most recent metrics CSV for a workload."""
    candidates = sorted(metrics_dir.glob(f"{prefix}*.csv"), key=os.path.getmtime, reverse=True)
    if not candidates:
        return []
    rows = []
    with open(candidates[0], newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            rows.append(row)
    return rows


def _safe_float(val, default=0.0):
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def _dataset_rows(data_path: Path) -> int:
    """Count lines in CSV (minus header)."""
    if not data_path.exists():
        return 0
    with open(data_path) as fh:
        return max(0, sum(1 for _ in fh) - 1)


# ---------------------------------------------------------------------------
# Single-driver baseline measurements
# ---------------------------------------------------------------------------

def _baseline_kmeans(data_path: Path, k: int, max_iter: int) -> dict:
    """Run single-driver K-Means baseline and return timing dict."""
    t0 = time.perf_counter()
    result = run_kmeans_baseline(
        data_path=str(data_path),
        k=k,
        max_iter=max_iter,
    )
    exec_time = time.perf_counter() - t0
    n_rows = _dataset_rows(data_path)
    return {
        "workload": "kmeans",
        "setup": "single_driver_baseline",
        "exec_time_s": round(exec_time, 6),
        "throughput_rows_per_s": round(n_rows / exec_time, 2) if exec_time > 0 else 0,
        "sync_time_mean_s": 0.0,
        "sync_time_max_s": 0.0,
        "sync_overhead_pct_mean": 0.0,
        "iterations_run": result.get("iterations", max_iter),
        "dataset_size_rows": n_rows,
    }


def _baseline_logreg(data_path: Path, epochs: int) -> dict:
    """Run single-driver LogReg baseline and return timing dict."""
    t0 = time.perf_counter()
    result = run_logreg_baseline(
        data_path=str(data_path),
        max_iter=epochs,
    )
    exec_time = time.perf_counter() - t0
    n_rows = _dataset_rows(data_path)
    return {
        "workload": "logreg",
        "setup": "single_driver_baseline",
        "exec_time_s": round(exec_time, 6),
        "throughput_rows_per_s": round(n_rows / exec_time, 2) if exec_time > 0 else 0,
        "sync_time_mean_s": 0.0,
        "sync_time_max_s": 0.0,
        "sync_overhead_pct_mean": 0.0,
        "iterations_run": result.get("iterations", epochs),
        "dataset_size_rows": n_rows,
    }


# ---------------------------------------------------------------------------
# MPI multi-driver measurements — read from metrics CSVs written by allreduce.py
# ---------------------------------------------------------------------------

def _mpi_kmeans_from_metrics(metrics_dir: Path, data_path: Path, max_iter: int) -> dict:
    """Parse the latest K-Means metrics CSV written by allreduce.py."""
    rows = _load_metrics_csv(metrics_dir, "kmeans")
    if not rows:
        return None
    sync_times = [_safe_float(r.get("sync_time_s", 0)) for r in rows]
    spark_times = [_safe_float(r.get("spark_time_s", 0)) for r in rows]
    iter_times = [_safe_float(r.get("iter_time_s", 0)) for r in rows]
    n_rows = _dataset_rows(data_path)
    total_exec = sum(iter_times)
    sync_mean = float(np.mean(sync_times)) if sync_times else 0.0
    sync_max = float(np.max(sync_times)) if sync_times else 0.0
    sync_pct = [
        100.0 * s / t if t > 0 else 0.0
        for s, t in zip(sync_times, iter_times)
    ]
    return {
        "workload": "kmeans",
        "setup": "mpi_multi_driver",
        "exec_time_s": round(total_exec, 6),
        "throughput_rows_per_s": round(n_rows / total_exec, 2) if total_exec > 0 else 0,
        "sync_time_mean_s": round(sync_mean, 6),
        "sync_time_max_s": round(sync_max, 6),
        "sync_overhead_pct_mean": round(float(np.mean(sync_pct)), 6) if sync_pct else 0.0,
        "iterations_run": len(rows),
        "dataset_size_rows": n_rows,
    }


def _mpi_logreg_from_metrics(metrics_dir: Path, data_path: Path, epochs: int) -> dict:
    """Parse the latest LogReg metrics CSV written by allreduce.py."""
    rows = _load_metrics_csv(metrics_dir, "logreg")
    if not rows:
        return None
    sync_times = [_safe_float(r.get("sync_time_s", 0)) for r in rows]
    spark_times = [_safe_float(r.get("spark_time_s", 0)) for r in rows]
    iter_times = [
        _safe_float(r.get("epoch_time_s", 0)) or
        _safe_float(r.get("iter_time_s", 0))
        for r in rows
    ]
    n_rows = _dataset_rows(data_path)
    total_exec = sum(iter_times)
    sync_mean = float(np.mean(sync_times)) if sync_times else 0.0
    sync_max = float(np.max(sync_times)) if sync_times else 0.0
    sync_pct = [
        100.0 * s / t if t > 0 else 0.0
        for s, t in zip(sync_times, iter_times)
    ]
    return {
        "workload": "logreg",
        "setup": "mpi_multi_driver",
        "exec_time_s": round(total_exec, 6),
        "throughput_rows_per_s": round(n_rows / total_exec, 2) if total_exec > 0 else 0,
        "sync_time_mean_s": round(sync_mean, 6),
        "sync_time_max_s": round(sync_max, 6),
        "sync_overhead_pct_mean": round(float(np.mean(sync_pct)), 6) if sync_pct else 0.0,
        "iterations_run": len(rows),
        "dataset_size_rows": n_rows,
    }


# ---------------------------------------------------------------------------
# Console reporter
# ---------------------------------------------------------------------------

CSV_FIELDS = [
    "workload", "setup", "exec_time_s", "throughput_rows_per_s",
    "sync_time_mean_s", "sync_time_max_s", "sync_overhead_pct_mean",
    "iterations_run", "dataset_size_rows",
]

HDR = (
    f"  {'workload':<10} {'setup':<25} {'exec_time_s':>12} "
    f"{'throughput':>14} {'sync_mean_s':>12} {'sync_pct_mean':>14}"
)


def _print_row(r: dict) -> None:
    print(
        f"  {r['workload']:<10} {r['setup']:<25} "
        f"{r['exec_time_s']:>12.3f} "
        f"{r['throughput_rows_per_s']:>14.1f} "
        f"{r['sync_time_mean_s']:>12.6f} "
        f"{r['sync_overhead_pct_mean']:>13.2f}%"
    )


def _print_comparison(rows: list[dict], workload: str) -> None:
    wrows = [r for r in rows if r["workload"] == workload]
    baseline = next((r for r in wrows if r["setup"] == "single_driver_baseline"), None)
    mpi      = next((r for r in wrows if r["setup"] == "mpi_multi_driver"), None)
    if not baseline or not mpi:
        return
    speedup = baseline["exec_time_s"] / mpi["exec_time_s"] if mpi["exec_time_s"] > 0 else 0
    tput_delta = mpi["throughput_rows_per_s"] / baseline["throughput_rows_per_s"] if baseline["throughput_rows_per_s"] > 0 else 0
    print(f"  {workload.upper()} speedup (exec_time): {speedup:.3f}x")
    print(f"  {workload.upper()} throughput ratio  : {tput_delta:.3f}x")
    print(f"  {workload.upper()} sync overhead (MPI): {mpi['sync_overhead_pct_mean']:.2f}%")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Issue #12 — Sync overhead benchmark: MPI multi-driver vs single-driver baseline."
    )
    parser.add_argument("--ranks",         type=int,   default=5,    help="Expected MPI size (informational, not enforced)")
    parser.add_argument("--kmeans-k",      type=int,   default=3,    help="K-Means clusters")
    parser.add_argument("--kmeans-iter",   type=int,   default=20,   help="K-Means max iterations")
    parser.add_argument("--logreg-epochs", type=int,   default=14,   help="LogReg epochs")
    parser.add_argument("--metrics-dir",   type=str,   default="metrics", help="Directory containing workload metrics CSVs")
    parser.add_argument("--results-dir",   type=str,   default="results", help="Output directory")
    parser.add_argument("--data-dir",      type=str,   default="data",    help="Dataset directory")
    parser.add_argument("--skip-kmeans",   action="store_true", help="Skip K-Means benchmark")
    parser.add_argument("--skip-logreg",   action="store_true", help="Skip LogReg benchmark")
    args = parser.parse_args()

    # Only rank 0 runs the benchmark logic
    if RANK != 0:
        if COMM:
            COMM.Barrier()
        return

    metrics_dir = Path(args.metrics_dir)
    results_dir = Path(args.results_dir)
    data_dir    = Path(args.data_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    kmeans_data = data_dir / "kmeans_dataset.csv"
    logreg_data = data_dir / "logreg_dataset.csv"

    rows: list[dict] = []
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    print()
    print("=" * 70)
    print("  Issue #12 — Sync Overhead Benchmark")
    print(f"  timestamp   : {ts}")
    print(f"  metrics dir : {metrics_dir}")
    print(f"  results dir : {results_dir}")
    print(f"  MPI ranks   : {SIZE} (expected {args.ranks})")
    print("=" * 70)
    print()

    # ── K-Means ──────────────────────────────────────────────────────────────
    if not args.skip_kmeans:
        print("[K-Means] Running single-driver baseline...")
        if kmeans_data.exists():
            km_bl = _baseline_kmeans(kmeans_data, args.kmeans_k, args.kmeans_iter)
            rows.append(km_bl)
            print(f"  baseline exec_time={km_bl['exec_time_s']:.3f}s  "
                  f"throughput={km_bl['throughput_rows_per_s']:.1f} rows/s")
        else:
            print(f"  WARNING: {kmeans_data} not found — skipping baseline. "
                  "Run scripts/generate_datasets.py first.")

        print("[K-Means] Reading MPI multi-driver metrics...")
        km_mpi = _mpi_kmeans_from_metrics(metrics_dir, kmeans_data, args.kmeans_iter)
        if km_mpi:
            rows.append(km_mpi)
            print(f"  MPI exec_time={km_mpi['exec_time_s']:.3f}s  "
                  f"sync_mean={km_mpi['sync_time_mean_s']:.6f}s  "
                  f"sync_pct={km_mpi['sync_overhead_pct_mean']:.2f}%")
        else:
            print(f"  WARNING: No K-Means metrics CSV found in {metrics_dir}. "
                  "Run mpirun -n 5 python -m mpj_spark.applications.kmeans.allreduce first.")

    # ── LogReg ────────────────────────────────────────────────────────────────
    if not args.skip_logreg:
        print("[LogReg] Running single-driver baseline...")
        if logreg_data.exists():
            lr_bl = _baseline_logreg(logreg_data, args.logreg_epochs)
            rows.append(lr_bl)
            print(f"  baseline exec_time={lr_bl['exec_time_s']:.3f}s  "
                  f"throughput={lr_bl['throughput_rows_per_s']:.1f} rows/s")
        else:
            print(f"  WARNING: {logreg_data} not found — skipping baseline. "
                  "Run scripts/generate_datasets.py first.")

        print("[LogReg] Reading MPI multi-driver metrics...")
        lr_mpi = _mpi_logreg_from_metrics(metrics_dir, logreg_data, args.logreg_epochs)
        if lr_mpi:
            rows.append(lr_mpi)
            print(f"  MPI exec_time={lr_mpi['exec_time_s']:.3f}s  "
                  f"sync_mean={lr_mpi['sync_time_mean_s']:.6f}s  "
                  f"sync_pct={lr_mpi['sync_overhead_pct_mean']:.2f}%")
        else:
            print(f"  WARNING: No LogReg metrics CSV found in {metrics_dir}. "
                  "Run mpirun -n 5 python -m mpj_spark.applications.logreg.allreduce first.")

    # ── Write CSV ─────────────────────────────────────────────────────────────
    out_path = results_dir / "sync_overhead_benchmark.csv"
    with open(out_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in CSV_FIELDS})
    print()
    print(f"  Written → {out_path}  ({len(rows)} rows)")

    # ── Console comparison table ──────────────────────────────────────────────
    print()
    print("=" * 70)
    print("  COMPARISON TABLE")
    print("=" * 70)
    print(HDR)
    print("  " + "-" * 90)
    for r in rows:
        _print_row(r)
    print()
    for wl in ["kmeans", "logreg"]:
        _print_comparison(rows, wl)
    print()
    print("  Aligns with Objective 2d evaluation baselines (B1 vs M3).")
    print("  Use results/sync_overhead_benchmark.csv as input to the allocator evaluation.")
    print("=" * 70)
    print()

    if COMM:
        COMM.Barrier()


if __name__ == "__main__":
    main()
