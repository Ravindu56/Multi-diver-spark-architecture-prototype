#!/usr/bin/env python3
# =============================================================================
# scripts/validate_parity.py
#
# Issue #10 — Data Integrity: Baseline Spark vs MPI Multi-Driver
#
# Usage (5 MPI ranks = 1 root + 4 workers):
#   mpirun --oversubscribe -n 5 \
#     --mca hwloc_base_binding_policy none \
#     -x PYTHONPATH \
#     python scripts/validate_parity.py
#
# CLI flags:
#   --skip-kmeans       skip K-Means parity check
#   --skip-logreg       skip Logistic Regression parity check
#   --tolerance FLOAT   delta threshold for PASS/FAIL  [default: 0.001]
#   --k INT             number of K-Means clusters     [default: 3]
#   --max-iter INT      max iterations                 [default: 20]
#   --report-dir PATH   directory for CSV report       [default: results/]
# =============================================================================
from __future__ import annotations

import argparse
import csv
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

from mpj_spark.config import (
    KMEANS_DATASET_PATH,
    LOGREG_DATASET_PATH,
    SHARED_STORAGE_PATH,
    TOTAL_CORES,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _banner(msg: str) -> None:
    if rank == 0:
        print("\n" + "=" * 70)
        print(f"  {msg}")
        print("=" * 70)


def _check(label: str, delta: float, tol: float) -> dict:
    status = "PASS" if delta <= tol else "FAIL"
    return {"metric": label, "delta": delta, "status": status}


# ---------------------------------------------------------------------------
# K-Means parity
# ---------------------------------------------------------------------------

def run_kmeans_parity(k: int, max_iter: int, tol: float) -> list[dict]:
    import numpy as np

    from mpj_spark.applications.baseline_kmeans import run_baseline_kmeans
    from mpj_spark.kmeans.driver import run_kmeans_driver

    _banner("[PARITY] K-Means — Running single-driver Spark BASELINE")
    baseline_result = None
    if rank == 0:
        baseline_result, _ = run_baseline_kmeans(
            KMEANS_DATASET_PATH,
            num_workers=size - 1,
            k=k,
            max_iter=max_iter,
        )
    baseline_result = comm.bcast(baseline_result, root=0)

    _banner("[PARITY] K-Means — Running MPI multi-driver")
    mpj_result = run_kmeans_driver(
        rank=rank,
        size=size,
        comm=comm,
        dataset_path=KMEANS_DATASET_PATH,
        k=k,
        max_iter=max_iter,
    )

    records: list[dict] = []
    if rank == 0:
        base_centres = np.array(baseline_result["centres"])
        mpj_centres = np.array(mpj_result["centres"])

        # Sort both by centroid L2 norm so comparison is order-independent
        base_sorted = base_centres[np.argsort(np.linalg.norm(base_centres, axis=1))]
        mpj_sorted = mpj_centres[np.argsort(np.linalg.norm(mpj_centres, axis=1))]

        for i, (bc, mc) in enumerate(zip(base_sorted, mpj_sorted)):
            delta = float(np.linalg.norm(bc - mc))
            records.append({**_check(f"centroid_C{i}_l2_delta", delta, tol), "workload": "kmeans"})

        wcss_delta = abs(baseline_result["wcss"] - mpj_result["wcss"]) / max(
            abs(baseline_result["wcss"]), 1e-9
        )
        records.append({**_check("wcss_relative_delta", wcss_delta, tol), "workload": "kmeans"})

    return records


# ---------------------------------------------------------------------------
# Logistic Regression parity
# ---------------------------------------------------------------------------

def run_logreg_parity(max_iter: int, tol: float) -> list[dict]:
    import numpy as np

    from mpj_spark.applications.baseline_logreg import run_baseline_logreg
    from mpj_spark.logreg.driver import run_logreg_driver

    _banner("[PARITY] Logistic Regression — Running single-driver Spark BASELINE")
    baseline_result = None
    if rank == 0:
        baseline_result, _ = run_baseline_logreg(
            LOGREG_DATASET_PATH,
            num_workers=size - 1,
            max_iter=max_iter,
        )
    baseline_result = comm.bcast(baseline_result, root=0)

    _banner("[PARITY] Logistic Regression — Running MPI multi-driver")
    mpj_result = run_logreg_driver(
        rank=rank,
        size=size,
        comm=comm,
        dataset_path=LOGREG_DATASET_PATH,
        max_iter=max_iter,
    )

    records: list[dict] = []
    if rank == 0:
        base_w = np.array(baseline_result["weights"])
        mpj_w = np.array(mpj_result["weights"])
        w_delta = float(np.linalg.norm(base_w - mpj_w))
        records.append({**_check("weight_vector_l2_delta", w_delta, tol), "workload": "logreg"})

        intercept_delta = abs(baseline_result["intercept"] - mpj_result["intercept"])
        records.append(
            {**_check("intercept_delta", intercept_delta, tol), "workload": "logreg"}
        )

    return records


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def write_report(records: list[dict], report_dir: str, run_id: str) -> None:
    Path(report_dir).mkdir(parents=True, exist_ok=True)
    path = Path(report_dir) / "parity_report.csv"
    fieldnames = ["run_id", "workload", "metric", "delta", "status"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in records:
            writer.writerow({"run_id": run_id, **r})
    print(f"\n  Report written → {path}")


def print_summary(records: list[dict], tol: float) -> None:
    passed = sum(1 for r in records if r["status"] == "PASS")
    total = len(records)
    verdict = "DATA INTEGRITY CONFIRMED" if passed == total else "DATA INTEGRITY FAILED"

    print("\n" + "=" * 70)
    print(f"  {'Workload':<10} {'Metric':<40} {'Delta':<15} {'Status'}")
    print("  " + "-" * 66)
    for r in records:
        print(
            f"  {r['workload']:<10} {r['metric']:<40} {r['delta']:<15.8f} {r['status']}"
        )
    print("=" * 70)
    print(f"  Result : {passed}/{total} checks passed")
    print(f"  Verdict: {verdict}")
    print("=" * 70)

    if passed < total:
        sys.exit(1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Issue #10 parity validation")
    parser.add_argument("--skip-kmeans", action="store_true")
    parser.add_argument("--skip-logreg", action="store_true")
    parser.add_argument("--tolerance", type=float, default=0.001)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--max-iter", type=int, default=20)
    parser.add_argument("--report-dir", default="results")
    args = parser.parse_args()

    # Use timezone-aware UTC (replaces deprecated datetime.utcnow())
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") if rank == 0 else None
    run_id = comm.bcast(run_id, root=0)

    if rank == 0:
        print("\n" + "=" * 70)
        print("  Issue #10 — Data Integrity: Baseline Spark vs MPI Multi-Driver")
        print(f"  MPI ranks   : {size}")
        print(f"  Tolerance   : {args.tolerance}")
        print(f"  KMeans path : {KMEANS_DATASET_PATH}")
        print(f"  LogReg path : {LOGREG_DATASET_PATH}")
        print("=" * 70)

    all_records: list[dict] = []

    if not args.skip_kmeans:
        records = run_kmeans_parity(k=args.k, max_iter=args.max_iter, tol=args.tolerance)
        if rank == 0:
            all_records.extend(records)

    if not args.skip_logreg:
        records = run_logreg_parity(max_iter=args.max_iter, tol=args.tolerance)
        if rank == 0:
            all_records.extend(records)

    if rank == 0:
        write_report(all_records, args.report_dir, run_id)
        print_summary(all_records, args.tolerance)


if __name__ == "__main__":
    main()
