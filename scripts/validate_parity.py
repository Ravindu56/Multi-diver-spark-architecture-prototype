# scripts/validate_parity.py
"""
Issue #10 — P3-05: Data Integrity Validation
Baseline single-driver Spark  vs  MPI multi-driver (N=5)

Run with:
    python scripts/generate_datasets.py          # once, before first run
    mpirun -n 5 python scripts/validate_parity.py
    mpirun -n 5 python scripts/validate_parity.py --tolerance 1e-3 --kmeans-k 3 --logreg-epochs 30
    mpirun -n 5 python scripts/validate_parity.py --skip-logreg    # K-Means only
    mpirun -n 5 python scripts/validate_parity.py --skip-kmeans    # LogReg only

Output:
    results/parity_report.csv  (append mode — multiple runs accumulate)
"""

import argparse
import csv
import os
from datetime import datetime

import numpy as np

try:
    from mpi4py import MPI
except ImportError:  # pragma: no cover
    raise SystemExit("[ERROR] mpi4py not found. Install: pip install mpi4py")

from mpj_spark.config import (
    KMEANS_DATASET_PATH,
    LOGREG_DATASET_PATH,
    SHARED_STORAGE_PATH,
)

RESULTS_DIR = os.path.join(SHARED_STORAGE_PATH, "results")
REPORT_PATH = os.path.join(RESULTS_DIR, "parity_report.csv")

REPORT_FIELDS = [
    "run_id",
    "num_workers",
    "workload",
    "metric",
    "baseline_value",
    "mpi_value",
    "delta",
    "within_tolerance",
    "tolerance",
]


# ── K-Means parity ────────────────────────────────────────────────────────────

def run_kmeans_parity(comm, rank, size, k, max_iter, tolerance):
    """Run baseline + MPI K-Means, compare centroids and WCSS. Returns records list (rank 0 only)."""
    records = []

    # ── Rank 0: single-driver Spark baseline ────────────────────────
    baseline_centres = None
    baseline_wcss = None
    if rank == 0:
        from mpj_spark.applications.baseline_kmeans import run_baseline_kmeans

        print("\n" + "=" * 70)
        print("  [PARITY] K-Means — Running single-driver Spark BASELINE")
        print("=" * 70)
        result, _ = run_baseline_kmeans(
            input_file_path=KMEANS_DATASET_PATH,
            num_workers=size,
            k=k,
            max_iter=max_iter,
        )
        baseline_centres = np.array(result["centres"])  # shape: (k, n_features)
        baseline_wcss = result["wcss"]

    # ── All ranks: MPI multi-driver ──────────────────────────────────
    comm.Barrier()
    print(f"  [PARITY rank {rank}] K-Means — Running MPI Allreduce (N={size})...")
    from mpj_spark.applications.kmeans.allreduce import run_kmeans_allreduce

    mpi_result = run_kmeans_allreduce(
        comm=comm,
        input_file=KMEANS_DATASET_PATH,
        k=k,
        max_iter=max_iter,
        seed=42,
    )

    # ── Rank 0: compare ──────────────────────────────────────────────
    if rank == 0:
        mpi_centres = np.array(mpi_result["centres"])  # shape: (k, n_features)
        mpi_wcss = mpi_result["wcss"]

        # Greedy nearest-neighbour matching — centroid order may differ between runs
        deltas = _match_centroids(baseline_centres, mpi_centres)
        for i, delta in enumerate(deltas):
            records.append(
                _record(
                    workload="kmeans",
                    metric=f"centroid_C{i}_l2_delta",
                    baseline_value=f"{np.linalg.norm(baseline_centres[i]):.6f}",
                    mpi_value=f"{np.linalg.norm(mpi_centres[i]):.6f}",
                    delta=delta,
                    within_tolerance=delta < tolerance,
                    tolerance=tolerance,
                )
            )

        # WCSS: allow 5% relative tolerance (cluster quality, not exact match)
        wcss_tol = baseline_wcss * 0.05
        wcss_delta = abs(baseline_wcss - mpi_wcss)
        records.append(
            _record(
                workload="kmeans",
                metric="wcss_relative_delta",
                baseline_value=f"{baseline_wcss:.4f}",
                mpi_value=f"{mpi_wcss:.4f}",
                delta=wcss_delta,
                within_tolerance=wcss_delta < wcss_tol,
                tolerance=f"{wcss_tol:.4f} (5%)",
            )
        )

    return records


def _match_centroids(ref: np.ndarray, cand: np.ndarray) -> list:
    """Greedy nearest-neighbour centroid matching. Returns list of L2 deltas."""
    used = set()
    deltas = []
    for r in ref:
        best_j, best_d = None, float("inf")
        for j, c in enumerate(cand):
            if j in used:
                continue
            d = float(np.linalg.norm(r - c))
            if d < best_d:
                best_d, best_j = d, j
        used.add(best_j)
        deltas.append(best_d)
    return deltas


# ── LogReg parity ─────────────────────────────────────────────────────────────

def run_logreg_parity(comm, rank, size, epochs, tolerance):
    """Run baseline + MPI LogReg, compare weight vector and intercept. Returns records list (rank 0 only)."""
    records = []

    # ── Rank 0: single-driver Spark baseline ────────────────────────
    baseline_weights = None
    baseline_intercept = None
    if rank == 0:
        from mpj_spark.applications.baseline_logreg import run_baseline_logreg

        print("\n" + "=" * 70)
        print("  [PARITY] LogReg — Running single-driver Spark BASELINE")
        print(f"  parity_iter = {epochs} epochs × {size} workers = {epochs * size} total steps")
        print("=" * 70)
        result, _ = run_baseline_logreg(
            input_file_path=LOGREG_DATASET_PATH,
            num_workers=size,
            parity_iter=epochs * size,   # fair: same total gradient steps as MPI multi-driver
        )
        baseline_weights = np.array(result["weight_vector"])
        baseline_intercept = float(result["intercept"])

    # ── All ranks: MPI multi-driver ──────────────────────────────────
    comm.Barrier()
    print(f"  [PARITY rank {rank}] LogReg — Running MPI Allreduce (N={size})...")
    from mpj_spark.applications.logreg.allreduce import run_logreg_allreduce

    mpi_result = run_logreg_allreduce(
        comm=comm,
        input_file=LOGREG_DATASET_PATH,
        epochs=epochs,
    )

    # ── Rank 0: compare ──────────────────────────────────────────────
    if rank == 0:
        mpi_weights = np.array(mpi_result["weight_vector"])
        mpi_intercept = float(mpi_result["intercept"])

        weight_delta = float(np.linalg.norm(baseline_weights - mpi_weights))
        intercept_delta = abs(baseline_intercept - mpi_intercept)

        records.append(
            _record(
                workload="logreg",
                metric="weight_vector_l2_delta",
                baseline_value=f"{np.linalg.norm(baseline_weights):.6f}",
                mpi_value=f"{np.linalg.norm(mpi_weights):.6f}",
                delta=weight_delta,
                within_tolerance=weight_delta < tolerance,
                tolerance=tolerance,
            )
        )
        records.append(
            _record(
                workload="logreg",
                metric="intercept_delta",
                baseline_value=f"{baseline_intercept:.6f}",
                mpi_value=f"{mpi_intercept:.6f}",
                delta=intercept_delta,
                within_tolerance=intercept_delta < tolerance,
                tolerance=tolerance,
            )
        )

    return records


# ── Helpers ───────────────────────────────────────────────────────────────────

def _record(workload, metric, baseline_value, mpi_value, delta, within_tolerance, tolerance):
    return {
        "workload": workload,
        "metric": metric,
        "baseline_value": baseline_value,
        "mpi_value": mpi_value,
        "delta": f"{delta:.8f}" if isinstance(delta, float) else str(delta),
        "within_tolerance": within_tolerance,
        "tolerance": tolerance,
    }


def write_report(records, run_id, num_workers):
    """Append results to parity_report.csv and print summary table."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    file_exists = os.path.exists(REPORT_PATH)
    with open(REPORT_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=REPORT_FIELDS)
        if not file_exists:
            writer.writeheader()
        for r in records:
            writer.writerow({"run_id": run_id, "num_workers": num_workers, **r})

    passed = sum(1 for r in records if r["within_tolerance"])
    total = len(records)

    print(f"\n{'=' * 70}")
    print(f"  PARITY REPORT — {run_id}  |  workers={num_workers}")
    print(f"{'=' * 70}")
    print(f"  {'Workload':<10} {'Metric':<38} {'Delta':<14} Status")
    print(f"  {'-' * 66}")
    for r in records:
        status = "PASS" if r["within_tolerance"] else "FAIL"
        print(f"  {r['workload']:<10} {r['metric']:<38} {r['delta']:<14} {status}")
    print(f"{'=' * 70}")
    print(f"  Result : {passed}/{total} checks passed")
    if passed == total:
        print("  Verdict: DATA INTEGRITY CONFIRMED")
    else:
        print(f"  Verdict: {total - passed} check(s) FAILED — review deltas above")
    print(f"{'=' * 70}")
    print(f"  Report saved -> {REPORT_PATH}\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Issue #10 — Baseline Spark vs MPI multi-driver data integrity validation"
    )
    parser.add_argument(
        "--tolerance", type=float, default=1e-3,
        help="L2 delta tolerance for centroid and weight comparisons (default: 1e-3)",
    )
    parser.add_argument("--kmeans-k", type=int, default=3, help="K for K-Means (default: 3)")
    parser.add_argument("--kmeans-iter", type=int, default=20, help="max_iter for K-Means (default: 20)")
    parser.add_argument("--logreg-epochs", type=int, default=30, help="Epochs for LogReg MPI (default: 30)")
    parser.add_argument("--skip-kmeans", action="store_true", help="Skip K-Means validation")
    parser.add_argument("--skip-logreg", action="store_true", help="Skip LogReg validation")
    args = parser.parse_args()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    run_id = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ") if rank == 0 else None

    if rank == 0:
        print(f"\n{'=' * 70}")
        print(f"  Issue #10 — Data Integrity: Baseline Spark vs MPI Multi-Driver")
        print(f"  MPI ranks   : {size}")
        print(f"  Tolerance   : {args.tolerance}")
        print(f"  KMeans path : {KMEANS_DATASET_PATH}")
        print(f"  LogReg path : {LOGREG_DATASET_PATH}")
        print(f"{'=' * 70}")

    all_records = []

    if not args.skip_kmeans:
        records = run_kmeans_parity(
            comm, rank, size,
            k=args.kmeans_k,
            max_iter=args.kmeans_iter,
            tolerance=args.tolerance,
        )
        if rank == 0:
            all_records.extend(records)

    if not args.skip_logreg:
        records = run_logreg_parity(
            comm, rank, size,
            epochs=args.logreg_epochs,
            tolerance=args.tolerance,
        )
        if rank == 0:
            all_records.extend(records)

    if rank == 0:
        write_report(all_records, run_id, num_workers=size)


if __name__ == "__main__":
    main()
