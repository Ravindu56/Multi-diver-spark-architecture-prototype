#!/usr/bin/env python3
# =============================================================================
# scripts/validate_parity.py
#
# Issue #10 — Data Integrity: Baseline Spark vs MPI Multi-Driver
#
# Parity criteria (academically correct):
#
#   K-Means  : wcss_relative_delta <= 0.01  (1 %)
#              Centroid L2 comparison removed — invalid when baseline and MPI
#              drivers initialise from different data pools (full vs. partition).
#              WCSS-based parity is the standard per Lloyd (1982) / k-means++.
#
#   LogReg   : accuracy_delta <= 0.03  (3 percentage points)
#              Weight-vector L2 and intercept checks removed — meaningless
#              between L-BFGS (baseline) and mini-batch SGD (MPI path).
#              Decision quality (accuracy on a held-out sample) is the correct
#              criterion when comparing two different optimisers.
#
# Usage (5 MPI ranks = 1 root + 4 workers):
#   mpirun --oversubscribe -n 5 \
#     --mca hwloc_base_binding_policy none \
#     python scripts/validate_parity.py
#
# CLI flags:
#   --skip-kmeans            skip K-Means parity check
#   --skip-logreg            skip Logistic Regression parity check
#   --kmeans-tol  FLOAT      WCSS relative-delta threshold      [default: 0.01]
#   --logreg-tol  FLOAT      accuracy delta threshold (0-1)     [default: 0.03]
#   --k           INT        number of K-Means clusters         [default: 3]
#   --max-iter    INT        max iterations                     [default: 20]
#   --report-dir  PATH       directory for parity CSV report    [default: results/]
#   --metrics-dir PATH       directory for per-rank metrics CSVs[default: metrics/]
#   --accuracy-sample INT    rows used for accuracy comparison  [default: 1000]
# =============================================================================
from __future__ import annotations

import argparse
import csv
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

from mpj_spark.config import (  # noqa: E402
    KMEANS_DATASET_PATH,
    LOGREG_DATASET_PATH,
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
    return {"metric": label, "delta": delta, "tolerance": tol, "status": status}

def _validate_runtime(
    kmeans_enabled: bool,
    logreg_enabled: bool,
    metrics_dir: Path,
    report_dir: Path,
) -> None:
    if size < 2:
        if rank == 0:
            print(
                "[ERROR] validate_parity.py requires at least 2 MPI ranks: "
                "one coordinator plus one driver."
            )
        raise SystemExit(2)

    if rank == 0:
        missing: list[str] = []

        if kmeans_enabled and not Path(KMEANS_DATASET_PATH).is_file():
            missing.append(f"K-Means dataset: {KMEANS_DATASET_PATH}")

        if logreg_enabled and not Path(LOGREG_DATASET_PATH).is_file():
            missing.append(f"LogReg dataset: {LOGREG_DATASET_PATH}")

        if missing:
            print("[ERROR] Required input files are missing:")
            for item in missing:
                print(f"  - {item}")
            print("Run: python /app/scripts/generate_datasets.py")
            raise SystemExit(2)

        metrics_dir.mkdir(parents=True, exist_ok=True)
        report_dir.mkdir(parents=True, exist_ok=True)

    comm.Barrier()

# ---------------------------------------------------------------------------
# K-Means parity
# ---------------------------------------------------------------------------


def run_kmeans_parity(
    k: int,
    max_iter: int,
    wcss_tol: float,
    metrics_dir: str,
) -> list[dict]:
    """
    Parity check for K-Means.

    Only the WCSS relative delta is checked.  Centroid-by-centroid L2
    comparison has been intentionally removed: the baseline initialises
    k-means++ on the full dataset while each MPI driver initialises on its
    local partition → different starting centroids → different local optima.
    WCSS measures objective-value equivalence and is independent of centroid
    label assignment or initialisation pool.
    """

    from mpj_spark.applications.baseline_kmeans import run_baseline_kmeans
    from mpj_spark.applications.kmeans.driver import run_kmeans_driver

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
        metrics_output_dir=metrics_dir,  # ← FIX: was never passed before
    )

    records: list[dict] = []
    if rank == 0:
        wcss_baseline = float(baseline_result["wcss"])
        wcss_mpi = float(mpj_result["wcss"])
        wcss_delta = abs(wcss_baseline - wcss_mpi) / max(abs(wcss_baseline), 1e-9)
        records.append(
            {**_check("wcss_relative_delta", wcss_delta, wcss_tol), "workload": "kmeans"}
        )

        print(
            f"\n  KMeans WCSS  baseline={wcss_baseline:.2f}  "
            f"mpi={wcss_mpi:.2f}  "
            f"rel_delta={wcss_delta:.6f}  tol={wcss_tol}"
        )

    return records


# ---------------------------------------------------------------------------
# Logistic Regression parity
# ---------------------------------------------------------------------------


def _predict_accuracy(
    weights: "np.ndarray", intercept: float, X: "np.ndarray", y: "np.ndarray"
) -> float:
    """Sigmoid prediction accuracy using numpy — no Spark dependency."""
    import numpy as np

    logits = X @ weights + intercept
    preds = (1.0 / (1.0 + np.exp(-logits)) >= 0.5).astype(int)
    return float(np.mean(preds == y))


def run_logreg_parity(
    max_iter: int,
    accuracy_tol: float,
    accuracy_sample: int,
    metrics_dir: str,
) -> list[dict]:
    """
    Parity check for Logistic Regression.

    Accuracy-delta replaces weight-vector L2 and intercept checks because:
      - The baseline uses MLlib L-BFGS (second-order, converges in ~15 steps).
      - The MPI path uses mini-batch SGD (first-order, needs many more epochs).
      - Raw weight-vector identity between two different optimisers is never a
        valid parity criterion.
      - The intercept in the MLlib result is from a standardised L-BFGS solve;
        the MPI path returns intercept=0.0 (bias folded into weights).
        Comparing these two numbers is meaningless.
    Decision quality (accuracy on a shared sample) is the correct criterion.
    """
    import numpy as np

    from mpj_spark.applications.baseline_logreg import run_baseline_logreg
    from mpj_spark.applications.logreg.driver import run_logreg_driver

    _banner("[PARITY] Logistic Regression — Running single-driver Spark BASELINE")
    baseline_result = None
    if rank == 0:
        baseline_result, _ = run_baseline_logreg(
            LOGREG_DATASET_PATH,
            num_workers=size - 1,
            cores_override=None,
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
        metrics_output_dir=metrics_dir,  # ← FIX: was never passed before
    )

    records: list[dict] = []
    if rank == 0:
        try:
            sample_X, sample_y = _load_sample_numpy(LOGREG_DATASET_PATH, n=accuracy_sample)

            base_w = np.array(baseline_result["weight_vector"])
            base_intercept = float(baseline_result.get("intercept", 0.0))
            mpj_w = np.array(mpj_result["weights"])
            mpj_intercept = float(mpj_result.get("intercept", 0.0))

            acc_baseline = _predict_accuracy(base_w, base_intercept, sample_X, sample_y)
            acc_mpi = _predict_accuracy(mpj_w, mpj_intercept, sample_X, sample_y)
            acc_delta = abs(acc_baseline - acc_mpi)

            print(
                f"\n  LogReg accuracy  baseline={acc_baseline:.4f}  "
                f"mpi={acc_mpi:.4f}  "
                f"delta={acc_delta:.6f}  tol={accuracy_tol}"
            )

            records.append(
                {
                    **_check("accuracy_delta", acc_delta, accuracy_tol),
                    "workload": "logreg",
                }
            )
        except Exception as exc:  # noqa: BLE001
            print(f"\n  [WARN] accuracy sample load failed: {exc}")
            records.append(
                {
                    "metric": "accuracy_delta",
                    "delta": -1.0,
                    "tolerance": accuracy_tol,
                    "status": "ERROR",
                    "workload": "logreg",
                }
            )

    return records


def _load_sample_numpy(dataset_path: str, n: int) -> "tuple[np.ndarray, np.ndarray]":
    """
    Load the first n rows of the CSV at dataset_path into (X, y) numpy arrays.
    Assumes the last column is the binary label (0/1 or -1/+1 mapped to 0/1).
    """
    import numpy as np

    rows = []
    with open(dataset_path, newline="") as f:
        f.readline()  # skip header
        for i, line in enumerate(f):
            if i >= n:
                break
            rows.append([float(v) for v in line.strip().split(",")])

    data = np.array(rows, dtype=np.float64)
    X = data[:, :-1]
    y_raw = data[:, -1].astype(int)
    # normalise labels to 0/1 in case dataset uses -1/+1
    y = ((y_raw + 1) // 2).clip(0, 1)
    return X, y


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def write_report(records: list[dict], report_dir: str, run_id: str) -> None:
    Path(report_dir).mkdir(parents=True, exist_ok=True)
    path = Path(report_dir) / "parity_report.csv"
    fieldnames = ["run_id", "workload", "metric", "delta", "tolerance", "status"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in records:
            writer.writerow({"run_id": run_id, **r})
    print(f"\n  Report written → {path}")


def print_summary(records: list[dict]) -> None:
    passed = sum(1 for r in records if r["status"] == "PASS")
    total = len(records)
    verdict = "DATA INTEGRITY CONFIRMED" if passed == total else "DATA INTEGRITY FAILED"

    print("\n" + "=" * 70)
    print(f"  {'Workload':<10} {'Metric':<35} {'Delta':<12} {'Tol':<8} {'Status'}")
    print("  " + "-" * 66)
    for r in records:
        delta_str = (
            f"{r['delta']:.8f}"
            if isinstance(r["delta"], float) and r["delta"] >= 0
            else str(r["delta"])
        )
        print(
            f"  {r['workload']:<10} {r['metric']:<35} {delta_str:<12} "
            f"{r.get('tolerance', ''):<8} {r['status']}"
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
    parser.add_argument(
        "--kmeans-tol",
        type=float,
        default=0.01,
        help="WCSS relative-delta threshold for K-Means PASS (default: 0.01 = 1%%)",
    )
    parser.add_argument(
        "--logreg-tol",
        type=float,
        default=0.03,
        help="Accuracy delta threshold for LogReg PASS (default: 0.03 = 3 pp)",
    )
    # Legacy alias
    parser.add_argument(
        "--tolerance",
        type=float,
        default=None,
        help="(legacy) sets both --kmeans-tol and --logreg-tol to the same value",
    )
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--max-iter", type=int, default=20)
    parser.add_argument(
        "--report-dir",
        default=os.getenv("MPJ_PARITY_REPORT_DIR", "/data/results/parity"),
    )

    parser.add_argument(
        "--metrics-dir",
        default=os.getenv("MPJ_PARITY_METRICS_DIR", "/data/metrics/parity"),
    )
    parser.add_argument(
        "--accuracy-sample",
        type=int,
        default=1000,
        help="Number of rows used for LogReg accuracy comparison (default: 1000)",
    )
    args = parser.parse_args()

    if args.skip_kmeans and args.skip_logreg:
        parser.error("At least one workload must be enabled.")

    if args.k <= 1:
        parser.error("--k must be greater than 1.")

    if args.max_iter <= 0:
        parser.error("--max-iter must be greater than 0.")

    if not 0.0 <= args.kmeans_tol <= 1.0:
        parser.error("--kmeans-tol must be in [0, 1].")

    if not 0.0 <= args.logreg_tol <= 1.0:
        parser.error("--logreg-tol must be in [0, 1].")

    _validate_runtime(
        kmeans_enabled=not args.skip_kmeans,
        logreg_enabled=not args.skip_logreg,
        metrics_dir=Path(args.metrics_dir),
        report_dir=Path(args.report_dir),
    )

    # --tolerance legacy override
    if args.tolerance is not None:
        args.kmeans_tol = args.tolerance
        args.logreg_tol = args.tolerance

    # Create metrics dir before any driver runs so all ranks can write to it
    Path(args.metrics_dir).mkdir(parents=True, exist_ok=True)

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") if rank == 0 else None
    run_id = comm.bcast(run_id, root=0)

    if rank == 0:
        print("\n" + "=" * 70)
        print("  Issue #10 — Data Integrity: Baseline Spark vs MPI Multi-Driver")
        print(f"  MPI ranks        : {size}")
        print(f"  KMeans tol (WCSS): {args.kmeans_tol}  (1 % relative delta)")
        print(f"  LogReg tol (acc) : {args.logreg_tol}  (3 pp accuracy delta)")
        print(f"  KMeans path      : {KMEANS_DATASET_PATH}")
        print(f"  LogReg path      : {LOGREG_DATASET_PATH}")
        print(f"  Metrics output   : {args.metrics_dir}/")
        print("=" * 70)

    all_records: list[dict] = []

    if not args.skip_kmeans:
        records = run_kmeans_parity(
            k=args.k,
            max_iter=args.max_iter,
            wcss_tol=args.kmeans_tol,
            metrics_dir=args.metrics_dir,
        )
        if rank == 0:
            all_records.extend(records)

    if not args.skip_logreg:
        records = run_logreg_parity(
            max_iter=args.max_iter,
            accuracy_tol=args.logreg_tol,
            accuracy_sample=args.accuracy_sample,
            metrics_dir=args.metrics_dir,
        )
        if rank == 0:
            all_records.extend(records)

    if rank == 0:
        write_report(all_records, args.report_dir, run_id)
        print_summary(all_records)

        # Remind the user to run the timing analysis next
        print(f"\n  Metrics written to: {args.metrics_dir}/")
        print(f"  Next step → python scripts/timing_analysis.py --metrics-dir {args.metrics_dir}")
        print()


if __name__ == "__main__":
    main()
