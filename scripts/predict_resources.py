#!/usr/bin/env python
# =============================================================================
# scripts/predict_resources.py
# Phase 6 — ask the trained resource predictor about one job (by hand)
#
#   python scripts/predict_resources.py --workload logreg --dataset-mb 100 \
#       --workers 2 --cores 1,2,4,8 --iterations 30 --features 20
#
# Prints one line per value of --cores (a comma list answers "what if I give
# it X cores?").  CPU is per driver, memory is the largest driver (every
# driver gets the same limit); "upper" = prediction + the calibrated margin
# of that workload.  --json prints the full
# PredictionResult records (mpj_spark/resource_prediction/predictor.py).
#
# CLI FLAGS
#   --model-dir PATH     trained models  (default: data/models/local_capped)
#   --model NAME         best_cv (default) | lookup | ridge | ridge_wl | rf
#   --workload NAME      kmeans | logreg | ...
#   --dataset-mb MB      --workers N (drivers)   --cores LIST
#   --heap-mb MB         only for models trained with the heap as input
#   --iterations N       planned iterations      --features N
#   --k N                K-Means clusters (default 0 = not K-Means)
#   --thread-cap N       BLAS/OpenMP threads (default: same as cores)
#   --trends NAME        policy (default) | local_only | off — growth trends
#                        beyond the measured range (predictor.TREND_POLICY)
#   --json               JSON output instead of a table
# =============================================================================

from __future__ import annotations

import argparse
import json
import os
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from mpj_spark.resource_prediction.predictor import (  # noqa: E402
    MODEL_CHOICES,
    TREND_CHOICES,
    PredictionRequest,
    ResourcePredictor,
)


def _parse_args(argv=None):
    p = argparse.ArgumentParser(description="Predict per-driver CPU and memory for one job.")
    p.add_argument(
        "--model-dir", default=os.path.join(_PROJECT_ROOT, "data", "models", "local_capped")
    )
    p.add_argument("--model", choices=MODEL_CHOICES, default="best_cv")
    p.add_argument("--workload", required=True)
    p.add_argument("--dataset-mb", type=float, required=True)
    p.add_argument("--workers", type=int, required=True)
    p.add_argument("--cores", required=True, help="cores per driver, e.g. 4 or 1,2,4,8")
    p.add_argument("--heap-mb", type=float)
    p.add_argument("--iterations", type=int)
    p.add_argument("--features", type=int)
    p.add_argument("--k", type=int, default=0)
    p.add_argument("--thread-cap", type=int)
    p.add_argument("--trends", choices=list(TREND_CHOICES), default="policy")
    p.add_argument("--json", action="store_true")
    return p.parse_args(argv)


def _cell(value, upper, digits):
    if value is None:
        return "-"
    return f"{value:.{digits}f} (≤{upper:.{digits}f})"


def main(argv=None):
    args = _parse_args(argv)
    predictor = ResourcePredictor.load(args.model_dir, args.model, args.trends)
    requests = [
        PredictionRequest(
            workload_type=args.workload,
            dataset_mb=args.dataset_mb,
            num_workers=args.workers,
            cores_per_driver=float(c),
            heap_mb_per_driver=args.heap_mb,
            iterations=args.iterations,
            num_features=args.features,
            num_clusters=args.k,
            thread_cap=args.thread_cap,
        )
        for c in args.cores.split(",")
    ]
    results = predictor.predict_many(requests)
    if args.json:
        print(json.dumps([r.to_dict() for r in results], indent=2))
        return results

    first = results[0]
    print(
        f"model {first.model_version} · training set {first.training_set} · "
        f"upper = {first.upper_quantile:.0%} margin · source {first.source}"
    )
    print("CPU per driver, memory of the largest driver; (≤x) = upper bound")
    print(
        f"{'cores':>5}  {'CPU p95 (cores)':>17}  {'CPU avg (cores)':>17}  "
        f"{'mem p95 (MB)':>19}  {'mem peak (MB)':>19}  status"
    )
    for req, res in zip(requests, results, strict=True):
        print(
            f"{req.cores_per_driver:>5g}  "
            f"{_cell(res.cpu_cores_p95, res.cpu_cores_p95_upper, 2):>17}  "
            f"{_cell(res.cpu_cores_mean, res.cpu_cores_mean_upper, 2):>17}  "
            f"{_cell(res.mem_mb_p95_max, res.mem_mb_p95_max_upper, 0):>19}  "
            f"{_cell(res.mem_mb_peak_max, res.mem_mb_peak_max_upper, 0):>19}  {res.status}"
        )
    remarks = dict.fromkeys(r for res in results for r in res.reasons + res.notes)
    for remark in remarks:
        print(f"  - {remark}")
    return results


if __name__ == "__main__":
    main()
