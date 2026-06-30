#!/usr/bin/env python3
"""
scripts/timing_analysis.py
==========================
Phase 3 — Controller Design Data Extraction

Reads the per-rank and aggregated metrics CSVs produced by
KMeansMetricsCollector and LogRegMetricsCollector after a
validate_parity.py run, derives all signals needed to design the
ML-aware resource allocator (Objectives 2a–2c), and writes four
structured output artefacts to results/timing/.

Controller-Relevant Signals
---------------------------
  spark_fraction    = spark_time_s / iter_time_s
                      > 0.6  → compute-bound  → allocate more CPU cores
                      < 0.4  → sync-bound     → reduce rank count / network tuning
                      0.4–0.6 → balanced

  sync_overhead_pct = sync_time_s / iter_time_s * 100
                      Primary bottleneck attribution metric.

  rank_variance_s   = max(iter_time_s across ranks) - min(iter_time_s across ranks)
                      Straggler detection signal for load-balance decisions.

  wcss_drop_rate    = (wcss[t-1] - wcss[t]) / wcss[t-1]  (K-Means only)
                      Convergence acceleration metric; used to decide early stop.

  loss_drop_rate    = (loss[t-1] - loss[t]) / loss[t-1]  (LogReg only)
                      Convergence acceleration metric.

  grad_norm_decay   = grad_norm[t] / grad_norm[0]  (LogReg only)
                      Normalised gradient decay; feeds LSTM feature vector.

Usage
-----
  python scripts/timing_analysis.py
  python scripts/timing_analysis.py --metrics-dir /path/to/metrics --out-dir /path/to/out
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        print(f"  [WARN] No data to write for {path.name} — skipping.")
        return
    fns = fieldnames or list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Written → {path}  ({len(rows)} rows)")


def _safe_float(val: Any, default: float = 0.0) -> float:
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def _describe(values: list[float]) -> dict[str, float]:
    """Return mean/max/min/std for a list of floats."""
    if not values:
        return {"mean": 0.0, "max": 0.0, "min": 0.0, "std": 0.0}
    return {
        "mean": round(statistics.mean(values), 6),
        "max":  round(max(values), 6),
        "min":  round(min(values), 6),
        "std":  round(statistics.pstdev(values), 6),
    }


# ---------------------------------------------------------------------------
# K-Means analysis
# ---------------------------------------------------------------------------

def analyse_kmeans(
    metrics_dir: Path,
    num_ranks: int,
) -> tuple[list[dict], list[dict], dict]:
    """
    Returns:
      iteration_detail  — list of per-iteration rows with derived signals
      rank_raw_rows     — flat list of all per-rank rows (for feature matrix)
      summary_stats     — dict of aggregate stats for timing_summary.csv
    """
    # Load aggregated CSV (written by rank 0)
    agg_path = metrics_dir / "kmeans_metrics_aggregated.csv"
    agg_rows = _read_csv(agg_path)

    # Load per-rank CSVs for rank variance
    per_rank: dict[int, list[dict]] = {}
    for r in range(num_ranks):
        rows = _read_csv(metrics_dir / f"kmeans_metrics_rank{r}.csv")
        if rows:
            per_rank[r] = rows

    if not agg_rows and not per_rank:
        print("  [WARN] No K-Means metrics found. Did you run validate_parity.py first?")
        return [], [], {}

    # If aggregated CSV is missing, build it from per-rank data
    if not agg_rows and per_rank:
        print("  [INFO] Aggregated K-Means CSV not found; computing from per-rank files.")
        n_iters = min(len(v) for v in per_rank.values())
        for i in range(n_iters):
            sync_vals, shift_vals, iter_vals, wcss_vals = [], [], [], []
            for rows in per_rank.values():
                row = rows[i]
                sync_vals.append(_safe_float(row.get("sync_overhead_pct", 0)))
                shift_vals.append(_safe_float(row.get("centroid_shift", 0)))
                iter_vals.append(_safe_float(row.get("iter_time_s", 0)))
                wcss_vals.append(_safe_float(row.get("global_wcss", 0)))
            agg_rows.append({
                "iteration": str(i + 1),
                "sync_overhead_pct_mean": str(round(statistics.mean(sync_vals), 4)),
                "sync_overhead_pct_max":  str(round(max(sync_vals), 4)),
                "sync_overhead_pct_min":  str(round(min(sync_vals), 4)),
                "centroid_shift_mean":    str(round(statistics.mean(shift_vals), 8)),
                "centroid_shift_max":     str(round(max(shift_vals), 8)),
                "centroid_shift_min":     str(round(min(shift_vals), 8)),
                "iter_time_s_mean":       str(round(statistics.mean(iter_vals), 6)),
                "iter_time_s_max":        str(round(max(iter_vals), 6)),
                "iter_time_s_min":        str(round(min(iter_vals), 6)),
                "global_wcss_mean":       str(round(statistics.mean(wcss_vals), 4)),
            })

    # --- Build iteration_detail with derived signals ---
    iteration_detail: list[dict] = []
    prev_wcss: float | None = None

    # Precompute per-rank iter_time_s per iteration for rank_variance
    rank_iter_times: dict[int, dict[int, float]] = {}
    for r, rows in per_rank.items():
        rank_iter_times[r] = {
            int(row["iteration"]): _safe_float(row.get("iter_time_s", 0))
            for row in rows
        }

    for agg in agg_rows:
        it = int(agg["iteration"])

        spark_time_mean = _safe_float(agg.get("spark_time_s_mean",
                          agg.get("spark_time_s", 0)))
        sync_time_mean  = _safe_float(agg.get("sync_time_s_mean",
                          agg.get("sync_time_s", 0)))
        iter_time_mean  = _safe_float(agg.get("iter_time_s_mean",
                          agg.get("iter_time_s", 0)))
        sync_pct_mean   = _safe_float(agg.get("sync_overhead_pct_mean", 0))
        shift_mean      = _safe_float(agg.get("centroid_shift_mean",
                          agg.get("centroid_shift", 0)))
        wcss_mean       = _safe_float(agg.get("global_wcss_mean",
                          agg.get("global_wcss", 0)))

        # spark_fraction: primary allocator decision signal
        spark_fraction = (
            round(spark_time_mean / iter_time_mean, 4)
            if iter_time_mean > 0 else 0.0
        )

        # Bottleneck classification
        if spark_fraction > 0.6:
            bottleneck = "compute"
        elif spark_fraction < 0.4:
            bottleneck = "sync"
        else:
            bottleneck = "balanced"

        # Rank variance (straggler signal)
        it_times_across_ranks = [
            rank_iter_times[r][it]
            for r in rank_iter_times
            if it in rank_iter_times[r]
        ]
        rank_variance_s = (
            round(max(it_times_across_ranks) - min(it_times_across_ranks), 6)
            if len(it_times_across_ranks) > 1 else 0.0
        )

        # WCSS drop rate (convergence acceleration)
        if prev_wcss is not None and prev_wcss > 0:
            wcss_drop_rate = round((prev_wcss - wcss_mean) / prev_wcss, 6)
        else:
            wcss_drop_rate = 0.0
        prev_wcss = wcss_mean

        # Re-read spark/sync from per-rank rank-0 if agg CSV lacks those columns
        if spark_time_mean == 0.0 and 0 in per_rank:
            r0_rows = {int(r["iteration"]): r for r in per_rank[0]}
            if it in r0_rows:
                spark_time_mean = _safe_float(r0_rows[it].get("spark_time_s", 0))
                sync_time_mean  = _safe_float(r0_rows[it].get("sync_time_s", 0))
                iter_time_mean  = _safe_float(r0_rows[it].get("iter_time_s", 0))
                sync_pct_mean   = _safe_float(r0_rows[it].get("sync_overhead_pct", 0))
                spark_fraction  = (
                    round(spark_time_mean / iter_time_mean, 4)
                    if iter_time_mean > 0 else 0.0
                )
                if spark_fraction > 0.6:
                    bottleneck = "compute"
                elif spark_fraction < 0.4:
                    bottleneck = "sync"
                else:
                    bottleneck = "balanced"

        iteration_detail.append({
            "iteration":          it,
            "spark_time_s":       round(spark_time_mean, 6),
            "sync_time_s":        round(sync_time_mean, 6),
            "iter_time_s":        round(iter_time_mean, 6),
            "sync_overhead_pct":  round(sync_pct_mean, 4),
            "spark_fraction":     spark_fraction,
            "bottleneck":         bottleneck,
            "centroid_shift":     round(shift_mean, 8),
            "global_wcss":        round(wcss_mean, 4),
            "wcss_drop_rate":     wcss_drop_rate,
            "rank_variance_s":    rank_variance_s,
        })

    # --- Flatten per-rank rows for feature matrix ---
    rank_raw_rows: list[dict] = []
    for r, rows in per_rank.items():
        for row in rows:
            rank_raw_rows.append({"workload": "kmeans", "rank": r, **row})

    # --- Summary stats ---
    spark_times  = [r["spark_time_s"]  for r in iteration_detail]
    sync_times   = [r["sync_time_s"]   for r in iteration_detail]
    iter_times   = [r["iter_time_s"]   for r in iteration_detail]
    sync_pcts    = [r["sync_overhead_pct"] for r in iteration_detail]
    spark_fracs  = [r["spark_fraction"] for r in iteration_detail]
    shifts       = [r["centroid_shift"] for r in iteration_detail]
    variances    = [r["rank_variance_s"] for r in iteration_detail]

    summary_stats = {
        "workload":              "kmeans",
        "iterations_run":        len(iteration_detail),
        "spark_time_mean_s":     _describe(spark_times)["mean"],
        "spark_time_max_s":      _describe(spark_times)["max"],
        "spark_time_std_s":      _describe(spark_times)["std"],
        "sync_time_mean_s":      _describe(sync_times)["mean"],
        "sync_time_max_s":       _describe(sync_times)["max"],
        "sync_time_std_s":       _describe(sync_times)["std"],
        "iter_time_mean_s":      _describe(iter_times)["mean"],
        "iter_time_max_s":       _describe(iter_times)["max"],
        "iter_time_std_s":       _describe(iter_times)["std"],
        "sync_overhead_pct_mean": _describe(sync_pcts)["mean"],
        "sync_overhead_pct_max":  _describe(sync_pcts)["max"],
        "spark_fraction_mean":   _describe(spark_fracs)["mean"],
        "spark_fraction_std":    _describe(spark_fracs)["std"],
        "centroid_shift_final":  shifts[-1] if shifts else 0.0,
        "rank_variance_mean_s":  _describe(variances)["mean"],
        "rank_variance_max_s":   _describe(variances)["max"],
        "bottleneck_mode":       _bottleneck_mode(spark_fracs),
    }

    return iteration_detail, rank_raw_rows, summary_stats


# ---------------------------------------------------------------------------
# LogReg analysis
# ---------------------------------------------------------------------------

def analyse_logreg(
    metrics_dir: Path,
    num_ranks: int,
) -> tuple[list[dict], list[dict], dict]:
    """
    Returns:
      epoch_detail   — per-epoch rows with all derived signals
      rank_raw_rows  — flat list of all per-rank rows
      summary_stats  — aggregate stats for timing_summary.csv
    """
    # Load per-rank CSVs
    per_rank: dict[int, list[dict]] = {}
    for r in range(num_ranks):
        rows = _read_csv(metrics_dir / f"logreg_rank{r}_epochs.csv")
        if rows:
            per_rank[r] = rows

    # Load aggregated CSV if available
    agg_path = metrics_dir / "logreg_all_ranks_epochs.csv"
    agg_rows = _read_csv(agg_path)

    if not per_rank and not agg_rows:
        print("  [WARN] No LogReg metrics found. Did you run validate_parity.py first?")
        return [], [], {}

    # Use rank-0 as the epoch reference (all ranks run the same epochs
    # due to synchronous Allreduce; rank-0 timings are representative)
    ref_rank = min(per_rank.keys()) if per_rank else None
    ref_rows = per_rank.get(ref_rank, [])

    # Build per-epoch timing across all ranks for variance
    rank_epoch_times: dict[int, dict[int, float]] = {}
    for r, rows in per_rank.items():
        rank_epoch_times[r] = {
            int(row["epoch"]): _safe_float(row.get("epoch_time_s", 0))
            for row in rows
        }

    epoch_detail: list[dict] = []
    first_grad_norm: float | None = None
    prev_loss: float | None = None

    for row in ref_rows:
        ep = int(row["epoch"])

        spark_time  = _safe_float(row.get("spark_time_s", 0))
        sync_time   = _safe_float(row.get("sync_time_s", 0))
        epoch_time  = _safe_float(row.get("epoch_time_s", 0))
        grad_norm   = _safe_float(row.get("grad_norm", 0))
        global_loss = _safe_float(row.get("global_loss", 0))
        weight_norm = _safe_float(row.get("weight_norm", 0))

        sync_overhead_pct = (
            round(sync_time / epoch_time * 100, 4) if epoch_time > 0 else 0.0
        )
        spark_fraction = (
            round(spark_time / epoch_time, 4) if epoch_time > 0 else 0.0
        )

        if spark_fraction > 0.6:
            bottleneck = "compute"
        elif spark_fraction < 0.4:
            bottleneck = "sync"
        else:
            bottleneck = "balanced"

        # Rank variance
        ep_times = [
            rank_epoch_times[r][ep]
            for r in rank_epoch_times
            if ep in rank_epoch_times[r]
        ]
        rank_variance_s = (
            round(max(ep_times) - min(ep_times), 6)
            if len(ep_times) > 1 else 0.0
        )

        # Loss drop rate
        if prev_loss is not None and prev_loss > 0:
            loss_drop_rate = round((prev_loss - global_loss) / prev_loss, 6)
        else:
            loss_drop_rate = 0.0
        prev_loss = global_loss

        # Gradient norm decay (normalised to epoch 0)
        if first_grad_norm is None:
            first_grad_norm = grad_norm if grad_norm > 0 else 1.0
        grad_norm_decay = (
            round(grad_norm / first_grad_norm, 6) if first_grad_norm > 0 else 0.0
        )

        epoch_detail.append({
            "epoch":             ep,
            "spark_time_s":      round(spark_time, 6),
            "sync_time_s":       round(sync_time, 6),
            "epoch_time_s":      round(epoch_time, 6),
            "sync_overhead_pct": sync_overhead_pct,
            "spark_fraction":    spark_fraction,
            "bottleneck":        bottleneck,
            "global_loss":       round(global_loss, 8),
            "loss_drop_rate":    loss_drop_rate,
            "grad_norm":         round(grad_norm, 8),
            "grad_norm_decay":   grad_norm_decay,
            "weight_norm":       round(weight_norm, 8),
            "rank_variance_s":   rank_variance_s,
        })

    # Flat rank rows
    rank_raw_rows: list[dict] = []
    for r, rows in per_rank.items():
        for row in rows:
            rank_raw_rows.append({"workload": "logreg", "rank": r, **row})

    spark_times  = [r["spark_time_s"]  for r in epoch_detail]
    sync_times   = [r["sync_time_s"]   for r in epoch_detail]
    epoch_times  = [r["epoch_time_s"]  for r in epoch_detail]
    sync_pcts    = [r["sync_overhead_pct"] for r in epoch_detail]
    spark_fracs  = [r["spark_fraction"] for r in epoch_detail]
    losses       = [r["global_loss"]   for r in epoch_detail]
    variances    = [r["rank_variance_s"] for r in epoch_detail]

    summary_stats = {
        "workload":              "logreg",
        "iterations_run":        len(epoch_detail),
        "spark_time_mean_s":     _describe(spark_times)["mean"],
        "spark_time_max_s":      _describe(spark_times)["max"],
        "spark_time_std_s":      _describe(spark_times)["std"],
        "sync_time_mean_s":      _describe(sync_times)["mean"],
        "sync_time_max_s":       _describe(sync_times)["max"],
        "sync_time_std_s":       _describe(sync_times)["std"],
        "iter_time_mean_s":      _describe(epoch_times)["mean"],
        "iter_time_max_s":       _describe(epoch_times)["max"],
        "iter_time_std_s":       _describe(epoch_times)["std"],
        "sync_overhead_pct_mean": _describe(sync_pcts)["mean"],
        "sync_overhead_pct_max":  _describe(sync_pcts)["max"],
        "spark_fraction_mean":   _describe(spark_fracs)["mean"],
        "spark_fraction_std":    _describe(spark_fracs)["std"],
        "final_loss":            losses[-1] if losses else 0.0,
        "rank_variance_mean_s":  _describe(variances)["mean"],
        "rank_variance_max_s":   _describe(variances)["max"],
        "bottleneck_mode":       _bottleneck_mode(spark_fracs),
    }

    return epoch_detail, rank_raw_rows, summary_stats


# ---------------------------------------------------------------------------
# Controller feature matrix
# ---------------------------------------------------------------------------

def build_feature_matrix(
    kmeans_detail: list[dict],
    logreg_detail: list[dict],
    kmeans_summary: dict,
    logreg_summary: dict,
) -> list[dict]:
    """
    Build a flattened feature vector per workload run suitable for an
    LSTM or regression-based resource demand predictor (Objective 2b).

    Each row represents ONE workload run (not one iteration).  The LSTM
    would consume the per-iteration sequences; this matrix provides the
    summary-level features used to train the regression model described
    in Objective 2b.

    Features
    --------
    Static (workload properties):
      workload_type      — kmeans | logreg  (categorical, 0/1 encoded)
      dataset_size_rows  — 540000 (fixed for current experiments)
      num_ranks          — 5
      iterations_run     — actual loop count

    Timing averages (Spark vs MPI split):
      spark_time_mean_s, sync_time_mean_s, iter_time_mean_s
      spark_time_std_s, sync_time_std_s, iter_time_std_s

    Bottleneck signals (allocator decision inputs):
      spark_fraction_mean, spark_fraction_std
      sync_overhead_pct_mean, sync_overhead_pct_max
      rank_variance_mean_s, rank_variance_max_s

    Convergence signals (iteration budget predictor input):
      final_convergence_value  — centroid_shift_final (kmeans) | final_loss (logreg)

    Controller target variables (what the allocator must predict):
      recommended_bottleneck   — compute | sync | balanced
      predictor_target_iter_s  — iter_time_mean_s (what LSTM predicts)
    """
    rows = []
    for summary, detail, label in [
        (kmeans_summary, kmeans_detail, "kmeans"),
        (logreg_summary, logreg_detail, "logreg"),
    ]:
        if not summary:
            continue
        row = {
            # Workload identity
            "workload_type":            label,
            "workload_type_encoded":    0 if label == "kmeans" else 1,
            "dataset_size_rows":        540000,
            "num_ranks":                5,
            "iterations_run":           summary.get("iterations_run", 0),

            # Spark vs MPI timing split
            "spark_time_mean_s":        summary.get("spark_time_mean_s", 0),
            "spark_time_std_s":         summary.get("spark_time_std_s", 0),
            "sync_time_mean_s":         summary.get("sync_time_mean_s", 0),
            "sync_time_std_s":          summary.get("sync_time_std_s", 0),
            "iter_time_mean_s":         summary.get("iter_time_mean_s", 0),
            "iter_time_std_s":          summary.get("iter_time_std_s", 0),

            # Bottleneck signals (primary allocator inputs)
            "spark_fraction_mean":      summary.get("spark_fraction_mean", 0),
            "spark_fraction_std":       summary.get("spark_fraction_std", 0),
            "sync_overhead_pct_mean":   summary.get("sync_overhead_pct_mean", 0),
            "sync_overhead_pct_max":    summary.get("sync_overhead_pct_max", 0),

            # Straggler / load-balance signals
            "rank_variance_mean_s":     summary.get("rank_variance_mean_s", 0),
            "rank_variance_max_s":      summary.get("rank_variance_max_s", 0),

            # Convergence signals
            "final_convergence_value":  (
                summary.get("centroid_shift_final", 0)
                if label == "kmeans"
                else summary.get("final_loss", 0)
            ),

            # Early-convergence profile: slope of first half vs second half
            "wcss_drop_rate_first_half":  _half_avg(detail, "wcss_drop_rate",   first=True),
            "wcss_drop_rate_second_half": _half_avg(detail, "wcss_drop_rate",   first=False),
            "loss_drop_rate_first_half":  _half_avg(detail, "loss_drop_rate",   first=True),
            "loss_drop_rate_second_half": _half_avg(detail, "loss_drop_rate",   first=False),
            "grad_norm_decay_final":      detail[-1].get("grad_norm_decay", 0) if detail else 0,

            # Controller target variables
            "bottleneck_mode":          summary.get("bottleneck_mode", "unknown"),
            "predictor_target_iter_s":  summary.get("iter_time_mean_s", 0),
        }
        rows.append(row)
    return rows


def _bottleneck_mode(spark_fracs: list[float]) -> str:
    """Return the dominant bottleneck classification across all iterations."""
    if not spark_fracs:
        return "unknown"
    compute = sum(1 for f in spark_fracs if f > 0.6)
    sync    = sum(1 for f in spark_fracs if f < 0.4)
    bal     = len(spark_fracs) - compute - sync
    majority = max([(compute, "compute"), (sync, "sync"), (bal, "balanced")],
                   key=lambda x: x[0])
    return majority[1]


def _half_avg(detail: list[dict], field: str, first: bool) -> float:
    """Mean of a field over the first or second half of iterations."""
    if not detail:
        return 0.0
    half = len(detail) // 2
    subset = detail[:half] if first else detail[half:]
    vals = [_safe_float(r.get(field, 0)) for r in subset]
    return round(statistics.mean(vals), 6) if vals else 0.0


# ---------------------------------------------------------------------------
# Pretty printer
# ---------------------------------------------------------------------------

def _print_section(title: str, data: dict) -> None:
    width = 52
    print(f"\n{'='*width}")
    print(f"  {title}")
    print(f"{'='*width}")
    for k, v in data.items():
        if isinstance(v, float):
            print(f"  {k:<38} {v:.6f}")
        else:
            print(f"  {k:<38} {v}")


def _print_iteration_table(rows: list[dict], workload: str, max_rows: int = 20) -> None:
    if not rows:
        return
    keys = ["iteration", "spark_time_s", "sync_time_s",
            "sync_overhead_pct", "spark_fraction", "bottleneck"]
    # For logreg, 'iteration' column is named 'epoch'
    if "epoch" in rows[0]:
        keys[0] = "epoch"
    print(f"\n  {workload.upper()} per-step timing (first {min(max_rows,len(rows))} rows)")
    header = "  ".join(f"{k:>18}" for k in keys)
    print("  " + header)
    print("  " + "-" * len(header))
    for row in rows[:max_rows]:
        line = "  ".join(
            f"{str(row.get(k, ''))[:18]:>18}" for k in keys
        )
        print("  " + line)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 3 timing analysis — extract controller design data"
    )
    parser.add_argument(
        "--metrics-dir",
        default="./metrics",
        help="Directory containing per-rank and aggregated metrics CSVs "
             "(default: ./metrics)",
    )
    parser.add_argument(
        "--out-dir",
        default="./results/timing",
        help="Output directory for analysis artefacts (default: ./results/timing)",
    )
    parser.add_argument(
        "--num-ranks",
        type=int,
        default=5,
        help="Number of MPI ranks used in the experiment (default: 5)",
    )
    args = parser.parse_args()

    metrics_dir = Path(args.metrics_dir)
    out_dir     = Path(args.out_dir)
    num_ranks   = args.num_ranks

    print(f"""
{'='*60}
  Phase 3 — Timing Analysis for Controller Design
  metrics dir : {metrics_dir}
  output dir  : {out_dir}
  num ranks   : {num_ranks}
{'='*60}""")

    # -----------------------------------------------------------------------
    # K-Means
    # -----------------------------------------------------------------------
    print("\n[1/4] Analysing K-Means metrics...")
    km_iter_detail, km_rank_rows, km_summary = analyse_kmeans(metrics_dir, num_ranks)

    if km_summary:
        _print_section("K-Means Timing Summary", km_summary)
        _print_iteration_table(km_iter_detail, "kmeans")

    # -----------------------------------------------------------------------
    # LogReg
    # -----------------------------------------------------------------------
    print("\n[2/4] Analysing Logistic Regression metrics...")
    lr_epoch_detail, lr_rank_rows, lr_summary = analyse_logreg(metrics_dir, num_ranks)

    if lr_summary:
        _print_section("LogReg Timing Summary", lr_summary)
        _print_iteration_table(lr_epoch_detail, "logreg")

    # -----------------------------------------------------------------------
    # Write outputs
    # -----------------------------------------------------------------------
    print("\n[3/4] Writing output files...")

    # timing_summary.csv — one row per workload
    summary_rows = [r for r in [km_summary, lr_summary] if r]
    summary_fields = [
        "workload", "iterations_run",
        "spark_time_mean_s", "spark_time_max_s", "spark_time_std_s",
        "sync_time_mean_s",  "sync_time_max_s",  "sync_time_std_s",
        "iter_time_mean_s",  "iter_time_max_s",  "iter_time_std_s",
        "sync_overhead_pct_mean", "sync_overhead_pct_max",
        "spark_fraction_mean", "spark_fraction_std",
        "rank_variance_mean_s", "rank_variance_max_s",
        "bottleneck_mode",
    ]
    _write_csv(out_dir / "timing_summary.csv", summary_rows, summary_fields)

    # kmeans_iteration_detail.csv
    if km_iter_detail:
        _write_csv(out_dir / "kmeans_iteration_detail.csv", km_iter_detail)

    # logreg_epoch_detail.csv
    if lr_epoch_detail:
        _write_csv(out_dir / "logreg_epoch_detail.csv", lr_epoch_detail)

    # controller_feature_matrix.csv
    print("\n[4/4] Building controller feature matrix...")
    feature_matrix = build_feature_matrix(
        km_iter_detail, lr_epoch_detail, km_summary, lr_summary
    )
    if feature_matrix:
        _write_csv(out_dir / "controller_feature_matrix.csv", feature_matrix)
        _print_section("Controller Feature Matrix (row 0)", feature_matrix[0])
        if len(feature_matrix) > 1:
            _print_section("Controller Feature Matrix (row 1)", feature_matrix[1])

    # -----------------------------------------------------------------------
    # Controller interpretation
    # -----------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("  CONTROLLER DESIGN INTERPRETATION")
    print(f"{'='*60}")
    for s in [km_summary, lr_summary]:
        if not s:
            continue
        wl   = s.get("workload", "?")
        bm   = s.get("bottleneck_mode", "unknown")
        spf  = s.get("spark_fraction_mean", 0)
        sop  = s.get("sync_overhead_pct_mean", 0)
        var  = s.get("rank_variance_max_s", 0)

        print(f"\n  Workload : {wl}")
        print(f"  Bottleneck mode      : {bm}")
        print(f"  Spark fraction (mean): {spf:.3f}")
        print(f"  Sync overhead (mean) : {sop:.2f} %")
        print(f"  Rank variance (max)  : {var:.4f} s")

        if bm == "compute":
            print("  => Allocator action  : INCREASE cores_per_worker (more Spark parallelism)")
        elif bm == "sync":
            print("  => Allocator action  : REDUCE num_ranks OR tune MPI transport (UCX)")
        else:
            print("  => Allocator action  : HOLD current allocation (balanced)")

        if var > 0.5:
            print(f"  => Straggler detected (rank_variance={var:.4f}s): "
                  "consider data re-partitioning")

    print(f"\n  Output artefacts written to: {out_dir}")
    print(f"  Use controller_feature_matrix.csv as input to Objective 2b model.")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
