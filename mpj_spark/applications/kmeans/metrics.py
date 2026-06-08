# =============================================================================
# mpj_spark/applications/kmeans/metrics.py
# Phase 3 — Issue #8 — Step 6: Metrics Collection
#
# PURPOSE
# -------
# Own all measurement, aggregation, and persistence for the K-Means
# multi-driver Allreduce experiment.  Provides one class:
#
#   KMeansMetricsCollector
#
# which is instantiated once per MPI rank inside run_kmeans_allreduce()
# and records every timing and convergence signal produced by the loop.
#
# RESEARCH METRICS COVERED
# -------------------------
# The project scope defines six primary performance metrics:
#
#   1. Execution Time        → iter_time_s   (total wall time per iteration)
#   2. CPU Utilisation       → logged by resource_profiler.py (Phase 2a)
#   3. Memory Utilisation    → logged by resource_profiler.py (Phase 2a)
#   4. Throughput            → derived: dataset_size / total_time_s
#   5. Synchronization Overhead → sync_time_s / iter_time_s  ×100  (%)
#   6. Convergence Rate      → centroid_shift series across iterations
#
# This module provides metrics 1, 4, 5, and 6 directly.
# Metrics 2 and 3 remain in the resource profiler (separate concern).
#
# PER-ITERATION FIELDS
# --------------------
#   iteration       int    — 1-based iteration counter
#   spark_time_s    float  — wall time for the Spark action only
#                            (compute_local_stats + WCSS reduce)
#                            isolates JVM/PySpark overhead from MPI overhead
#   sync_time_s     float  — wall time inside Allreduce + WCSS Allreduce
#                            + convergence Bcast ONLY
#                            = pure MPI synchronisation cost
#   iter_time_s     float  — total iteration wall time
#                            (spark_time_s + sync_time_s + overhead)
#   centroid_shift  float  — Frobenius norm ||new - old||_F
#                            (convergence signal; approaches 0 at convergence)
#   global_wcss     float  — Within-Cluster Sum of Squares aggregated
#                            across all ranks via Allreduce
#                            (quality signal; should decrease monotonically)
#
# RUN-LEVEL FIELDS
# ----------------
#   rank            int    — this MPI rank
#   total_time_s    float  — wall time from Barrier to spark.stop()
#   iterations_run  int    — number of iterations actually executed
#   converged       bool   — True if tol was reached, False if max_iter hit
#   k               int    — number of clusters
#   dataset_size    int    — total data points across all ranks
#   num_ranks       int    — MPI world size
#   throughput      float  — dataset_size / total_time_s  (points/sec)
#
# OUTPUT FILES (per rank)
# -----------------------
#   {output_dir}/kmeans_metrics_rank{rank}.csv   — per-iteration CSV
#   {output_dir}/kmeans_metrics_rank{rank}.json  — full dict (run + iters)
#   {output_dir}/kmeans_metrics_aggregated.csv   — rank-aggregated summary
#                                                  (written by rank 0 only)
#
# WHY SEPARATE spark_time_s AND sync_time_s?
# ------------------------------------------
# The project's Synchronization Overhead metric is defined as the fraction
# of iteration time spent in MPI collectives (Allreduce + Bcast), NOT in
# Spark computation.  Conflating the two would make it impossible to
# attribute bottlenecks correctly:
#
#   High sync_time_s  → MPI layer is the bottleneck (network, serialisation)
#   High spark_time_s → Spark/JVM is the bottleneck (GC, scheduling)
#   High iter_time_s with low both → Python/numpy overhead
#
# The separation follows the measurement methodology in:
#   Theodorakopoulos et al. (2025) — Spark MLlib resource prediction
#   Saleh et al. (2025)           — MPJ-Spark timing decomposition
# =============================================================================

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# Required per-iteration field names in insertion order
_ITER_FIELDS = [
    "iteration",
    "spark_time_s",
    "sync_time_s",
    "iter_time_s",
    "centroid_shift",
    "global_wcss",
]

# Required run-level field names
_RUN_FIELDS = [
    "rank",
    "total_time_s",
    "iterations_run",
    "converged",
    "k",
    "dataset_size",
    "num_ranks",
    "throughput",
]


class KMeansMetricsCollector:
    """
    Accumulate, compute, and persist K-Means per-iteration and run-level
    performance metrics for one MPI rank.

    Usage pattern in run_kmeans_allreduce()
    ----------------------------------------
        collector = KMeansMetricsCollector(rank=rank, output_dir="./metrics")

        for iteration in range(1, max_iter + 1):
            t_iter = time.perf_counter()

            t_spark = time.perf_counter()
            local_sums, local_counts = compute_local_stats(...)
            local_wcss = _compute_local_wcss(...)
            spark_time = time.perf_counter() - t_spark

            t_sync = time.perf_counter()
            new_centroids = allreduce_centroids(...)
            ...  # WCSS Allreduce
            converged, shift = check_and_broadcast(...)
            sync_time = time.perf_counter() - t_sync

            iter_time = time.perf_counter() - t_iter

            collector.record_iteration(
                iteration=iteration,
                spark_time_s=spark_time,
                sync_time_s=sync_time,
                iter_time_s=iter_time,
                centroid_shift=shift,
                global_wcss=global_wcss,
            )

        collector.record_run(
            total_time_s=..., iterations_run=len(metrics),
            converged=converged, k=k,
            dataset_size=total_points, num_ranks=size,
        )
        collector.to_csv()
        collector.to_json()
    """

    def __init__(self, rank: int, output_dir: str = "./metrics") -> None:
        self.rank        = rank
        self.output_dir  = Path(output_dir)
        self._iterations: List[Dict] = []
        self._run:        Optional[Dict] = None
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    # Recording
    # -----------------------------------------------------------------------

    def record_iteration(
        self,
        iteration: int,
        spark_time_s: float,
        sync_time_s: float,
        iter_time_s: float,
        centroid_shift: float,
        global_wcss: float,
    ) -> None:
        """
        Append one iteration's timing and quality metrics.

        Parameters map 1-to-1 to _ITER_FIELDS.  All timing values are in
        seconds from time.perf_counter() differences.  Values are stored
        rounded to 6 decimal places for timing and 8 for shift/wcss to
        keep CSV files human-readable without losing precision needed for
        convergence analysis.
        """
        row = {
            "iteration"     : int(iteration),
            "spark_time_s"  : round(float(spark_time_s),   6),
            "sync_time_s"   : round(float(sync_time_s),    6),
            "iter_time_s"   : round(float(iter_time_s),    6),
            "centroid_shift": round(float(centroid_shift), 8),
            "global_wcss"   : round(float(global_wcss),    4),
        }
        self._iterations.append(row)
        logger.debug(
            "[rank %d] iter=%d  spark=%.4fs  sync=%.4fs  iter=%.4fs  "
            "shift=%.6f  wcss=%.2f",
            self.rank, iteration,
            spark_time_s, sync_time_s, iter_time_s,
            centroid_shift, global_wcss,
        )

    def record_run(
        self,
        total_time_s: float,
        iterations_run: int,
        converged: bool,
        k: int,
        dataset_size: int,
        num_ranks: int,
    ) -> None:
        """
        Record run-level summary.  Call exactly once after the loop exits.
        Computes throughput = dataset_size / total_time_s.
        """
        throughput = round(dataset_size / total_time_s, 2) if total_time_s > 0 else 0.0
        self._run = {
            "rank"          : self.rank,
            "total_time_s"  : round(float(total_time_s),  4),
            "iterations_run": int(iterations_run),
            "converged"     : bool(converged),
            "k"             : int(k),
            "dataset_size"  : int(dataset_size),
            "num_ranks"     : int(num_ranks),
            "throughput"    : throughput,
        }
        logger.info(
            "[rank %d] Run complete: %d iters, converged=%s, "
            "total=%.2fs, throughput=%.0f pts/s",
            self.rank, iterations_run, converged, total_time_s, throughput,
        )

    # -----------------------------------------------------------------------
    # Derived metrics
    # -----------------------------------------------------------------------

    def sync_overhead_pct(self) -> List[float]:
        """
        Synchronization Overhead (%) per iteration.

        Defined as: sync_time_s / iter_time_s * 100

        This is the primary metric for evaluating MPI coordination cost
        relative to total iteration cost.  A value near 100% means the
        iteration is almost entirely spent in MPI collectives — indicates
        very small local data partitions or network bottleneck.
        A value near 0% means Spark computation dominates.

        Returns an empty list if no iterations have been recorded.
        """
        result = []
        for row in self._iterations:
            if row["iter_time_s"] > 0:
                pct = row["sync_time_s"] / row["iter_time_s"] * 100.0
            else:
                pct = 0.0
            result.append(round(pct, 4))
        return result

    def convergence_rate(self) -> List[float]:
        """
        Centroid shift series across all recorded iterations.

        The convergence rate is the speed at which the centroid_shift
        decreases toward tol.  A steep early drop indicates fast convergence
        (typical for well-separated clusters); a slow monotonic decline
        indicates slow convergence (overlapping clusters or poor initialisation).

        The series can be plotted as: iteration (x) vs centroid_shift (y)
        on a log scale to visualise exponential convergence behaviour.

        Returns a list of float in iteration order.
        """
        return [row["centroid_shift"] for row in self._iterations]

    def wcss_series(self) -> List[float]:
        """
        Global WCSS values across iterations.
        Should be monotonically non-increasing for correct Allreduce sync.
        A WCSS increase between iterations signals a sync bug or empty-
        cluster reinitialisation event.
        """
        return [row["global_wcss"] for row in self._iterations]

    def summary_table(self) -> List[Dict]:
        """
        Return a list of dicts enriched with sync_overhead_pct for
        display or logging.  Each dict contains all _ITER_FIELDS plus
        'sync_overhead_pct'.  Rows are in iteration order.
        """
        overheads = self.sync_overhead_pct()
        table = []
        for row, overhead in zip(self._iterations, overheads):
            enriched = dict(row)
            enriched["sync_overhead_pct"] = overhead
            table.append(enriched)
        return table

    # -----------------------------------------------------------------------
    # Persistence
    # -----------------------------------------------------------------------

    def to_csv(self) -> Path:
        """
        Write per-iteration metrics to a CSV file.

        Path: {output_dir}/kmeans_metrics_rank{rank}.csv

        Columns: iteration, spark_time_s, sync_time_s, iter_time_s,
                 centroid_shift, global_wcss, sync_overhead_pct

        Returns the Path object of the written file.
        """
        path = self.output_dir / f"kmeans_metrics_rank{self.rank}.csv"
        overheads = self.sync_overhead_pct()
        fieldnames = _ITER_FIELDS + ["sync_overhead_pct"]

        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row, overhead in zip(self._iterations, overheads):
                writer.writerow({**row, "sync_overhead_pct": overhead})

        logger.info("[rank %d] Metrics CSV written → %s", self.rank, path)
        return path

    def to_json(self) -> Path:
        """
        Write full metrics dict (run summary + per-iteration rows) to JSON.

        Path: {output_dir}/kmeans_metrics_rank{rank}.json

        Structure:
            {
              "run": { ...run-level fields... },
              "iterations": [ {iteration row}, ... ],
              "derived": {
                "sync_overhead_pct":  [...],
                "convergence_rate":   [...],
                "wcss_series":        [...]
              }
            }

        Returns the Path object of the written file.
        """
        path = self.output_dir / f"kmeans_metrics_rank{self.rank}.json"
        payload = {
            "run"       : self._run or {},
            "iterations": self._iterations,
            "derived"   : {
                "sync_overhead_pct": self.sync_overhead_pct(),
                "convergence_rate" : self.convergence_rate(),
                "wcss_series"      : self.wcss_series(),
            },
        }
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)

        logger.info("[rank %d] Metrics JSON written → %s", self.rank, path)
        return path

    # -----------------------------------------------------------------------
    # Cross-rank aggregation  (called on rank 0 only after all ranks finish)
    # -----------------------------------------------------------------------

    @staticmethod
    def aggregate_across_ranks(
        output_dir: str,
        num_ranks: int,
    ) -> Path:
        """
        Read all per-rank CSV files and produce an aggregated summary.

        Called by rank 0 only, after all ranks have written their CSVs.
        Computes mean, max, and min of sync_overhead_pct and centroid_shift
        across all ranks, per iteration.  Writes the result to:

            {output_dir}/kmeans_metrics_aggregated.csv

        Aggregation is iteration-aligned: ranks must have run the same
        number of iterations (enforced by the synchronous Allreduce loop
        — all ranks exit on the same iteration by construction).

        Columns in aggregated CSV:
            iteration,
            sync_overhead_pct_mean, sync_overhead_pct_max, sync_overhead_pct_min,
            centroid_shift_mean,    centroid_shift_max,    centroid_shift_min,
            iter_time_s_mean,       iter_time_s_max,       iter_time_s_min,
            global_wcss_mean

        Returns the Path object of the written file.
        """
        import statistics

        out = Path(output_dir)

        # Load all per-rank CSVs
        all_rows: Dict[int, List[Dict]] = {}  # rank → list of row dicts
        for r in range(num_ranks):
            csv_path = out / f"kmeans_metrics_rank{r}.csv"
            if not csv_path.exists():
                logger.warning("Rank %d CSV not found at %s — skipping", r, csv_path)
                continue
            with open(csv_path, newline="") as f:
                all_rows[r] = list(csv.DictReader(f))

        if not all_rows:
            raise FileNotFoundError(
                f"No per-rank CSV files found in {output_dir}"
            )

        # Iteration count from the first available rank
        first_rank_rows = next(iter(all_rows.values()))
        n_iters = len(first_rank_rows)

        agg_rows = []
        for i in range(n_iters):
            sync_overheads = []
            shifts         = []
            iter_times     = []
            wcsss          = []

            for rank_rows in all_rows.values():
                if i >= len(rank_rows):
                    continue
                row = rank_rows[i]
                sync_overheads.append(float(row["sync_overhead_pct"]))
                shifts.append(float(row["centroid_shift"]))
                iter_times.append(float(row["iter_time_s"]))
                wcsss.append(float(row["global_wcss"]))

            agg_rows.append({
                "iteration"              : i + 1,
                "sync_overhead_pct_mean": round(statistics.mean(sync_overheads), 4),
                "sync_overhead_pct_max" : round(max(sync_overheads), 4),
                "sync_overhead_pct_min" : round(min(sync_overheads), 4),
                "centroid_shift_mean"   : round(statistics.mean(shifts), 8),
                "centroid_shift_max"    : round(max(shifts), 8),
                "centroid_shift_min"    : round(min(shifts), 8),
                "iter_time_s_mean"      : round(statistics.mean(iter_times), 6),
                "iter_time_s_max"       : round(max(iter_times), 6),
                "iter_time_s_min"       : round(min(iter_times), 6),
                "global_wcss_mean"      : round(statistics.mean(wcsss), 4),
            })

        agg_path = out / "kmeans_metrics_aggregated.csv"
        fieldnames = list(agg_rows[0].keys()) if agg_rows else []
        with open(agg_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(agg_rows)

        logger.info("Aggregated metrics written → %s", agg_path)
        return agg_path
