# =============================================================================
# mpj_spark/applications/kmeans/allreduce.py
# Phase 3 — Issue #8 — Step 4 (refactored Steps 5 + 6)
#
# CHANGE LOG (Step 6 integration)
# --------------------------------
# - Instantiates KMeansMetricsCollector at the top of run_kmeans_allreduce.
# - Added spark_time_s measurement window: time.perf_counter() bracket
#   around compute_local_stats() + _compute_local_wcss() only — isolates
#   the Spark action cost from the MPI synchronisation window.
# - sync_time_s window now starts before allreduce_centroids() and ends
#   after check_and_broadcast() — covers the full MPI-collective region.
# - Calls collector.record_iteration() inside the loop.
# - Calls collector.record_run() + collector.to_csv() + collector.to_json()
#   after the loop, before spark.stop().
# - rank 0 calls KMeansMetricsCollector.aggregate_across_ranks() at the end.
# - The bare `metrics: List[Dict]` list is replaced by the collector;
#   run return dict now reads from collector._iterations and collector._run.
# - All Step 4 and Step 5 logic (Allreduce, empty-cluster guard, WCSS
#   Allreduce, convergence check, comm.Barrier) is unchanged.
# =============================================================================

from __future__ import annotations

import logging
import time
from typing import Dict, List

import numpy as np

logger = logging.getLogger(__name__)

_EMPTY_CLUSTER_THRESHOLD = 0.5


# ===========================================================================
# Core Allreduce primitive  (unchanged from Steps 4 + 5)
# ===========================================================================

def allreduce_centroids(
    comm,
    rank: int,
    local_sums: np.ndarray,
    local_counts: np.ndarray,
    points_rdd,
) -> np.ndarray:
    """
    Aggregate local centroid sums and counts across all MPI ranks and
    compute the new global centroid array.  Buffer-level MPI.SUM Allreduce
    on local_sums (K×D) and local_counts (K,).  Empty-cluster guard: rank 0
    samples a replacement and Bcasts it.
    """
    from mpi4py import MPI

    k, d = local_sums.shape
    global_sums   = np.zeros_like(local_sums)
    global_counts = np.zeros_like(local_counts)

    comm.Allreduce([local_sums,   MPI.DOUBLE], [global_sums,   MPI.DOUBLE], op=MPI.SUM)
    comm.Allreduce([local_counts, MPI.DOUBLE], [global_counts, MPI.DOUBLE], op=MPI.SUM)

    global_centroids = np.zeros((k, d), dtype=np.float64)
    empty_clusters: List[int] = []

    for j in range(k):
        if global_counts[j] >= _EMPTY_CLUSTER_THRESHOLD:
            global_centroids[j] = global_sums[j] / global_counts[j]
        else:
            empty_clusters.append(j)
            logger.warning(
                "[rank %d] Cluster %d empty (global_count=%.1f) — reinitialising.",
                rank, j, global_counts[j],
            )

    for j in empty_clusters:
        if rank == 0:
            sample      = points_rdd.takeSample(False, 1, seed=int(time.time() * 1000))
            replacement = np.array(sample[0], dtype=np.float64)
        else:
            replacement = np.zeros(d, dtype=np.float64)
        comm.Bcast([replacement, MPI.DOUBLE], root=0)
        global_centroids[j] = replacement
        logger.info("[rank %d] Cluster %d reinitialised → %s", rank, j, replacement[:4])

    return global_centroids


# ===========================================================================
# Full K-Means Allreduce runner  (Steps 2–6 orchestration)
# ===========================================================================

def run_kmeans_allreduce(
    comm,
    rank: int,
    size: int,
    input_file: str,
    k: int = 3,
    max_iter: int = 20,
    tol: float = 1e-4,
    seed: int = 42,
    cores_override: int | None = None,
    metrics_output_dir: str = "./metrics",
) -> Dict:
    """
    Full multi-driver K-Means with synchronous Allreduce centroid sync
    and per-iteration metrics collection (Step 6).

    New parameter
    -------------
    metrics_output_dir : str
        Directory where per-rank CSV and JSON files are written.
        Rank 0 also writes the aggregated cross-rank CSV here.
        Default: './metrics' (relative to the working directory).

    Returns
    -------
    dict:
        global_centroids  : list[list[float]]
        iterations_run    : int
        converged         : bool
        metrics           : list[dict]  — one row per iteration with all
                            6 fields + sync_overhead_pct
        run_summary       : dict        — run-level fields
        rank              : int
        total_time_s      : float
    """
    from mpi4py import MPI
    from mpj_spark.applications.kmeans.partition import partition_and_init_spark
    from mpj_spark.applications.kmeans.local_iteration import (
        load_partition_rdd,
        init_centroids,
        compute_local_stats,
    )
    from mpj_spark.applications.kmeans.convergence import check_and_broadcast
    from mpj_spark.applications.kmeans.metrics import KMeansMetricsCollector

    t_total_start = time.perf_counter()

    # Step 6: one collector per rank, writes to metrics_output_dir
    collector = KMeansMetricsCollector(rank=rank, output_dir=metrics_output_dir)

    # ---- Step 2: partition + scatter + Spark session -------------------
    partition_path, spark = partition_and_init_spark(
        comm=comm, rank=rank, size=size,
        input_file=input_file, num_workers=size,
        cores_override=cores_override,
    )

    # ---- Step 3 setup: load RDD + init centroids -----------------------
    points_rdd   = load_partition_rdd(spark, partition_path)
    centroids    = init_centroids(points_rdd, k=k, seed=seed)
    total_points = points_rdd.count()  # used for throughput calculation

    comm.Barrier()
    logger.info("[rank %d] Barrier passed — entering iteration loop", rank)

    prev_centroids = np.zeros_like(centroids)
    converged      = False

    # ---- Main loop: Steps 3 + 4 + 5 + 6 interleaved -------------------
    for iteration in range(1, max_iter + 1):
        t_iter_start = time.perf_counter()

        # --------------------------------------------------------------- #
        # Step 3 window: Spark action only                                 #
        # Timed separately so spark_time_s ≠ sync_time_s in the output.   #
        # --------------------------------------------------------------- #
        t_spark_start = time.perf_counter()
        local_sums, local_counts = compute_local_stats(points_rdd, centroids)
        local_wcss = _compute_local_wcss(points_rdd, centroids)
        spark_time = time.perf_counter() - t_spark_start

        # --------------------------------------------------------------- #
        # Step 4 + 5 window: MPI collectives only                          #
        # Starts at first Allreduce; ends after convergence Bcast.         #
        # --------------------------------------------------------------- #
        t_sync_start = time.perf_counter()

        new_centroids = allreduce_centroids(
            comm, rank, local_sums, local_counts, points_rdd
        )

        # WCSS Allreduce (scalar — counted inside the sync window)
        global_wcss_arr = np.array([local_wcss], dtype=np.float64)
        global_wcss_buf = np.zeros(1, dtype=np.float64)
        comm.Allreduce(
            [global_wcss_arr, MPI.DOUBLE],
            [global_wcss_buf, MPI.DOUBLE],
            op=MPI.SUM,
        )
        global_wcss = float(global_wcss_buf[0])

        # Step 5: convergence check + broadcast (inside sync window)
        converged, centroid_shift = check_and_broadcast(
            comm, rank, new_centroids, prev_centroids, tol, iteration
        )

        sync_time = time.perf_counter() - t_sync_start
        iter_time = time.perf_counter() - t_iter_start

        # --------------------------------------------------------------- #
        # Step 6: record this iteration                                    #
        # --------------------------------------------------------------- #
        collector.record_iteration(
            iteration=iteration,
            spark_time_s=spark_time,
            sync_time_s=sync_time,
            iter_time_s=iter_time,
            centroid_shift=centroid_shift,
            global_wcss=global_wcss,
        )

        logger.info(
            "[rank %d] iter=%d  spark=%.4fs  sync=%.4fs  iter=%.4fs  "
            "shift=%.6f  wcss=%.2f",
            rank, iteration, spark_time, sync_time, iter_time,
            centroid_shift, global_wcss,
        )

        prev_centroids = centroids
        centroids      = new_centroids

        if converged:
            logger.info(
                "[rank %d] Converged at iteration %d (shift=%.6f < tol=%.6f)",
                rank, iteration, centroid_shift, tol,
            )
            break

    # ---- Step 6: run-level record + file output ------------------------
    total_time = time.perf_counter() - t_total_start

    collector.record_run(
        total_time_s=total_time,
        iterations_run=len(collector._iterations),
        converged=converged,
        k=k,
        dataset_size=total_points,
        num_ranks=size,
    )
    collector.to_csv()
    collector.to_json()

    # Rank 0 produces the cross-rank aggregated CSV after all ranks have
    # written their individual files.  A Barrier ensures rank 0 does not
    # read files that other ranks have not yet flushed.
    comm.Barrier()
    if rank == 0:
        try:
            KMeansMetricsCollector.aggregate_across_ranks(
                output_dir=metrics_output_dir,
                num_ranks=size,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Aggregation failed (non-fatal): %s", exc)

    # ---- Cleanup -------------------------------------------------------
    points_rdd.unpersist()
    spark.stop()

    return {
        "global_centroids": centroids.tolist(),
        "iterations_run"  : len(collector._iterations),
        "converged"       : converged,
        "metrics"         : collector.summary_table(),
        "run_summary"     : collector._run,
        "rank"            : rank,
        "total_time_s"    : round(total_time, 4),
    }


# ===========================================================================
# Internal helper — local WCSS  (unchanged)
# ===========================================================================

def _compute_local_wcss(points_rdd, centroids: np.ndarray) -> float:
    _centroids = centroids

    def _wcss_partition(points_iter):
        total = 0.0
        for pt in points_iter:
            diff     = _centroids - pt
            sq_dists = np.einsum('ij,ij->i', diff, diff)
            total   += float(np.min(sq_dists))
        yield total

    return float(points_rdd.mapPartitions(_wcss_partition).sum())
