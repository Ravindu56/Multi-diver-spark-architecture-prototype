# =============================================================================
# mpj_spark/applications/kmeans/allreduce.py
# Phase 3 — Issue #8 — Step 4 (refactored for Step 5)
#
# CHANGE LOG (Step 5 refactor)
# ----------------------------
# - Replaced the inlined convergence block in run_kmeans_allreduce() with
#   a single call to check_and_broadcast() from convergence.py.
# - Removes 12 lines of duplicated shift/flag/bcast arithmetic from the
#   loop body.  All convergence logic now lives in convergence.py.
# - metrics dict is unchanged — centroid_shift is still recorded per iter.
# - All other Step 4 logic (buffer-level Allreduce, empty-cluster guard,
#   WCSS aggregation, comm.Barrier()) is unchanged.
# =============================================================================

from __future__ import annotations

import logging
import time
from typing import Dict, List

import numpy as np

logger = logging.getLogger(__name__)

_EMPTY_CLUSTER_THRESHOLD = 0.5


# ===========================================================================
# Core Allreduce primitive  (unchanged from Step 4)
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
    compute the new global centroid array.

    See Step 4 commit for full algorithm description.
    Buffer-level MPI.SUM Allreduce on local_sums (K×D) and local_counts (K,).
    Empty-cluster guard: rank 0 samples a replacement and Bcasts it.
    """
    from mpi4py import MPI

    k, d = local_sums.shape
    global_sums   = np.zeros_like(local_sums)
    global_counts = np.zeros_like(local_counts)

    comm.Allreduce(
        [local_sums,   MPI.DOUBLE],
        [global_sums,  MPI.DOUBLE],
        op=MPI.SUM,
    )
    comm.Allreduce(
        [local_counts,   MPI.DOUBLE],
        [global_counts,  MPI.DOUBLE],
        op=MPI.SUM,
    )

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
            sample = points_rdd.takeSample(False, 1, seed=int(time.time() * 1000))
            replacement = np.array(sample[0], dtype=np.float64)
        else:
            replacement = np.zeros(d, dtype=np.float64)
        comm.Bcast([replacement, MPI.DOUBLE], root=0)
        global_centroids[j] = replacement
        logger.info("[rank %d] Cluster %d reinitialised → %s", rank, j, replacement[:4])

    return global_centroids


# ===========================================================================
# Full K-Means Allreduce runner  (Steps 2 + 3 + 4 + 5 orchestration)
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
) -> Dict:
    """
    Full multi-driver K-Means with synchronous Allreduce centroid sync.
    Orchestrates Steps 2 → 3 → 4 → 5 in sequence.

    Returns
    -------
    dict:
        global_centroids  : list[list[float]]
        iterations_run    : int
        converged         : bool
        metrics           : list[dict]  — keys: iteration, sync_time_s,
                            iter_time_s, centroid_shift, global_wcss
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

    t_total_start = time.perf_counter()
    metrics: List[Dict] = []

    # ---- Step 2: partition + scatter + Spark session -------------------
    partition_path, spark = partition_and_init_spark(
        comm=comm, rank=rank, size=size,
        input_file=input_file, num_workers=size,
        cores_override=cores_override,
    )

    # ---- Step 3 setup: load RDD + init centroids -----------------------
    points_rdd = load_partition_rdd(spark, partition_path)
    centroids  = init_centroids(points_rdd, k=k, seed=seed)

    # Barrier: all ranks must finish RDD cache + Spark init before loop
    comm.Barrier()
    logger.info("[rank %d] Barrier passed — entering iteration loop", rank)

    prev_centroids = np.zeros_like(centroids)
    converged      = False

    # ---- Main loop: Steps 3 + 4 + 5 interleaved -----------------------
    for iteration in range(1, max_iter + 1):
        t_iter_start = time.perf_counter()

        # Step 3: one Spark action → local sums + counts
        local_sums, local_counts = compute_local_stats(points_rdd, centroids)
        local_wcss = _compute_local_wcss(points_rdd, centroids)

        # Step 4: Allreduce → new global centroids
        t_sync_start  = time.perf_counter()
        new_centroids = allreduce_centroids(
            comm, rank, local_sums, local_counts, points_rdd
        )

        # Aggregate WCSS across all ranks (one extra scalar Allreduce)
        global_wcss_arr = np.array([local_wcss], dtype=np.float64)
        global_wcss_buf = np.zeros(1, dtype=np.float64)
        comm.Allreduce(
            [global_wcss_arr, MPI.DOUBLE],
            [global_wcss_buf, MPI.DOUBLE],
            op=MPI.SUM,
        )
        global_wcss = float(global_wcss_buf[0])

        # Step 5: convergence check + broadcast
        # check_and_broadcast() issues comm.Bcast unconditionally so
        # all ranks execute it on every iteration (collective correctness).
        converged, centroid_shift = check_and_broadcast(
            comm, rank, new_centroids, prev_centroids, tol, iteration
        )

        t_sync_end = time.perf_counter()
        sync_time  = t_sync_end - t_sync_start
        iter_time  = time.perf_counter() - t_iter_start

        metrics.append({
            "iteration"     : iteration,
            "sync_time_s"   : round(sync_time, 6),
            "iter_time_s"   : round(iter_time, 6),
            "centroid_shift": round(centroid_shift, 8),
            "global_wcss"   : round(global_wcss, 4),
        })

        logger.info(
            "[rank %d] iter=%d  shift=%.6f  wcss=%.2f  sync=%.4fs  iter=%.4fs",
            rank, iteration, centroid_shift, global_wcss, sync_time, iter_time,
        )

        prev_centroids = centroids
        centroids      = new_centroids

        if converged:
            logger.info(
                "[rank %d] Converged at iteration %d (shift=%.6f < tol=%.6f)",
                rank, iteration, centroid_shift, tol,
            )
            break

    # ---- Cleanup -------------------------------------------------------
    points_rdd.unpersist()
    spark.stop()

    return {
        "global_centroids": centroids.tolist(),
        "iterations_run"  : len(metrics),
        "converged"       : converged,
        "metrics"         : metrics,
        "rank"            : rank,
        "total_time_s"    : round(time.perf_counter() - t_total_start, 4),
    }


# ===========================================================================
# Internal helper — local WCSS  (unchanged from Step 4)
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
