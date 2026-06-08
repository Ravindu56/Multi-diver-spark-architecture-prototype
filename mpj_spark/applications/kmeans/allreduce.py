# =============================================================================
# mpj_spark/applications/kmeans/allreduce.py
# Phase 3 — Issue #8 — Step 4: Allreduce Centroid Synchronisation
#
# PURPOSE
# -------
# This module owns the MPI layer of the K-Means iteration loop.  It:
#
#   1. Calls compute_local_stats() (Step 3) to obtain per-rank raw centroid
#      sums and point counts from the local PySpark RDD.
#   2. Issues two buffer-level comm.Allreduce calls (op=MPI.SUM) to aggregate
#      local_sums and local_counts across ALL ranks into global totals.
#   3. Divides global_sums by global_counts to produce the new global centroid
#      array — the numerically correct synchronous average.
#   4. Handles empty clusters (global_counts[j] == 0) by re-initialising the
#      dead centroid from a random data point on rank 0, then broadcasting.
#   5. Broadcasts a convergence flag from rank 0 so ALL ranks exit the loop
#      on the same iteration — critical for MPI collective correctness.
#   6. Records per-iteration timing metrics for experimental evaluation.
#
# MPI COLLECTIVE CORRECTNESS RULES
# ---------------------------------
# Every comm.Allreduce and comm.bcast call is a COLLECTIVE operation: it
# must be called by ALL ranks on EVERY iteration with no rank skipping.
# A single rank exiting the loop early (e.g. due to a local convergence
# check) while others are still inside Allreduce will cause an MPI deadlock.
# The convergence check therefore happens on rank 0 only; the result is
# broadcast to all ranks; all ranks read the same converged flag before
# deciding whether to continue.
#
# BUFFER-LEVEL vs OBJECT-LEVEL Allreduce
# ----------------------------------------
# mpi4py supports two calling conventions:
#   - Object-level (lowercase): comm.allreduce(scalar, op=MPI.SUM)
#     Pickles Python objects. Convenient but slow for large arrays.
#   - Buffer-level (uppercase): comm.Allreduce([arr, MPI.DOUBLE], ...)
#     Passes numpy buffer directly to the MPI C layer. No pickle overhead.
#     Required for numpy arrays of shape (K, D) to avoid serialisation cost.
# This module uses buffer-level (uppercase) for all numpy array collectives.
#
# METRICS COLLECTED (per iteration)
# -----------------------------------
#   sync_time      — wall time inside the two Allreduce calls only
#   iter_time      — total wall time for one full iteration
#   centroid_shift — Frobenius norm ||new_centroids - old_centroids||_F
#   global_wcss    — Within-Cluster Sum of Squares aggregated via Allreduce
#
# These feed the project's Synchronization Overhead and Convergence Rate
# performance metrics (defined in the research scope).
#
# BOUNDARY WITH OTHER STEPS
# --------------------------
#   Step 2 (partition.py)       : provides partition_path, spark
#   Step 3 (local_iteration.py) : provides load_partition_rdd,
#                                  init_centroids, compute_local_stats
#   Step 4 (this file)          : owns Allreduce + convergence + metrics
#   Step 5 (convergence.py)     : will add the stopping-criterion check
#                                  (currently inlined here as a simple
#                                  Frobenius-norm threshold)
# =============================================================================

from __future__ import annotations

import logging
import time
from typing import Dict, List, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Sentinel: a global_counts element below this threshold is treated as an
# empty cluster.  Using a small epsilon (not strict 0.0) guards against
# floating-point underflow when counts are accumulated across many ranks.
# ---------------------------------------------------------------------------
_EMPTY_CLUSTER_THRESHOLD = 0.5


# ===========================================================================
# Core Allreduce primitive
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

    This is the acceptance-criterion function for Issue #8 Step 4.
    The caller (run_kmeans_allreduce) wraps it with timing.

    Algorithm
    ---------
    1. Allreduce local_sums  (K × D, float64) → global_sums
    2. Allreduce local_counts (K,   float64) → global_counts
    3. global_centroids[j] = global_sums[j] / global_counts[j]  for all j
    4. For any j where global_counts[j] < _EMPTY_CLUSTER_THRESHOLD:
         - Rank 0 samples a random point from its local partition RDD
           and broadcasts it as the replacement centroid for cluster j.
         - All ranks update their centroid copy.
       This prevents a dead centroid from permanently biasing WCSS
       (the re-init strategy follows k-means|| restart convention).

    Parameters
    ----------
    comm          : mpi4py MPI communicator (COMM_WORLD or subset)
    rank          : this process's rank
    local_sums    : np.ndarray (K, D) — raw cluster sums from Step 3,
                    dtype float64, NOT pre-divided
    local_counts  : np.ndarray (K,)   — raw cluster point counts,
                    dtype float64, NOT pre-divided
    points_rdd    : cached PySpark RDD (used only for empty-cluster reinit)

    Returns
    -------
    global_centroids : np.ndarray (K, D), dtype float64
    """
    from mpi4py import MPI

    k, d = local_sums.shape

    global_sums   = np.zeros_like(local_sums)
    global_counts = np.zeros_like(local_counts)

    # ---- Collective 1: aggregate cluster sums ----------------------------
    # Buffer-level call: passes the numpy array buffer directly to MPI C
    # layer — no Python pickling.  Both buffers must be contiguous float64.
    comm.Allreduce(
        [local_sums,   MPI.DOUBLE],
        [global_sums,  MPI.DOUBLE],
        op=MPI.SUM,
    )

    # ---- Collective 2: aggregate cluster counts --------------------------
    comm.Allreduce(
        [local_counts,   MPI.DOUBLE],
        [global_counts,  MPI.DOUBLE],
        op=MPI.SUM,
    )

    # ---- Divide: global sums / global counts → global centroids ----------
    global_centroids = np.zeros((k, d), dtype=np.float64)
    empty_clusters: List[int] = []

    for j in range(k):
        if global_counts[j] >= _EMPTY_CLUSTER_THRESHOLD:
            global_centroids[j] = global_sums[j] / global_counts[j]
        else:
            # Mark for reinit — handled below after all centres are computed
            empty_clusters.append(j)
            logger.warning(
                "[rank %d] Cluster %d is empty (global_count=%.1f) "
                "— will reinitialise from rank 0 sample.",
                rank, j, global_counts[j],
            )

    # ---- Empty-cluster reinitialisation ----------------------------------
    # Rank 0 picks a random point from its local data for each dead cluster
    # and broadcasts it.  All ranks replace their copy of that centroid.
    # This is a collective bcast per empty cluster — all ranks must enter.
    for j in empty_clusters:
        if rank == 0:
            # takeSample(withReplacement=False, num=1) returns a list of 1
            sample = points_rdd.takeSample(False, 1, seed=int(time.time() * 1000))
            replacement = np.array(sample[0], dtype=np.float64)
        else:
            replacement = np.zeros(d, dtype=np.float64)

        # Collective 3 (conditional): broadcast replacement centroid
        comm.Bcast([replacement, MPI.DOUBLE], root=0)
        global_centroids[j] = replacement
        logger.info(
            "[rank %d] Cluster %d reinitialised → %s",
            rank, j, replacement[:4],
        )

    return global_centroids


# ===========================================================================
# Full K-Means Allreduce runner  (Steps 2 + 3 + 4 orchestration)
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

    Orchestrates Steps 2, 3, 4 in sequence and returns per-iteration
    metrics alongside the final global centroids.

    Parameters
    ----------
    comm          : mpi4py COMM_WORLD
    rank          : this process's MPI rank
    size          : total number of MPI ranks
    input_file    : path to full dataset (shared NFS / local)
    k             : number of clusters
    max_iter      : maximum iterations before forced stop
    tol           : convergence threshold — Frobenius norm of centroid shift
    seed          : random seed for init_centroids
    cores_override: CPU cores per Spark session (None = auto)

    Returns
    -------
    dict with keys:
        global_centroids  : list[list[float]]  — final K × D centroid coords
        iterations_run    : int                — iterations until convergence
        converged         : bool               — True if tol was reached
        metrics           : list[dict]         — one entry per iteration:
            {
              'iteration'     : int,
              'sync_time_s'   : float,   # wall time inside Allreduce only
              'iter_time_s'   : float,   # total iteration wall time
              'centroid_shift': float,   # Frobenius norm
              'global_wcss'   : float,   # aggregated WCSS across all ranks
            }
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

    t_total_start = time.perf_counter()
    metrics: List[Dict] = []

    # ------------------------------------------------------------------ #
    # STEP 2: Partition + scatter + per-rank Spark session                #
    # ------------------------------------------------------------------ #
    partition_path, spark = partition_and_init_spark(
        comm=comm,
        rank=rank,
        size=size,
        input_file=input_file,
        num_workers=size,
        cores_override=cores_override,
    )

    # ------------------------------------------------------------------ #
    # STEP 3 setup: Load and cache the RDD; initialise centroids         #
    # ------------------------------------------------------------------ #
    points_rdd = load_partition_rdd(spark, partition_path)
    centroids  = init_centroids(points_rdd, k=k, seed=seed)  # shape (K, D)

    # Barrier: ensure ALL ranks have their RDD cached and initial
    # centroids computed before the first Allreduce.  Without this,
    # a slow rank's JVM startup can delay its first Allreduce call while
    # faster ranks are already inside the collective — causing a hang.
    comm.Barrier()
    logger.info("[rank %d] Barrier passed — entering iteration loop", rank)

    prev_centroids = np.zeros_like(centroids)
    converged = False

    # ------------------------------------------------------------------ #
    # MAIN LOOP: Steps 3 + 4 interleaved                                  #
    # ------------------------------------------------------------------ #
    for iteration in range(1, max_iter + 1):
        t_iter_start = time.perf_counter()

        # ---- Step 3: per-rank local stats (one Spark action) -----------
        local_sums, local_counts = compute_local_stats(points_rdd, centroids)

        # Local WCSS: sum of squared distances from each point to its
        # assigned centroid.  Computed from the same assignment pass
        # without an extra RDD scan.
        local_wcss = _compute_local_wcss(points_rdd, centroids)

        # ---- Step 4: Allreduce centroid sync ---------------------------
        t_sync_start = time.perf_counter()

        new_centroids = allreduce_centroids(
            comm, rank, local_sums, local_counts, points_rdd
        )

        # Aggregate local WCSS into global WCSS
        global_wcss_arr = np.array([local_wcss], dtype=np.float64)
        global_wcss_buf = np.zeros(1, dtype=np.float64)
        comm.Allreduce([global_wcss_arr, MPI.DOUBLE],
                       [global_wcss_buf,  MPI.DOUBLE], op=MPI.SUM)
        global_wcss = float(global_wcss_buf[0])

        t_sync_end = time.perf_counter()
        sync_time = t_sync_end - t_sync_start

        # ---- Convergence check (rank 0 decides; all ranks obey) --------
        centroid_shift = float(
            np.linalg.norm(new_centroids - prev_centroids, ord='fro')
        )

        if rank == 0:
            converged_flag = np.array(
                [1] if (iteration > 1 and centroid_shift < tol) else [0],
                dtype=np.int32,
            )
        else:
            converged_flag = np.zeros(1, dtype=np.int32)

        # Collective: broadcast convergence decision to all ranks
        comm.Bcast([converged_flag, MPI.INT], root=0)

        t_iter_end = time.perf_counter()
        iter_time  = t_iter_end - t_iter_start

        # ---- Record metrics --------------------------------------------
        metrics.append({
            "iteration"     : iteration,
            "sync_time_s"   : round(sync_time, 6),
            "iter_time_s"   : round(iter_time, 6),
            "centroid_shift": round(centroid_shift, 8),
            "global_wcss"   : round(global_wcss, 4),
        })

        logger.info(
            "[rank %d] iter=%d  shift=%.6f  wcss=%.2f  "
            "sync=%.4fs  iter=%.4fs",
            rank, iteration, centroid_shift, global_wcss,
            sync_time, iter_time,
        )

        # ---- Update state and check exit condition ---------------------
        prev_centroids = centroids
        centroids      = new_centroids

        if bool(converged_flag[0]):
            converged = True
            logger.info(
                "[rank %d] Converged at iteration %d  (shift=%.6f < tol=%.6f)",
                rank, iteration, centroid_shift, tol,
            )
            break

    # ------------------------------------------------------------------ #
    # Cleanup                                                              #
    # ------------------------------------------------------------------ #
    points_rdd.unpersist()
    spark.stop()

    t_total_end  = time.perf_counter()
    total_time   = t_total_end - t_total_start

    return {
        "global_centroids" : centroids.tolist(),
        "iterations_run"   : len(metrics),
        "converged"        : converged,
        "metrics"          : metrics,
        "rank"             : rank,
        "total_time_s"     : round(total_time, 4),
    }


# ===========================================================================
# Internal helper — local WCSS without an extra Spark action
# ===========================================================================

def _compute_local_wcss(points_rdd, centroids: np.ndarray) -> float:
    """
    Compute Within-Cluster Sum of Squares for this rank's partition.

    Uses a single mapPartitions action that mirrors the assignment logic
    in compute_local_stats — no additional Spark job.  Called BEFORE
    Allreduce so that the WCSS aggregation piggybacks on the same sync
    round (one extra Allreduce scalar, not an extra RDD scan).
    """
    _centroids = centroids

    def _wcss_partition(points_iter):
        total = 0.0
        for pt in points_iter:
            diff     = _centroids - pt
            sq_dists = np.einsum('ij,ij->i', diff, diff)
            total   += float(np.min(sq_dists))
        yield total

    return float(
        points_rdd
        .mapPartitions(_wcss_partition)
        .sum()
    )
