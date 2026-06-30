# =============================================================================
# mpj_spark/applications/logreg/local_gradient.py
# Phase 3 — Issue #9 — Steps 3 & 4: SparkSession Init Contract +
#                                     Local Gradient via PySpark RDD
#
# CHANGE LOG (global standardisation fix)
# ----------------------------------------
# Problem: the MPI path had no feature standardisation while the MLlib
# baseline uses standardization=True (default), which fits a StandardScaler
# on the full training set.  Per-rank local scaling would produce different
# feature spaces on each rank making the Allreduce average meaningless.
#
# Fix: _broadcast_global_stats() aggregates feature sum + sum-of-squares
# across all ranks via a single Spark pass on rank 0, derives global
# mean/std, and comm.Bcast both arrays to all ranks before the cache
# is built.  load_and_cache_rdd() applies normalisation inline so the
# cached RDD contains standardised (features, label) pairs.  All ranks
# now use IDENTICAL normalisation parameters — equivalent to MLlib's
# full-dataset StandardScaler.
#
# load_and_cache_rdd() signature change:
#   OLD: load_and_cache_rdd(spark, partition_path, num_features)
#   NEW: load_and_cache_rdd(spark, partition_path, num_features, comm, rank)
# allreduce.py call site updated accordingly.
#
# All other logic (parse_row, compute_gradient_spark, cores_per_worker)
# is UNCHANGED.
#
# PURPOSE
# -------
# Step 3 (SparkSession Init Contract)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Documents and enforces the required SparkSession creation pattern for the
# LogReg MPI-Allreduce runner.  The session is NOT created here — it was
# already created in partition.py (Step 2c) via build_spark_session().  This
# module makes the CPU allocation formula explicit so the runner script can
# use it directly without reimplementing the cores_per_worker calculation.
#
# Step 4 (Local Gradient Computation)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Implements compute_gradient_spark() — each Spark driver reads its shard
# and computes the gradient of the log-loss using PySpark RDD operations:
#
#   sc.textFile(partition_path)
#     .map(parse_row)            <- (features: np.ndarray, label: float)
#     .map(local_grad)           <- per-sample gradient vector
#     .reduce(lambda a, b: a+b)  <- sum of all per-sample gradients
#   / n                          <- normalised by shard size
#
# WHY NOT USE MLlib LogisticRegression.fit()?
# -------------------------------------------
# MLlib runs the FULL iterative SGD loop internally with no hook to intercept
# gradient state between epochs.  The MPI-Allreduce architecture requires
# control of each gradient step so that comm.Allreduce can synchronise
# weight updates globally before the next epoch begins.
#
# WHY NORMALISE BY n (shard size) AND NOT GLOBAL N?
# ---------------------------------------------------
# Each rank computes: grad_local = (1/n_local) * sum_local(grad_i)
# Step 5 (allreduce.py) does:  comm.Allreduce(grad_local, global_grad, MPI.SUM)
#                              global_grad /= size   (average over ranks)
# The two-level normalisation (per-shard then per-rank average) is equivalent
# to a weighted average if all shards are equal-sized.
#
# ZERO MPI IMPORTS (in non-standardisation helpers)
# --------------------------------------------------
# parse_row and compute_gradient_spark have no mpi4py dependency.
# _broadcast_global_stats receives comm as a parameter so the MPI layer
# stays in allreduce.py; this module still has no top-level MPI import.
# =============================================================================

from __future__ import annotations

import logging
import os

import numpy as np
from pyspark import RDD
from pyspark.sql import SparkSession

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# STEP 3 — SparkSession Init Contract
# ---------------------------------------------------------------------------

def cores_per_worker(size: int) -> int:
    """
    Return the number of CPU cores to allocate per Spark session.

    Formula (C3-fix from mpj_spark_prototype_v2.py):
        total_cores      = os.cpu_count() or 4
        cores_per_worker = max(1, total_cores // size)
    """
    total_cores = os.cpu_count() or 4
    return max(1, total_cores // size)


# ---------------------------------------------------------------------------
# STEP 4a — Global statistics broadcast (standardisation)
# ---------------------------------------------------------------------------

def _broadcast_global_stats(
    comm,
    rank: int,
    data_rdd_raw: RDD,
    num_features: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute global feature mean and std across ALL ranks via a single
    Spark aggregation on rank 0, then comm.Bcast to every rank.

    This mirrors MLlib's StandardScaler(withMean=True, withStd=True)
    fitted on the FULL training set, making the MPI gradient path
    numerically equivalent to the baseline.

    Algorithm
    ---------
    Each rank already holds a cached RDD of (features, label) pairs in
    *raw* (un-normalised) space passed as data_rdd_raw.  To compute global
    statistics without a second file read:

      1. Rank 0 aggregates (sum_x, sum_x2, n) from its local partition
         using a single RDD.aggregate() action.
         - zero_val  = (zeros_D, zeros_D, 0)
         - seqOp(acc, row): acc[0] += x; acc[1] += x**2; acc[2] += 1
         - combOp(a, b): element-wise add all three

      2. All ranks send their (sum_x, sum_x2, n_local) to rank 0 via
         comm.reduce (not Allreduce — avoids N redundant Spark jobs).

      3. Rank 0 computes:
           global_mean = total_sum / total_n
           global_var  = total_sum_sq / total_n - global_mean**2
           global_std  = sqrt(clip(global_var, 0)) floored at 1e-8

      4. global_mean and global_std are packed into one flat buffer and
         comm.Bcast to all ranks.

    Parameters
    ----------
    comm          : mpi4py.MPI.Intracomm
    rank          : int
    data_rdd_raw  : RDD of (np.ndarray shape (D,), float) in raw feature space
    num_features  : int — D

    Returns
    -------
    global_mean : np.ndarray shape (D,)
    global_std  : np.ndarray shape (D,)  (floored at 1e-8)
    """
    from mpi4py import MPI

    D = num_features

    # Step 1: each rank aggregates its local (sum_x, sum_x2, n) via Spark
    zero_val = (np.zeros(D, dtype=np.float64),
                np.zeros(D, dtype=np.float64),
                0)

    def seq_op(acc, row):
        x, _ = row
        return (acc[0] + x, acc[1] + x * x, acc[2] + 1)

    def comb_op(a, b):
        return (a[0] + b[0], a[1] + b[1], a[2] + b[2])

    local_sum_x, local_sum_x2, local_n = data_rdd_raw.aggregate(
        zero_val, seq_op, comb_op
    )

    # Step 2: reduce all ranks' stats to rank 0 using MPI.SUM
    # Pack into a single flat buffer: [sum_x (D), sum_x2 (D), n (1)]
    send_buf = np.empty(2 * D + 1, dtype=np.float64)
    send_buf[:D]       = local_sum_x
    send_buf[D:2*D]    = local_sum_x2
    send_buf[2*D]      = float(local_n)

    recv_buf = np.zeros(2 * D + 1, dtype=np.float64)
    comm.Reduce([send_buf, MPI.DOUBLE], [recv_buf, MPI.DOUBLE],
                op=MPI.SUM, root=0)

    # Step 3: rank 0 computes global mean + std
    if rank == 0:
        total_n      = max(recv_buf[2*D], 1.0)  # guard against empty dataset
        global_mean  = recv_buf[:D]   / total_n
        global_var   = recv_buf[D:2*D] / total_n - global_mean ** 2
        global_std   = np.sqrt(np.clip(global_var, 0.0, None))
        global_std   = np.where(global_std < 1e-8, 1.0, global_std)  # floor

        logger.info(
            "[rank 0] Global feature stats computed  "
            "n=%.0f  mean[:3]=%s  std[:3]=%s",
            total_n,
            np.round(global_mean[:3], 4),
            np.round(global_std[:3], 4),
        )
    else:
        global_mean = np.zeros(D, dtype=np.float64)
        global_std  = np.ones(D,  dtype=np.float64)

    # Step 4: Bcast mean and std as a single flat buffer
    stats_buf = np.empty(2 * D, dtype=np.float64)
    if rank == 0:
        stats_buf[:D]  = global_mean
        stats_buf[D:]  = global_std
    comm.Bcast([stats_buf, MPI.DOUBLE], root=0)

    global_mean = stats_buf[:D].copy()
    global_std  = stats_buf[D:].copy()

    logger.info(
        "[rank %d] Standardisation params received  mean[:3]=%s  std[:3]=%s",
        rank,
        np.round(global_mean[:3], 4),
        np.round(global_std[:3], 4),
    )
    return global_mean, global_std


# ---------------------------------------------------------------------------
# STEP 4a — Data Loading (with standardisation)
# ---------------------------------------------------------------------------

def parse_row(line: str, num_features: int) -> tuple[np.ndarray, float] | None:
    """
    Parse one CSV line into (feature_vector, label).

    Format expected:  f0,f1,...,f{D-1},label   (label is last column)
    Stray header rows and ragged lines are silently dropped.
    """
    try:
        parts = line.strip().split(",")
        if len(parts) != num_features + 1:
            return None
        vals = [float(p) for p in parts]
        features = np.array(vals[:-1], dtype=np.float64)
        label = float(vals[-1])
        return features, label
    except (ValueError, IndexError):
        return None


def load_and_cache_rdd(
    spark: SparkSession,
    partition_path: str,
    num_features: int,
    comm,
    rank: int,
) -> RDD:
    """
    Load this rank's CSV shard into a cached RDD of (features, label) tuples
    with features standardised to zero mean and unit variance using GLOBAL
    statistics broadcast from rank 0.

    Standardisation strategy
    ------------------------
    1. Parse the raw file into a temporary un-standardised RDD.
    2. Call _broadcast_global_stats() to obtain global_mean and global_std
       that are IDENTICAL on all ranks.
    3. Apply normalisation inline:  x_norm = (x - global_mean) / global_std
    4. Cache the normalised RDD; trigger the cache with count().

    This is equivalent to MLlib's default StandardScaler fitted on the
    full training set, making the MPI gradient path comparable to the
    baseline LogisticRegression(standardization=True).

    Parameters
    ----------
    spark          : active SparkSession for this rank
    partition_path : path to this rank's CSV shard
    num_features   : number of feature columns (excl. label)
    comm           : mpi4py.MPI.Intracomm  (needed for Bcast)
    rank           : int  (this rank's MPI rank)

    Returns
    -------
    RDD of (np.ndarray shape (D,), float) pairs, cached and standardised.
    """
    sc = spark.sparkContext
    _num_features = num_features

    # Step 1: raw RDD — parsed but NOT yet standardised
    raw_rdd = (
        sc.textFile(partition_path)
        .map(lambda line: parse_row(line, _num_features))
        .filter(lambda row: row is not None)
        .cache()
    )
    raw_rdd.count()  # materialise raw cache before stats aggregation

    # Step 2: compute and broadcast global mean/std across all MPI ranks
    global_mean, global_std = _broadcast_global_stats(comm, rank, raw_rdd, num_features)

    # Step 3: apply standardisation and build the final cached RDD
    _mean = global_mean
    _std  = global_std

    data_rdd = (
        raw_rdd
        .map(lambda row: ((row[0] - _mean) / _std, row[1]))
        .cache()
    )

    n = data_rdd.count()  # trigger standardised cache
    raw_rdd.unpersist()   # release the un-standardised cache

    logger.info(
        "[rank %d] Loaded, standardised, and cached %d rows from '%s' (%d features)",
        rank, n, partition_path, num_features,
    )
    return data_rdd


# ---------------------------------------------------------------------------
# STEP 4b — Local Gradient Computation (unchanged)
# ---------------------------------------------------------------------------

def compute_gradient_spark(
    data_rdd: RDD,
    w: np.ndarray,
) -> tuple[np.ndarray, int]:
    """
    Compute the local gradient of the log-loss over this rank's data shard.

    RDD pipeline
    ------------
      data_rdd  (cached standardised (features, label) pairs)
        .map(local_grad)          <- per-sample gradient vector, shape (D,)
        .reduce(lambda a,b: a+b)  <- sum, shape (D,)
      / n                         <- normalised by shard size

    Parameters
    ----------
    data_rdd : cached RDD of (np.ndarray shape (D,), float) tuples
               Features are already standardised by load_and_cache_rdd().
    w        : np.ndarray, shape (D,)
               Current weight vector in normalised feature space.

    Returns
    -------
    grad_local : np.ndarray, shape (D,) — normalised local gradient
    n          : int — number of rows in this shard
    """
    _w = w

    def local_grad(row: tuple[np.ndarray, float]) -> np.ndarray:
        x, y = row
        pred = 1.0 / (1.0 + np.exp(-float(np.dot(x, _w))))
        return x * (pred - y)

    grad_sum = data_rdd.map(local_grad).reduce(lambda a, b: a + b)
    n = data_rdd.count()
    grad_local = grad_sum / float(n)

    logger.info(
        "[rank local] Gradient norm: %.6f  (n=%d)",
        float(np.linalg.norm(grad_local)),
        n,
    )
    return grad_local, n
