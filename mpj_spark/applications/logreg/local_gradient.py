# =============================================================================
# mpj_spark/applications/logreg/local_gradient.py
# Phase 3 — Issue #9 — Steps 3 & 4: SparkSession Init Contract +
#                                     Local Gradient via PySpark RDD
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
# The formula:
#     total_cores      = os.cpu_count() or 4
#     cores_per_worker = max(1, total_cores // size)
#
# This is identical to the local[{cores_per_worker}] pattern from
# mpj_spark_prototype_v2.py (the C3-fix) and to what build_spark_session()
# applies when cores_override is None and TOTAL_CORES is set via config.py.
# The runner passes cores_per_worker as cores_override= so the allocation
# is fully explicit and reproducible across all MPI ranks.
#
# Step 4 (Local Gradient Computation)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Implements compute_gradient_spark() — each Spark driver reads its shard
# and computes the gradient of the log-loss using PySpark RDD operations:
#
#   sc.textFile(partition_path)
#     .map(parse_row)            ← (features: np.ndarray, label: float)
#     .map(local_grad)           ← per-sample gradient vector
#     .reduce(lambda a, b: a+b)  ← sum of all per-sample gradients
#   / n                          ← normalised by shard size
#
# This is the same sc.textFile → map → reduce chain used in the v2 WordCount
# workload.  The only logreg-specific part is parse_row + local_grad.
#
# WHY NOT USE MLlib LogisticRegression.fit()?
# -------------------------------------------
# The same reason as kmeans/local_iteration.py avoiding MLlib KMeans.fit():
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
# to a weighted average if all shards are equal-sized — which dynamic_partition
# guarantees.  This avoids passing global N through the MPI layer.
#
# DATA FORMAT
# -----------
# Partition files are numeric CSV rows:  f0,f1,...,f{D-1},label
# Label is the LAST column.  A stray header row ("f0,f1,...,label") in
# partition 0 is silently dropped by parse_row via the float-cast guard.
#
# ZERO MPI IMPORTS
# ----------------
# This module has no mpi4py dependency.  The MPI layer lives in allreduce.py
# (Steps 5–6).  This clean boundary makes unit-testing straightforward:
# just call compute_gradient_spark() with a mock SparkContext.
#
# USAGE (called from allreduce.py — Steps 5–6)
# -----------------------------------------------
#   from mpj_spark.applications.logreg.local_gradient import (
#       cores_per_worker,
#       load_and_cache_rdd,
#       compute_gradient_spark,
#   )
#
#   # SparkSession was already created in partition.py Step 2c:
#   #   spark = build_spark_session(app_name=..., cores_override=cores_per_worker(size), ...)
#
#   data_rdd = load_and_cache_rdd(spark, partition_path, num_features)
#   grad     = compute_gradient_spark(spark.sparkContext, data_rdd, w)
# =============================================================================

from __future__ import annotations

import logging
import os
from typing import Tuple

import numpy as np
from pyspark import RDD
from pyspark.sql import SparkSession

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# STEP 3 — SparkSession Init Contract
# ---------------------------------------------------------------------------
# The formula below is the C3-fix pattern from mpj_spark_prototype_v2.py.
# Call cores_per_worker(size) in the runner and pass the result as
# cores_override= to build_spark_session() (already done in partition.py).
# Exposed here so allreduce.py can assert the value matches at runtime.
# ---------------------------------------------------------------------------

def cores_per_worker(size: int) -> int:
    """
    Return the number of CPU cores to allocate per Spark session.

    Formula (C3-fix from mpj_spark_prototype_v2.py):
        total_cores      = os.cpu_count() or 4
        cores_per_worker = max(1, total_cores // size)

    This evenly partitions the host's logical cores across all MPI ranks
    so that the sum of all Spark sessions' parallelism equals the total
    available concurrency on the machine.

    Parameters
    ----------
    size : int — total number of MPI ranks (comm.Get_size())

    Returns
    -------
    int — number of cores for each rank's local[N] Spark master
    """
    total_cores = os.cpu_count() or 4
    return max(1, total_cores // size)


# ---------------------------------------------------------------------------
# STEP 4a — Data Loading
# ---------------------------------------------------------------------------

def parse_row(line: str, num_features: int) -> Tuple[np.ndarray, float] | None:
    """
    Parse one CSV line into (feature_vector, label).

    Format expected:  f0,f1,...,f{D-1},label   (label is last column)

    Stray header rows (e.g. "f0,f1,...,label" written by the CSV generator
    into partition 0 by round-robin) are silently dropped when the first
    token cannot be cast to float.  This mirrors the NULL-filter in the
    existing logreg.py baseline.

    Rows with the wrong column count are also dropped so that ragged lines
    from interrupted writes do not poison the gradient.

    Parameters
    ----------
    line        : str — one raw text line from sc.textFile()
    num_features: int — expected number of feature columns (excl. label)

    Returns
    -------
    (np.ndarray shape (num_features,), float)  or  None if line is invalid
    """
    try:
        parts = line.strip().split(",")
        if len(parts) != num_features + 1:
            return None
        vals = [float(p) for p in parts]       # raises ValueError for header
        features = np.array(vals[:-1], dtype=np.float64)  # all but last
        label    = float(vals[-1])
        return features, label
    except (ValueError, IndexError):
        return None  # silently drop header row and malformed lines


def load_and_cache_rdd(spark: SparkSession, partition_path: str, num_features: int) -> RDD:
    """
    Load this rank's CSV shard into a cached RDD of (features, label) tuples.

    The RDD is cached with MEMORY_AND_DISK so that repeated scans across
    epochs read from memory rather than re-parsing the file each time.
    A count() action is triggered immediately to warm the cache before the
    first epoch, ensuring epoch timing reflects computation cost, not I/O.

    Parameters
    ----------
    spark          : active SparkSession for this rank (created in partition.py)
    partition_path : str — path to this rank's CSV shard
    num_features   : int — number of feature columns (excl. label)

    Returns
    -------
    RDD of (np.ndarray, float) pairs, cached in memory.
    """
    sc = spark.sparkContext

    # sc.textFile() is the same entry point as in v2 WordCount:
    # Spark reads the file in chunks, distributes lines across cores_per_worker
    # Spark partitions, and returns an RDD of raw text lines.
    _num_features = num_features  # capture for closure serialisation

    data_rdd = (
        sc.textFile(partition_path)
          .map(lambda line: parse_row(line, _num_features))
          .filter(lambda row: row is not None)
          .cache()
    )

    # Trigger the cache: force Spark to read the file and materialise the RDD
    # now so the first epoch's timing is clean.
    n = data_rdd.count()
    logger.info(
        "[rank local] Loaded and cached %d rows from '%s' (%d features)",
        n,
        partition_path,
        num_features,
    )
    return data_rdd


# ---------------------------------------------------------------------------
# STEP 4b — Local Gradient Computation (core of this file)
# ---------------------------------------------------------------------------

def compute_gradient_spark(
    data_rdd: RDD,
    w: np.ndarray,
) -> Tuple[np.ndarray, int]:
    """
    Compute the local gradient of the log-loss over this rank's data shard.

    This uses the sc.textFile → map → reduce chain from v2 WordCount,
    applied to the logistic regression gradient formula:

        sigmoid(x, w) = 1 / (1 + exp(-x · w))
        grad_i        = x_i * (sigmoid(x_i, w) - y_i)    [per sample]
        grad_local    = mean over shard of grad_i          [normalised]

    RDD pipeline
    ------------
      data_rdd  (cached (features, label) pairs)
        .map(local_grad)          ← per-sample gradient vector, shape (D,)
        .reduce(lambda a,b: a+b)  ← sum of all per-sample gradients, shape (D,)
      / n                         ← divide by shard size = normalised gradient

    Parameters
    ----------
    data_rdd : cached RDD of (np.ndarray shape (D,), float) tuples
               Must be the RDD returned by load_and_cache_rdd().
    w        : np.ndarray, shape (D,)
               Current weight vector (broadcast to tasks via closure;
               D is small enough that sc.broadcast() is not needed at
               prototype scale — consistent with kmeans/local_iteration.py).

    Returns
    -------
    grad_local : np.ndarray, shape (D,)
        Normalised local gradient.  NOT yet globally averaged — Step 5
        (allreduce.py) will call comm.Allreduce and divide by size:

            comm.Allreduce(grad_local, global_grad, op=MPI.SUM)
            global_grad /= size   ← average over all ranks (synchronous SGD)
            w -= learning_rate * global_grad

    n : int
        Number of data rows in this rank's shard (for logging / metrics).

    NOTE ON NUMERICAL STABILITY
    ---------------------------
    exp(-x · w) can overflow to inf when x · w is very negative (sigmoid → 0)
    or underflow when x · w is very positive (sigmoid → 1).  np.exp clips
    silently via IEEE 754 rules so the gradient never becomes NaN in normal
    operation.  For ill-conditioned inputs (very large feature magnitudes),
    the caller should normalise features at dataset-generation time.
    """
    _w = w  # local alias so Spark can serialise the closure cleanly

    def local_grad(row: Tuple[np.ndarray, float]) -> np.ndarray:
        """
        Per-sample gradient: x * (sigmoid(x · w) - y)

        Parameters
        ----------
        row : (features np.ndarray shape (D,), label float)

        Returns
        -------
        np.ndarray shape (D,) — per-sample gradient contribution
        """
        x, y = row
        # Logistic sigmoid prediction
        pred = 1.0 / (1.0 + np.exp(-float(np.dot(x, _w))))
        # Gradient of binary cross-entropy w.r.t. w: x * (pred - y)
        return x * (pred - y)

    # One Spark job: map per-sample gradient then reduce (sum) across all rows.
    # This is the same sc.textFile → map → reduce chain as v2 WordCount:
    # the only difference is the map function (local_grad vs split/flatMap)
    # and the reduce function (np.add vs int addition).
    grad_sum = (
        data_rdd
        .map(local_grad)
        .reduce(lambda a, b: a + b)   # np.ndarray + np.ndarray element-wise
    )

    # Count is already known from load_and_cache_rdd but we re-read it here
    # so this function is self-contained (count() hits the cache, no I/O).
    n = data_rdd.count()

    # Normalise by shard size so ranks with slightly unequal partition sizes
    # (dynamic_partition guarantees ~equal, not exact equal) contribute
    # proportionally rather than by raw row count.
    grad_local = grad_sum / float(n)

    logger.info(
        "[rank local] Gradient norm: %.6f  (n=%d)",
        float(np.linalg.norm(grad_local)),
        n,
    )

    return grad_local, n
