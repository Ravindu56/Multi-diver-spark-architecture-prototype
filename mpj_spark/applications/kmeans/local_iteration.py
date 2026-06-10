# =============================================================================
# mpj_spark/applications/kmeans/local_iteration.py
# Phase 3 — Issue #8 — Step 3: Local K-Means Centroid Computation Per Rank
#
# PURPOSE
# -------
# Provide the per-rank, per-iteration K-Means computation that produces
# LOCAL centroid sums and point counts for each cluster.  These raw
# aggregates are NOT averaged here — they are passed directly to Step 4
# (comm.Allreduce) so that all ranks contribute to a single global average.
#
# WHY NOT USE MLlib KMeans.fit() FROM kmeans.py?
# -----------------------------------------------
# The existing kmeans.py calls KMeans.fit() which runs the FULL iterative
# loop internally inside the JVM and returns a finished converged model.
# That is correct for single-driver Spark but incompatible with the
# Allreduce architecture because:
#   - There is no hook to intercept centroid state between iterations
#   - The JVM-side loop cannot be paused for an MPI collective
#   - KMeansModel.clusterCenters are already averaged — the raw per-cluster
#     sums and counts needed for a numerically correct global average are
#     discarded inside Spark
#
# DESIGN: MANUAL ITERATIVE LOOP
# ------------------------------
# We implement K-Means from scratch using PySpark RDD operations:
#
#   centroids (numpy array, shape K x D)
#       ↓
#   assign_and_sum()  →  one Spark mapPartitions + reduceByKey action
#       ↓
#   (local_sums: K x D,  local_counts: K)   ← RETURNED to caller
#       ↓
#   Step 4: comm.Allreduce(local_sums, global_sums, op=MPI.SUM)
#           comm.Allreduce(local_counts, global_counts, op=MPI.SUM)
#       ↓
#   global_centroids = global_sums / global_counts[:, np.newaxis]
#       ↓
#   repeat
#
# This means this file has ZERO MPI imports — it is a pure Spark/numpy
# module.  The MPI layer lives entirely in Step 4 (allreduce.py).
# Keeping the boundary clean makes unit testing straightforward.
#
# DATA FORMAT ASSUMPTION
# ----------------------
# Partition files are numeric CSV (comma-separated floats, no header).
# This is consistent with the dataset generator in mpj_spark_prototype_v2.py
# and with the CSV loading logic in baseline_kmeans.py.
#
# USAGE (called from Step 4 runner — allreduce.py)
# -------------------------------------------------
#   from mpj_spark.applications.kmeans.local_iteration import (
#       load_partition_rdd,
#       init_centroids,
#       compute_local_stats,
#   )
#
#   points_rdd = load_partition_rdd(spark, partition_path)
#   centroids  = init_centroids(points_rdd, k=K, seed=42)
#
#   # Inside the iteration loop (Step 4 handles Allreduce between calls):
#   local_sums, local_counts = compute_local_stats(points_rdd, centroids)
# =============================================================================

from __future__ import annotations

import logging

import numpy as np
from pyspark import RDD
from pyspark.sql import SparkSession

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. Data loading
# ---------------------------------------------------------------------------


def load_partition_rdd(spark: SparkSession, partition_path: str) -> RDD:
    """
    Load a numeric CSV partition file into a cached numpy-row RDD.

    Each line is parsed into a numpy float64 array.  Malformed lines
    (non-numeric tokens, empty lines, mismatched column counts) are
    silently dropped — consistent with VectorAssembler(handleInvalid='skip')
    used in the baseline.

    The RDD is cached so that repeated scans during K-Means iterations
    read from memory rather than re-parsing the file each time.

    Parameters
    ----------
    spark          : active SparkSession for this rank
    partition_path : path to the rank's CSV shard

    Returns
    -------
    RDD of numpy.ndarray, shape (D,), dtype float64
    Cached in memory (StorageLevel.MEMORY_AND_DISK).
    """
    raw_rdd = spark.sparkContext.textFile(partition_path)

    # Detect number of features from first valid line so we can filter
    # ragged rows without collecting the whole RDD to the driver.
    first_line = raw_rdd.first()
    n_features = len(first_line.strip().split(","))

    def _parse_line(line: str):
        """
        Parse one CSV line → numpy array.  Returns None for bad lines.
        Only rows with exactly n_features numeric columns are kept.
        """
        try:
            vals = [float(x) for x in line.strip().split(",") if x.strip()]
            if len(vals) != n_features:
                return None
            return np.array(vals, dtype=np.float64)
        except ValueError:
            return None

    points_rdd = raw_rdd.map(_parse_line).filter(lambda x: x is not None).cache()

    # Trigger the cache by counting — ensures data is loaded into memory
    # before the first iteration begins so timing measurements (Step 4)
    # reflect computation cost, not I/O latency.
    row_count = points_rdd.count()
    logger.info(
        "[rank local] Loaded %d rows from '%s' (%d features)",
        row_count,
        partition_path,
        n_features,
    )
    return points_rdd


# ---------------------------------------------------------------------------
# 2. Centroid initialisation  (k-means++ single-pass)
# ---------------------------------------------------------------------------


def init_centroids(points_rdd: RDD, k: int, seed: int = 42) -> np.ndarray:
    """
    Initialise K cluster centroids using k-means++ sampling.

    Strategy
    --------
    Full k-means++ (Arthur & Vassilvitskii, 2007) requires O(k) full RDD
    scans which is expensive for large partitions.  We use a pragmatic
    approximation that preserves the spread guarantee:

      1. Sample min(10k, row_count) points from the RDD into driver RAM.
         (10k points is sufficient for stable initialisation up to ~1M rows)
      2. Run k-means++ selection on the in-memory sample using numpy.
         This avoids k full Spark actions and runs in microseconds.

    This approach is consistent with how Spark's own k-means|| initialises
    on a subsample when the dataset is large.

    NOTE ON GLOBAL CONSISTENCY
    --------------------------
    In the Allreduce architecture every rank runs init_centroids() on its
    LOCAL partition independently.  The initial centroids will differ per
    rank.  This is intentional — Step 4 (comm.Allreduce) will aggregate
    the per-rank local_sums and local_counts to produce a GLOBAL centroid
    after the first iteration, overwriting the local init.  Convergence to
    a shared global model is achieved through the Allreduce loop, not
    through a common initialisation.

    If a globally synchronised init is required (e.g. for reproducibility
    benchmarks), rank 0 can run init_centroids() and broadcast the result
    via comm.bcast() before the iteration loop.  That is an optional
    enhancement; it is NOT required for the acceptance criterion of Issue #8.

    Parameters
    ----------
    points_rdd : cached RDD of numpy arrays, shape (D,)
    k          : number of clusters
    seed       : random seed for reproducibility

    Returns
    -------
    numpy.ndarray, shape (k, D), dtype float64
    """
    rng = np.random.default_rng(seed)

    # --- Collect a bounded sample to driver ---
    row_count = points_rdd.count()
    sample_size = min(10 * k, row_count)
    # takeSample(withReplacement, num, seed) is a Spark action that returns
    # a Python list of the RDD elements — here: a list of numpy arrays.
    sample = np.array(
        points_rdd.takeSample(False, sample_size, seed=seed),
        dtype=np.float64,
    )  # shape: (sample_size, D)

    # --- k-means++ selection on sample ---
    # Pick first centre uniformly at random
    idx = rng.integers(0, len(sample))
    centres = [sample[idx]]

    for _ in range(k - 1):
        # Squared distance from each sample point to the nearest existing centre
        dists = np.array(
            [min(np.sum((pt - c) ** 2) for c in centres) for pt in sample]
        )  # shape: (sample_size,)

        # Probability proportional to D^2 distance (k-means++ rule)
        probs = dists / dists.sum()
        chosen_idx = rng.choice(len(sample), p=probs)
        centres.append(sample[chosen_idx])

    centroids = np.array(centres, dtype=np.float64)  # shape: (k, D)
    logger.info(
        "[rank local] init_centroids: k=%d, D=%d, sample_size=%d",
        k,
        centroids.shape[1],
        sample_size,
    )
    return centroids


# ---------------------------------------------------------------------------
# 3. Per-iteration local centroid stats  (the core Step 3 output)
# ---------------------------------------------------------------------------


def compute_local_stats(
    points_rdd: RDD,
    centroids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute LOCAL centroid sums and point counts for this rank's data shard.

    This is the single Spark action executed per K-Means iteration.  It
    does NOT compute the final centroid averages — it returns the raw sums
    and counts so that Step 4 (comm.Allreduce) can aggregate them globally
    across all ranks before division.

    Algorithm (one mapPartitions + reduceByKey action)
    --------------------------------------------------
    For each data point p in the partition:
      1. Assign p to the nearest centroid: j = argmin_j ||p - centroids[j]||^2
      2. Accumulate: cluster_sums[j] += p
                     cluster_counts[j] += 1

    The accumulation is done inside mapPartitions (once per Spark partition
    slice) and then reduced across slices with reduceByKey.  This minimises
    the number of Python objects created compared to a per-row map.

    Parameters
    ----------
    points_rdd : cached RDD of numpy arrays, shape (D,)
                 Must be the RDD returned by load_partition_rdd().
    centroids  : numpy.ndarray, shape (K, D)
                 Current global centroid estimates (broadcast to all Spark
                 tasks via closure — K and D are small enough that explicit
                 sc.broadcast() is not required for prototype scale).

    Returns
    -------
    local_sums   : numpy.ndarray, shape (K, D)
        Sum of all data points assigned to each cluster on THIS rank's shard.
        NOT divided by count — ready for comm.Allreduce(op=MPI.SUM).

    local_counts : numpy.ndarray, shape (K,)
        Number of data points assigned to each cluster on THIS rank's shard.
        NOT normalised — ready for comm.Allreduce(op=MPI.SUM).

    Post-Allreduce usage (Step 4)
    -----------------------------
        global_centroids = global_sums / global_counts[:, np.newaxis]

    IMPORTANT: clusters with global_counts[j] == 0 must be handled by
    the caller (Step 4) before division.  This file raises no ZeroDivision
    error — it only produces raw sums and counts.
    """
    k, d = centroids.shape

    # Capture centroids as a local variable so Spark can serialise the
    # closure without pickling the entire calling frame.
    _centroids = centroids

    def _map_partition(points_iter):
        """
        Process one Spark partition slice: assign each point to its nearest
        centroid and accumulate local sums and counts.

        Yields (cluster_id, (point_sum, count)) pairs to reduceByKey.
        """
        # Local accumulators — one per cluster
        sums = np.zeros((k, d), dtype=np.float64)
        counts = np.zeros(k, dtype=np.int64)

        for pt in points_iter:
            # Vectorised nearest-centroid assignment:
            # diff shape: (K, D);  sq_dists shape: (K,)
            diff = _centroids - pt  # broadcast subtract
            sq_dists = np.einsum("ij,ij->i", diff, diff)  # row-wise dot product
            j = int(np.argmin(sq_dists))
            sums[j] += pt
            counts[j] += 1

        # Yield one (cluster_id, (sum_vec, count)) per cluster that has >= 1 point
        for j in range(k):
            if counts[j] > 0:
                yield (j, (sums[j], int(counts[j])))

    def _reduce_partition_stats(a, b):
        """Merge two (sum_vec, count) tuples from different Spark partition slices."""
        return (a[0] + b[0], a[1] + b[1])

    # One Spark action: mapPartitions → reduceByKey → collect
    # Result: list of (cluster_id, (sum_vec, count)) for clusters with >= 1 point
    raw_results = (
        points_rdd.mapPartitions(_map_partition).reduceByKey(_reduce_partition_stats).collect()
    )

    # Assemble into dense K-indexed arrays.
    # Clusters with zero local points remain as zeros — the Allreduce
    # of all-zero sums across all ranks is still zero, which is correct:
    # global_counts[j] == 0 means no point in the entire cluster globally,
    # and the convergence check in Step 4 will handle the ZeroDivision guard.
    local_sums = np.zeros((k, d), dtype=np.float64)
    local_counts = np.zeros(k, dtype=np.float64)  # float64 for Allreduce compat

    for cluster_id, (sum_vec, count) in raw_results:
        local_sums[cluster_id] = sum_vec
        local_counts[cluster_id] = float(count)

    return local_sums, local_counts
