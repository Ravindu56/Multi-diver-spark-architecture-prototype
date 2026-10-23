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
#       |
#   assign_and_sum()  ->  one Spark mapPartitions + reduceByKey action
#       |
#   (local_sums: K x D,  local_counts: K)   <- RETURNED to caller
#       |
#       |
#   assign_and_sum()  ->  one Spark mapPartitions + reduceByKey action
#       |
#   (local_sums: K x D,  local_counts: K)   <- RETURNED to caller
#       |
#   Step 4: comm.Allreduce(local_sums, global_sums, op=MPI.SUM)
#           comm.Allreduce(local_counts, global_counts, op=MPI.SUM)
#       |
#       |
#   global_centroids = global_sums / global_counts[:, np.newaxis]
#       |
#       |
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
#
# CHANGE LOG
# ----------
# perf(vectorise): the per-partition assignment loop ran one numpy call chain per
#   POINT (about 6 us of interpreter overhead each, ~3 s per epoch per 540k rows),
#   which dominated every K-Means sync mode.  compute_local_stats() now assigns
#   points in numpy blocks (rows x K x D temporaries, bounded by _BLOCK_BUDGET
#   elements).  Same return contract; cluster counts are identical and the sums
#   differ only by float summation order (~1e-15 relative).  New
#   compute_local_wcss() does the same for the per-round WCSS pass.
# perf(block-cache): every pass still un-pickled ~540k per-row numpy objects from the
#   cached RDD (~0.5-2 s per pass; six passes per round).  load_partition_rdd() now also
#   builds a cached twin RDD holding ONE (rows x D) float64 array per Spark partition and
#   attaches it to the returned RDD as ``_mpj_blocks``.  compute_local_stats() and
#   compute_local_wcss() use it when present and fall back to the per-row path otherwise
#   (RDD-like test fakes).  Results are identical; row-based consumers (count,
#   takeSample in init_centroids) are unchanged.
# =============================================================================



from __future__ import annotations

import itertools
import logging


import numpy as np
from pyspark import RDD
from pyspark.sql import SparkSession


logger = logging.getLogger(__name__)

# Upper bound on the float64 elements of the (rows x K x D) difference tensor built per
# block; keeps each temporary around 16 MB regardless of K and D.
_BLOCK_BUDGET = 2_000_000

# Attribute under which load_partition_rdd() stores the cached 2-D block twin of the rows.
_BLOCKS_ATTR = "_mpj_blocks"


# ---------------------------------------------------------------------------
# 0. Vectorised block helpers (module level: shipped to Spark workers by reference)
# ---------------------------------------------------------------------------


def _rows_per_chunk(k: int, d: int) -> int:
    return max(256, _BLOCK_BUDGET // max(1, k * d))


def _iter_blocks(points_iter, k: int, d: int):
    """Yield (rows, D) float64 blocks from an iterator of (D,) points."""
    rows = _rows_per_chunk(k, d)
    while True:
        chunk = list(itertools.islice(points_iter, rows))
        if not chunk:
            return
        yield np.asarray(chunk, dtype=np.float64)


def _iter_chunks(block_iter, k: int, d: int):
    """Yield row-chunks (views, no copy) of already-built (rows, D) blocks."""
    rows = _rows_per_chunk(k, d)
    for block in block_iter:
        for start in range(0, block.shape[0], rows):
            yield block[start : start + rows]


def _squared_distances(block: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """(rows, K) exact squared distances (difference form, no cancellation)."""
    diff = block[:, None, :] - centroids[None, :, :]
    return np.einsum("mkd,mkd->mk", diff, diff)


def _rows_to_block(points_iter):
    """Collapse one partition of (D,) rows into a single (rows, D) float64 array."""
    rows = list(points_iter)
    if rows:
        yield np.asarray(rows, dtype=np.float64)


def attach_blocks(points_rdd: RDD) -> RDD:
    """Build + cache the one-array-per-partition twin of ``points_rdd`` and attach it."""
    blocks = points_rdd.mapPartitions(_rows_to_block).cache()
    blocks.count()  # materialise now so timed iterations measure computation
    setattr(points_rdd, _BLOCKS_ATTR, blocks)
    return points_rdd


def _chunk_source(points_rdd):
    """Return (source_rdd, chunker): block twin when available, else the per-row RDD."""
    blocks = getattr(points_rdd, _BLOCKS_ATTR, None)
    if blocks is not None:
        return blocks, _iter_chunks
    return points_rdd, _iter_blocks


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
    read from memory rather than re-parsing the file each time.  A second,
    block-shaped cached twin (one (rows, D) array per partition) is attached as
    ``_mpj_blocks`` for the vectorised passes (see attach_blocks()).


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
        Parse one CSV line -> numpy array.  Returns None for bad lines.
        Parse one CSV line -> numpy array.  Returns None for bad lines.
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
    attach_blocks(points_rdd)
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


      1. Sample min(max(10k, 200), row_count) points from the RDD into
         driver RAM.  (10k points is sufficient for stable initialisation
         up to ~1M rows)

      1. Sample min(max(10k, 200), row_count) points from the RDD into
         driver RAM.  (10k points is sufficient for stable initialisation
         up to ~1M rows)
      2. Run k-means++ selection on the in-memory sample using numpy.
         This avoids k full Spark actions and runs in microseconds.



    This approach is consistent with how Spark's own k-means|| initialises
    on a subsample when the dataset is large.


    ROBUSTNESS
    ----------
    Spark's takeSample(withReplacement=False) is a bounded Bernoulli
    sampler: it retries at most a few times to fill the requested quota
    and otherwise returns whatever it collected.  When the requested size
    is a tiny fraction of the RDD (e.g. 30 of 1.2M rows, draw probability
    ~2.5e-5) an empty or under-full sample is a legitimate outcome and
    was observed as ValueError('high <= 0') from rng.integers().  This
    function therefore:
      * raises a clear error on a truly empty RDD,
      * floors the draw at 200 rows (empty-outcome probability becomes
        negligible and 200 rows is still free at this scale),
      * retries the draw up to twice with varied seeds,
      * falls back to a deterministic take() before k-means++ selection,
      * falls back to uniform selection if the sampled points are all
        identical (dists.sum() == 0 would otherwise yield NaN probs).



    ROBUSTNESS
    ----------
    Spark's takeSample(withReplacement=False) is a bounded Bernoulli
    sampler: it retries at most a few times to fill the requested quota
    and otherwise returns whatever it collected.  When the requested size
    is a tiny fraction of the RDD (e.g. 30 of 1.2M rows, draw probability
    ~2.5e-5) an empty or under-full sample is a legitimate outcome and
    was observed as ValueError('high <= 0') from rng.integers().  This
    function therefore:
      * raises a clear error on a truly empty RDD,
      * floors the draw at 200 rows (empty-outcome probability becomes
        negligible and 200 rows is still free at this scale),
      * retries the draw up to twice with varied seeds,
      * falls back to a deterministic take() before k-means++ selection,
      * falls back to uniform selection if the sampled points are all
        identical (dists.sum() == 0 would otherwise yield NaN probs).


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
    if row_count == 0:
        raise ValueError(
            "init_centroids: input RDD is empty — check the partition file "
            "and the CSV parsing filters before sampling."
        )
    sample_size = min(max(10 * k, 200), row_count)

    # takeSample(withReplacement, num, seed) is a Spark action that returns
    # a Python list of the RDD elements — here: a list of numpy arrays.
    sample_rows = points_rdd.takeSample(False, sample_size, seed=seed)
    if len(sample_rows) < k:
        # Rare path: the bounded sampler came up short. Retry twice with
        # varied seeds; if still short, fall back to a deterministic
        # head-collect so initialisation can never fail on a non-empty RDD.
        for attempt in range(2):
            alt_seed = None if seed is None else int(seed) + 101 + attempt
            sample_rows = points_rdd.takeSample(False, sample_size, seed=alt_seed)
            if len(sample_rows) >= k:
                break
        if len(sample_rows) < k:
            logger.warning(
                "[rank local] takeSample returned %d rows (wanted >= %d); "
                "falling back to deterministic take(%d)",
                len(sample_rows),
                k,
                sample_size,
            )
            sample_rows = points_rdd.take(sample_size)

    sample = np.array(sample_rows, dtype=np.float64)  # shape: (sample_size, D)

    # --- k-means++ selection on sample ---
    # Pick first centre uniformly at random
    idx = rng.integers(0, len(sample))
    centres = [sample[idx]]


    for _ in range(k - 1):
        # Squared distance from each sample point to the nearest existing centre
        dists = np.array(
            [min(np.sum((pt - c) ** 2) for c in centres) for pt in sample]
        )  # shape: (sample_size,)

        # Probability proportional to D^2 distance (k-means++ rule);
        # if every sampled point coincides with an existing centre the sum
        # is zero — fall back to a uniform pick instead of NaN probabilities.
        total = dists.sum()
        if total > 0:
            chosen_idx = rng.choice(len(sample), p=dists / total)
        else:
            chosen_idx = rng.integers(0, len(sample))
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
    slice) in numpy BLOCKS of points and then reduced across slices with
    reduceByKey.  When the RDD carries the cached block twin (_mpj_blocks), the
    blocks are used directly; otherwise rows are batched on the fly.


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
    _centroids = np.asarray(centroids, dtype=np.float64)
    source, chunker = _chunk_source(points_rdd)

    def _map_partition(points_iter):
        """
        Process one Spark partition slice: assign each point to its nearest
        centroid (in numpy blocks) and accumulate local sums and counts.


        Yields (cluster_id, (point_sum, count)) pairs to reduceByKey.
        """
        sums = np.zeros((k, d), dtype=np.float64)
        counts = np.zeros(k, dtype=np.int64)

        for block in chunker(points_iter, k, d):
            labels = _squared_distances(block, _centroids).argmin(axis=1)
            counts += np.bincount(labels, minlength=k)
            for j in range(k):
                mask = labels == j
                if mask.any():
                    sums[j] += block[mask].sum(axis=0)

        # Yield one (cluster_id, (sum_vec, count)) per cluster that has >= 1 point
        for j in range(k):
            if counts[j] > 0:
                yield (j, (sums[j], int(counts[j])))


    def _reduce_partition_stats(a, b):
        """Merge two (sum_vec, count) tuples from different Spark partition slices."""
        return (a[0] + b[0], a[1] + b[1])

    # One Spark action: mapPartitions -> reduceByKey -> collect
    # Result: list of (cluster_id, (sum_vec, count)) for clusters with >= 1 point
    raw_results = (
        source.mapPartitions(_map_partition).reduceByKey(_reduce_partition_stats).collect()
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


def compute_local_wcss(points_rdd: RDD, centroids: np.ndarray) -> float:
    """
    Sum of squared distances from every point on this rank's shard to its nearest
    centroid, computed in numpy blocks (one mapPartitions + sum action).
    """
    k, d = centroids.shape
    _centroids = np.asarray(centroids, dtype=np.float64)
    source, chunker = _chunk_source(points_rdd)

    def _wcss_partition(points_iter):
        total = 0.0
        for block in chunker(points_iter, k, d):
            total += float(_squared_distances(block, _centroids).min(axis=1).sum())
        yield total

    return float(source.mapPartitions(_wcss_partition).sum())
