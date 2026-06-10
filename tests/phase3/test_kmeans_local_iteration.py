# =============================================================================
# tests/phase3/test_kmeans_local_iteration.py
# Phase 3 — Issue #8 — Step 3: Local Iteration Unit Tests
#
# WHAT IS TESTED
# --------------
#   1. init_centroids() returns the correct shape (k, D)
#   2. init_centroids() returns centroids within the data range
#   3. compute_local_stats() returns local_sums shape (K, D)
#      and local_counts shape (K,)
#   4. compute_local_stats() sum correctness on a known 2D dataset:
#      with perfectly separated clusters, each rank's local_sums should
#      match the ground-truth column sums for points in that cluster
#   5. compute_local_stats() with all points in one cluster: the other
#      cluster's sums and counts must be zero (zero-count cluster guard)
#   6. RDD is re-usable across multiple calls (cached RDD not consumed)
#
# HOW TO RUN (no MPI needed — pure PySpark unit tests)
# ------------------------------------------------------
#   python -m pytest tests/phase3/test_kmeans_local_iteration.py -v
#
# These tests start a local[1] SparkSession.  They run WITHOUT mpirun
# because local_iteration.py has zero MPI imports.
# =============================================================================

from __future__ import annotations

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# SparkSession fixture — one session shared across all tests in this module
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def spark():
    from pyspark.sql import SparkSession

    sess = (
        SparkSession.builder.appName("test-kmeans-local-iteration")
        .master("local[1]")
        .config("spark.driver.memory", "512m")
        .config("spark.ui.enabled", "false")
        .getOrCreate()
    )
    sess.sparkContext.setLogLevel("ERROR")
    yield sess
    sess.stop()


# ---------------------------------------------------------------------------
# Shared test data: two perfectly separated 2D clusters
# Cluster 0: points near (1.0, 1.0)
# Cluster 1: points near (9.0, 9.0)
# With centroids exactly at the cluster means, every point is assigned
# to its correct cluster — making expected sums/counts deterministic.
# ---------------------------------------------------------------------------
_CLUSTER0 = np.array([[1.0, 1.0], [1.1, 0.9], [0.9, 1.1], [1.0, 1.0]], dtype=np.float64)
_CLUSTER1 = np.array([[9.0, 9.0], [9.1, 8.9], [8.9, 9.1], [9.0, 9.0]], dtype=np.float64)
_ALL_POINTS = np.vstack([_CLUSTER0, _CLUSTER1])  # shape (8, 2)

# Exact centroids — every point assigned unambiguously
_CENTROIDS_EXACT = np.array([[1.0, 1.0], [9.0, 9.0]], dtype=np.float64)

# Expected sums and counts for the local partition (all 8 points on one rank)
_EXPECTED_SUMS = np.array(
    [
        _CLUSTER0.sum(axis=0),  # cluster 0 sum
        _CLUSTER1.sum(axis=0),  # cluster 1 sum
    ],
    dtype=np.float64,
)
_EXPECTED_COUNTS = np.array([len(_CLUSTER0), len(_CLUSTER1)], dtype=np.float64)


@pytest.fixture(scope="module")
def points_rdd(spark):
    """Cached RDD of numpy arrays from _ALL_POINTS."""
    rdd = spark.sparkContext.parallelize([row for row in _ALL_POINTS]).cache()
    rdd.count()  # materialise cache
    return rdd


# ---------------------------------------------------------------------------
# Test 1 — init_centroids shape
# ---------------------------------------------------------------------------
def test_init_centroids_shape(spark, points_rdd):
    from mpj_spark.applications.kmeans.local_iteration import init_centroids

    k, D = 2, 2
    centroids = init_centroids(points_rdd, k=k, seed=42)
    assert centroids.shape == (
        k,
        D,
    ), f"Expected shape ({k}, {D}), got {centroids.shape}"


# ---------------------------------------------------------------------------
# Test 2 — init_centroids values within data range
# ---------------------------------------------------------------------------
def test_init_centroids_within_data_range(spark, points_rdd):
    from mpj_spark.applications.kmeans.local_iteration import init_centroids

    centroids = init_centroids(points_rdd, k=2, seed=42)
    data_min = _ALL_POINTS.min()
    data_max = _ALL_POINTS.max()
    assert np.all(
        centroids >= data_min - 1e-9
    ), f"Centroid value below data min ({data_min}): {centroids}"
    assert np.all(
        centroids <= data_max + 1e-9
    ), f"Centroid value above data max ({data_max}): {centroids}"


# ---------------------------------------------------------------------------
# Test 3 — compute_local_stats output shapes
# ---------------------------------------------------------------------------
def test_compute_local_stats_shapes(spark, points_rdd):
    from mpj_spark.applications.kmeans.local_iteration import compute_local_stats

    k, D = 2, 2
    local_sums, local_counts = compute_local_stats(points_rdd, _CENTROIDS_EXACT)
    assert local_sums.shape == (
        k,
        D,
    ), f"local_sums: expected ({k}, {D}), got {local_sums.shape}"
    assert local_counts.shape == (k,), f"local_counts: expected ({k},), got {local_counts.shape}"


# ---------------------------------------------------------------------------
# Test 4 — compute_local_stats sum correctness on known dataset
# ---------------------------------------------------------------------------
def test_compute_local_stats_correct_sums(spark, points_rdd):
    from mpj_spark.applications.kmeans.local_iteration import compute_local_stats

    local_sums, local_counts = compute_local_stats(points_rdd, _CENTROIDS_EXACT)

    np.testing.assert_array_almost_equal(
        local_sums,
        _EXPECTED_SUMS,
        decimal=10,
        err_msg="local_sums do not match expected cluster column sums",
    )
    np.testing.assert_array_almost_equal(
        local_counts,
        _EXPECTED_COUNTS,
        decimal=10,
        err_msg="local_counts do not match expected cluster sizes",
    )


# ---------------------------------------------------------------------------
# Test 5 — zero-count cluster: sums and counts stay zero
# ---------------------------------------------------------------------------
def test_zero_count_cluster_stays_zero(spark):
    """
    When a centroid is so far from all data points that no point is nearest
    to it, that cluster's local_sums row and local_counts element must be
    zero — not NaN, not garbage.
    """
    from mpj_spark.applications.kmeans.local_iteration import compute_local_stats

    # All 4 points are near (1,1); the second centroid is at (1000, 1000)
    close_points = np.array([[1.0, 1.0], [1.1, 0.9], [0.9, 1.1]], dtype=np.float64)
    rdd = spark.sparkContext.parallelize(list(close_points)).cache()
    rdd.count()

    # centroid 1 is unreachable from any data point
    centroids = np.array([[1.0, 1.0], [1000.0, 1000.0]], dtype=np.float64)
    local_sums, local_counts = compute_local_stats(rdd, centroids)

    assert (
        local_counts[1] == 0.0
    ), f"Expected local_counts[1] == 0 for unreachable centroid, got {local_counts[1]}"
    np.testing.assert_array_equal(
        local_sums[1],
        np.zeros(2),
        err_msg="local_sums[1] should be zeros for empty cluster",
    )
    rdd.unpersist()


# ---------------------------------------------------------------------------
# Test 6 — RDD is re-usable (cached, not consumed after first call)
# ---------------------------------------------------------------------------
def test_rdd_reusable_across_iterations(spark, points_rdd):
    """
    compute_local_stats() must not consume or invalidate the cached RDD.
    Calling it twice must return identical results — simulating two
    K-Means iterations reading the same partition data.
    """
    from mpj_spark.applications.kmeans.local_iteration import compute_local_stats

    sums1, counts1 = compute_local_stats(points_rdd, _CENTROIDS_EXACT)
    sums2, counts2 = compute_local_stats(points_rdd, _CENTROIDS_EXACT)

    np.testing.assert_array_equal(
        sums1,
        sums2,
        err_msg="local_sums differ across two calls — RDD may have been consumed",
    )
    np.testing.assert_array_equal(
        counts1,
        counts2,
        err_msg="local_counts differ across two calls — RDD may have been consumed",
    )
