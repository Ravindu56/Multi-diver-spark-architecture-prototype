# tests/logreg/test_local_gradient.py
#
# Unit tests for:
#   mpj_spark.applications.logreg.local_gradient.cores_per_worker()
#   mpj_spark.applications.logreg.local_gradient.parse_row()
#   mpj_spark.applications.logreg.local_gradient.compute_gradient_spark()
#
# Spark strategy:
#   compute_gradient_spark() receives an RDD, so we test it with a REAL
#   local[1] SparkSession (see conftest.py spark fixture).  The fixture is
#   module-scoped so the JVM starts once for this whole file.
#   cores_per_worker() and parse_row() have zero MPI/Spark dependencies
#   and are tested without any fixtures.

import os
from unittest.mock import patch

import numpy as np
import pytest

from mpj_spark.applications.logreg.local_gradient import (
    cores_per_worker,
    parse_row,
    compute_gradient_spark,
)


# ---------------------------------------------------------------------------
# cores_per_worker — pure arithmetic, no fixtures needed
# ---------------------------------------------------------------------------

class TestCoresPerWorker:

    def test_single_rank_gets_all_cores(self):
        """size=1 must return all available cores (capped at total)."""
        with patch("mpj_spark.applications.logreg.local_gradient.os.cpu_count",
                   return_value=8):
            assert cores_per_worker(1) == 8

    def test_even_split_across_ranks(self):
        """8 cores / 4 ranks = 2 cores per worker."""
        with patch("mpj_spark.applications.logreg.local_gradient.os.cpu_count",
                   return_value=8):
            assert cores_per_worker(4) == 2

    def test_floor_division_rounds_down(self):
        """7 cores / 3 ranks = 2 (floor), not 3."""
        with patch("mpj_spark.applications.logreg.local_gradient.os.cpu_count",
                   return_value=7):
            assert cores_per_worker(3) == 2

    def test_minimum_is_one(self):
        """Never returns 0 even when ranks > cores."""
        with patch("mpj_spark.applications.logreg.local_gradient.os.cpu_count",
                   return_value=2):
            assert cores_per_worker(100) == 1

    def test_cpu_count_none_defaults_to_four(self):
        """When os.cpu_count() returns None, formula uses 4."""
        with patch("mpj_spark.applications.logreg.local_gradient.os.cpu_count",
                   return_value=None):
            # 4 // 2 = 2
            assert cores_per_worker(2) == 2


# ---------------------------------------------------------------------------
# parse_row — pure text parsing, no fixtures needed
# ---------------------------------------------------------------------------

class TestParseRow:

    def test_valid_row_returns_features_and_label(self):
        row = parse_row("1.0,2.0,3.0,1.0", num_features=3)
        assert row is not None
        features, label = row
        np.testing.assert_array_equal(features, [1.0, 2.0, 3.0])
        assert label == 1.0

    def test_header_row_returns_none(self):
        """A text header like 'f0,f1,f2,label' cannot be cast to float."""
        assert parse_row("f0,f1,f2,label", num_features=3) is None

    def test_wrong_column_count_returns_none(self):
        """Rows with too few columns are silently dropped."""
        assert parse_row("1.0,2.0", num_features=3) is None

    def test_too_many_columns_returns_none(self):
        """Rows with too many columns are silently dropped."""
        assert parse_row("1.0,2.0,3.0,4.0,1.0", num_features=3) is None

    def test_whitespace_stripped(self):
        """Leading/trailing whitespace on the line should not cause failure."""
        row = parse_row("  0.5,0.5,1.0  ", num_features=2)
        assert row is not None
        features, label = row
        np.testing.assert_allclose(features, [0.5, 0.5])
        assert label == 1.0

    def test_negative_values_and_zero_label(self):
        """Negative features and label=0 must parse correctly."""
        row = parse_row("-1.5,0.0,0.0", num_features=2)
        assert row is not None
        features, label = row
        assert features[0] == pytest.approx(-1.5)
        assert label == 0.0

    def test_empty_line_returns_none(self):
        assert parse_row("", num_features=3) is None


# ---------------------------------------------------------------------------
# compute_gradient_spark — requires Spark fixture
# ---------------------------------------------------------------------------

class TestComputeGradientSpark:
    """Uses a real local[1] SparkSession via the spark conftest fixture."""

    def _make_rdd(self, spark, rows):
        """Parallelise a list of (np.ndarray, float) tuples into an RDD."""
        return spark.sparkContext.parallelize(rows)

    def test_output_shape_matches_feature_dimension(self, spark):
        """grad_local must have shape (D,) matching the feature dimension."""
        D = 5
        rng = np.random.default_rng(42)
        rows = [(rng.uniform(size=D), float(rng.integers(0, 2))) for _ in range(20)]
        rdd = self._make_rdd(spark, rows)
        w = np.zeros(D)

        grad, n = compute_gradient_spark(rdd, w)

        assert grad.shape == (D,)
        assert n == 20

    def test_gradient_direction_with_zero_weights(self):
        """
        With w=0, sigmoid(x·w) = 0.5 for every sample.
        grad_i = x_i * (0.5 - y_i).
        For all-positive features and label=1: grad_i = x_i*(0.5-1) < 0.
        So the gradient should be negative (weights need to increase).
        """
        rows = [
            (np.array([1.0, 1.0]), 1.0),
            (np.array([2.0, 2.0]), 1.0),
        ]
        rdd = spark.sparkContext.parallelize(rows)
        w = np.zeros(2)

        grad, n = compute_gradient_spark(rdd, w)

        assert np.all(grad < 0), "Expected negative gradient for positive features + label=1"
        assert n == 2

    def test_gradient_is_zero_for_perfect_prediction(self):
        """
        When the model perfectly predicts (sigmoid output ~= label),
        the gradient should be near zero.
        w·x >> 0 and y=1 → pred≈1, (pred-y)≈0 → grad≈0.
        """
        rows = [
            (np.array([1.0, 0.0]), 1.0),
        ]
        rdd = spark.sparkContext.parallelize(rows)
        # Very large positive weight for feature 0 → sigmoid(100) ≈ 1.0
        w = np.array([100.0, 0.0])

        grad, _ = compute_gradient_spark(rdd, w)

        np.testing.assert_allclose(grad, np.zeros(2), atol=1e-6)

    def test_count_returned_equals_rdd_size(self, spark):
        """The second return value must equal the number of rows in the RDD."""
        rows = [(np.array([1.0, 2.0]), 0.0)] * 7
        rdd = self._make_rdd(spark, rows)
        w = np.zeros(2)

        _, n = compute_gradient_spark(rdd, w)

        assert n == 7

    def test_gradient_is_finite_for_extreme_weights(self, spark):
        """Large weights must not produce NaN or Inf in the gradient."""
        rows = [(np.array([0.5, -0.5]), 1.0),
                (np.array([0.3,  0.7]), 0.0)]
        rdd = self._make_rdd(spark, rows)
        w = np.array([1e6, -1e6])

        grad, _ = compute_gradient_spark(rdd, w)

        assert np.all(np.isfinite(grad))

    # Retrieve the spark fixture via method argument for the class
    @pytest.fixture(autouse=False)
    def spark(self, spark):
        return spark
