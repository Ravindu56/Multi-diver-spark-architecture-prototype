"""Unit tests for kmeans/common.py (shared local step)."""

from __future__ import annotations

import numpy as np
import pytest

from mpj_spark.applications.kmeans import common


class _Mapped:
    def __init__(self, vals):
        self.vals = vals

    def sum(self):
        return sum(self.vals)


class _FakeRDD:
    def __init__(self, pts):
        self.pts = [np.asarray(p, dtype=np.float64) for p in pts]

    def map(self, f):
        return _Mapped([f(p) for p in self.pts])


def _stats(rdd, cents):
    k, d = cents.shape
    sums, counts = np.zeros((k, d)), np.zeros(k)
    for p in rdd.pts:
        j = int(np.argmin(((cents - p) ** 2).sum(axis=1)))
        sums[j] += p
        counts[j] += 1
    return sums, counts


def _blobs():
    rng = np.random.default_rng(0)
    a = rng.normal(0.0, 0.1, (40, 2))
    b = rng.normal(5.0, 0.1, (40, 2))
    return _FakeRDD(np.vstack([a, b]))


def test_local_lloyd_converges_to_blob_means_and_keeps_input():
    rdd = _blobs()
    init = np.array([[0.5, 0.5], [4.0, 4.0]])
    snap = init.copy()
    cents, sums, counts, wcss = common.local_lloyd(rdd, init, 5, stats_fn=_stats)
    assert np.array_equal(init, snap)
    assert sorted(counts.tolist()) == [40.0, 40.0]
    assert np.allclose(sorted(cents[:, 0]), [0.0, 5.0], atol=0.1)
    assert wcss == pytest.approx(common.local_wcss(rdd, cents))
    assert wcss < 2.0


def test_empty_cluster_keeps_position():
    rdd = _FakeRDD([[0.0, 0.0], [0.1, 0.0]])
    init = np.array([[0.0, 0.0], [100.0, 100.0]])
    cents, _, counts, _ = common.local_lloyd(rdd, init, 3, stats_fn=_stats)
    assert counts[1] == 0
    assert np.allclose(cents[1], [100.0, 100.0])


def test_epochs_must_be_positive():
    with pytest.raises(ValueError):
        common.local_lloyd(_blobs(), np.zeros((2, 2)), 0, stats_fn=_stats)


def test_unpack_stats_tuple_and_dict():
    s, c, w = common.unpack_stats((np.ones((2, 2)), np.ones(2)))
    assert w == 0.0 and s.shape == (2, 2)
    s, c, w = common.unpack_stats({"sums": np.ones((2, 2)), "counts": [1, 2], "wcss": 3.0})
    assert w == 3.0 and c.tolist() == [1.0, 2.0]
    assert common.unpack_stats((np.ones((1, 1)), np.ones(1), 7.0))[2] == 7.0


def test_centroid_shift():
    assert common.centroid_shift(np.zeros((2, 2)), np.ones((2, 2))) == pytest.approx(2.0)
