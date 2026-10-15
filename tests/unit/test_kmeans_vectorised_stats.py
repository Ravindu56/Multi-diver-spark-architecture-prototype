"""Vectorised K-Means local stats / WCSS must reproduce the per-point reference."""

from __future__ import annotations

import numpy as np
import pytest

from mpj_spark.applications.kmeans import common, local_iteration
from mpj_spark.applications.kmeans.local_iteration import compute_local_stats, compute_local_wcss


class _KV:
    def __init__(self, pairs):
        self.pairs = pairs

    def reduceByKey(self, f):
        acc = {}
        for key, val in self.pairs:
            acc[key] = f(acc[key], val) if key in acc else val
        return _KV(list(acc.items()))

    def collect(self):
        return list(self.pairs)


class _Nums:
    def __init__(self, vals):
        self.vals = vals

    def sum(self):
        return sum(self.vals)


class _PartRDD:
    """Minimal partitioned RDD: mapPartitions / reduceByKey / collect / sum."""

    def __init__(self, parts):
        self.parts = parts

    def mapPartitions(self, f):
        out = []
        for part in self.parts:
            out.extend(f(iter(part)))
        if out and isinstance(out[0], tuple):
            return _KV(out)
        return _Nums(out) if out else _KV([])


class _MapOnlyRDD:
    """Fake without mapPartitions (the per-point fallback path)."""

    def __init__(self, pts):
        self.pts = pts

    def map(self, f):
        return _Nums([f(p) for p in self.pts])


def _reference_stats(parts, cents):
    k, d = cents.shape
    sums, counts = np.zeros((k, d)), np.zeros(k)
    for part in parts:
        for p in part:
            diff = cents - p
            j = int(np.argmin(np.einsum("ij,ij->i", diff, diff)))
            sums[j] += p
            counts[j] += 1
    return sums, counts


def _parts(n_parts=3, rows=700, d=5, seed=0):
    rng = np.random.default_rng(seed)
    return [[rng.normal(0, 3, d) for _ in range(rows)] for _ in range(n_parts)]


def test_stats_match_per_point_reference_across_blocks_and_partitions(monkeypatch):
    monkeypatch.setattr(local_iteration, "_BLOCK_BUDGET", 1)
    parts = _parts()
    cents = np.random.default_rng(1).normal(0, 3, (4, 5))
    sums, counts = compute_local_stats(_PartRDD(parts), cents)
    ref_sums, ref_counts = _reference_stats(parts, cents)
    assert np.array_equal(counts, ref_counts)
    assert np.allclose(sums, ref_sums, rtol=1e-12, atol=1e-9)
    assert counts.sum() == sum(len(p) for p in parts)


def test_empty_cluster_has_zero_count_and_sum():
    parts = [[np.array([0.0, 0.0]), np.array([0.1, 0.0])]]
    cents = np.array([[0.0, 0.0], [100.0, 100.0]])
    sums, counts = compute_local_stats(_PartRDD(parts), cents)
    assert counts.tolist() == [2.0, 0.0]
    assert np.allclose(sums[1], 0.0)


def test_empty_and_single_row_partitions_are_handled():
    parts = [[], [np.array([1.0, 1.0])], []]
    cents = np.array([[0.0, 0.0], [5.0, 5.0]])
    sums, counts = compute_local_stats(_PartRDD(parts), cents)
    assert counts.tolist() == [1.0, 0.0]
    assert np.allclose(sums[0], [1.0, 1.0])


def test_wcss_matches_brute_force(monkeypatch):
    monkeypatch.setattr(local_iteration, "_BLOCK_BUDGET", 1)
    parts = _parts(seed=3)
    cents = np.random.default_rng(4).normal(0, 3, (3, 5))
    allpts = np.vstack([np.vstack(p) for p in parts])
    expected = float(((allpts[:, None, :] - cents[None]) ** 2).sum(-1).min(axis=1).sum())
    assert compute_local_wcss(_PartRDD(parts), cents) == pytest.approx(expected, rel=1e-12)


def test_common_local_wcss_uses_block_path_for_real_rdds_and_falls_back_otherwise(monkeypatch):
    parts = _parts(n_parts=2, rows=50, seed=5)
    cents = np.random.default_rng(6).normal(0, 3, (3, 5))
    allpts = [p for part in parts for p in part]
    expected = float(((np.vstack(allpts)[:, None] - cents[None]) ** 2).sum(-1).min(axis=1).sum())

    calls = []
    real = common.compute_local_wcss
    monkeypatch.setattr(
        common, "compute_local_wcss", lambda rdd, c: calls.append(1) or real(rdd, c)
    )
    assert common.local_wcss(_PartRDD(parts), cents) == pytest.approx(expected, rel=1e-12)
    assert calls == [1]

    assert common.local_wcss(_MapOnlyRDD(allpts), cents) == pytest.approx(expected, rel=1e-12)
    assert calls == [1]


def test_fedavg_local_wcss_uses_block_path_for_real_rdds_and_falls_back_otherwise():
    from mpj_spark.applications.kmeans import fedavg_mpi_run

    parts = _parts(n_parts=2, rows=50, seed=7)
    cents = np.random.default_rng(8).normal(0, 3, (3, 5))
    allpts = [p for part in parts for p in part]
    expected = float(((np.vstack(allpts)[:, None] - cents[None]) ** 2).sum(-1).min(axis=1).sum())

    # _PartRDD has mapPartitions but NO map(): only the block path can handle it.
    assert fedavg_mpi_run._local_wcss(_PartRDD(parts), cents) == pytest.approx(expected, rel=1e-12)
    assert fedavg_mpi_run._local_wcss(_MapOnlyRDD(allpts), cents) == pytest.approx(
        expected, rel=1e-12
    )
