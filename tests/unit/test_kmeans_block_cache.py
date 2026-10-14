"""K-Means block cache: one (rows, D) array per partition must give identical results."""

from __future__ import annotations

import numpy as np
import pytest

from mpj_spark.applications.kmeans import local_iteration
from mpj_spark.applications.kmeans.local_iteration import (
    _BLOCKS_ATTR,
    _rows_to_block,
    attach_blocks,
    compute_local_stats,
    compute_local_wcss,
)


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


class _FakeRDD:
    """Partitioned RDD fake; ``forbid`` makes any mapPartitions call fail the test."""

    def __init__(self, parts, forbid=False):
        self.parts = parts
        self.forbid = forbid

    def mapPartitions(self, f):
        if self.forbid:
            raise AssertionError("the per-row RDD must not be scanned when blocks exist")
        outs = [list(f(iter(p))) for p in self.parts]
        flat = [x for o in outs for x in o]
        if flat and isinstance(flat[0], np.ndarray):
            return _FakeRDD(outs)
        if flat and isinstance(flat[0], tuple):
            return _KV(flat)
        return _Nums(flat) if flat else _KV([])

    def cache(self):
        return self

    def count(self):
        return sum(len(p) for p in self.parts)


def _parts(n_parts=3, rows=700, d=5, seed=0):
    rng = np.random.default_rng(seed)
    return [[rng.normal(0, 3, d) for _ in range(rows)] for _ in range(n_parts)]


def _reference(parts, cents):
    k, d = cents.shape
    sums, counts = np.zeros((k, d)), np.zeros(k)
    for part in parts:
        for p in part:
            diff = cents - p
            j = int(np.argmin(np.einsum("ij,ij->i", diff, diff)))
            sums[j] += p
            counts[j] += 1
    return sums, counts


def test_attach_blocks_builds_one_array_per_partition_with_identical_rows():
    parts = _parts(n_parts=2, rows=40)
    rdd = attach_blocks(_FakeRDD(parts))
    blocks = getattr(rdd, _BLOCKS_ATTR)
    assert [len(p) for p in blocks.parts] == [1, 1]
    for blk_part, rows in zip(blocks.parts, parts, strict=True):
        assert blk_part[0].shape == (40, 5)
        assert np.array_equal(blk_part[0], np.asarray(rows))


def test_rows_to_block_skips_empty_partitions():
    assert list(_rows_to_block(iter([]))) == []


def test_stats_from_blocks_match_per_point_reference(monkeypatch):
    monkeypatch.setattr(local_iteration, "_BLOCK_BUDGET", 1)  # force several chunks per block
    parts = _parts()
    cents = np.random.default_rng(1).normal(0, 3, (4, 5))
    rdd = attach_blocks(_FakeRDD(parts))
    sums, counts = compute_local_stats(rdd, cents)
    ref_sums, ref_counts = _reference(parts, cents)
    assert np.array_equal(counts, ref_counts)
    assert np.allclose(sums, ref_sums, rtol=1e-12, atol=1e-9)


def test_wcss_from_blocks_matches_brute_force():
    parts = _parts(seed=3)
    cents = np.random.default_rng(4).normal(0, 3, (3, 5))
    allpts = np.vstack([np.vstack(p) for p in parts])
    expected = float(((allpts[:, None, :] - cents[None]) ** 2).sum(-1).min(axis=1).sum())
    rdd = attach_blocks(_FakeRDD(parts))
    assert compute_local_wcss(rdd, cents) == pytest.approx(expected, rel=1e-12)


def test_passes_scan_the_block_twin_not_the_per_row_rdd():
    rdd = attach_blocks(_FakeRDD(_parts(n_parts=2, rows=30)))
    rdd.forbid = True  # any further scan of the per-row RDD would raise
    cents = np.random.default_rng(5).normal(0, 3, (3, 5))
    compute_local_stats(rdd, cents)
    compute_local_wcss(rdd, cents)


def test_without_blocks_the_per_row_path_still_works():
    parts = _parts(n_parts=2, rows=50, seed=7)
    cents = np.random.default_rng(8).normal(0, 3, (3, 5))
    sums, counts = compute_local_stats(_FakeRDD(parts), cents)
    ref_sums, ref_counts = _reference(parts, cents)
    assert np.array_equal(counts, ref_counts)
    assert np.allclose(sums, ref_sums, rtol=1e-12, atol=1e-9)
