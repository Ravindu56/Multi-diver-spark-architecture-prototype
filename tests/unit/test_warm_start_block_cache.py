"""spark_loss_grad block cache: same objective, but the per-row RDD is scanned only once."""

from __future__ import annotations

import numpy as np
import pytest

from mpj_spark.core import warm_start_lr as ws


class _FakeSparkRDD:
    """Partitioned RDD fake with the pyspark calls spark_loss_grad uses."""

    def __init__(self, parts):
        self.parts = parts
        self.scans = 0

    def mapPartitions(self, f):
        self.scans += 1
        return _FakeSparkRDD([list(f(iter(p))) for p in self.parts])

    def map(self, f):
        return _FakeSparkRDD([[f(x) for x in p] for p in self.parts])

    def cache(self):
        return self

    def count(self):
        return sum(len(p) for p in self.parts)

    def fold(self, zero, op):
        acc = zero
        for part in self.parts:
            for item in part:
                acc = op(acc, item)
        return acc


def _data(n_parts=3, rows=120, d=4, seed=0, empty_last=False):
    rng = np.random.default_rng(seed)
    parts = []
    for _ in range(n_parts):
        parts.append([(rng.normal(size=d), float(rng.integers(0, 2))) for _ in range(rows)])
    if empty_last:
        parts.append([])
    X = np.vstack([r[0] for p in parts for r in p])
    y = np.asarray([r[1] for p in parts for r in p])
    return parts, X, y


def _thetas(d, count=3, seed=1):
    rng = np.random.default_rng(seed)
    return [rng.normal(size=d + 1) for _ in range(count)]


def test_block_path_matches_numpy_objective_including_an_empty_partition(monkeypatch):
    monkeypatch.setattr(ws, "_is_spark_rdd", lambda rdd: True)
    parts, X, y = _data(empty_last=True)
    fg = ws.spark_loss_grad(_FakeSparkRDD(parts), 4, 0.01, len(y))
    ref = ws.numpy_loss_grad(X, y, 0.01)
    for theta in _thetas(4):
        f, g = fg(theta)
        f_ref, g_ref = ref(theta)
        assert f == pytest.approx(f_ref, rel=1e-12)
        assert np.allclose(g, g_ref, rtol=1e-10, atol=1e-12)


def test_block_path_scans_the_per_row_rdd_once_not_once_per_evaluation(monkeypatch):
    monkeypatch.setattr(ws, "_is_spark_rdd", lambda rdd: True)
    parts, _, y = _data()
    rdd = _FakeSparkRDD(parts)
    fg = ws.spark_loss_grad(rdd, 4, 0.01, len(y))
    for theta in _thetas(4, count=5):
        fg(theta)
    assert rdd.scans == 1


def test_non_spark_rdd_keeps_the_per_row_path(monkeypatch):
    monkeypatch.setattr(ws, "_is_spark_rdd", lambda rdd: False)
    parts, X, y = _data()
    rdd = _FakeSparkRDD(parts)
    fg = ws.spark_loss_grad(rdd, 4, 0.01, len(y))
    ref = ws.numpy_loss_grad(X, y, 0.01)
    thetas = _thetas(4, count=3)
    for theta in thetas:
        assert fg(theta)[0] == pytest.approx(ref(theta)[0], rel=1e-12)
    assert rdd.scans == len(thetas)


def test_warm_lbfgs_through_the_block_path_converges_like_numpy(monkeypatch):
    monkeypatch.setattr(ws, "_is_spark_rdd", lambda rdd: True)
    parts, X, y = _data(n_parts=2, rows=300)
    fg_block = ws.spark_loss_grad(_FakeSparkRDD(parts), 4, 0.01, len(y))
    fg_np = ws.numpy_loss_grad(X, y, 0.01)
    w_b, b_b, info_b = ws.warm_lbfgs(fg_block, None, 0.0, 4, 8)
    w_n, b_n, info_n = ws.warm_lbfgs(fg_np, None, 0.0, 4, 8)
    assert np.allclose(w_b, w_n, atol=1e-9) and b_b == pytest.approx(b_n, abs=1e-9)
    assert info_b["loss_end"] == pytest.approx(info_n["loss_end"], rel=1e-12)
    assert info_b["loss_end"] < info_b["loss_start"]
