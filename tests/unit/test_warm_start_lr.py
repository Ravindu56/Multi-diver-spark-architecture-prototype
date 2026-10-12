"""Tests for the warm-start solver (pure NumPy; no Spark/MPI needed)."""
from __future__ import annotations

import numpy as np
import pytest
from scipy.optimize import check_grad

from mpj_spark.core.warm_start_lr import numpy_loss_grad, partition_stats, regularised, warm_lbfgs


def _data(n=4000, d=6, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    w_true = rng.normal(size=d)
    y = (rng.random(n) < 1 / (1 + np.exp(-(X @ w_true + 0.3)))).astype(float)
    return X, y


def test_gradient_matches_finite_differences():
    X, y = _data(300, 5)
    fg = numpy_loss_grad(X, y, 0.01)
    th = np.random.default_rng(1).normal(size=6)
    err = check_grad(lambda t: fg(t)[0], lambda t: fg(t)[1], th)
    assert err < 1e-5


def test_cold_refit_each_round_never_moves():
    """Reproduces the production bug: refitting from zero every round gives the
    same model every round (zero weight change after round 1)."""
    X, y = _data()
    fg = numpy_loss_grad(X, y, 0.01)
    w1, b1, _ = warm_lbfgs(fg, None, 0.0, 6, max_iter=5)
    w2, b2, _ = warm_lbfgs(fg, None, 0.0, 6, max_iter=5)
    assert np.allclose(w1, w2) and b1 == b2


def test_warm_start_keeps_improving_then_converges():
    X, y = _data()
    fg = numpy_loss_grad(X, y, 0.01)
    w, b, losses, deltas = None, 0.0, [], []
    for _ in range(10):
        prev = None if w is None else w.copy()
        w, b, info = warm_lbfgs(fg, w, b, 6, max_iter=2)
        losses.append(info["loss_end"])
        if prev is not None:
            deltas.append(float(np.linalg.norm(w - prev)))
    assert all(l2 <= l1 + 1e-12 for l1, l2 in zip(losses, losses[1:]))  # monotone
    assert deltas[0] > 1e-6                    # still moving after round 1 (cold refit: 0)
    assert deltas[-1] < deltas[0]              # and settling
    ref_w, ref_b, ref = warm_lbfgs(fg, None, 0.0, 6, max_iter=500)
    assert losses[-1] - ref["loss_end"] < 1e-3 # approaches the converged optimum


def test_fedavg_with_warm_start_approaches_centralised_optimum():
    """2 workers on disjoint halves, FedAvg each round (row-weighted)."""
    X, y = _data(6000, 6, seed=3)
    parts = [(X[:3000], y[:3000]), (X[3000:], y[3000:])]
    gw, gb = np.zeros(6), 0.0
    losses = []
    for _ in range(12):
        locals_ = []
        for Xp, yp in parts:
            fg = numpy_loss_grad(Xp, yp, 0.01)
            w, b, _ = warm_lbfgs(fg, gw, gb, 6, max_iter=3)
            locals_.append((w, b, len(yp)))
        n = sum(c for *_, c in locals_)
        gw = sum(w * c for w, _, c in locals_) / n
        gb = sum(b * c for _, b, c in locals_) / n
        losses.append(numpy_loss_grad(X, y, 0.01)(np.concatenate([gw, [gb]]))[0])
    ref = warm_lbfgs(numpy_loss_grad(X, y, 0.01), None, 0.0, 6, max_iter=500)[2]["loss_end"]
    assert losses[-1] - ref < 5e-3
    assert losses[-1] < losses[0]


def test_wrong_shape_rejected_not_swallowed():
    X, y = _data(100, 4)
    with pytest.raises(ValueError):
        warm_lbfgs(numpy_loss_grad(X, y, 0.01), np.zeros(3), 0.0, 4, max_iter=2)


def test_partition_sums_equal_whole():
    X, y = _data(500, 4)
    th = np.random.default_rng(2).normal(size=5)
    s_all, g_all, n_all = partition_stats(X, y, th)
    s1, g1, n1 = partition_stats(X[:200], y[:200], th)
    s2, g2, n2 = partition_stats(X[200:], y[200:], th)
    assert n1 + n2 == n_all
    assert s1 + s2 == pytest.approx(s_all)
    assert np.allclose(g1 + g2, g_all)
    f, g = regularised(s_all, g_all.copy(), n_all, th, 0.01)
    assert np.isfinite(f) and g.shape == th.shape
