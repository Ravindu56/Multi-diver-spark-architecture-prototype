"""mpj_spark/core/warm_start_lr.py
Warm-startable local logistic-regression solver for the sync modes.

WHY THIS EXISTS
---------------
pyspark.ml.classification.LogisticRegression has no initial-weights setter
(verified on PySpark 4.2.0: hasattr(LogisticRegression, "setInitialWeights")
is False).  The sync modes called ``lr.setInitialWeights(...)`` inside
``try/except Exception: pass``, so the AttributeError was swallowed and every
round refit from zero.  Result: identical local models each round, zero weight
change after round 1, and "convergence parity" that no synchronisation could
have caused.

This module runs a few L-BFGS-B steps STARTING FROM the supplied weights.
The loss/gradient is either plain NumPy (tests, small data) or computed by
Spark (one pass per function evaluation, partition-vectorised, tree-reduced).

Objective (identical for every mode, so models are comparable):
    f(w, b) = mean_i logloss(y_i, x_i.w + b) + 0.5 * reg_param * ||w||^2
The intercept is not regularised.  No feature standardisation (state this when
comparing with MLlib models, which standardise by default).
"""
from __future__ import annotations

from collections.abc import Callable

import numpy as np
from scipy.optimize import minimize

LossGrad = Callable[[np.ndarray], tuple[float, np.ndarray]]


def _sigmoid(z: np.ndarray) -> np.ndarray:
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    e = np.exp(z[~pos])
    out[~pos] = e / (1.0 + e)
    return out


def partition_stats(X: np.ndarray, y: np.ndarray, theta: np.ndarray) -> tuple[float, np.ndarray, int]:
    """Summed (not averaged) log-loss, gradient and row count for one block.

    theta = [w_0..w_{d-1}, b].  Pure NumPy so it can run inside a Spark
    partition and in unit tests identically.
    """
    w, b = theta[:-1], theta[-1]
    z = X @ w + b
    # log(1+exp(z)) - y*z, stable
    loss = float(np.sum(np.logaddexp(0.0, z) - y * z))
    r = _sigmoid(z) - y
    grad = np.concatenate([X.T @ r, [r.sum()]])
    return loss, grad, int(X.shape[0])


def regularised(loss_sum: float, grad_sum: np.ndarray, n: int, theta: np.ndarray, reg: float):
    w = theta[:-1]
    f = loss_sum / n + 0.5 * reg * float(w @ w)
    g = grad_sum / n
    g[:-1] += reg * w
    return f, g


def numpy_loss_grad(X: np.ndarray, y: np.ndarray, reg: float) -> LossGrad:
    def fg(theta: np.ndarray):
        s, g, n = partition_stats(X, y, theta)
        return regularised(s, g, n, theta, reg)

    return fg


def spark_loss_grad(rdd, num_features: int, reg: float, n_rows: int) -> LossGrad:
    """rdd: RDD of (features ndarray(d,), label float).  One Spark pass per call."""

    def fg(theta: np.ndarray):
        bc = theta.copy()

        def per_partition(it):
            rows = list(it)
            if not rows:
                return
            X = np.asarray([r[0] for r in rows], dtype=np.float64)
            y = np.asarray([r[1] for r in rows], dtype=np.float64)
            yield partition_stats(X, y, bc)

        zero = (0.0, np.zeros(num_features + 1), 0)
        s, g, n = rdd.mapPartitions(per_partition).fold(
            zero, lambda a, c: (a[0] + c[0], a[1] + c[1], a[2] + c[2])
        )
        return regularised(s, g, n if n else n_rows, theta, reg)

    return fg


def warm_lbfgs(
    loss_grad: LossGrad,
    weights: np.ndarray | None,
    intercept: float,
    num_features: int,
    max_iter: int,
) -> tuple[np.ndarray, float, dict]:
    """Run <= max_iter L-BFGS-B iterations from (weights, intercept).

    Returns (weights, intercept, info) with info = {loss_start, loss_end, n_eval}.
    ``weights=None`` means a cold start from zero (first round only).
    """
    w0 = np.zeros(num_features) if weights is None else np.asarray(weights, dtype=np.float64)
    if w0.shape != (num_features,):
        raise ValueError(f"warm-start weights have shape {w0.shape}, expected ({num_features},)")
    theta0 = np.concatenate([w0, [float(intercept)]])
    loss_start, _ = loss_grad(theta0)
    res = minimize(loss_grad, theta0, jac=True, method="L-BFGS-B", options={"maxiter": int(max_iter)})
    theta = res.x
    return theta[:-1].copy(), float(theta[-1]), {
        "loss_start": float(loss_start),
        "loss_end": float(res.fun),
        "n_eval": int(res.nfev),
    }
