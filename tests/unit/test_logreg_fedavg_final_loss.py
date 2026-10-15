"""FedAvg-MPI returns ``final_loss`` like gossip / async-PS / hybrid do."""

from __future__ import annotations

import numpy as np
import pytest

from mpj_spark.core.warm_start_lr import numpy_loss_grad

# ``patched`` is a pytest fixture defined in the sibling module; importing it
# registers it here (used via usefixtures, so no argument shadows the import).
from .test_logreg_fedavg_warm_start import _make_data, _run_workers, patched  # noqa: F401

REG = 0.01


def _theta(res):
    return np.concatenate([res["weight_vector"], [res["intercept"]]])


@pytest.mark.usefixtures("patched")
def test_final_loss_is_loss_of_returned_model_on_own_partition(tmp_path):
    X, y = _make_data()
    parts = [(X[:1500], y[:1500]), (X[1500:], y[1500:])]
    res = _run_workers(["p0", "p1"], tmp_path, max_iter=6, local_epochs=3, loss_tol=0)
    for r, (Xp, yp) in zip(res, parts, strict=True):
        expected = numpy_loss_grad(Xp, yp, REG)(_theta(r))[0]
        assert r["final_loss"] == pytest.approx(expected, rel=1e-9)


@pytest.mark.usefixtures("patched")
def test_row_weighted_mean_of_final_loss_is_the_exact_global_loss(tmp_path):
    X, y = _make_data()
    res = _run_workers(["p0", "p1"], tmp_path, max_iter=6, local_epochs=3, loss_tol=0)
    assert np.allclose(res[0]["weight_vector"], res[1]["weight_vector"])  # same model on every rank
    rows = [r["row_count"] for r in res]
    mean = sum(r["final_loss"] * n for r, n in zip(res, rows, strict=True)) / sum(rows)
    assert mean == pytest.approx(numpy_loss_grad(X, y, REG)(_theta(res[0]))[0], rel=1e-9)


@pytest.mark.usefixtures("patched")
def test_final_loss_not_above_loss_entering_the_first_round(tmp_path):
    res = _run_workers(["p0", "p1"], tmp_path, max_iter=6, local_epochs=3, loss_tol=0)
    first = res[0]["iter_metrics"][0]["global_loss"]
    assert res[0]["final_loss"] < first - 0.05
