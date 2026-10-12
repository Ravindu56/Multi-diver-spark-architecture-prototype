"""Early stop on global-loss convergence for LogReg FedAvg-MPI."""

from __future__ import annotations

import numpy as np
import pytest

from mpj_spark.applications.logreg.fedavg_mpi_run import _loss_converged
from mpj_spark.core.warm_start_lr import numpy_loss_grad, warm_lbfgs

# ``patched`` is a pytest fixture defined in the sibling module; importing it
# registers it here (used via usefixtures, so no argument shadows the import).
from .test_logreg_fedavg_warm_start import NF, _make_data, _run_workers, patched  # noqa: F401


def test_loss_converged_rules():
    assert _loss_converged(None, 0.5, 1e-4) is False  # first round: nothing to compare
    assert _loss_converged(0.5, 0.5, 1e-4) is True  # identical
    assert _loss_converged(0.023133, 0.022901, 1e-4) is False  # still improving (1%)
    assert _loss_converged(0.022901, 0.0229011, 1e-4) is True  # ~4e-6 relative
    assert _loss_converged(0.5, 0.5, 0.0) is False  # tol 0 disables
    assert _loss_converged(0.5, 0.5, -1.0) is False
    assert _loss_converged(0.0, 0.0, 1e-4) is True  # zero loss, no division error


@pytest.mark.usefixtures("patched")
def test_stops_early_on_every_rank_in_the_same_round(tmp_path):
    res = _run_workers(["p0", "p1"], tmp_path, max_iter=20, local_epochs=3, loss_tol=1e-4)
    assert res[0]["converged"] is True and res[1]["converged"] is True
    assert res[0]["stopped_early"] is True
    assert res[0]["iterations_done"] == res[1]["iterations_done"] < 20
    assert res[0]["iterations_done"] >= 3  # needs a few rounds before the loss settles
    assert len(res[0]["iter_metrics"]) == res[0]["iterations_done"]
    assert np.allclose(res[0]["weight_vector"], res[1]["weight_vector"])


@pytest.mark.usefixtures("patched")
def test_early_stopped_model_is_near_the_centralised_optimum(tmp_path):
    X, y = _make_data()
    res = _run_workers(["p0", "p1"], tmp_path, max_iter=30, local_epochs=3, loss_tol=1e-4)
    fg = numpy_loss_grad(X, y, 0.01)
    final = fg(np.concatenate([res[0]["weight_vector"], [res[0]["intercept"]]]))[0]
    ref = warm_lbfgs(fg, None, 0.0, NF, 500)[2]["loss_end"]
    assert final - ref < 5e-3


@pytest.mark.usefixtures("patched")
def test_loss_tol_zero_runs_all_rounds(tmp_path):
    res = _run_workers(["p0", "p1"], tmp_path, max_iter=7, local_epochs=3, loss_tol=0)
    assert res[0]["iterations_done"] == 7
    assert res[0]["converged"] is False and res[0]["stopped_early"] is False
