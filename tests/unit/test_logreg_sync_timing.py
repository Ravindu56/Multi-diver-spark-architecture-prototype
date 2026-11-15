"""LogReg Allreduce runner: sync_time_s covers MPI calls only (P5-05 / #30).

Before the fix, sync_time_s included check_loss_convergence(), which runs two
Spark actions over the whole partition. These tests pin the split: the loss
pass is timed as loss_time_s, reuses the gradient pass's row count, and the
loss/convergence arithmetic is unchanged.

CI-safe: mpi4py is a stub (no libmpi), Spark is replaced by an in-memory RDD.
"""

from __future__ import annotations

import csv
import functools
import sys
import time
import types

import numpy as np
import pytest

from mpj_spark.applications.logreg import allreduce as ar
from mpj_spark.applications.logreg import local_gradient as lg
from mpj_spark.applications.logreg import partition as lp
from mpj_spark.applications.logreg.metrics import LogRegMetricsCollector


@pytest.fixture
def fake_mpi(monkeypatch):
    """Stand-in mpi4py so the lazy `from mpi4py import MPI` works without libmpi."""
    mpi = types.ModuleType("mpi4py.MPI")
    mpi.DOUBLE = object()
    mpi.SUM = object()
    pkg = types.ModuleType("mpi4py")
    pkg.MPI = mpi
    monkeypatch.setitem(sys.modules, "mpi4py", pkg)
    monkeypatch.setitem(sys.modules, "mpi4py.MPI", mpi)


class FakeComm:
    """MPI stand-in where every rank holds the same values.

    Allreduce(SUM) therefore returns size * send, which is what a real job with
    identical ranks would produce; bcast returns rank 0's object.
    """

    def __init__(self, size: int = 1):
        self.size = size
        self.allreduce_calls = 0

    def Allreduce(self, send, recv, op=None):  # noqa: N802 (mpi4py API)
        self.allreduce_calls += 1
        recv[0][...] = send[0] * self.size

    def bcast(self, obj, root=0):
        return obj

    def Barrier(self):  # noqa: N802 (mpi4py API)
        pass


class FakeRDD:
    """Minimal RDD: map/reduce/count over a Python list, counting count() calls."""

    def __init__(self, rows, counter=None):
        self._rows = list(rows)
        self.counter = counter if counter is not None else {"count": 0}

    def map(self, fn):
        return FakeRDD([fn(r) for r in self._rows], self.counter)

    def reduce(self, fn):
        return functools.reduce(fn, self._rows)

    def count(self):
        self.counter["count"] += 1
        return len(self._rows)

    def unpersist(self):
        pass


def _rows(n=200, d=4, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    y = (X[:, 0] + 0.5 * rng.normal(size=n) > 0).astype(float)
    return [(X[i], float(y[i])) for i in range(n)]


def _reference_loss(rows, w):
    """Mean cross-entropy, written independently of the module under test."""
    eps = 1e-15
    total = 0.0
    for x, y in rows:
        p = 1.0 / (1.0 + np.exp(-float(np.dot(x, w))))
        p = float(np.clip(p, eps, 1.0 - eps))
        total += -(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))
    return total / len(rows)


# ---------------------------------------------------------------------------
# compute_local_loss
# ---------------------------------------------------------------------------


def test_compute_local_loss_matches_reference_and_skips_count_when_n_given():
    rows = _rows()
    rdd = FakeRDD(rows)
    w = np.array([0.3, -0.1, 0.05, 0.2])

    loss = ar.compute_local_loss(rdd, w, n_local=len(rows))

    assert loss == pytest.approx(_reference_loss(rows, w), rel=1e-12)
    assert rdd.counter["count"] == 0  # no extra Spark pass for the row count


def test_compute_local_loss_counts_when_n_not_given():
    rows = _rows(50)
    rdd = FakeRDD(rows)
    w = np.zeros(4)

    loss = ar.compute_local_loss(rdd, w)

    assert loss == pytest.approx(np.log(2.0), rel=1e-12)  # p = 0.5 for w = 0
    assert rdd.counter["count"] == 1


# ---------------------------------------------------------------------------
# allreduce_loss_and_check / check_loss_convergence
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("size", [1, 3])
def test_allreduce_loss_and_check_averages_and_decides_convergence(fake_mpi, size):
    comm = FakeComm(size)

    converged, global_loss = ar.allreduce_loss_and_check(
        comm, rank=0, size=size, local_loss=0.65, prev_loss=0.65005, tol=1e-4, epoch=3
    )
    assert global_loss == pytest.approx(0.65)
    assert converged is True

    converged, _ = ar.allreduce_loss_and_check(
        comm, rank=0, size=size, local_loss=0.65, prev_loss=0.66, tol=1e-4, epoch=4
    )
    assert converged is False


def test_check_loss_convergence_wrapper_is_unchanged(fake_mpi):
    """The kept wrapper returns what the split functions return."""
    rows = _rows()
    w = np.array([0.1, 0.2, -0.3, 0.0])

    old = ar.check_loss_convergence(FakeComm(), 0, 1, FakeRDD(rows), w, 1.0, 1e-4, 0)
    new = ar.allreduce_loss_and_check(
        FakeComm(), 0, 1, ar.compute_local_loss(FakeRDD(rows), w), 1.0, 1e-4, 0
    )

    assert old == new
    assert old[1] == pytest.approx(_reference_loss(rows, w), rel=1e-12)


# ---------------------------------------------------------------------------
# run_logreg_allreduce: the loss pass lands in loss_time_s, not sync_time_s
# ---------------------------------------------------------------------------

LOSS_PASS_S = 0.05


@pytest.fixture
def stubbed_runner(monkeypatch, fake_mpi):
    rows = _rows(300, d=3, seed=1)
    counter = {"count": 0}
    rdd = FakeRDD(rows, counter)

    class FakeSpark:
        def stop(self):
            pass

    monkeypatch.setattr(
        lp, "partition_and_init_spark", lambda **_: ("/unused", FakeSpark(), 3), raising=True
    )
    monkeypatch.setattr(lg, "load_and_cache_rdd", lambda *a, **k: rdd, raising=True)
    monkeypatch.setattr(lg, "cores_per_worker", lambda size: 1, raising=True)

    real_loss = ar.compute_local_loss

    def slow_loss(data_rdd, w, n_local=None):
        time.sleep(LOSS_PASS_S)  # stands in for a full Spark pass
        return real_loss(data_rdd, w, n_local)

    monkeypatch.setattr(ar, "compute_local_loss", slow_loss)
    return rows, counter


def test_loss_pass_is_timed_separately_from_sync(stubbed_runner, tmp_path):
    rows, counter = stubbed_runner
    comm = FakeComm(size=1)

    result = ar.run_logreg_allreduce(
        comm=comm,
        rank=0,
        size=1,
        input_file="/unused.csv",
        max_epochs=3,
        learning_rate=0.5,
        tol=0.0,  # never converge: run all epochs
        metrics_output_dir=str(tmp_path),
    )

    assert result["run_summary"]["epochs_run"] == 3
    with open(tmp_path / "logreg_rank0_epochs.csv", newline="") as fh:
        epochs = list(csv.DictReader(fh))

    assert len(epochs) == 3
    assert list(epochs[0].keys()) == LogRegMetricsCollector.EPOCH_FIELDS
    for row in epochs:
        loss_t, sync_t = float(row["loss_time_s"]), float(row["sync_time_s"])
        spark_t, epoch_t = float(row["spark_time_s"]), float(row["epoch_time_s"])
        assert loss_t >= LOSS_PASS_S
        assert sync_t < LOSS_PASS_S  # the loss pass is no longer inside sync
        assert spark_t + loss_t + sync_t <= epoch_t + 1e-6

    # two Allreduce calls per epoch (gradient, loss); the loss pass reuses the
    # gradient pass's row count: one count() per epoch from
    # compute_gradient_spark plus one from record_run, none from the loss pass
    assert comm.allreduce_calls == 2 * 3
    assert counter["count"] == 3 + 1


def test_record_epoch_loss_time_defaults_to_zero(tmp_path):
    """Existing callers that do not pass loss_time_s keep working."""
    c = LogRegMetricsCollector(rank=0, output_dir=str(tmp_path))
    c.record_epoch(
        epoch=1,
        spark_time_s=0.1,
        sync_time_s=0.01,
        epoch_time_s=0.2,
        grad_norm=0.5,
        global_loss=0.69,
        weight_norm=0.1,
    )
    assert c._epochs[0]["loss_time_s"] == 0.0
    assert LogRegMetricsCollector.EPOCH_FIELDS[-1] == "loss_time_s"
