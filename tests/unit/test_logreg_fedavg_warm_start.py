"""End-to-end wiring tests for LogReg FedAvg with the warm-start solver.

Spark is replaced by in-memory fakes (FakeRDD / FakeDF) and MPI by a threaded
2-rank bus, so these run in CI without a JVM or libmpi.  They exercise the real
``run()`` loop: warm start, FedAvg averaging, metrics, accuracy.
"""
from __future__ import annotations

import csv
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from mpj_spark.applications.logreg import fedavg_mpi_run as fm
from mpj_spark.core.warm_start_lr import numpy_loss_grad, spark_loss_grad, warm_lbfgs

NF = 5


class FakeRDD:
    def __init__(self, partitions):
        self.partitions = partitions

    def map(self, fn):
        return FakeRDD([[fn(x) for x in p] for p in self.partitions])

    def mapPartitions(self, fn):
        return FakeRDD([list(fn(iter(p))) for p in self.partitions])

    def fold(self, zero, op):
        acc = zero
        for p in self.partitions:
            part = zero
            for x in p:
                part = op(part, x)
            acc = op(acc, part)
        return acc

    def sum(self):
        return sum(x for p in self.partitions for x in p)

    def count(self):
        return sum(len(p) for p in self.partitions)

    def cache(self):
        return self

    def unpersist(self):
        return self


class FakeDF:
    def __init__(self, rows):
        self.rows = rows

    def filter(self, *_):
        return self

    def dropna(self):
        return self

    def withColumn(self, *_):
        return self

    def cache(self):
        return self

    def unpersist(self):
        return self

    def count(self):
        return len(self.rows)

    def select(self, *_):
        return self

    @property
    def rdd(self):
        half = max(1, len(self.rows) // 2)
        return FakeRDD([self.rows[:half], [], self.rows[half:]])  # includes an empty partition


class FakeCol:
    def isNotNull(self):
        return self

    def cast(self, _):
        return self


def _make_data(n=3000, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, NF))
    w_true = rng.normal(size=NF)
    y = (rng.random(n) < 1 / (1 + np.exp(-(X @ w_true + 0.3)))).astype(float)
    return X, y


class Bus:
    """Threaded gather/bcast for N ranks."""

    def __init__(self, n):
        self.n = n
        self.barrier = threading.Barrier(n)
        self.slots = [None] * n
        self.shared = None


class FakeComm:
    def __init__(self, bus, rank):
        self.bus, self.rank = bus, rank

    def gather(self, obj, root=0):
        self.bus.slots[self.rank] = obj
        self.bus.barrier.wait()
        out = list(self.bus.slots) if self.rank == root else None
        self.bus.barrier.wait()
        return out

    def bcast(self, obj, root=0):
        if self.rank == root:
            self.bus.shared = obj
        self.bus.barrier.wait()
        out = self.bus.shared
        self.bus.barrier.wait()
        return out


@pytest.fixture
def patched(monkeypatch):
    X, y = _make_data()
    halves = {"p0": (X[:1500], y[:1500]), "p1": (X[1500:], y[1500:])}

    def read_csv(path, **_):
        Xp, yp = halves[path]
        return FakeDF([tuple(list(map(float, r)) + [float(l)]) for r, l in zip(Xp, yp, strict=True)])

    fake_spark = SimpleNamespace(read=SimpleNamespace(csv=read_csv))
    monkeypatch.setattr(fm.SparkSession, "getActiveSession", staticmethod(lambda: fake_spark))
    monkeypatch.setattr(fm, "F", SimpleNamespace(col=lambda _n: FakeCol()))
    monkeypatch.setattr(fm, "IntegerType", lambda: None)
    return X, y


def _run_workers(paths, tmp_path, **kw):
    n = len(paths)
    bus = Bus(n)
    results, errors = [None] * n, []

    def work(r):
        try:
            results[r] = fm.run(
                partition_path=paths[r], comm=FakeComm(bus, r), rank=r, num_workers=n,
                num_features=NF, results_dir=str(tmp_path), **kw,
            )
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)
            bus.barrier.abort()

    ts = [threading.Thread(target=work, args=(r,)) for r in range(n)]
    for t in ts:
        t.start()
    for t in ts:
        t.join(timeout=60)
    assert not errors, errors
    return results


def test_weights_keep_moving_after_round_one_and_loss_falls(patched, tmp_path):
    X, y = patched
    res = _run_workers(["p0", "p1"], tmp_path, max_iter=8, local_epochs=3, reg_param=0.01)
    m = res[0]["iter_metrics"]
    assert m[1]["weight_delta"] > 1e-6                                   # the old bug: exactly 0
    gl = [r["global_loss"] for r in m]
    # FedAvg is not a descent method: allow tiny upticks from averaging, require real progress
    assert all(b <= a + 1e-4 for a, b in zip(gl, gl[1:], strict=False))
    assert gl[-1] < gl[0] - 0.05
    ref = warm_lbfgs(numpy_loss_grad(X, y, 0.01), None, 0.0, NF, 500)[2]["loss_end"]
    final = numpy_loss_grad(X, y, 0.01)(
        np.concatenate([res[0]["weight_vector"], [res[0]["intercept"]]])
    )[0]
    assert final - ref < 5e-3                                            # near centralised optimum


def test_all_ranks_agree_on_global_model(patched, tmp_path):
    res = _run_workers(["p0", "p1"], tmp_path, max_iter=4, local_epochs=3)
    assert np.allclose(res[0]["weight_vector"], res[1]["weight_vector"])
    assert res[0]["intercept"] == res[1]["intercept"]
    assert 0.5 < res[0]["train_accuracy"] <= 1.0


def test_metrics_csv_has_loss_columns(patched, tmp_path):
    _run_workers(["p0", "p1"], tmp_path, max_iter=3, local_epochs=2)
    with open(tmp_path / "worker_0_fedavg_mpi_metrics.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 3
    for col in ("global_loss", "loss_start", "loss_end", "n_eval", "solver", "weight_delta"):
        assert col in rows[0]
    assert rows[0]["solver"] == "warm_lbfgs"


def test_spark_loss_grad_matches_numpy_incl_empty_partition():
    X, y = _make_data(600)
    rows = [(X[i], float(y[i])) for i in range(len(y))]
    rdd = FakeRDD([rows[:250], [], rows[250:]])
    th = np.random.default_rng(5).normal(size=NF + 1)
    f1, g1 = spark_loss_grad(rdd, NF, 0.01, len(rows))(th)
    f2, g2 = numpy_loss_grad(X, y, 0.01)(th)
    assert f1 == pytest.approx(f2)
    assert np.allclose(g1, g2)


def test_correct_count_and_weighted_mean():
    X = np.array([[1.0, 0.0], [-1.0, 0.0], [2.0, 0.0]])
    y = [1.0, 0.0, 0.0]
    rdd = FakeRDD([[(X[0], y[0])], [], [(X[1], y[1]), (X[2], y[2])]])
    assert fm._correct_count(rdd, [1.0, 0.0], 0.0) == 2
    assert fm._weighted_mean([1.0, 3.0], [1, 3]) == pytest.approx(2.5)
    assert fm._weighted_mean([2.0, 4.0], [0, 0]) == pytest.approx(3.0)
