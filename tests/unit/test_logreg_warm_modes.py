"""Wiring tests for the warm-started LogReg gossip / async-PS / hybrid workers.

Spark is replaced by in-memory fakes and MPI by threaded in-process channels
(ring mailboxes, an async PS built on the real ``core.async_ps._serve``, an
Allreduce stand-in plus a scalar PS).  They run the real ``run()`` loops, so a
return to the swallowed-setInitialWeights cold refit would fail them.
"""

from __future__ import annotations

import queue
import sys
import threading
import types
from types import SimpleNamespace

import numpy as np
import pytest
from pyspark.sql import SparkSession

from mpj_spark.applications.logreg import async_ps_run, gossip_run, hybrid_run
from mpj_spark.applications.logreg import warm_common as wc
from mpj_spark.core.async_ps import _serve as async_serve
from mpj_spark.core.warm_start_lr import numpy_loss_grad, warm_lbfgs

from .test_logreg_fedavg_warm_start import NF, FakeCol, FakeDF, _make_data

ROUNDS = 8
EPOCHS = 3
REG = 0.01


def _rows(Xp, yp):
    return [tuple(list(map(float, r)) + [float(lab)]) for r, lab in zip(Xp, yp, strict=True)]


@pytest.fixture
def quarters(monkeypatch):
    X, y = _make_data(4000)
    parts = {f"p{i}": (X[i * 1000 : (i + 1) * 1000], y[i * 1000 : (i + 1) * 1000]) for i in range(4)}
    fake_spark = SimpleNamespace(read=SimpleNamespace(csv=lambda path, **_: FakeDF(_rows(*parts[path]))))
    monkeypatch.setattr(SparkSession, "getActiveSession", staticmethod(lambda: fake_spark))
    monkeypatch.setattr(wc, "F", SimpleNamespace(col=lambda _n: FakeCol()))
    monkeypatch.setattr(wc, "IntegerType", lambda: None)
    return parts


def _union(parts, names):
    X = np.vstack([parts[n][0] for n in names])
    y = np.concatenate([parts[n][1] for n in names])
    return X, y


def _gap(parts, names, w, b):
    """loss(model on the union of the partitions) - loss(centralised optimum)."""
    X, y = _union(parts, names)
    fg = numpy_loss_grad(X, y, REG)
    ref = warm_lbfgs(fg, None, 0.0, NF, 500)[2]["loss_end"]
    return fg(np.concatenate([np.asarray(w, dtype=float), [b]]))[0] - ref


def _run_threads(targets):
    results, errors = [None] * len(targets), []

    def work(i, fn):
        try:
            results[i] = fn()
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    ts = [threading.Thread(target=work, args=(i, fn), daemon=True) for i, fn in enumerate(targets)]
    for t in ts:
        t.start()
    for t in ts:
        t.join(timeout=90)
    assert not errors, errors
    assert all(not t.is_alive() for t in ts), "a worker thread hung"
    return results


def _assert_moves_and_improves(metrics):
    assert metrics[1]["weight_delta"] > 1e-6  # the old bug: exactly 0 after round 1
    first, last = float(metrics[0]["loss_start"]), float(metrics[-1]["loss_start"])
    assert last < first - 0.05  # loss of the model entering the round falls


# ───────────────────────── gossip ─────────────────────────
class _RingComm:
    def __init__(self, boxes, rank):
        self.boxes, self.rank = boxes, rank

    def sendrecv(self, sendobj, dest, sendtag, source, recvtag):
        self.boxes[(self.rank, dest)].put(sendobj)
        return self.boxes[(source, self.rank)].get(timeout=30)


def test_gossip_warm_start_reaches_consensus_near_optimum(quarters, tmp_path):
    n = 3
    boxes = {(a, b): queue.Queue() for a in range(n) for b in range(n)}
    names = [f"p{i}" for i in range(n)]
    res = _run_threads(
        [
            lambda r=r: gossip_run.run(
                names[r], _RingComm(boxes, r), r, n, max_iter=ROUNDS, local_epochs=EPOCHS,
                num_features=NF, reg_param=REG, results_dir=str(tmp_path),
            )
            for r in range(n)
        ]
    )
    _assert_moves_and_improves(res[0]["iter_metrics"])
    assert _gap(quarters, names, res[0]["weight_vector"], res[0]["intercept"]) < 5e-3
    w = np.array([r["weight_vector"] for r in res])
    assert np.abs(w - w.mean(axis=0)).max() < 5e-2  # workers are near consensus
    assert all(np.isfinite(r["final_loss"]) for r in res)
    assert res[0]["iter_metrics"][0]["solver"] == "warm_lbfgs"


# ───────────────────────── async PS ─────────────────────────
class _AsyncNet:
    def __init__(self, n):
        self.up = queue.Queue()
        self.down = {r: queue.Queue() for r in range(1, n + 1)}


class _AsyncRoot:
    def __init__(self, net):
        self.net = net

    def Iprobe(self, source, tag):  # noqa: N802
        return not self.net.up.empty()

    def recv(self, source, tag):
        return self.net.up.get()

    def send(self, msg, dest, tag):
        self.net.down[dest].put(msg)


class _AsyncWorker:
    def __init__(self, net, rank):
        self.net, self.rank = net, rank

    def send(self, msg, dest, tag):
        self.net.up.put(msg)

    def recv(self, source, tag):
        return self.net.down[self.rank].get(timeout=60)


def test_async_ps_warm_start_improves_loss(quarters, tmp_path):
    n = 2
    net = _AsyncNet(n)
    names = ["p0", "p1"]
    targets = [
        lambda: async_serve(
            _AsyncRoot(net), num_workers=n, num_features=NF, num_iterations=ROUNDS,
            any_source="ANY", poll_interval_s=0.0005,
        )
    ] + [
        lambda r=r: async_ps_run.run(
            names[r - 1], _AsyncWorker(net, r), r, n, max_iter=ROUNDS, local_epochs=EPOCHS,
            num_features=NF, reg_param=REG, results_dir=str(tmp_path),
        )
        for r in range(1, n + 1)
    ]
    res = _run_threads(targets)
    served, workers = res[0], res[1:]
    assert served["global_version"] == n * ROUNDS
    _assert_moves_and_improves(workers[0]["iter_metrics"])
    assert _gap(quarters, names, served["weight_vector"], served["intercept"]) < 2e-2
    assert all(np.isfinite(r["final_loss"]) for r in workers)


# ───────────────────────── hybrid PS + Allreduce ─────────────────────────
class _Collective:
    def __init__(self, n):
        self.n = n
        self.barrier = threading.Barrier(n)
        self.slots = [None] * n


class _HybridWorkerComm:
    def __init__(self, coll, rank):
        self.coll, self.rank = coll, rank

    def allgather(self, obj):
        self.coll.slots[self.rank] = obj
        self.coll.barrier.wait()
        out = list(self.coll.slots)
        self.coll.barrier.wait()
        return out

    def Allreduce(self, send, recv, op=None):  # noqa: N802
        self.coll.slots[self.rank] = np.array(send, copy=True)
        self.coll.barrier.wait()
        recv[:] = np.sum(self.coll.slots, axis=0)
        self.coll.barrier.wait()


class _ScalarNet:
    def __init__(self, n):
        self.up = {r: queue.Queue() for r in range(1, n + 1)}
        self.down = {r: queue.Queue() for r in range(1, n + 1)}


class _HybridRootComm:
    def __init__(self, net, world_rank):
        self.net, self.world_rank = net, world_rank

    def send(self, msg, dest, tag):
        self.net.up[self.world_rank].put(msg)

    def recv(self, source, tag):
        return self.net.down[self.world_rank].get(timeout=60)


def _scalar_ps(net, n, rounds):
    """Row-weighted intercept mean per round, replied to every worker (the real PS contract)."""
    for it in range(rounds):
        msgs = [net.up[r].get(timeout=60) for r in range(1, n + 1)]
        rows = sum(m["row_count"] for m in msgs)
        mean = sum(m["intercept"] * m["row_count"] for m in msgs) / rows
        for r in range(1, n + 1):
            net.down[r].put({"intercept": mean, "iteration": it})


def test_hybrid_warm_start_workers_agree_and_improve(quarters, tmp_path, monkeypatch):
    stub = types.ModuleType("mpi4py")
    stub.MPI = SimpleNamespace(SUM="SUM")
    monkeypatch.setitem(sys.modules, "mpi4py", stub)
    n = 2
    coll, net = _Collective(n), _ScalarNet(n)
    names = ["p0", "p1"]
    targets = [lambda: _scalar_ps(net, n, ROUNDS)] + [
        lambda r=r: hybrid_run.run(
            names[r], _HybridWorkerComm(coll, r), r, n, root_comm=_HybridRootComm(net, r + 1),
            max_iter=ROUNDS, local_epochs=EPOCHS, num_features=NF, reg_param=REG,
            results_dir=str(tmp_path),
        )
        for r in range(n)
    ]
    res = _run_threads(targets)[1:]
    _assert_moves_and_improves(res[0]["iter_metrics"])
    assert np.allclose(res[0]["weight_vector"], res[1]["weight_vector"])
    assert _gap(quarters, names, res[0]["weight_vector"], res[0]["intercept"]) < 5e-3


# ───────────────────────── shared helper ─────────────────────────
def test_load_points_rejects_empty_partition():
    spark = SimpleNamespace(read=SimpleNamespace(csv=lambda *a, **k: FakeDF([])))
    import mpj_spark.applications.logreg.warm_common as mod

    saved = (mod.F, mod.IntegerType)
    mod.F, mod.IntegerType = SimpleNamespace(col=lambda _n: FakeCol()), (lambda: None)
    try:
        with pytest.raises(RuntimeError, match="empty"):
            wc.load_points(spark, "x.csv", NF, "Test Worker")
    finally:
        mod.F, mod.IntegerType = saved
