"""K-Means hybrid and async-PS modes, driven against the REAL root-side server loops
(core/hybrid_ps._serve and core/async_ps._serve) with an in-process fake MPI world."""

from __future__ import annotations

import threading
import time
from queue import Empty, Queue

import numpy as np
import pytest

from mpj_spark.applications.kmeans import async_ps_run, hybrid_run
from mpj_spark.applications.kmeans.async_ps_run import run_kmeans_async_ps
from mpj_spark.applications.kmeans.hybrid_run import run_kmeans_hybrid
from mpj_spark.core import async_ps, hybrid_ps

ANY = object()


class _Mapped:
    def __init__(self, vals):
        self.vals = vals

    def sum(self):
        return sum(self.vals)


class _FakeRDD:
    def __init__(self, pts):
        self.pts = [np.asarray(p, dtype=np.float64) for p in pts]

    def count(self):
        return len(self.pts)

    def map(self, f):
        return _Mapped([f(p) for p in self.pts])


def _stats(rdd, cents):
    k, d = cents.shape
    sums, counts = np.zeros((k, d)), np.zeros(k)
    for p in rdd.pts:
        j = int(np.argmin(((cents - p) ** 2).sum(axis=1)))
        sums[j] += p
        counts[j] += 1
    return sums, counts


class _World:
    def __init__(self, n):
        self.n = n
        self.up = [Queue() for _ in range(n + 1)]
        self.down = [Queue() for _ in range(n + 1)]
        self.bc = [Queue() for _ in range(n)]
        self.slots = [None] * n
        self.barrier = threading.Barrier(n)


class _WorkerComm:
    """Worker sub-communicator (ranks 0..n-1)."""

    def __init__(self, world, rank):
        self.w, self.rank = world, rank

    def bcast(self, obj, root=0):
        if self.rank == root:
            for r in range(self.w.n):
                if r != root:
                    self.w.bc[r].put(obj)
            return obj
        return self.w.bc[self.rank].get(timeout=10)

    def allgather(self, x):
        self.w.slots[self.rank] = x
        self.w.barrier.wait(timeout=10)
        out = list(self.w.slots)
        self.w.barrier.wait(timeout=10)
        return out

    def Allreduce(self, send, recv, op=None):
        self.w.slots[self.rank] = send.copy()
        self.w.barrier.wait(timeout=10)
        recv[:] = sum(self.w.slots)
        self.w.barrier.wait(timeout=10)


class _WorkerRoot:
    """COMM_WORLD as seen by worker world rank `wr` (root is rank 0)."""

    def __init__(self, world, wr):
        self.w, self.wr = world, wr

    def send(self, msg, dest, tag):
        self.w.up[self.wr].put(msg)

    def recv(self, source, tag):
        return self.w.down[self.wr].get(timeout=10)


class _RootSide:
    """COMM_WORLD as seen by the root parameter-server thread."""

    def __init__(self, world):
        self.w = world

    def Iprobe(self, source, tag):
        return any(not q.empty() for q in self.w.up[1:])

    def recv(self, source, tag):
        if source is ANY:
            for q in self.w.up[1:]:
                try:
                    return q.get_nowait()
                except Empty:
                    continue
            raise AssertionError("recv(ANY) without a pending message")
        return self.w.up[source].get(timeout=10)

    def send(self, msg, dest, tag):
        self.w.down[dest].put(msg)


def _data():
    rng = np.random.default_rng(11)
    blobs = [rng.normal(c, 0.4, (200, 2)) for c in ([0, 0], [6, 0], [3, 5])]
    pts = np.vstack(blobs)
    rng.shuffle(pts)
    return pts


def _wcss(pts, cents):
    d = ((pts[:, None, :] - np.asarray(cents)[None]) ** 2).sum(-1)
    return float(d.min(axis=1).sum())


def _lloyd_reference(pts, init, iters=40):
    c = np.array(init, dtype=float)
    for _ in range(iters):
        a = ((pts[:, None, :] - c[None]) ** 2).sum(-1).argmin(1)
        for j in range(len(c)):
            if (a == j).any():
                c[j] = pts[a == j].mean(0)
    return c


def _fedavg_reference(halves, init, rounds, epochs):
    """Row-weighted FedAvg of centroids after `epochs` local Lloyd steps per round."""
    total = sum(len(h) for h in halves)
    c = np.array(init, dtype=float)
    for _ in range(rounds):
        local = []
        for h in halves:
            lc = c.copy()
            for _ in range(epochs):
                a = ((h[:, None, :] - lc[None]) ** 2).sum(-1).argmin(1)
                for j in range(len(lc)):
                    if (a == j).any():
                        lc[j] = h[a == j].mean(0)
            local.append(lc)
        c = sum(len(h) / total * lc for h, lc in zip(halves, local, strict=True))
    return c


def _drive(monkeypatch, tmp_path, module, runner, serve, rounds, uneven=False, **extra):
    pts = _data()
    halves = [pts[:150], pts[150:]] if uneven else [pts[0::2], pts[1::2]]
    init = pts[[0, 1, 2]]
    monkeypatch.setattr(module, "init_centroids", lambda rdd, k, seed: np.asarray(init))
    world = _World(2)
    server, results, errors = {}, {}, []

    def _server():
        try:
            server["res"] = serve(_RootSide(world))
        except Exception as exc:  # pragma: no cover - surfaced below
            errors.append(exc)

    def _worker(r):
        try:
            results[r] = runner(
                _WorkerComm(world, r),
                _WorkerRoot(world, r + 1),
                r,
                r + 1,
                2,
                "unused",
                k=3,
                max_iter=rounds,
                local_epochs=3,
                metrics_output_dir=str(tmp_path),
                points_rdd=_FakeRDD(halves[r]),
                stats_fn=_stats,
                **extra,
            )
        except Exception as exc:  # pragma: no cover - surfaced below
            errors.append(exc)

    threads = [threading.Thread(target=_server)] + [
        threading.Thread(target=_worker, args=(r,)) for r in range(2)
    ]
    t0 = time.time()
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)
    assert not errors, errors
    assert time.time() - t0 < 30
    return pts, init, halves, results, server["res"]


def test_hybrid_equals_row_weighted_fedavg_and_reports_exact_global_wcss(monkeypatch, tmp_path):
    rounds = 6
    last_local = {}
    orig = hybrid_run.local_lloyd

    def _record(rdd, cents, epochs, **kw):
        out = orig(rdd, cents, epochs, **kw)
        last_local[threading.current_thread().name] = out[3]
        return out

    monkeypatch.setattr(hybrid_run, "local_lloyd", _record)
    pts, init, halves, res, srv = _drive(
        monkeypatch,
        tmp_path,
        hybrid_run,
        run_kmeans_hybrid,
        lambda c: hybrid_ps._serve(c, num_workers=2, num_iterations=rounds),
        rounds,
        uneven=True,
        mpi_sum="SUM",
    )
    centres = res[0]["centres"]
    assert np.allclose(centres, res[1]["centres"], atol=1e-9)
    assert np.allclose(centres, _fedavg_reference(halves, init, rounds, 3), atol=1e-9)
    assert sum(r["wcss"] for r in res.values()) == pytest.approx(_wcss(pts, centres))
    assert res[0]["global_wcss"] == pytest.approx(sum(last_local.values()))
    assert res[0]["global_wcss"] == pytest.approx(res[1]["global_wcss"])
    assert srv["iterations_done"] == rounds
    assert res[0]["iterations_done"] == rounds and res[0]["sync_mode"] == "hybrid_ps_allreduce"


def test_scalar_ps_encoding_gives_the_exact_global_wcss():
    world = _World(2)
    rows, wcss = [300, 100], [30.0, 50.0]
    for r in (1, 2):
        world.up[r].put({"intercept": wcss[r - 1] / rows[r - 1], "row_count": rows[r - 1]})
    hybrid_ps._serve(_RootSide(world), num_workers=2, num_iterations=1)
    reply = world.down[1].get_nowait()["intercept"]
    assert reply * sum(rows) == pytest.approx(sum(wcss))


def test_async_ps_trains_through_the_real_server_and_stays_near_optimum(monkeypatch, tmp_path):
    rounds = 6
    pts, init, _halves, res, srv = _drive(
        monkeypatch,
        tmp_path,
        async_ps_run,
        run_kmeans_async_ps,
        lambda c: async_ps._serve(
            c, num_workers=2, num_features=3, num_iterations=rounds, any_source=ANY
        ),
        rounds,
    )
    assert srv["global_version"] == 2 * rounds
    assert np.asarray(srv["weight_vector"]).shape == (3, 2)
    assert np.asarray(res[0]["centres"]).shape == (3, 2)
    ref = _wcss(pts, _lloyd_reference(pts, init))
    assert _wcss(pts, srv["weight_vector"]) == pytest.approx(ref, rel=0.05)
    assert all(r["iterations_done"] == rounds and r["final_version"] > 0 for r in res.values())
    assert res[0]["sync_mode"] == "ps_async"


@pytest.mark.parametrize("runner", [run_kmeans_hybrid, run_kmeans_async_ps])
def test_both_communicators_are_required(runner):
    with pytest.raises(RuntimeError, match="requires both"):
        runner(None, object(), 0, 1, 2, "x.csv")
    with pytest.raises(RuntimeError, match="requires both"):
        runner(object(), None, 0, 1, 2, "x.csv")
