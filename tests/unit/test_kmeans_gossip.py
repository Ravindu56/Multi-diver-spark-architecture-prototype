"""Gossip K-Means: two threaded workers on disjoint halves reach the single-machine optimum."""

from __future__ import annotations

import threading
from queue import Queue

import numpy as np
import pytest

from mpj_spark.applications.kmeans import gossip_run
from mpj_spark.applications.kmeans.gossip_run import mix_centroids, run_kmeans_gossip


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


class _Net:
    def __init__(self, n):
        self.n = n
        self.boxes = {(s, d): Queue() for s in range(n) for d in range(n)}
        self.bc = [Queue() for _ in range(n)]


class _FakeComm:
    def __init__(self, net, rank):
        self.net, self.rank = net, rank

    def bcast(self, obj, root=0):
        if self.rank == root:
            for r in range(self.net.n):
                if r != root:
                    self.net.bc[r].put(obj)
            return obj
        return self.net.bc[self.rank].get(timeout=10)

    def sendrecv(self, payload, dest, sendtag, source, recvtag):
        self.net.boxes[(self.rank, dest)].put(payload)
        return self.net.boxes[(source, self.rank)].get(timeout=10)


def _data():
    rng = np.random.default_rng(7)
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


def _run_two(monkeypatch, tmp_path, init, halves, **kw):
    monkeypatch.setattr(gossip_run, "init_centroids", lambda rdd, k, seed: np.asarray(init))
    net = _Net(2)
    results, errors = {}, []

    def _work(r):
        try:
            results[r] = run_kmeans_gossip(
                _FakeComm(net, r),
                r,
                2,
                "unused",
                k=3,
                max_iter=kw.get("max_iter", 8),
                local_epochs=kw.get("local_epochs", 3),
                metrics_output_dir=str(tmp_path),
                points_rdd=_FakeRDD(halves[r]),
                stats_fn=_stats,
                tol=kw.get("tol", 0.0),
            )
        except Exception as exc:  # pragma: no cover - surfaced below
            errors.append(exc)

    threads = [threading.Thread(target=_work, args=(r,)) for r in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)
    assert not errors, errors
    return results


def test_two_workers_match_single_machine_wcss(monkeypatch, tmp_path):
    pts = _data()
    halves = [pts[0::2], pts[1::2]]
    init = pts[[0, 1, 2]]
    res = _run_two(monkeypatch, tmp_path, init, halves)

    ref = _wcss(pts, _lloyd_reference(pts, init))
    got = _wcss(pts, res[0]["centres"])
    assert got == pytest.approx(ref, rel=0.02)
    assert np.allclose(res[0]["centres"], res[1]["centres"], atol=1e-9)
    assert res[0]["iterations_done"] == 8
    assert res[0]["sync_mode"] == "gossip"
    assert res[0]["row_count"] + res[1]["row_count"] == len(pts)


def test_mix_aligns_permuted_neighbour_before_averaging():
    own = np.array([[0.0, 0.0], [10.0, 10.0]])
    peer = {"centroids": [[10.2, 10.2], [0.2, 0.2]], "row_count": 50}
    mixed = mix_centroids(own, 50, [peer])
    assert np.allclose(mixed, [[0.1, 0.1], [10.1, 10.1]])


def test_mix_weights_by_row_count():
    own = np.array([[0.0, 0.0], [10.0, 10.0]])
    peer = {"centroids": [[1.0, 1.0], [11.0, 11.0]], "row_count": 300}
    mixed = mix_centroids(own, 100, [peer])
    assert np.allclose(mixed, [[0.75, 0.75], [10.75, 10.75]])


def test_tol_freezes_refit_but_keeps_exchanging(monkeypatch, tmp_path):
    pts = _data()
    halves = [pts[0::2], pts[1::2]]
    res = _run_two(monkeypatch, tmp_path, pts[[0, 1, 2]], halves, max_iter=10, tol=1e-3)
    assert all(r["converged"] for r in res.values())
    assert all(r["iterations_done"] == 10 for r in res.values())


def test_requires_sub_communicator():
    with pytest.raises(RuntimeError, match="sub-communicator"):
        run_kmeans_gossip(None, 0, 2, "x.csv")
