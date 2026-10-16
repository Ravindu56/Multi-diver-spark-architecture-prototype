"""K-Means FedAvg must return a per-partition ``wcss``.

The root's aggregate_kmeans_results() sums ``wcss`` across workers.  FedAvg used to return
the already-global sum, so the reported Total WCSS was multiplied by the worker count.
"""

from __future__ import annotations

import sys
import threading
from queue import Queue
from types import SimpleNamespace

import numpy as np
import pytest

from mpj_spark.applications.kmeans import fedavg_mpi_run as mod


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
        self.up = Queue()
        self.down = [Queue() for _ in range(n)]
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

    def Bcast(self, buf, root=0):
        arr = buf[0]
        if self.rank == root:
            for r in range(self.net.n):
                if r != root:
                    self.net.down[r].put(arr.copy())
        else:
            arr[:] = self.net.down[self.rank].get(timeout=10)

    def gather(self, obj, root=0):
        if self.rank != root:
            self.net.up.put((self.rank, obj))
            return None
        items = {root: obj}
        for _ in range(self.net.n - 1):
            r, o = self.net.up.get(timeout=10)
            items[r] = o
        return [items[r] for r in range(self.net.n)]


def _data():
    rng = np.random.default_rng(3)
    blobs = [rng.normal(c, 0.4, (150, 2)) for c in ([0, 0], [6, 0], [3, 5])]
    pts = np.vstack(blobs)
    rng.shuffle(pts)
    return pts


def _global_wcss(pts, cents):
    d = ((pts[:, None, :] - np.asarray(cents)[None]) ** 2).sum(-1)
    return float(d.min(axis=1).sum())


def test_returned_wcss_is_per_partition_so_root_sum_is_the_global_wcss(monkeypatch, tmp_path):
    pts = _data()
    halves = {"h0": pts[0::2], "h1": pts[1::2]}
    monkeypatch.setitem(sys.modules, "mpi4py", SimpleNamespace(MPI=SimpleNamespace(DOUBLE="d")))
    monkeypatch.setattr(mod, "load_partition_rdd", lambda spark, path: _FakeRDD(halves[path]))
    monkeypatch.setattr(mod, "init_centroids", lambda rdd, k, seed: pts[:k].copy())
    monkeypatch.setattr(mod, "compute_local_stats", _stats)
    monkeypatch.setattr(
        "pyspark.sql.SparkSession.getActiveSession", classmethod(lambda cls: object())
    )

    net = _Net(2)
    results, errors = {}, []

    def _work(r):
        try:
            results[r] = mod.run_kmeans_fedavg_mpi(
                _FakeComm(net, r),
                r,
                2,
                f"h{r}",
                k=3,
                max_iter=4,
                local_epochs=2,
                tol=0.0,
                metrics_output_dir=str(tmp_path),
            )
        except Exception as exc:  # pragma: no cover - surfaced below
            errors.append(exc)

    threads = [threading.Thread(target=_work, args=(r,)) for r in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)
    assert not errors, errors

    centres = results[0]["centres"]
    assert np.allclose(centres, results[1]["centres"])
    root_total = sum(r["wcss"] for r in results.values())
    assert root_total == pytest.approx(_global_wcss(pts, centres), rel=1e-9)
    assert results[0]["global_wcss"] == results[1]["global_wcss"]
    assert results[0]["wcss"] != pytest.approx(root_total)
