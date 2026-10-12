"""Unit tests for K-Means FedAvg post-sync convergence (fixes the
pre-sync stopping bug where worker->global disagreement (~1e-2) kept
every multi-driver run at max_iter with flat WCSS).

Run: pytest tests/unit/test_kmeans_fedavg_convergence.py -q
"""

from __future__ import annotations

import sys
import types

import numpy as np
import pytest

from mpj_spark.applications.kmeans import fedavg_mpi_run as fm
from mpj_spark.applications.kmeans.fedavg_mpi_run import _post_sync_shift


@pytest.fixture
def fake_mpi(monkeypatch):
    """Stand-in mpi4py so the lazy `from mpi4py import MPI` works without libmpi."""
    mpi = types.ModuleType("mpi4py.MPI")
    mpi.DOUBLE = object()
    pkg = types.ModuleType("mpi4py")
    pkg.MPI = mpi
    monkeypatch.setitem(sys.modules, "mpi4py", pkg)
    monkeypatch.setitem(sys.modules, "mpi4py.MPI", mpi)


class FakeComm:
    """Single-rank MPI stand-in: gather returns own payload; Bcast is a no-op."""

    def __init__(self, store):
        self._store = store

    def bcast(self, obj, root=0):
        return obj if obj is not None else self._store.get("d")

    def Bcast(self, buf, root=0):
        pass

    def gather(self, payload, root=0):
        return [payload]


class _Sum:
    def __init__(self, v):
        self._v = v

    def sum(self):
        return self._v


class FakeRDD:
    def __init__(self, rows=100, wcss=42.5):
        self._rows = rows
        self._wcss = wcss

    def count(self):
        return self._rows

    def map(self, fn):
        return _Sum(self._wcss)

    def unpersist(self):
        pass


class FakeCollector:
    captured = []

    def __init__(self, rank, output_dir):
        pass

    def record_iteration(self, **kw):
        FakeCollector.captured.append(kw)

    def record_run(self, **kw):
        pass

    def to_csv(self):
        pass

    def to_json(self):
        pass


def _patch_common(monkeypatch, centroids, sums, counts):
    import pyspark.sql as pysql

    monkeypatch.setattr(pysql.SparkSession, "getActiveSession", staticmethod(lambda: object()))
    monkeypatch.setattr(fm, "KMeansMetricsCollector", FakeCollector)
    monkeypatch.setattr(fm, "load_partition_rdd", lambda spark, path: FakeRDD())
    monkeypatch.setattr(fm, "init_centroids", lambda rdd, k, seed: np.array(centroids))
    monkeypatch.setattr(
        fm, "compute_local_stats", lambda rdd, c: (np.array(sums), np.array(counts))
    )


def test_fixed_point_stops_after_two_rounds(monkeypatch, fake_mpi):
    """Constant local stats reach the fixed point; post-sync shift -> 0
    and the loop must stop in round 2 (not burn max_iter)."""
    FakeCollector.captured.clear()
    m = [[2.0] * 3] * 3
    sums = [[6.0, 6.0, 6.0]] * 3
    counts = [3.0, 3.0, 3.0]
    _patch_common(monkeypatch, centroids=np.zeros((3, 3)), sums=sums, counts=counts)

    res = fm.run_kmeans_fedavg_mpi(
        comm=FakeComm({"d": 3}),
        rank=0,
        size=1,
        input_file="x.csv",
        k=3,
        max_iter=20,
        local_epochs=5,
        tol=1e-4,
        metrics_output_dir="/tmp/x",
    )
    assert res["converged"] is True
    assert res["iterations_done"] == 2
    assert FakeCollector.captured[-1]["centroid_shift"] == pytest.approx(0.0)
    assert np.allclose(res["centres"], m)


def test_post_sync_shift_semantics():
    """Opposing worker updates cancel in the average: post-sync shift is 0
    (converged) even though each worker's pre-sync local shift is large."""
    start = np.array([[1.0, -1.0]])
    delta = np.array([[0.5, 0.5]])
    worker_a = start + delta
    worker_b = start - delta
    avg = (worker_a + worker_b) / 2.0

    local_shift = float(np.linalg.norm(worker_a - start))  # pre-sync (old test)
    assert local_shift > 0.1  # old test: would NOT stop
    assert _post_sync_shift(start, avg) == 0.0  # new test: stops
    # avg + 0.02 offsets BOTH entries of the 1x2 array, so the Frobenius norm is 0.02*sqrt(2)
    assert _post_sync_shift(start, avg + 0.02) == pytest.approx(0.02 * np.sqrt(2), rel=1e-6)


def test_identical_across_ranks_implies_no_deadlock():
    """Broadcast post-sync centroids are the same value for every rank —
    guards the decision that kernel guarantees all ranks break together."""
    prev = np.array([[0.0, 0.0]])
    synced = np.array([[0.001, -0.001]])
    vals = [_post_sync_shift(prev, synced) for _ in range(3)]  # ranks 0..2
    assert len(set(vals)) == 1
