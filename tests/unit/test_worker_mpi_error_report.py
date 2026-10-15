"""A worker that fails inside run_worker_core must still report to the root.

Without the catch, the worker dies before sending TAG_RESULT / TAG_TIMING and the
root blocks in comm.recv() forever (observed: K-Means gossip guard -> exit 124).
"""

from __future__ import annotations

from mpj_spark.workers import worker_mpi


class _FakeSpark:
    def stop(self):
        pass


class _FakeLogger:
    def __init__(self, worker_id=None):
        pass

    def log_worker_timing(self, **kwargs):
        pass


class _FakeComm:
    def __init__(self):
        self.sent = []

    def Get_rank(self):
        return 1

    def recv(self, source=0, tag=0):
        if tag == worker_mpi.TAG_CONFIG:
            return {
                "partition_path": "x.csv",
                "app": "kmeans",
                "num_workers": 1,
                "sync_mode": "gossip",
            }
        return True

    def send(self, obj, dest=0, tag=0):
        self.sent.append((tag, obj))

    def Split(self, color, key):
        return object()


def _patch(monkeypatch, core):
    monkeypatch.setattr(worker_mpi, "build_spark_session", lambda **kw: _FakeSpark())
    monkeypatch.setattr(worker_mpi, "DevLogger", _FakeLogger)
    monkeypatch.setattr(worker_mpi, "run_worker_core", core)


def test_failing_worker_still_sends_result_and_timing(monkeypatch):
    def _boom(**kwargs):
        raise RuntimeError("boom: not wired")

    _patch(monkeypatch, _boom)
    comm = _FakeComm()
    worker_mpi.run_worker_mpi(comm)

    by_tag = dict(comm.sent)
    assert worker_mpi.TAG_READY in by_tag
    result = by_tag[worker_mpi.TAG_RESULT]
    assert result["status"] == "error"
    assert "boom: not wired" in result["error"]
    assert result["result"] is None
    timing = by_tag[worker_mpi.TAG_TIMING]
    assert timing["processing_time"] >= 0.0
    assert timing["init_time"] >= 0.0


def test_succeeding_worker_still_reports_success(monkeypatch):
    def _ok(**kwargs):
        return {
            "status": "success",
            "result": {"x": 1},
            "timing": {"worker_id": 0, "load_time": 0.0, "processing_time": 1.5},
        }

    _patch(monkeypatch, _ok)
    comm = _FakeComm()
    worker_mpi.run_worker_mpi(comm)

    by_tag = dict(comm.sent)
    assert by_tag[worker_mpi.TAG_RESULT]["status"] == "success"
    assert by_tag[worker_mpi.TAG_RESULT]["result"] == {"x": 1}
    assert by_tag[worker_mpi.TAG_TIMING]["processing_time"] == 1.5
