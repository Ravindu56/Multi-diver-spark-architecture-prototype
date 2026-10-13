"""run_root_mpi must exit non-zero when any worker reports an error.

Before: it printed 'Aborting.' and returned None, so mpirun exited 0 and scripts
could record a failed run as a pass.
"""

from __future__ import annotations

import pytest

from mpj_spark.core import root_mpi


class _FakeLogger:
    def __init__(self, worker_id=None):
        pass

    def log_run(self, **kwargs):
        pass


class _FakeComm:
    def __init__(self, num_workers):
        self.num_workers = num_workers
        self.sent = []

    def Get_rank(self):
        return 0

    def Get_size(self):
        return self.num_workers + 1

    def send(self, obj, dest=0, tag=0):
        self.sent.append((dest, tag))

    def recv(self, source=0, tag=0):
        if tag == root_mpi.TAG_READY:
            return "ready"
        if tag == root_mpi.TAG_RESULT:
            return {
                "worker_id": source - 1,
                "status": "error",
                "error": "boom: not wired",
                "result": None,
            }
        return {"worker_id": source - 1, "init_time": 0.0, "load_time": 0.0, "processing_time": 0.0}

    def Split(self, color, key):
        return None


def test_worker_failure_makes_root_exit_with_status_1(monkeypatch, capsys):
    monkeypatch.setattr(
        "mpj_spark.core.root_process.dynamic_partition",
        lambda input_file, n, data_dir: [f"part_{i}.csv" for i in range(n)],
    )
    monkeypatch.setattr("mpj_spark.utils.dev_logger.DevLogger", _FakeLogger)

    comm = _FakeComm(num_workers=2)
    with pytest.raises(SystemExit) as excinfo:
        root_mpi.run_root_mpi(
            comm,
            input_file="x.csv",
            num_workers=2,
            app="kmeans",
            sync_mode="ps_sync_fedavg_mpi",
            cores_override=1,
        )

    assert excinfo.value.code == 1
    out = capsys.readouterr().out
    assert "2 worker(s) failed. Aborting." in out
    assert "boom: not wired" in out
