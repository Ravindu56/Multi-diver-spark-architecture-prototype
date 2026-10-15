"""run_root_mpi must exit non-zero when any worker reports an error.

Before: it printed 'Aborting.' and returned None, so mpirun exited 0 and scripts
could record a failed run as a pass.
"""

from __future__ import annotations

import importlib
import sys
from types import SimpleNamespace

import pytest

import mpj_spark.core as core

_MODULE = "mpj_spark.core.root_mpi"
_MISSING = object()


@pytest.fixture
def root_mpi(monkeypatch):
    """Import root_mpi against a fake mpi4py.

    root_mpi imports mpi4py at module level.  CI runners have the mpi4py wheel but no
    libmpi, so a plain import fails at collection.  Nothing else in the unit suite imports
    root_mpi, so it is removed again afterwards.
    """
    orig_mod = sys.modules.pop(_MODULE, None)
    orig_attr = getattr(core, "root_mpi", _MISSING)
    monkeypatch.setitem(
        sys.modules, "mpi4py", SimpleNamespace(MPI=SimpleNamespace(UNDEFINED=-32766))
    )
    module = importlib.import_module(_MODULE)
    yield module
    sys.modules.pop(_MODULE, None)
    if orig_mod is not None:
        sys.modules[_MODULE] = orig_mod
    if orig_attr is _MISSING:
        if hasattr(core, "root_mpi"):
            delattr(core, "root_mpi")
    else:
        core.root_mpi = orig_attr


class _FakeLogger:
    def __init__(self, worker_id=None):
        pass

    def log_run(self, **kwargs):
        pass


class _FakeComm:
    def __init__(self, module, num_workers):
        self.module = module
        self.num_workers = num_workers
        self.sent = []

    def Get_rank(self):
        return 0

    def Get_size(self):
        return self.num_workers + 1

    def send(self, obj, dest=0, tag=0):
        self.sent.append((dest, tag))

    def recv(self, source=0, tag=0):
        if tag == self.module.TAG_READY:
            return "ready"
        if tag == self.module.TAG_RESULT:
            return {
                "worker_id": source - 1,
                "status": "error",
                "error": "boom: not wired",
                "result": None,
            }
        return {"worker_id": source - 1, "init_time": 0.0, "load_time": 0.0, "processing_time": 0.0}

    def Split(self, color, key):
        return None


def test_worker_failure_makes_root_exit_with_status_1(root_mpi, monkeypatch, capsys):
    monkeypatch.setattr(
        "mpj_spark.core.root_process.dynamic_partition",
        lambda input_file, n, data_dir: [f"part_{i}.csv" for i in range(n)],
    )
    monkeypatch.setattr("mpj_spark.utils.dev_logger.DevLogger", _FakeLogger)

    comm = _FakeComm(root_mpi, num_workers=2)
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
