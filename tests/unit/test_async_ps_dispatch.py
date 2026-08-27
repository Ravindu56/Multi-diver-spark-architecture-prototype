"""tests/unit/test_async_ps_dispatch.py
Unit tests for P3-09 dispatch wiring (Issue #61).

CI-safe: no mpi4py runtime, no JVM, no Spark execution.  Verifies that
sync_mode="ps_async" is routed from run_worker_core() to
applications/logreg/async_ps_run.run() over the COMM_WORLD root channel
(root_comm), and that the multiprocessing transport (root_comm=None)
fails fast with a clear error.
"""

import inspect

import pytest

from mpj_spark.core.sync_modes import MODE_PS_ASYNC
from mpj_spark.workers.worker_process import run_worker_core


def _cfg(**overrides):
    cfg = {
        "app": "logreg",
        "sync_mode": MODE_PS_ASYNC,
        "num_workers": 2,
        "results_dir": "results",
        "logreg_iter": 3,
        "logreg_features": 4,
        "logreg_reg_param": 0.01,
    }
    cfg.update(overrides)
    return cfg


class TestWorkerCoreContract:
    def test_run_worker_core_accepts_root_comm(self):
        params = inspect.signature(run_worker_core).parameters
        assert "root_comm" in params
        assert params["root_comm"].default is None


class TestAsyncPsDispatch:
    def test_ps_async_requires_mpi_root_channel(self):
        """root_comm=None (multiprocessing transport) must fail fast."""
        with pytest.raises(RuntimeError, match="ps_async"):
            run_worker_core(
                worker_id=0,
                partition_path="/nonexistent/partition.csv",
                spark=None,
                worker_config=_cfg(),
                root_comm=None,
            )

    def test_ps_async_dispatch_uses_comm_world_and_one_based_rank(self, monkeypatch):
        """async_ps_run.run() must receive COMM_WORLD, not the worker sub-comm,
        and the COMM_WORLD rank (worker_id + 1), not the 0-based worker_id
        used by the collective (gather/bcast) sync modes."""
        import mpj_spark.applications.logreg.async_ps_run as async_ps_run

        calls = {}

        def _fake_run(**kwargs):
            calls.update(kwargs)
            return {
                "weight_vector": [0.0] * kwargs["num_features"],
                "intercept": 0.0,
                "train_accuracy": 0.0,
                "row_count": 0,
                "iterations_done": 0,
                "partition_path": kwargs["partition_path"],
                "iter_metrics": [],
                "sync_mode": MODE_PS_ASYNC,
            }

        monkeypatch.setattr(async_ps_run, "run", _fake_run)
        sentinel = object()

        run_worker_core(
            worker_id=0,
            partition_path="/nonexistent/partition.csv",
            spark=None,
            worker_config=_cfg(),
            root_comm=sentinel,
        )

        assert calls["comm"] is sentinel  # COMM_WORLD, not the worker sub-comm
        assert calls["rank"] == 1  # COMM_WORLD rank = worker_id + 1
        assert calls["num_workers"] == 2
        assert calls["max_iter"] == 3
        assert calls["num_features"] == 4
