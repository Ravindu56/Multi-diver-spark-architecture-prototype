"""K-Means must refuse sync modes it does not implement (no silent allreduce)."""

from __future__ import annotations

import pytest

from mpj_spark.core.sync_modes import MODE_GOSSIP, MODE_HYBRID_PS_ALLREDUCE, MODE_PS_ASYNC
from mpj_spark.workers.worker_process import run_worker_core


@pytest.mark.parametrize("mode", [MODE_PS_ASYNC, MODE_HYBRID_PS_ALLREDUCE])
def test_unwired_kmeans_modes_raise_instead_of_running_allreduce(mode):
    cfg = {"app": "kmeans", "num_workers": 2, "sync_mode": mode}
    with pytest.raises(RuntimeError, match="not wired"):
        run_worker_core(
            worker_id=0,
            partition_path="x.csv",
            spark=None,
            worker_config=cfg,
            comm=object(),
            root_comm=object(),
        )


def test_gossip_without_worker_subcomm_refuses_allreduce_fallback():
    cfg = {"app": "kmeans", "num_workers": 2, "sync_mode": MODE_GOSSIP}
    with pytest.raises(RuntimeError, match="not wired"):
        run_worker_core(
            worker_id=0, partition_path="x.csv", spark=None, worker_config=cfg, comm=None
        )


def test_gossip_dispatches_to_run_kmeans_gossip(monkeypatch):
    seen = {}

    def _fake(**kwargs):
        seen.update(kwargs)
        return {"centres": []}

    monkeypatch.setattr("mpj_spark.applications.kmeans.gossip_run.run_kmeans_gossip", _fake)
    comm = object()
    cfg = {
        "app": "kmeans",
        "num_workers": 3,
        "sync_mode": MODE_GOSSIP,
        "kmeans_k": 4,
        "kmeans_max_iter": 7,
        "gossip_fanout": None,
        "results_dir": "r",
    }
    out = run_worker_core(
        worker_id=1, partition_path="p.csv", spark=None, worker_config=cfg, comm=comm
    )

    assert out["status"] == "success"
    assert seen["comm"] is comm
    assert (seen["rank"], seen["size"], seen["k"], seen["max_iter"]) == (1, 3, 4, 7)
    assert seen["input_file"] == "p.csv"
    assert seen["fanout"] is None
