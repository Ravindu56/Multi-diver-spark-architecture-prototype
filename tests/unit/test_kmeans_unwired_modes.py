"""K-Means sync-mode dispatch: wired modes reach their runner; the rest refuse to fall back."""

from __future__ import annotations

import pytest

from mpj_spark.core.sync_modes import MODE_GOSSIP, MODE_HYBRID_PS_ALLREDUCE, MODE_PS_ASYNC
from mpj_spark.workers.worker_process import run_worker_core


@pytest.mark.parametrize("mode", [MODE_PS_ASYNC, MODE_HYBRID_PS_ALLREDUCE])
@pytest.mark.parametrize("comm,root_comm", [(None, object()), (object(), None), (None, None)])
def test_ps_modes_without_both_channels_refuse_allreduce_fallback(mode, comm, root_comm):
    cfg = {"app": "kmeans", "num_workers": 2, "sync_mode": mode}
    with pytest.raises(RuntimeError, match="not wired"):
        run_worker_core(
            worker_id=0,
            partition_path="x.csv",
            spark=None,
            worker_config=cfg,
            comm=comm,
            root_comm=root_comm,
        )


def test_gossip_without_worker_subcomm_refuses_allreduce_fallback():
    cfg = {"app": "kmeans", "num_workers": 2, "sync_mode": MODE_GOSSIP}
    with pytest.raises(RuntimeError, match="not wired"):
        run_worker_core(
            worker_id=0, partition_path="x.csv", spark=None, worker_config=cfg, comm=None
        )


def _cfg(mode):
    return {
        "app": "kmeans",
        "num_workers": 3,
        "sync_mode": mode,
        "kmeans_k": 4,
        "kmeans_max_iter": 7,
        "gossip_fanout": None,
        "results_dir": "r",
    }


def test_gossip_dispatches_to_run_kmeans_gossip(monkeypatch):
    seen = {}

    def _fake(**kwargs):
        seen.update(kwargs)
        return {"centres": []}

    monkeypatch.setattr("mpj_spark.applications.kmeans.gossip_run.run_kmeans_gossip", _fake)
    comm = object()
    out = run_worker_core(
        worker_id=1,
        partition_path="p.csv",
        spark=None,
        worker_config=_cfg(MODE_GOSSIP),
        comm=comm,
    )

    assert out["status"] == "success"
    assert seen["comm"] is comm
    assert (seen["rank"], seen["size"], seen["k"], seen["max_iter"]) == (1, 3, 4, 7)
    assert seen["input_file"] == "p.csv"
    assert seen["fanout"] is None


@pytest.mark.parametrize(
    "mode,module,func",
    [
        (MODE_HYBRID_PS_ALLREDUCE, "hybrid_run", "run_kmeans_hybrid"),
        (MODE_PS_ASYNC, "async_ps_run", "run_kmeans_async_ps"),
    ],
)
def test_ps_modes_dispatch_with_both_communicators(monkeypatch, mode, module, func):
    seen = {}

    def _fake(**kwargs):
        seen.update(kwargs)
        return {"centres": []}

    monkeypatch.setattr(f"mpj_spark.applications.kmeans.{module}.{func}", _fake)
    comm, root_comm = object(), object()
    out = run_worker_core(
        worker_id=1,
        partition_path="p.csv",
        spark=None,
        worker_config=_cfg(mode),
        comm=comm,
        root_comm=root_comm,
    )

    assert out["status"] == "success"
    assert seen["comm"] is comm and seen["root_comm"] is root_comm
    assert (seen["rank"], seen["world_rank"], seen["size"]) == (1, 2, 3)
    assert (seen["k"], seen["max_iter"], seen["input_file"]) == (4, 7, "p.csv")
