"""K-Means must refuse sync modes it does not implement (no silent allreduce)."""

from __future__ import annotations

import pytest

from mpj_spark.core.sync_modes import MODE_GOSSIP, MODE_HYBRID_PS_ALLREDUCE, MODE_PS_ASYNC
from mpj_spark.workers.worker_process import run_worker_core


@pytest.mark.parametrize("mode", [MODE_GOSSIP, MODE_PS_ASYNC, MODE_HYBRID_PS_ALLREDUCE])
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
