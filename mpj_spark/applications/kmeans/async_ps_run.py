"""mpj_spark/applications/kmeans/async_ps_run.py
Asynchronous parameter-server synchronisation for K-Means (FedAsync mixing).

Each worker runs `local_epochs` Lloyd iterations from its current model, sends the whole
(k x d) centroid matrix to the root parameter server over COMM_WORLD P2P (TAG 30 up /
31 down) and continues from the refreshed global model that the server replies with.  The
server (core/async_ps._serve) never waits for stragglers: it blends every update into the
global model on arrival with alpha = server_lr / (1 + staleness).  Its mixing code is
shape-agnostic, so the centroid matrix is carried in the ``weights`` field and the unused
``intercept`` field is 0.0.

Labels stay aligned without Hungarian matching: the seed centroids come from one setup
bcast on the worker sub-comm, and every worker always continues from the server's model.

The server serves exactly max_iter updates per worker, so there is no early stop.
Import-safe on CI: no mpi4py import here.

Returned ``centres`` are the model from the worker's LAST server reply; workers finish at
different versions, so they can differ slightly.  ``wcss`` is the WCSS of those centres on
THIS worker's partition (the root sums it), so the total is approximate unless the workers'
final models agree.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from mpj_spark.applications.kmeans.common import centroid_shift, local_lloyd, local_wcss
from mpj_spark.applications.kmeans.local_iteration import init_centroids, load_partition_rdd
from mpj_spark.applications.kmeans.metrics import KMeansMetricsCollector
from mpj_spark.core.sync_modes import MODE_PS_ASYNC

TAG_ALLREDUCE_UP = 30
TAG_ALLREDUCE_DOWN = 31


def run_kmeans_async_ps(
    comm,
    root_comm,
    rank: int,
    world_rank: int,
    size: int,
    input_file: str,
    k: int = 3,
    max_iter: int = 20,
    local_epochs: int = 5,
    seed: int = 42,
    metrics_output_dir: str = "./metrics",
    sync_mode: str = MODE_PS_ASYNC,
    tag_up: int = TAG_ALLREDUCE_UP,
    tag_down: int = TAG_ALLREDUCE_DOWN,
    points_rdd=None,
    stats_fn=None,
) -> dict[str, Any]:
    """K-Means with async PS.  `comm` = worker sub-comm (setup bcast), `root_comm` = COMM_WORLD."""
    if comm is None or root_comm is None:
        raise RuntimeError(
            "[KMeans Async PS] ps_async requires both the worker sub-communicator (comm, "
            "seed broadcast) and COMM_WORLD (root_comm, parameter server)."
        )

    if points_rdd is None:
        from pyspark.sql import SparkSession

        spark = SparkSession.getActiveSession()
        if spark is None:
            raise RuntimeError(f"[KMeans Async Worker {world_rank}] No active SparkSession found.")
        points_rdd = load_partition_rdd(spark, input_file)
    row_count = int(points_rdd.count())

    lloyd_kwargs = {} if stats_fn is None else {"stats_fn": stats_fn}
    collector = KMeansMetricsCollector(rank=world_rank, output_dir=metrics_output_dir)

    seed_c = init_centroids(points_rdd, k=k, seed=seed).tolist() if rank == 0 else None
    centroids = np.asarray(comm.bcast(seed_c, root=0), dtype=np.float64)

    base_version = 0
    local_wcss_val = float("inf")
    iter_times: list[float] = []
    print(
        f"[KMeans Async Worker {world_rank}] k={k} rounds={max_iter} "
        f"local_epochs={local_epochs} rows={row_count:,} mode=ps_async (FedAsync)"
    )

    for it in range(max_iter):
        t_iter = time.perf_counter()
        prev = centroids.copy()

        t_spark = time.perf_counter()
        local_c, _sums, _counts, local_wcss_val = local_lloyd(
            points_rdd, centroids, local_epochs, **lloyd_kwargs
        )
        spark_time = time.perf_counter() - t_spark

        t_ps = time.perf_counter()
        root_comm.send(
            {
                "weights": local_c.tolist(),
                "intercept": 0.0,
                "base_version": base_version,
                "rank": world_rank,
                "worker_round": it,
                "row_count": row_count,
            },
            dest=0,
            tag=tag_up,
        )
        reply = root_comm.recv(source=0, tag=tag_down)
        ps_time = time.perf_counter() - t_ps

        centroids = np.asarray(reply["weights"], dtype=np.float64)
        base_version = int(reply["global_version"])
        staleness = int(reply.get("staleness", 0))
        shift = centroid_shift(prev, centroids)
        iter_time = time.perf_counter() - t_iter
        iter_times.append(iter_time)
        collector.record_iteration(
            iteration=it + 1,
            spark_time_s=spark_time,
            sync_time_s=ps_time,
            iter_time_s=iter_time,
            centroid_shift=shift,
            global_wcss=local_wcss_val,
        )
        print(
            f"[KMeans Async Worker {world_rank}] round {it + 1}/{max_iter}  ({iter_time:.3f}s)  "
            f"shift={shift:.6f}  local_WCSS={local_wcss_val:.2f}  staleness={staleness}  "
            f"v{base_version}  ps={ps_time:.4f}s"
        )

    final_wcss = local_wcss(points_rdd, centroids)
    collector.record_run(
        total_time_s=sum(iter_times),
        iterations_run=len(iter_times),
        converged=False,
        k=k,
        dataset_size=row_count,
        num_ranks=size,
    )
    collector.to_csv()
    collector.to_json()

    return {
        "centres": centroids.tolist(),
        "wcss": final_wcss,
        "global_wcss": local_wcss_val,
        "row_count": row_count,
        "iterations_done": len(iter_times),
        "converged": False,
        "sync_mode": sync_mode,
        "final_version": base_version,
    }
