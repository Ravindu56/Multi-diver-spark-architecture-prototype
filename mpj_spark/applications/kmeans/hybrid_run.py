"""mpj_spark/applications/kmeans/hybrid_run.py
Hybrid PS + Allreduce synchronisation for K-Means (mirrors logreg/hybrid_run.py).

Per round:
  1. DENSE centroids -> comm.Allreduce(SUM) over the WORKER sub-communicator.  Each
     worker contributes frac_i * centroids_i with frac_i = rows_i / total_rows, so the
     collective directly yields the row-weighted mean (FedAvg of centroids).  Row counts
     are static, so they are exchanged ONCE at init via allgather.
  2. SCALAR metadata (WCSS) -> the existing root scalar parameter server over COMM_WORLD
     P2P (hybrid_ps._serve, TAG 30 up / 31 down).  That server returns the row-weighted
     mean of one float per worker, so each worker sends local_wcss / row_count in the
     ``intercept`` field; the reply times total_rows is the exact global WCSS.

Labels stay aligned without Hungarian matching: every worker starts every round from the
same Allreduce result, and the seed centroids come from one setup bcast on the sub-comm.

No early stop: the root scalar PS serves exactly max_iter rounds, so a worker leaving
early would stall it.  Import-safe on CI: mpi4py is imported lazily inside the runner.

Returned ``wcss`` is the WCSS of the final centroids on THIS worker's partition (the root
sums it); ``global_wcss`` is the last round's global pre-sync WCSS from the PS channel.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from mpj_spark.applications.kmeans.common import centroid_shift, local_lloyd, local_wcss
from mpj_spark.applications.kmeans.local_iteration import init_centroids, load_partition_rdd
from mpj_spark.applications.kmeans.metrics import KMeansMetricsCollector
from mpj_spark.core.sync_modes import MODE_HYBRID_PS_ALLREDUCE

TAG_ALLREDUCE_UP = 30
TAG_ALLREDUCE_DOWN = 31


def run_kmeans_hybrid(
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
    sync_mode: str = MODE_HYBRID_PS_ALLREDUCE,
    tag_up: int = TAG_ALLREDUCE_UP,
    tag_down: int = TAG_ALLREDUCE_DOWN,
    points_rdd=None,
    stats_fn=None,
    mpi_sum=None,
) -> dict[str, Any]:
    """K-Means with hybrid sync.  `comm` = worker sub-comm, `root_comm` = COMM_WORLD."""
    if comm is None or root_comm is None:
        raise RuntimeError(
            "[KMeans Hybrid] hybrid_ps_allreduce requires both the worker sub-communicator "
            "(comm) and COMM_WORLD (root_comm)."
        )
    if mpi_sum is None:
        from mpi4py import MPI  # lazy: module must import without libmpi (CI)

        mpi_sum = MPI.SUM

    if points_rdd is None:
        from pyspark.sql import SparkSession

        spark = SparkSession.getActiveSession()
        if spark is None:
            raise RuntimeError(f"[KMeans Hybrid Worker {world_rank}] No active SparkSession found.")
        points_rdd = load_partition_rdd(spark, input_file)
    row_count = int(points_rdd.count())

    lloyd_kwargs = {} if stats_fn is None else {"stats_fn": stats_fn}
    collector = KMeansMetricsCollector(rank=world_rank, output_dir=metrics_output_dir)

    all_rows = comm.allgather(row_count)
    total_rows = int(sum(all_rows))
    frac = row_count / total_rows if total_rows > 0 else 1.0 / max(1, len(all_rows))

    seed_c = init_centroids(points_rdd, k=k, seed=seed).tolist() if rank == 0 else None
    centroids = np.asarray(comm.bcast(seed_c, root=0), dtype=np.float64)

    global_wcss = float("inf")
    iter_times: list[float] = []
    print(
        f"[KMeans Hybrid Worker {world_rank}] k={k} rounds={max_iter} "
        f"local_epochs={local_epochs} rows={row_count:,} "
        "mode=hybrid_ps_allreduce (dense=Allreduce, scalars=PS)"
    )

    for it in range(max_iter):
        t_iter = time.perf_counter()
        prev = centroids.copy()

        t_spark = time.perf_counter()
        local_c, _sums, _counts, local_wcss_val = local_lloyd(
            points_rdd, centroids, local_epochs, **lloyd_kwargs
        )
        spark_time = time.perf_counter() - t_spark

        t_allreduce = time.perf_counter()
        send_c = np.ascontiguousarray(frac * local_c, dtype=np.float64)
        glob_c = np.zeros_like(send_c)
        comm.Allreduce(send_c, glob_c, op=mpi_sum)
        allreduce_time = time.perf_counter() - t_allreduce

        t_ps = time.perf_counter()
        per_point = local_wcss_val / row_count if row_count > 0 else 0.0
        root_comm.send(
            {
                "intercept": per_point,
                "row_count": row_count,
                "rank": world_rank,
                "worker_round": it,
            },
            dest=0,
            tag=tag_up,
        )
        reply = root_comm.recv(source=0, tag=tag_down)
        ps_time = time.perf_counter() - t_ps

        centroids = glob_c
        global_wcss = float(reply["intercept"]) * total_rows
        shift = centroid_shift(prev, centroids)
        iter_time = time.perf_counter() - t_iter
        iter_times.append(iter_time)
        collector.record_iteration(
            iteration=it + 1,
            spark_time_s=spark_time,
            sync_time_s=allreduce_time + ps_time,
            iter_time_s=iter_time,
            centroid_shift=shift,
            global_wcss=global_wcss,
        )
        print(
            f"[KMeans Hybrid Worker {world_rank}] round {it + 1}/{max_iter}  ({iter_time:.3f}s)  "
            f"shift={shift:.6f}  WCSS={global_wcss:.2f}  "
            f"allreduce={allreduce_time:.4f}s  ps={ps_time:.4f}s"
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
        "global_wcss": global_wcss,
        "row_count": row_count,
        "iterations_done": len(iter_times),
        "converged": False,
        "sync_mode": sync_mode,
    }
