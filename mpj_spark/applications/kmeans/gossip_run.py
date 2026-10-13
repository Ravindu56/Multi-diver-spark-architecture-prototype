"""mpj_spark/applications/kmeans/gossip_run.py
Decentralized gossip synchronisation for K-Means (ring diffusion).

Per round each worker runs `local_epochs` Lloyd iterations on its own partition
(mpj_spark.applications.kmeans.common.local_lloyd), then exchanges
{centroids, row_count, rank} with its fixed-ring neighbours over the WORKER
sub-communicator and mixes: every neighbour's centroids are first label-aligned
to this worker's own centroids (Hungarian), then averaged with row-count weights.

No root coordinator and no collective in the sync path.  The only collective is a
one-time setup bcast of the seed centroids so every worker starts with the same
labelling.  With tol > 0 a worker whose post-mix shift drops below tol freezes its
local refit but keeps exchanging, so neighbours never block on a stopped peer.

Import-safe on CI: mpi4py is never imported here; pyspark only when no RDD is passed.

Metrics note: `global_wcss` in the per-round CSV is the WCSS of THIS worker's model on
THIS worker's partition (there is no global view in a decentralized mode).  The root
sums the workers' final `wcss` values; treat that total as approximate unless workers
have reached consensus.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from mpj_spark.applications.kmeans.common import centroid_shift, local_lloyd, local_wcss
from mpj_spark.applications.kmeans.local_iteration import init_centroids, load_partition_rdd
from mpj_spark.applications.kmeans.metrics import KMeansMetricsCollector
from mpj_spark.core.gossip_protocol import TAG_GOSSIP_EXCHANGE, gossip_exchange, ring_neighbors
from mpj_spark.core.root_process import align_centres_hungarian
from mpj_spark.core.sync_modes import MODE_GOSSIP


def mix_centroids(own: np.ndarray, own_rows: int, received: list[dict[str, Any]]) -> np.ndarray:
    """Row-count-weighted mix of own centroids with label-aligned neighbour centroids."""
    total = float(own_rows + sum(int(m["row_count"]) for m in received))
    if total <= 0:
        uniform = 1.0 / (1 + len(received))
        frac_self, fracs = uniform, [uniform] * len(received)
    else:
        frac_self = own_rows / total
        fracs = [int(m["row_count"]) / total for m in received]
    mixed = frac_self * np.asarray(own, dtype=np.float64)
    for frac, msg in zip(fracs, received, strict=True):
        aligned, _ = align_centres_hungarian(np.asarray(own).tolist(), msg["centroids"])
        mixed = mixed + frac * np.asarray(aligned, dtype=np.float64)
    return mixed


def run_kmeans_gossip(
    comm,
    rank: int,
    size: int,
    input_file: str,
    k: int = 3,
    max_iter: int = 20,
    local_epochs: int = 5,
    fanout: int | None = 1,
    tol: float = 0.0,
    seed: int = 42,
    metrics_output_dir: str = "./metrics",
    sync_mode: str = MODE_GOSSIP,
    tag: int = TAG_GOSSIP_EXCHANGE,
    points_rdd=None,
    stats_fn=None,
) -> dict[str, Any]:
    """K-Means with decentralized ring gossip.  `comm` is the worker sub-communicator."""
    if comm is None:
        raise RuntimeError(
            "[KMeans Gossip] sync_mode='gossip' requires the MPI worker sub-communicator."
        )

    if points_rdd is None:
        from pyspark.sql import SparkSession

        spark = SparkSession.getActiveSession()
        if spark is None:
            raise RuntimeError(f"[KMeans Gossip Rank {rank}] No active SparkSession found.")
        points_rdd = load_partition_rdd(spark, input_file)
    row_count = int(points_rdd.count())

    lloyd_kwargs = {} if stats_fn is None else {"stats_fn": stats_fn}
    collector = KMeansMetricsCollector(rank=rank, output_dir=metrics_output_dir)

    peers = ring_neighbors(rank, size, fanout)
    if not peers and size > 1:
        raise RuntimeError(f"[KMeans Gossip Rank {rank}] ring produced no neighbours.")

    # One-time setup: rank 0 seeds, everyone starts from the same labelled centroids.
    seed_c = init_centroids(points_rdd, k=k, seed=seed).tolist() if rank == 0 else None
    centroids = np.asarray(comm.bcast(seed_c, root=0), dtype=np.float64)

    frozen = False
    converged = False
    local_wcss_val = float("inf")
    iter_times: list[float] = []
    print(
        f"[KMeans Gossip Rank {rank}] k={k} rounds={max_iter} local_epochs={local_epochs} "
        f"rows={row_count:,} peers={peers} mode=gossip (decentralized)"
    )

    for it in range(max_iter):
        t_iter = time.perf_counter()
        prev = centroids.copy()

        t_spark = time.perf_counter()
        if not frozen:
            centroids, _sums, _counts, local_wcss_val = local_lloyd(
                points_rdd, centroids, local_epochs, **lloyd_kwargs
            )
        spark_time = time.perf_counter() - t_spark

        t_sync = time.perf_counter()
        payload = {"centroids": centroids.tolist(), "row_count": row_count, "rank": rank}
        received = gossip_exchange(comm, payload, rank, size, fanout=fanout, tag=tag)
        centroids = mix_centroids(centroids, row_count, received)
        sync_time = time.perf_counter() - t_sync

        shift = centroid_shift(prev, centroids)
        iter_time = time.perf_counter() - t_iter
        iter_times.append(iter_time)
        collector.record_iteration(
            iteration=it + 1,
            spark_time_s=spark_time,
            sync_time_s=sync_time,
            iter_time_s=iter_time,
            centroid_shift=shift,
            global_wcss=local_wcss_val,
        )
        print(
            f"[KMeans Gossip Rank {rank}] round {it + 1}/{max_iter}  ({iter_time:.3f}s)  "
            f"shift={shift:.6f}  local_WCSS={local_wcss_val:.2f}  peers={len(received)}"
            f"{'  [frozen]' if frozen else ''}"
        )
        if tol > 0.0 and not frozen and shift < tol:
            frozen = converged = True
            print(f"[KMeans Gossip Rank {rank}] converged at round {it + 1} - local refit frozen")

    # Final WCSS of the returned (mixed) model on this worker's partition.
    final_wcss = local_wcss(points_rdd, centroids)
    collector.record_run(
        total_time_s=sum(iter_times),
        iterations_run=len(iter_times),
        converged=converged,
        k=k,
        dataset_size=row_count,
        num_ranks=size,
    )
    collector.to_csv()
    collector.to_json()

    return {
        "centres": centroids.tolist(),
        "wcss": final_wcss,
        "row_count": row_count,
        "iterations_done": len(iter_times),
        "converged": converged,
        "sync_mode": sync_mode,
    }
