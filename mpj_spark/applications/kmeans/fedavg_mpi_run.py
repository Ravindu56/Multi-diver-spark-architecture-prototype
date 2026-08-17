# ================================================================
# mpj_spark/applications/kmeans/fedavg_mpi_run.py
# Phase 3 — P3-08: K-Means periodic FedAvg over native MPI collectives
# University of Jaffna — 2022/E/033 & 2022/E/090
# ================================================================
from __future__ import annotations

import time
import numpy as np
from mpi4py import MPI

from mpj_spark.applications.kmeans.local_iteration import load_partition_rdd, init_centroids, compute_local_stats
from mpj_spark.applications.kmeans.metrics import KMeansMetricsCollector
from mpj_spark.core.sync_modes import MODE_PS_SYNC_FEDAVG_MPI
from mpj_spark.core.root_process import align_centres_hungarian


def run_kmeans_fedavg_mpi(
    comm,
    rank: int,
    size: int,
    input_file: str,
    k: int = 3,
    max_iter: int = 20,
    local_epochs: int = 5,
    tol: float = 1e-4,
    seed: int = 42,
    metrics_output_dir: str = "./metrics",
    sync_mode: str = MODE_PS_SYNC_FEDAVG_MPI,
) -> dict:
    """K-Means periodic FedAvg using native MPI gather/bcast collectives."""
    from pyspark.sql import SparkSession

    spark = SparkSession.getActiveSession()
    if spark is None:
        raise RuntimeError(f"[K-Means FedAvg MPI Rank {rank}] No active SparkSession found.")

    collector = KMeansMetricsCollector(rank=rank, output_dir=metrics_output_dir)

    # Step 1: Load RDD
    t_load = time.perf_counter()
    points_rdd = load_partition_rdd(spark, input_file)
    load_time = time.perf_counter() - t_load

    # Step 2: Rank 0 init centroids and Bcast (same as K-Means Allreduce)
    if rank == 0:
        centroids = init_centroids(points_rdd, k=k, seed=seed)
        d = centroids.shape[1]
    else:
        d = None

    d = comm.bcast(d, root=0)
    if rank != 0:
        centroids = np.zeros((k, d), dtype=np.float64)

    comm.Bcast([centroids, MPI.DOUBLE], root=0)

    # Step 3: Iterate periodic FedAvg
    for it in range(max_iter):
        t_iter_start = time.perf_counter()

        # Local optimization (E iterations)
        t_spark = time.perf_counter()
        for _ in range(local_epochs):
            stats = compute_local_stats(points_rdd, centroids)
            cluster_sums = stats["cluster_sums"]
            cluster_counts = stats["cluster_counts"]
            for j in range(k):
                if cluster_counts[j] > 0:
                    centroids[j] = cluster_sums[j] / cluster_counts[j]
        spark_time_s = time.perf_counter() - t_spark

        # MPI Collective Sync (gather -> Hungarian align + weighted avg -> bcast)
        t_sync = time.perf_counter()
        payload = {
            "centroids": centroids.tolist(),
            "row_count": points_rdd.count(),
            "cluster_counts": cluster_counts.tolist(),
            "rank": rank,
        }

        gathered = comm.gather(payload, root=0)
        global_payload = None
        if rank == 0 and gathered is not None:
            total_rows = sum(g["row_count"] for g in gathered)
            ref_centres = gathered[0]["centroids"]
            aligned_results = [gathered[0]]
            for g in gathered[1:]:
                aligned_centres, perm = align_centres_hungarian(ref_centres, g["centroids"])
                aligned_results.append({**g, "centroids": aligned_centres})

            avg_centroids = np.zeros((k, d))
            for g in aligned_results:
                frac = g["row_count"] / total_rows
                avg_centroids += frac * np.array(g["centroids"])

            global_payload = {
                "centroids": avg_centroids.tolist(),
                "iteration": it,
            }

        result_payload = comm.bcast(global_payload, root=0)
        centroids = np.array(result_payload["centroids"])
        sync_time_s = time.perf_counter() - t_sync

        iter_time_s = time.perf_counter() - t_iter_start

        collector.record_iteration(
            iteration=it + 1,
            spark_time_s=spark_time_s,
            sync_time_s=sync_time_s,
            iter_time_s=iter_time_s,
            centroid_shift=0.0,
            global_wcss=0.0,
        )

    # Metrics write
    collector.record_run(
        total_time_s=load_time + sum([r["iter_time_s"] for r in collector._iterations]),
        iterations_run=max_iter,
        converged=False,
        dataset_size=points_rdd.count(),
        num_ranks=size,
        k=k,
        tol=tol,
    )
    collector.to_csv()
    collector.to_json()

    return {
        "centres": centroids.tolist(),
        "wcss": 0.0,
        "iterations_done": max_iter,
        "sync_mode": sync_mode,
    }
