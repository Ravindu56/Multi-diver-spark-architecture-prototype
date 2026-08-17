# ================================================================
# mpj_spark/applications/kmeans/fedavg_mpi_run.py
# Phase 3 — P3-08: K-Means periodic FedAvg over native MPI collectives
# University of Jaffna — 2022/E/033 & 2022/E/090
#
# CHANGE LOG
# ----------
# fix(p3-08): compute_local_stats() returns a tuple, not a dict.
#   _unpack_stats() handles both (sums, counts[, wcss]) tuples and
#   dict shapes defensively.  centroid_shift (Frobenius) and global
#   WCSS are now computed per round instead of stubbed 0.0 values.
# ================================================================
from __future__ import annotations

import time

import numpy as np
from mpi4py import MPI

from mpj_spark.applications.kmeans.local_iteration import (
    compute_local_stats,
    init_centroids,
    load_partition_rdd,
)
from mpj_spark.applications.kmeans.metrics import KMeansMetricsCollector
from mpj_spark.core.root_process import align_centres_hungarian
from mpj_spark.core.sync_modes import MODE_PS_SYNC_FEDAVG_MPI


def _unpack_stats(stats):
    """Normalise compute_local_stats() output to (cluster_sums, cluster_counts, local_wcss).

    Handles both tuple returns — (sums, counts) or (sums, counts, wcss) —
    and dict returns with cluster_sums/cluster_counts keys.
    """
    if isinstance(stats, dict):
        sums = np.asarray(stats.get("cluster_sums", stats.get("sums")), dtype=np.float64)
        counts = np.asarray(stats.get("cluster_counts", stats.get("counts")), dtype=np.float64)
        wcss = float(stats.get("local_wcss", stats.get("wcss", 0.0)))
        return sums, counts, wcss
    sums = np.asarray(stats[0], dtype=np.float64)
    counts = np.asarray(stats[1], dtype=np.float64)
    wcss = float(stats[2]) if len(stats) >= 3 else 0.0
    return sums, counts, wcss


def _local_wcss(points_rdd, centroids: np.ndarray) -> float:
    """Sum of squared distances from each point to its nearest centroid."""
    c = centroids  # closure for Spark serialisation

    def _sq_dist(p):
        return float(np.min(np.sum((c - np.asarray(p, dtype=np.float64)) ** 2, axis=1)))

    return float(points_rdd.map(_sq_dist).sum())


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
    """K-Means periodic FedAvg using native MPI gather/bcast collectives.

    Protocol per synchronisation round:
      1. E local Lloyd iterations on this worker's partition
      2. comm.gather() of (centroids, row_count, cluster_counts, local_wcss)
      3. Aggregator (worker rank 0 of the worker sub-communicator) aligns
         each worker's centroid labels via Hungarian matching against the
         round's reference, computes the row-weighted FedAvg centroid set,
         and aggregates total WCSS
      4. comm.bcast() of the global centroids + WCSS back to all workers

    Convergence: stop early when Frobenius centroid shift < tol.
    """
    from pyspark.sql import SparkSession

    spark = SparkSession.getActiveSession()
    if spark is None:
        raise RuntimeError(f"[K-Means FedAvg MPI Rank {rank}] No active SparkSession found.")

    collector = KMeansMetricsCollector(rank=rank, output_dir=metrics_output_dir)

    # Step 1: load partition RDD
    t_load = time.perf_counter()
    points_rdd = load_partition_rdd(spark, input_file)
    load_time = time.perf_counter() - t_load
    row_count = points_rdd.count()

    # Step 2: rank 0 of this communicator computes seed centroids, Bcast
    if rank == 0:
        centroids = init_centroids(points_rdd, k=k, seed=seed)
        d = int(centroids.shape[1])
    else:
        d = None

    d = comm.bcast(d, root=0)
    if rank != 0:
        centroids = np.zeros((k, d), dtype=np.float64)
    comm.Bcast([centroids, MPI.DOUBLE], root=0)

    converged = False
    global_wcss = float("inf")
    cluster_sums = np.zeros((k, d), dtype=np.float64)
    cluster_counts = np.zeros(k, dtype=np.float64)
    iter_times: list[float] = []

    # Step 3: periodic FedAvg rounds
    for it in range(max_iter):
        t_iter_start = time.perf_counter()
        prev_centroids = centroids.copy()

        # ── Local Lloyd iterations ────────────────────────────────────
        t_spark = time.perf_counter()
        local_wcss = 0.0
        for _ in range(local_epochs):
            cluster_sums, cluster_counts, local_wcss = _unpack_stats(
                compute_local_stats(points_rdd, centroids)
            )
            for j in range(k):
                if cluster_counts[j] > 0:
                    centroids[j] = cluster_sums[j] / cluster_counts[j]
                # empty cluster: retain previous position
        spark_time_s = time.perf_counter() - t_spark

        centroid_shift = float(np.linalg.norm(centroids - prev_centroids))

        # ── MPI collective sync: gather → align → FedAvg → bcast ──────
        t_sync = time.perf_counter()
        payload = {
            "centroids": centroids.tolist(),
            "row_count": row_count,
            "cluster_counts": cluster_counts.tolist(),
            "local_wcss": local_wcss,
        }

        gathered = comm.gather(payload, root=0)
        global_payload = None
        if rank == 0 and gathered is not None:
            total_rows = sum(g["row_count"] for g in gathered)
            ref_centres = gathered[0]["centroids"]
            aligned = [gathered[0]]
            for g in gathered[1:]:
                aligned_centres, _ = align_centres_hungarian(ref_centres, g["centroids"])
                aligned.append({**g, "centroids": aligned_centres})

            avg_centroids = np.zeros((k, d), dtype=np.float64)
            for g in aligned:
                frac = g["row_count"] / total_rows if total_rows > 0 else 1.0 / len(aligned)
                avg_centroids += frac * np.asarray(g["centroids"], dtype=np.float64)

            global_payload = {
                "centroids": avg_centroids.tolist(),
                "global_wcss": sum(g["local_wcss"] for g in gathered),
                "iteration": it,
            }

        result_payload = comm.bcast(global_payload, root=0)
        centroids = np.asarray(result_payload["centroids"], dtype=np.float64)
        global_wcss = float(result_payload["global_wcss"])
        sync_time_s = time.perf_counter() - t_sync

        iter_time_s = time.perf_counter() - t_iter_start
        iter_times.append(iter_time_s)

        collector.record_iteration(
            iteration=it + 1,
            spark_time_s=spark_time_s,
            sync_time_s=sync_time_s,
            iter_time_s=iter_time_s,
            centroid_shift=centroid_shift,
            global_wcss=global_wcss,
        )

        print(
            f"[KMeans FedAvg MPI Rank {rank}] round {it + 1}/{max_iter}  "
            f"({iter_time_s:.3f}s)  shift={centroid_shift:.6f}  "
            f"WCSS={global_wcss:.2f}"
        )

        if centroid_shift < tol:
            converged = True
            break

    total_time_s = load_time + sum(iter_times)

    collector.record_run(
        total_time_s=total_time_s,
        iterations_run=len(iter_times),
        converged=converged,
        dataset_size=row_count,
        num_ranks=size,
        k=k,
        tol=tol,
    )
    collector.to_csv()
    collector.to_json()

    return {
        "centres": centroids.tolist(),
        "wcss": global_wcss,
        "row_count": row_count,
        "iterations_done": len(iter_times),
        "converged": converged,
        "sync_mode": sync_mode,
    }
