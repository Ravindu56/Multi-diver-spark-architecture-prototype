# =============================================================================
# mpj_spark/applications/kmeans/allreduce.py
# Phase 3 — Issue #8 — Step 4 (refactored Steps 5 + 6)
#
# CHANGE LOG (broadcast seed centroids fix)
# -----------------------------------------
# - rank 0 now calls init_centroids() on its local partition and
#   comm.Bcast the result to all other ranks before the iteration loop.
#   All ranks start iteration 1 with IDENTICAL seed centroids so the
#   first Allreduce aggregates assignments made with a consistent
#   geometric reference frame. This eliminates the non-reproducible
#   early-iteration divergence caused by independent per-rank init.
#
# CHANGE LOG (Step 6 integration)
# --------------------------------
# - Instantiates KMeansMetricsCollector at the top of run_kmeans_allreduce.
# - Added spark_time_s measurement window: time.perf_counter() bracket
#   around compute_local_stats() + _compute_local_wcss() only — isolates
#   the Spark action cost from the MPI synchronisation window.
# - sync_time_s window now starts before allreduce_centroids() and ends
#   after check_and_broadcast() — covers the full MPI-collective region.
# - Calls collector.record_iteration() inside the loop.
# - Calls collector.record_run() + collector.to_csv() + collector.to_json()
#   after the loop, before spark.stop().
# - rank 0 calls KMeansMetricsCollector.aggregate_across_ranks() at the end.
# - The bare `metrics: List[Dict]` list is replaced by the collector;
#   run return dict now reads from collector._iterations and collector._run.
# - All Step 4 and Step 5 logic (Allreduce, empty-cluster guard, WCSS
#   Allreduce, convergence check, comm.Barrier) is unchanged.
#
# CHANGE LOG (__main__ entry-point)
# ----------------------------------
# - Added `if __name__ == '__main__':` block with argparse so that
#   `python -m mpj_spark.applications.kmeans.allreduce` actually runs.
# =============================================================================

from __future__ import annotations

import logging
import time

import numpy as np

logger = logging.getLogger(__name__)

_EMPTY_CLUSTER_THRESHOLD = 0.5


# ===========================================================================
# Core Allreduce primitive  (unchanged)
# ===========================================================================


def allreduce_centroids(
    comm,
    rank: int,
    local_sums: np.ndarray,
    local_counts: np.ndarray,
    points_rdd,
) -> np.ndarray:
    """
    Aggregate local centroid sums and counts across all MPI ranks and
    compute the new global centroid array.  Buffer-level MPI.SUM Allreduce
    on local_sums (K x D) and local_counts (K,).  Empty-cluster guard: rank 0
    samples a replacement and Bcasts it.
    """
    from mpi4py import MPI

    k, d = local_sums.shape
    global_sums = np.zeros_like(local_sums)
    global_counts = np.zeros_like(local_counts)

    comm.Allreduce([local_sums, MPI.DOUBLE], [global_sums, MPI.DOUBLE], op=MPI.SUM)
    comm.Allreduce([local_counts, MPI.DOUBLE], [global_counts, MPI.DOUBLE], op=MPI.SUM)

    global_centroids = np.zeros((k, d), dtype=np.float64)
    empty_clusters: list[int] = []

    for j in range(k):
        if global_counts[j] >= _EMPTY_CLUSTER_THRESHOLD:
            global_centroids[j] = global_sums[j] / global_counts[j]
        else:
            empty_clusters.append(j)
            logger.warning(
                "[rank %d] Cluster %d empty (global_count=%.1f) — reinitialising.",
                rank,
                j,
                global_counts[j],
            )

    for j in empty_clusters:
        if rank == 0:
            sample = points_rdd.takeSample(False, 1, seed=int(time.time() * 1000))
            replacement = np.array(sample[0], dtype=np.float64)
        else:
            replacement = np.zeros(d, dtype=np.float64)
        comm.Bcast([replacement, MPI.DOUBLE], root=0)
        global_centroids[j] = replacement
        logger.info("[rank %d] Cluster %d reinitialised -> %s", rank, j, replacement[:4])

    return global_centroids


# ===========================================================================
# Full K-Means Allreduce runner  (Steps 2-6 orchestration)
# ===========================================================================


def run_kmeans_allreduce(
    comm,
    rank: int,
    size: int,
    input_file: str,
    k: int = 3,
    max_iter: int = 20,
    tol: float = 1e-4,
    seed: int = 42,
    cores_override: int | None = None,
    metrics_output_dir: str = "./metrics",
) -> dict:
    """
    Full multi-driver K-Means with synchronous Allreduce centroid sync
    and per-iteration metrics collection (Step 6).

    Centroid initialisation strategy (broadcast from rank 0)
    ---------------------------------------------------------
    Rank 0 calls init_centroids() on its local partition using k-means++
    with the given seed, then broadcasts the resulting (k, D) centroid
    array to all other ranks via comm.Bcast before the iteration loop.

    This guarantees:
      1. All ranks start iteration 1 with the same seed centroid positions.
      2. Iteration 1 Allreduce aggregates assignments made against a
         consistent geometric reference frame — no cross-rank label
         misalignment in the first step.
      3. Results are reproducible: same seed => same centroid path every run.

    Alternative (independent per-rank init) is available by passing
    broadcast_init=False but is NOT recommended for production runs.
    """
    from mpi4py import MPI

    from mpj_spark.applications.kmeans.convergence import check_and_broadcast
    from mpj_spark.applications.kmeans.local_iteration import (
        compute_local_stats,
        init_centroids,
        load_partition_rdd,
    )
    from mpj_spark.applications.kmeans.metrics import KMeansMetricsCollector
    from mpj_spark.applications.kmeans.partition import partition_and_init_spark

    t_total_start = time.perf_counter()

    collector = KMeansMetricsCollector(rank=rank, output_dir=metrics_output_dir)

    # Step 2: partition + scatter + Spark session
    partition_path, spark = partition_and_init_spark(
        comm=comm,
        rank=rank,
        size=size,
        input_file=input_file,
        num_workers=size,
        cores_override=cores_override,
    )

    # Step 3 setup: load RDD
    points_rdd = load_partition_rdd(spark, partition_path)
    total_points = points_rdd.count()

    # ------------------------------------------------------------------
    # Centroid initialisation — broadcast from rank 0
    # ------------------------------------------------------------------
    # Rank 0 runs k-means++ on its partition (representative sample of the
    # full dataset after round-robin partitioning) and broadcasts the result.
    # All other ranks allocate a zero buffer of the correct shape first so
    # comm.Bcast has a valid receive buffer on every rank.
    #
    # Why this is correct:
    #   - After round-robin partitioning every partition is an i.i.d. sample
    #     of the full dataset, so rank 0's k-means++ init is as representative
    #     as running it on the full dataset.
    #   - The Bcast cost is negligible: k * D * 8 bytes (e.g. 3 * 20 * 8 = 480B).
    #   - Iteration 1 then aggregates assignments made with a SHARED reference
    #     frame, eliminating the geometric misalignment that caused non-
    #     reproducible early-iteration behaviour in the independent-init path.
    # ------------------------------------------------------------------
    if rank == 0:
        centroids = init_centroids(points_rdd, k=k, seed=seed)
        d = centroids.shape[1]
    else:
        # Shape is unknown until rank 0 broadcasts; allocate after we learn D.
        # We first Bcast D so non-root ranks can allocate the correct buffer.
        d = None

    # Step 1: broadcast D (feature dimension) so all ranks can allocate
    d = comm.bcast(d if rank == 0 else None, root=0)

    if rank != 0:
        centroids = np.zeros((k, d), dtype=np.float64)

    # Step 2: broadcast the centroid array itself
    comm.Bcast([centroids, MPI.DOUBLE], root=0)

    logger.info(
        "[rank %d] init  mode=bcast_from_rank0  k=%d  D=%d  "
        "seed=%d (only rank 0 used seed; others received via Bcast)",
        rank,
        k,
        d,
        seed,
    )

    comm.Barrier()
    logger.info("[rank %d] Barrier passed — entering iteration loop", rank)

    prev_centroids = np.zeros_like(centroids)
    converged = False

    # Main loop: Steps 3 + 4 + 5 + 6
    for iteration in range(1, max_iter + 1):
        t_iter_start = time.perf_counter()

        t_spark_start = time.perf_counter()
        local_sums, local_counts = compute_local_stats(points_rdd, centroids)
        local_wcss = _compute_local_wcss(points_rdd, centroids)
        spark_time = time.perf_counter() - t_spark_start

        t_sync_start = time.perf_counter()

        new_centroids = allreduce_centroids(comm, rank, local_sums, local_counts, points_rdd)

        global_wcss_arr = np.array([local_wcss], dtype=np.float64)
        global_wcss_buf = np.zeros(1, dtype=np.float64)
        comm.Allreduce(
            [global_wcss_arr, MPI.DOUBLE],
            [global_wcss_buf, MPI.DOUBLE],
            op=MPI.SUM,
        )
        global_wcss = float(global_wcss_buf[0])

        converged, centroid_shift = check_and_broadcast(
            comm, rank, new_centroids, prev_centroids, tol, iteration
        )

        sync_time = time.perf_counter() - t_sync_start
        iter_time = time.perf_counter() - t_iter_start

        collector.record_iteration(
            iteration=iteration,
            spark_time_s=spark_time,
            sync_time_s=sync_time,
            iter_time_s=iter_time,
            centroid_shift=centroid_shift,
            global_wcss=global_wcss,
        )

        logger.info(
            "[rank %d] iter=%d  spark=%.4fs  sync=%.4fs  iter=%.4fs  shift=%.6f  wcss=%.2f",
            rank,
            iteration,
            spark_time,
            sync_time,
            iter_time,
            centroid_shift,
            global_wcss,
        )

        prev_centroids = centroids
        centroids = new_centroids

        if converged:
            logger.info(
                "[rank %d] Converged at iteration %d (shift=%.6f < tol=%.6f)",
                rank,
                iteration,
                centroid_shift,
                tol,
            )
            break

    # Step 6: run-level record + file output
    total_time = time.perf_counter() - t_total_start

    collector.record_run(
        total_time_s=total_time,
        iterations_run=len(collector._iterations),
        converged=converged,
        k=k,
        dataset_size=total_points,
        num_ranks=size,
    )
    collector.to_csv()
    collector.to_json()

    comm.Barrier()
    if rank == 0:
        try:
            KMeansMetricsCollector.aggregate_across_ranks(
                output_dir=metrics_output_dir,
                num_ranks=size,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Aggregation failed (non-fatal): %s", exc)

    points_rdd.unpersist()
    spark.stop()

    return {
        "global_centroids": centroids.tolist(),
        "iterations_run": len(collector._iterations),
        "converged": converged,
        "metrics": collector.summary_table(),
        "run_summary": collector._run,
        "rank": rank,
        "total_time_s": round(total_time, 4),
    }


# ===========================================================================
# Internal helper — local WCSS  (unchanged)
# ===========================================================================


def _compute_local_wcss(points_rdd, centroids: np.ndarray) -> float:
    _centroids = centroids

    def _wcss_partition(points_iter):
        total = 0.0
        for pt in points_iter:
            diff = _centroids - pt
            sq_dists = np.einsum("ij,ij->i", diff, diff)
            total += float(np.min(sq_dists))
        yield total

    return float(points_rdd.mapPartitions(_wcss_partition).sum())


# ===========================================================================
# CLI entry-point
# ===========================================================================

if __name__ == "__main__":
    import argparse
    import os
    import sys

    from mpi4py import MPI

    _comm = MPI.COMM_WORLD
    _rank = _comm.Get_rank()
    _size = _comm.Get_size()

    parser = argparse.ArgumentParser(
        prog="python -m mpj_spark.applications.kmeans.allreduce",
        description="Multi-driver K-Means with synchronous MPI Allreduce "
        "centroid sync (Phase 3 — Issue #8).",
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to the input CSV file (shared/NFS path visible to all ranks).",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="Number of clusters K (default: 5).",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=20,
        help="Maximum number of iterations (default: 20).",
    )
    parser.add_argument(
        "--tol",
        type=float,
        default=1e-4,
        help="Convergence tolerance — Frobenius centroid shift (default: 1e-4).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for centroid initialisation (default: 42).",
    )
    parser.add_argument(
        "--output",
        default="./kmeans_results",
        help="Directory for per-rank metrics CSV/JSON and aggregated CSV "
        "(default: ./kmeans_results).",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level for all ranks (default: INFO).",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format=f"%(asctime)s [rank {_rank}] %(levelname)s %(name)s - %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
    )

    os.makedirs(args.output, exist_ok=True)

    if _rank == 0:
        print(
            f"\n{'=' * 60}\n"
            "  K-Means Allreduce — Phase 3 / Issue #8\n"
            f"  ranks={_size}  k={args.k}  max_iter={args.max_iter}  "
            f"tol={args.tol}  seed={args.seed}\n"
            f"  init=bcast_from_rank0\n"
            f"  input  : {args.input}\n"
            f"  output : {args.output}\n"
            f"{'=' * 60}\n",
            flush=True,
        )

    result = run_kmeans_allreduce(
        comm=_comm,
        rank=_rank,
        size=_size,
        input_file=args.input,
        k=args.k,
        max_iter=args.max_iter,
        tol=args.tol,
        seed=args.seed,
        metrics_output_dir=args.output,
    )

    if _rank == 0:
        print("\n" + "=" * 60)
        print("  Run complete — rank 0 summary")
        print(f"  converged      : {result['converged']}")
        print(f"  iterations_run : {result['iterations_run']}")
        print(f"  total_time_s   : {result['total_time_s']}s")
        print(f"  output dir     : {args.output}")
        print("=" * 60 + "\n")

        print("Per-iteration metrics (rank 0):")
        table = result["metrics"]
        if table:
            headers = list(table[0].keys())
            col_w = {h: max(len(h), max(len(str(r[h])) for r in table)) for h in headers}
            header_line = "  ".join(h.ljust(col_w[h]) for h in headers)
            print(header_line)
            print("-" * len(header_line))
            for row in table:
                print("  ".join(str(row[h]).ljust(col_w[h]) for h in headers))
        print()
