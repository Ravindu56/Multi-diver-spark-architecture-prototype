# =============================================================================
# mpj_spark/applications/kmeans/partition.py
# Phase 3 — Issue #8 — Step 2: Dataset Partitioning and Per-Rank Spark Session
#
# PURPOSE
# -------
# Provide a single entry-point function — partition_and_init_spark() — that:
#
#   1. On rank 0 (root): uses MPJSparkFileManager.dynamic_partition() to split
#      the input dataset into N equal shards on shared storage, then distributes
#      the partition metadata dict to every rank via comm.scatter().
#
#   2. On all ranks (including root): receives its own partition metadata,
#      resolves the partition file path, and creates a rank-local PySpark
#      session via build_spark_session().
#
# DESIGN DECISIONS
# ----------------
# a) WHY comm.scatter() and not N individual comm.send(tag=10) calls?
#    The existing mpj_spark_mpi.py uses per-worker send(tag=10) for the
#    general-purpose coordinator.  For K-Means we want a clean, testable
#    partition module that is independent of the coordinator scaffolding.
#    comm.scatter() is semantically identical for equal-sized payloads and
#    is the idiomatic mpi4py collective for 1-to-all data distribution.
#
# b) WHY create the Spark session AFTER comm.scatter()?
#    JVM startup (SparkContext.__init__) grabs OS threads and opens sockets.
#    Initiating this on all ranks simultaneously, before MPI has finished
#    distributing work, causes resource contention that manifests as
#    intermittent "Address already in use" errors on the Spark driver port.
#    Sequencing: scatter first → every rank knows its data → then JVM starts.
#
# c) WHY does root also get a Spark session?
#    In the current single-machine prototype the root rank doubles as worker 0.
#    In Phase 5 (multi-node Docker Swarm) rank 0 will be a pure coordinator
#    and the `if rank == 0` guard in worker code will skip Spark init.
#    Keeping root symmetric here simplifies Step 3 (all ranks run the same
#    centroid computation loop).
#
# USAGE (called from the K-Means runner — Step 3)
# -----------------------------------------------
#   from mpj_spark.applications.kmeans.partition import partition_and_init_spark
#
#   partition_path, spark = partition_and_init_spark(
#       comm=comm,
#       rank=rank,
#       size=size,
#       input_file=args.input,
#       num_workers=size,      # every rank is a worker
#       cores_override=args.cores,
#   )
#
# RETURNS
#   partition_path : str   — absolute path to this rank's partition file
#   spark          : SparkSession — rank-local Spark session (local[N])
# =============================================================================

from __future__ import annotations

import os
import logging
from typing import Tuple

from pyspark.sql import SparkSession

from mpj_spark.config import SHARED_STORAGE_PATH
from mpj_spark.core.file_manager import MPJSparkFileManager
from mpj_spark.workers.spark_session import build_spark_session

logger = logging.getLogger(__name__)


def partition_and_init_spark(
    comm,
    rank: int,
    size: int,
    input_file: str,
    num_workers: int | None = None,
    cores_override: int | None = None,
    memory_fraction: float = 0.75,
    shared_storage_path: str = SHARED_STORAGE_PATH,
) -> Tuple[str, SparkSession]:
    """
    Partition the dataset on rank 0 and scatter metadata; every rank
    creates its own isolated PySpark session over its data shard.

    Parameters
    ----------
    comm             : mpi4py.MPI.Intracomm — COMM_WORLD (or any comm)
    rank             : int — this process's MPI rank
    size             : int — total number of MPI ranks
    input_file       : str — path to the full input dataset (NFS / local)
    num_workers      : int — number of shards to create (default: size)
    cores_override   : int — CPU cores per Spark session (default: auto)
    memory_fraction  : float — fraction of total RAM available for Spark
    shared_storage_path : str — root of shared storage (NFS mount point)

    Returns
    -------
    (partition_path, spark) : (str, SparkSession)
    """
    n_workers = num_workers if num_workers is not None else size

    # ------------------------------------------------------------------ #
    # STEP 2a — Root partitions the dataset                               #
    # ------------------------------------------------------------------ #
    # Only rank 0 performs I/O.  It uses the existing streaming two-pass  #
    # MPJSparkFileManager which writes N partition files to shared storage #
    # and returns a list of metadata dicts (one per worker).              #
    # ------------------------------------------------------------------ #
    if rank == 0:
        logger.info(
            "[rank 0] Partitioning '%s' into %d shards → %s",
            input_file,
            n_workers,
            shared_storage_path,
        )
        file_manager = MPJSparkFileManager(shared_storage_path=shared_storage_path)
        metadata_list = file_manager.dynamic_partition(
            input_file_path=input_file,
            num_workers=n_workers,
        )
        # comm.scatter() needs a list of exactly `size` elements.
        # If n_workers < size (unusual) pad with None; receiver must guard.
        scatter_payload = metadata_list[:size]  # one dict per rank
        while len(scatter_payload) < size:
            scatter_payload.append(None)

        logger.info(
            "[rank 0] Partition complete.  Sizes: %s",
            [m["num_lines"] for m in metadata_list],
        )
    else:
        scatter_payload = None  # non-root: placeholder, filled by scatter

    # ------------------------------------------------------------------ #
    # STEP 2b — Scatter partition metadata to all ranks                   #
    # ------------------------------------------------------------------ #
    # comm.scatter() is a collective: every rank (including root) must    #
    # call it.  Root distributes scatter_payload[i] → rank i.            #
    # Non-root ranks receive their own dict automatically.                #
    # ------------------------------------------------------------------ #
    my_metadata = comm.scatter(scatter_payload, root=0)

    if my_metadata is None:
        raise RuntimeError(
            f"[rank {rank}] received None from scatter — "
            "num_workers may be less than MPI size."
        )

    partition_path: str = my_metadata["partition_path"]

    if not os.path.exists(partition_path):
        raise FileNotFoundError(
            f"[rank {rank}] Partition file not found: {partition_path}\n"
            "Ensure all ranks share the same filesystem "
            "(NFS in Docker, Lustre in HPC)."
        )

    logger.info(
        "[rank %d] Received partition: %s  (%d lines, %d bytes)",
        rank,
        partition_path,
        my_metadata["num_lines"],
        my_metadata["file_size_bytes"],
    )

    # ------------------------------------------------------------------ #
    # STEP 2c — Create rank-local PySpark session                         #
    # ------------------------------------------------------------------ #
    # Spark context is started AFTER scatter completes so that JVM        #
    # startup on all ranks does not race with MPI partition distribution. #
    # build_spark_session() allocates memory proportionally:             #
    #   heap_mb = total_ram * memory_fraction / n_workers                 #
    # and uses native BLAS + G1GC (already configured in spark_session.py)#
    # ------------------------------------------------------------------ #
    spark = build_spark_session(
        app_name=f"KMeans-rank{rank}",
        cores_override=cores_override,
        num_workers=n_workers,
        memory_fraction=memory_fraction,
    )

    logger.info(
        "[rank %d] SparkSession ready.  Master: %s",
        rank,
        spark.sparkContext.master,
    )

    return partition_path, spark
