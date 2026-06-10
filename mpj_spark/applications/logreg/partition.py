# =============================================================================
# mpj_spark/applications/logreg/partition.py
# Phase 3 — Issue #9 — Steps 1 & 2: MPI Init Rule + Dataset Partitioning
#
# PURPOSE
# -------
# Provide a single entry-point function — partition_and_init_spark() — that:
#
#   Step 1 (MPI Init Rule)
#   ~~~~~~~~~~~~~~~~~~~~~~
#   Documents and enforces the critical ordering constraint: SparkSession
#   must be created AFTER MPI.COMM_WORLD is initialised and AFTER
#   comm.scatter() completes.  This module is imported inside the MPI
#   process (after comm = MPI.COMM_WORLD) so the rule is always respected.
#
#   Step 2a — Root partitions the dataset
#   ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#   On rank 0: uses MPJSparkFileManager.dynamic_partition() to split the
#   input CSV into N equal shards on shared storage and returns a list of
#   metadata dicts (one per rank).
#
#   Step 2b — Scatter partition metadata
#   ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#   comm.scatter() distributes metadata_list[i] → rank i.  Every rank
#   (including root) participates as a collective call.
#
#   Step 2c — Per-rank SparkSession creation
#   ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#   Each rank creates its own isolated PySpark session via
#   build_spark_session() AFTER scatter completes, avoiding the JVM-startup
#   race condition documented in kmeans/partition.py design note (b).
#
# LOGREG-SPECIFIC ADDITIONS vs KMEANS/partition.py
# -------------------------------------------------
# 1. num_features is propagated through my_metadata so that the caller
#    (allreduce.py Step 3) can pass it directly to logreg.run() without
#    re-reading the CSV header.  Rank 0 detects num_features by peeking
#    at the first non-header line of the input file.
#
# 2. The stray-header problem noted in logreg.py is caused by
#    MPJSparkFileManager's round-robin streaming: line 0 (the CSV header)
#    is assigned to partition 0.  This module records which rank received
#    the stray header in my_metadata["has_header_line"] = True so that
#    logreg.run() can apply its existing NULL-filter without surprises.
#
# USAGE
# -----
#   from mpi4py import MPI
#   comm = MPI.COMM_WORLD          # Step 1: MPI init FIRST
#   rank = comm.Get_rank()
#   size = comm.Get_size()
#
#   # SparkSession is created inside partition_and_init_spark, AFTER scatter
#   from mpj_spark.applications.logreg.partition import partition_and_init_spark
#
#   partition_path, spark, num_features = partition_and_init_spark(
#       comm=comm, rank=rank, size=size, input_file=args.input,
#   )
#
# RETURNS
#   partition_path : str          — absolute path to this rank's CSV shard
#   spark          : SparkSession — rank-local Spark session (local[N])
#   num_features   : int          — number of feature columns (excl. label)
# =============================================================================

from __future__ import annotations

import logging
import os

from pyspark.sql import SparkSession

from mpj_spark.config import SHARED_STORAGE_PATH
from mpj_spark.core.file_manager import MPJSparkFileManager
from mpj_spark.workers.spark_session import build_spark_session

logger = logging.getLogger(__name__)


def _detect_num_features(input_file: str) -> int:
    """
    Peek at the first non-blank line of the CSV to count feature columns.
    Assumes format: f0,f1,...,f{N-1},label  (last column is always 'label').
    Returns N (number of feature columns, excluding the label).
    """
    with open(input_file, encoding="utf-8") as fh:
        for raw_line in fh:
            line = raw_line.strip()
            if not line:
                continue
            cols = line.split(",")
            # If the first line is a header row (contains non-numeric text),
            # skip it and try the next line.
            try:
                float(cols[0])
            except ValueError:
                continue
            # cols[-1] is label; everything before it is a feature
            return len(cols) - 1
    raise RuntimeError(
        f"Could not detect num_features from '{input_file}'. "
        "Ensure the file contains at least one data row."
    )


def partition_and_init_spark(
    comm,
    rank: int,
    size: int,
    input_file: str,
    num_workers: int | None = None,
    cores_override: int | None = None,
    memory_fraction: float = 0.75,
    shared_storage_path: str = SHARED_STORAGE_PATH,
) -> tuple[str, SparkSession, int]:
    """
    Step 1 & 2 entry point for the LogReg MPI-Allreduce runner.

    MPI INIT RULE (Step 1)
    ~~~~~~~~~~~~~~~~~~~~~~
    This function must be called *after* comm = MPI.COMM_WORLD has been
    obtained in the caller.  It never calls MPI.Init() itself — mpi4py
    handles that on import.  The SparkSession is created inside this
    function, AFTER comm.scatter(), enforcing the required ordering:

        MPI.COMM_WORLD  →  comm.scatter()  →  SparkSession

    Parameters
    ----------
    comm             : mpi4py.MPI.Intracomm — COMM_WORLD (or any intracomm)
    rank             : int — this process's MPI rank
    size             : int — total number of MPI ranks
    input_file       : str — path to full input CSV (NFS / local shared FS)
    num_workers      : int — shards to create; defaults to size
    cores_override   : int — CPU cores per Spark session (None = auto)
    memory_fraction  : float — fraction of total RAM for Spark heap
    shared_storage_path : str — root directory for partition files

    Returns
    -------
    partition_path : str          — path to this rank's CSV partition file
    spark          : SparkSession — rank-local isolated Spark session
    num_features   : int          — feature count (for schema construction)
    """
    n_workers = num_workers if num_workers is not None else size

    # ------------------------------------------------------------------ #
    # STEP 2a — Rank 0: partition the CSV and detect num_features         #
    # ------------------------------------------------------------------ #
    # Only rank 0 performs disk I/O.  MPJSparkFileManager.dynamic_partition
    # uses the same streaming round-robin partitioner as in the WordCount
    # and K-Means workloads (no full-file RAM load, exactly N writers open
    # simultaneously).
    #
    # LOGREG NOTE: The CSV header row ("f0,f1,...,label") is assigned to
    # partition 0 by the round-robin (it is line index 0, 0 % N == 0).
    # We flag this in metadata so logreg.run() knows which partition gets
    # the stray header and can apply its NULL-filter reliably.
    # ------------------------------------------------------------------ #
    if rank == 0:
        logger.info(
            "[rank 0] Detecting num_features from '%s'.",
            input_file,
        )
        num_features_detected = _detect_num_features(input_file)
        logger.info(
            "[rank 0] Detected %d feature columns.",
            num_features_detected,
        )

        logger.info(
            "[rank 0] Partitioning '%s' into %d shards → '%s'",
            input_file,
            n_workers,
            shared_storage_path,
        )
        file_manager = MPJSparkFileManager(shared_storage_path=shared_storage_path)
        metadata_list = file_manager.dynamic_partition(
            input_file_path=input_file,
            num_workers=n_workers,
        )

        # Annotate each metadata dict with logreg-specific fields.
        for i, meta in enumerate(metadata_list):
            meta["num_features"] = num_features_detected
            # Rank 0's partition gets the round-robin line 0 (CSV header).
            meta["has_header_line"] = i == 0

        # comm.scatter() requires a list of exactly `size` elements.
        scatter_payload = metadata_list[:size]
        while len(scatter_payload) < size:
            scatter_payload.append(None)

        logger.info(
            "[rank 0] Partition complete. Lines per shard: %s",
            [m["num_lines"] for m in metadata_list],
        )
    else:
        scatter_payload = None  # placeholder; filled by scatter below

    # ------------------------------------------------------------------ #
    # STEP 2b — Scatter partition metadata to every rank (collective)     #
    # ------------------------------------------------------------------ #
    # comm.scatter() is a blocking collective: every rank must call it.   #
    # Rank 0 sends scatter_payload[i] to rank i; non-root ranks pass None #
    # as sendobj (ignored by MPI).  After this call, my_metadata is a     #
    # fully populated dict on every rank.
    # ------------------------------------------------------------------ #
    my_metadata = comm.scatter(scatter_payload, root=0)

    if my_metadata is None:
        raise RuntimeError(
            f"[rank {rank}] received None from scatter — "
            "num_workers may be less than MPI world size."
        )

    partition_path: str = my_metadata["partition_path"]
    num_features: int = my_metadata["num_features"]

    if not os.path.exists(partition_path):
        raise FileNotFoundError(
            f"[rank {rank}] Partition file not found: '{partition_path}'.\n"
            "All ranks must share the same filesystem "
            "(NFS volume in Docker, Lustre in HPC)."
        )

    logger.info(
        "[rank %d] Received partition: '%s'  (%d lines, %d bytes, "
        "num_features=%d, has_header=%s)",
        rank,
        partition_path,
        my_metadata["num_lines"],
        my_metadata["file_size_bytes"],
        num_features,
        my_metadata.get("has_header_line", False),
    )

    # ------------------------------------------------------------------ #
    # STEP 2c — Create rank-local PySpark session (AFTER scatter)         #
    # ------------------------------------------------------------------ #
    # SparkSession is started here — after every rank has received its    #
    # partition metadata — so JVM initialisation on all ranks does not    #
    # race with MPI collective communication.  This is the core ordering  #
    # constraint from Step 1: MPI scatter completes before any JVM starts.#
    #                                                                      #
    # build_spark_session() allocates memory proportionally:             #
    #   heap_mb = total_ram * memory_fraction / n_workers                 #
    # and configures G1GC + native BLAS (already in spark_session.py).   #
    # ------------------------------------------------------------------ #
    spark = build_spark_session(
        app_name=f"LogReg-MPI-rank{rank}",
        cores_override=cores_override,
        num_workers=n_workers,
        memory_fraction=memory_fraction,
    )

    logger.info(
        "[rank %d] SparkSession ready.  Master: %s",
        rank,
        spark.sparkContext.master,
    )

    return partition_path, spark, num_features
