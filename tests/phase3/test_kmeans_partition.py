# =============================================================================
# tests/phase3/test_kmeans_partition.py
# Phase 3 — Issue #8 — Step 2: Partition & Spark Session Tests
#
# WHAT IS TESTED
# --------------
#   1. comm.scatter() distributes metadata so every rank receives a dict
#      with the expected keys (partition_id, partition_path, num_lines, …)
#
#   2. Every rank's partition file exists on the shared filesystem
#      (validates that MPJSparkFileManager wrote to the shared path and
#      all ranks can see it — the NFS correctness check in Docker)
#
#   3. The partition file is non-empty for a non-trivial input dataset
#
#   4. Each rank's partition covers a distinct, non-overlapping shard
#      (total lines across all ranks == total lines in the input file)
#
#   5. The PySpark session is live and usable:
#      spark.sparkContext.parallelize([rank]) collects correctly — proves
#      that the JVM started cleanly inside the MPI rank process.
#
# HOW TO RUN
# ----------
# Tests 1-4 (no Spark, any rank count):
#   mpirun --oversubscribe -n 3 python -m pytest \
#       tests/phase3/test_kmeans_partition.py -v -s
#
# Tests 5-6 (Spark JVM tests) — MUST be run in isolation to avoid
# SparkContext lifecycle pollution from other test modules:
#   mpirun --oversubscribe -n 3 python -m pytest \
#       tests/phase3/test_kmeans_partition.py -v -s
#
# NOTE: these tests perform real I/O and start PySpark, so they are
# integration tests.  They will create files under a temp dir.
# Run `pytest tests/phase3/ --ignore=tests/phase3/test_kmeans_partition.py`
# for pure unit tests without Spark.
# =============================================================================

from __future__ import annotations

import os
import tempfile

import pytest
from mpi4py import MPI

from mpj_spark.applications.kmeans.partition import partition_and_init_spark

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# ---------------------------------------------------------------------------
# Module-level dataset setup (no Spark here — safe at import time)
# ---------------------------------------------------------------------------
_DATASET_LINES = 30  # enough to give >= 1 line per rank for up to 10 ranks


def _make_dataset(tmp_dir: str) -> str:
    """Rank 0 creates a synthetic text dataset; returns its path."""
    path = os.path.join(tmp_dir, "test_input.txt")
    with open(path, "w") as fh:
        for i in range(_DATASET_LINES):
            fh.write(f"word{i} alpha beta gamma delta\n")
    return path


# ---------------------------------------------------------------------------
# Module-level shared state: one temp dir + dataset path visible to all tests.
# Created by rank 0 and broadcast to all ranks so every rank uses the same
# path (required: partition() reads from a shared filesystem).
# ---------------------------------------------------------------------------
_tmp_dir_obj = None  # only rank 0 creates this
_tmp_dir = None  # broadcast to all
_dataset_path = None  # broadcast to all

if rank == 0:
    _tmp_dir_obj = tempfile.TemporaryDirectory()
    _tmp_dir = _tmp_dir_obj.name
    _dataset_path = _make_dataset(_tmp_dir)

_tmp_dir = comm.bcast(_tmp_dir, root=0)
_dataset_path = comm.bcast(_dataset_path, root=0)


# ---------------------------------------------------------------------------
# Session fixture for Spark JVM tests
# ---------------------------------------------------------------------------
# WHY A SESSION FIXTURE?
# ----------------------
# When partition_and_init_spark() is called at module scope (e.g. inside
# import time (module scope), the SparkContext is created during pytest
# collection before other test modules run.  When those modules finish,
# PySpark's atexit hooks may stop the JVM, leaving _spark._jsc = None
# by the time Spark tests actually execute.  A session fixture runs
# after collection and is guaranteed alive for the duration of the
# session, then torn down cleanly via yield.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def spark_partition_session():
    """Partition the dataset and start Spark; yield (partition_path, spark)."""
    partition_path, spark = partition_and_init_spark(
        comm=comm,
        rank=rank,
        size=size,
        input_file=_dataset_path,
        num_workers=size,
    )
    yield partition_path, spark
    spark.stop()


# ---------------------------------------------------------------------------
# Tests 1–4 — partition metadata and file visibility (no Spark)
# ---------------------------------------------------------------------------


def test_scatter_metadata_keys():
    """
    partition_and_init_spark() uses comm.scatter() to distribute metadata.
    After scatter, every rank should receive a metadata dict with the
    standard keys: partition_id, partition_path, num_lines, total_lines.
    """
    # Import only; does not start Spark (metadata path only)
    from mpj_spark.applications.kmeans.partition import _scatter_partition_metadata

    meta = _scatter_partition_metadata(
        comm=comm,
        rank=rank,
        size=size,
        input_file=_dataset_path,
        output_dir=_tmp_dir,
    )
    required = {"partition_id", "partition_path", "num_lines", "total_lines"}
    missing = required - set(meta.keys())
    assert not missing, f"[rank {rank}] Scatter metadata missing keys: {missing}"


def test_partition_file_exists():
    """
    Every rank's partition file must exist on the shared filesystem after
    partition_and_init_spark().  In Docker Swarm with NFS, this validates
    that the shared volume is correctly mounted and the file manager wrote
    to a path visible to all ranks.
    """
    from mpj_spark.applications.kmeans.partition import _scatter_partition_metadata

    meta = _scatter_partition_metadata(
        comm=comm,
        rank=rank,
        size=size,
        input_file=_dataset_path,
        output_dir=_tmp_dir,
    )
    path = meta["partition_path"]
    assert os.path.exists(path), f"[rank {rank}] Partition file not found: {path}"


def test_partition_file_non_empty():
    """
    Each rank's partition file must contain at least one line.
    An empty partition would cause compute_local_stats() to return a
    zero-count cluster on every iteration — triggering the reinit guard.
    """
    from mpj_spark.applications.kmeans.partition import _scatter_partition_metadata

    meta = _scatter_partition_metadata(
        comm=comm,
        rank=rank,
        size=size,
        input_file=_dataset_path,
        output_dir=_tmp_dir,
    )
    path = meta["partition_path"]
    with open(path) as f:
        lines = f.readlines()
    assert len(lines) > 0, f"[rank {rank}] Partition file is empty: {path}"


def test_total_lines_equal_input():
    """
    The sum of num_lines across all ranks must equal the total number of
    lines in the input dataset.  Ensures the partitioner produces a
    complete, non-overlapping cover of the input.
    """
    from mpj_spark.applications.kmeans.partition import _scatter_partition_metadata

    meta = _scatter_partition_metadata(
        comm=comm,
        rank=rank,
        size=size,
        input_file=_dataset_path,
        output_dir=_tmp_dir,
    )
    local_lines = meta["num_lines"]
    total_local = comm.reduce(local_lines, op=MPI.SUM, root=0)
    if rank == 0:
        assert total_local == _DATASET_LINES, (
            f"Total partitioned lines ({total_local}) != "
            f"input dataset lines ({_DATASET_LINES})"
        )


# ---------------------------------------------------------------------------
# Tests 5–6 — Spark JVM session (skipped if size < 2)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    MPI.COMM_WORLD.Get_size() < 2,
    reason="Spark JVM tests require >= 2 MPI ranks",
)
def test_spark_session_live(spark_partition_session):
    """
    The PySpark session returned by partition_and_init_spark() must be
    alive and usable.  spark.sparkContext.parallelize([rank]).collect()
    is the minimal end-to-end Spark action that proves the JVM started
    cleanly inside the MPI rank process.
    """
    _, spark = spark_partition_session
    try:
        result = spark.sparkContext.parallelize([rank]).collect()
        assert result == [
            rank
        ], f"[rank {rank}] Spark parallelize/collect failed: got {result}"
    except Exception as e:
        pytest.fail(f"[rank {rank}] Spark session is not usable: {e}")


@pytest.mark.skipif(
    MPI.COMM_WORLD.Get_size() < 2,
    reason="Spark JVM tests require >= 2 MPI ranks",
)
def test_partition_rdd_readable(spark_partition_session):
    """
    The partition file written by partition_and_init_spark() must be
    readable as a Spark RDD.  spark.sparkContext.textFile(partition_path)
    must return an RDD with at least one element.
    """
    partition_path, spark = spark_partition_session
    try:
        rdd = spark.sparkContext.textFile(partition_path)
        count = rdd.count()
        assert (
            count > 0
        ), f"[rank {rank}] Partition RDD is empty for file: {partition_path}"
    except Exception as e:
        pytest.fail(f"[rank {rank}] Failed to read partition as RDD: {e}")
