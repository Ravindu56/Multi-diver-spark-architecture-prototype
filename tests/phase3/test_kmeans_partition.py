# =============================================================================
# tests/phase3/test_kmeans_partition.py
# Phase 3 — MPI partition integration tests
#
# ALL tests in this file require >= 2 MPI ranks and are guarded with
# @_NEEDS_MPI.  Under plain `pytest` (size=1) every test is SKIPPED.
#
# HOW TO RUN
#   mpirun --oversubscribe -n 3 python -m pytest \
#       tests/phase3/test_kmeans_partition.py -v -s
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

_NEEDS_MPI = pytest.mark.skipif(
    size < 2,
    reason="MPI partition tests require >= 2 MPI ranks — "
    "re-launch with: mpirun --oversubscribe -n 3 python -m pytest tests/phase3/test_kmeans_partition.py",
)

# ---------------------------------------------------------------------------
# Module-level dataset setup
# ---------------------------------------------------------------------------
_DATASET_LINES = 30


def _make_dataset(tmp_dir: str) -> str:
    path = os.path.join(tmp_dir, "test_input.txt")
    with open(path, "w") as fh:
        for i in range(_DATASET_LINES):
            fh.write(f"word{i} alpha beta gamma delta\n")
    return path


_tmp_dir_obj = None
_tmp_dir = None
_dataset_path = None

if rank == 0:
    _tmp_dir_obj = tempfile.TemporaryDirectory()
    _tmp_dir = _tmp_dir_obj.name
    _dataset_path = _make_dataset(_tmp_dir)

_tmp_dir = comm.bcast(_tmp_dir, root=0)
_dataset_path = comm.bcast(_dataset_path, root=0)


# ---------------------------------------------------------------------------
# Session fixture for Spark JVM tests
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
# Tests 1–4 — partition metadata and file visibility (all MPI-only)
# Driven through the public partition_and_init_spark() API.
# ---------------------------------------------------------------------------


@_NEEDS_MPI
def test_scatter_metadata_keys(spark_partition_session):
    """
    partition_and_init_spark() must return a non-empty partition path string
    for every rank, confirming comm.scatter() distributed metadata correctly.
    """
    partition_path, _ = spark_partition_session
    assert isinstance(partition_path, str) and len(partition_path) > 0, (
        f"[rank {rank}] Expected a non-empty partition path, got: {partition_path!r}"
    )


@_NEEDS_MPI
def test_partition_file_exists(spark_partition_session):
    """
    Every rank's partition file must exist on the shared filesystem.
    In Docker Swarm with NFS this validates the shared volume is mounted
    and the file manager wrote to a path visible to all ranks.
    """
    partition_path, _ = spark_partition_session
    assert os.path.exists(partition_path), (
        f"[rank {rank}] Partition file not found: {partition_path}"
    )


@_NEEDS_MPI
def test_partition_file_non_empty(spark_partition_session):
    """
    Each rank's partition file must contain at least one line.
    An empty partition causes compute_local_stats() to return zero-count
    clusters on every iteration, triggering the reinit guard.
    """
    partition_path, _ = spark_partition_session
    with open(partition_path) as f:
        lines = f.readlines()
    assert len(lines) > 0, f"[rank {rank}] Partition file is empty: {partition_path}"


@_NEEDS_MPI
def test_total_lines_equal_input(spark_partition_session):
    """
    Sum of line counts across all ranks must equal the total input lines.
    Ensures the partitioner produces a complete, non-overlapping cover.
    """
    partition_path, _ = spark_partition_session
    with open(partition_path) as f:
        local_lines = sum(1 for _ in f)
    total = comm.reduce(local_lines, op=MPI.SUM, root=0)
    if rank == 0:
        assert total == _DATASET_LINES, (
            f"Total partitioned lines ({total}) != input lines ({_DATASET_LINES})"
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
    alive and usable.
    """
    _, spark = spark_partition_session
    try:
        result = spark.sparkContext.parallelize([rank]).collect()
        assert result == [rank], f"[rank {rank}] Spark parallelize/collect failed: got {result}"
    except Exception as e:
        pytest.fail(f"[rank {rank}] Spark session is not usable: {e}")


@pytest.mark.skipif(
    MPI.COMM_WORLD.Get_size() < 2,
    reason="Spark JVM tests require >= 2 MPI ranks",
)
def test_partition_rdd_readable(spark_partition_session):
    """
    The partition file must be readable as a Spark RDD with >= 1 element.
    """
    partition_path, spark = spark_partition_session
    try:
        rdd = spark.sparkContext.textFile(partition_path)
        count = rdd.count()
        assert count > 0, f"[rank {rank}] Partition RDD is empty: {partition_path}"
    except Exception as e:
        pytest.fail(f"[rank {rank}] Failed to read partition as RDD: {e}")
