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


# Rank 0 creates a temp dir + dataset and broadcasts the paths
if rank == 0:
    _tmp_dir = tempfile.mkdtemp(prefix="mpjspark_test_")
    _input_path = _make_dataset(_tmp_dir)
else:
    _tmp_dir = None
    _input_path = None

_tmp_dir = comm.bcast(_tmp_dir, root=0)
_input_path = comm.bcast(_input_path, root=0)

# Shared storage dir lives inside the temp dir — does not pollute the repo.
_shared_storage = os.path.join(_tmp_dir, "shared_storage")


# ---------------------------------------------------------------------------
# Session-scoped fixture: initialise partition + SparkSession ONCE per
# pytest session, AFTER collection is complete.
#
# WHY fixture and not module scope:
#   partition_and_init_spark() starts a PySpark JVM.  If called at module
#   import time (module scope), the SparkContext is created during pytest
#   collection before other test modules run.  When those modules finish,
#   PySpark's atexit hooks may stop the JVM, leaving _spark._jsc = None
#   by the time Spark tests actually execute.  A session fixture runs
#   after collection and is guaranteed alive for the duration of the
#   session, then torn down cleanly via yield.
# ---------------------------------------------------------------------------
from mpj_spark.applications.kmeans.partition import partition_and_init_spark


@pytest.fixture(scope="session")
def spark_partition_session():
    """Partition the dataset and start Spark; yield (partition_path, spark)."""
    partition_path, spark = partition_and_init_spark(
        comm=comm,
        rank=rank,
        size=size,
        input_file=_input_path,
        num_workers=size,
        shared_storage_path=_shared_storage,
    )
    yield partition_path, spark
    # Teardown: stop Spark cleanly so the JVM exits before the next
    # test module's SparkContext (if any) is created.
    try:
        spark.stop()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Convenience: module-level partition_path/spark derived from the fixture
# for tests 1-4 that don't need Spark but do need partition_path.
#
# Tests 5-6 (Spark tests) receive the fixture directly as an argument.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Test 1 — Scatter delivered a valid metadata dict to every rank
# ---------------------------------------------------------------------------
def test_scatter_metadata_keys(spark_partition_session):
    """
    Every rank must receive a metadata dict with the exact keys produced
    by MPJSparkFileManager.dynamic_partition().
    """
    partition_path, _spark = spark_partition_session
    assert os.path.isabs(partition_path) or os.path.exists(
        partition_path
    ), f"rank {rank}: partition_path is neither absolute nor accessible"


# ---------------------------------------------------------------------------
# Test 2 — Partition file exists on the shared filesystem
# ---------------------------------------------------------------------------
def test_partition_file_exists(spark_partition_session):
    """
    The file written by rank 0 must be visible to every rank.
    Failure here means NFS is not mounted or the shared storage path
    is not consistent across ranks.
    """
    partition_path, _spark = spark_partition_session
    assert os.path.exists(partition_path), (
        f"rank {rank}: partition file not found at '{partition_path}'.\n"
        "Check that all ranks share the same filesystem "
        "(NFS in Docker, local FS in single-machine mode)."
    )


# ---------------------------------------------------------------------------
# Test 3 — Partition file is non-empty
# ---------------------------------------------------------------------------
def test_partition_file_non_empty(spark_partition_session):
    """Each rank must have at least one line in its shard."""
    partition_path, _spark = spark_partition_session
    size_bytes = os.path.getsize(partition_path)
    assert size_bytes > 0, (
        f"rank {rank}: partition file is empty (0 bytes): {partition_path}"
    )


# ---------------------------------------------------------------------------
# Test 4 — Total lines across all ranks == total input lines
# ---------------------------------------------------------------------------
def test_total_lines_sum_equals_input(spark_partition_session):
    """
    Collect the line count from every rank and assert the global sum
    equals the known input size.  Uses comm.allreduce() — the same
    collective that will be used for centroid sync in Step 4.
    """
    partition_path, _spark = spark_partition_session
    local_line_count = sum(1 for _ in open(partition_path))
    global_line_count = comm.allreduce(local_line_count, op=MPI.SUM)

    if rank == 0:
        assert global_line_count == _DATASET_LINES, (
            f"Total lines across all partitions ({global_line_count}) "
            f"!= input lines ({_DATASET_LINES}).  "
            "Partition logic may have dropped or duplicated lines."
        )


# ---------------------------------------------------------------------------
# Tests 5 & 6 — Spark JVM tests
# These require an active SparkContext.  Guarded with skipif so that
# running without mpirun (size == 1) produces a clear SKIP rather than
# a confusing JVM error.
# ---------------------------------------------------------------------------
@pytest.mark.skipif(
    size < 2,
    reason=(
        "Spark JVM tests require at least 2 MPI ranks. "
        "Re-run with: mpirun --oversubscribe -n 3 python -m pytest "
        "tests/phase3/test_kmeans_partition.py -v"
    ),
)
def test_spark_session_alive(spark_partition_session):
    """
    Perform a minimal Spark action on each rank's local SparkSession.
    If the JVM failed to start or the session is stale, collect() will
    raise — catching it gives a clearer error message than a JVM crash.
    """
    _partition_path, spark = spark_partition_session
    try:
        result = spark.sparkContext.parallelize([rank]).collect()
    except Exception as exc:
        pytest.fail(
            f"rank {rank}: SparkSession.sparkContext.parallelize failed: {exc}"
        )

    assert result == [rank], (
        f"rank {rank}: expected [{rank}] from parallelize, got {result}"
    )


@pytest.mark.skipif(
    size < 2,
    reason=(
        "Spark JVM tests require at least 2 MPI ranks. "
        "Re-run with: mpirun --oversubscribe -n 3 python -m pytest "
        "tests/phase3/test_kmeans_partition.py -v"
    ),
)
def test_spark_reads_partition_file(spark_partition_session):
    """
    Each rank reads its own partition file into a Spark RDD and counts
    the lines.  This is the exact first operation the K-Means runner
    (Step 3) will perform before computing local centroid sums.
    """
    partition_path, spark = spark_partition_session
    try:
    	 rdd = spark.sparkContext.textFile(partition_path)
    	 line_count = rdd.count()
    except Exception as exc:
        pytest.fail(
            f"rank {rank}: textFile('{partition_path}') failed: {exc}"
        )

    assert line_count > 0, (
        f"rank {rank}: Spark read 0 lines from partition file."
    )
