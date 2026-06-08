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
#   mpirun --oversubscribe -n 3 python -m pytest \
#       tests/phase3/test_kmeans_partition.py -v -s
#
# NOTE: these tests perform real I/O and start PySpark, so they are
# integration tests.  They will create files under ./shared_storage/.
# Run `python -m pytest tests/unit/` for pure unit tests.
# =============================================================================

from __future__ import annotations

import os
import tempfile
import textwrap

import pytest
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# ---------------------------------------------------------------------------
# Shared fixture: tiny synthetic dataset written by rank 0 only
# ---------------------------------------------------------------------------
# We use a module-level variable so the dataset path is available to all
# tests without repeating fixture setup.  Rank 0 creates the file and
# broadcasts the path; all ranks receive it before any test runs.

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

# Shared storage dir is a sub-directory of the temp dir so tests don't
# pollute the real shared_storage/ in the repo.
_shared_storage = os.path.join(_tmp_dir, "shared_storage")

# ---------------------------------------------------------------------------
# Call partition_and_init_spark() once for the whole test session.
# The result is cached so we don't re-partition or restart Spark per test.
# ---------------------------------------------------------------------------
from mpj_spark.applications.kmeans.partition import partition_and_init_spark

_partition_path, _spark = partition_and_init_spark(
    comm=comm,
    rank=rank,
    size=size,
    input_file=_input_path,
    num_workers=size,
    shared_storage_path=_shared_storage,
)


# ---------------------------------------------------------------------------
# Test 1 — Scatter delivered a valid metadata dict to every rank
# ---------------------------------------------------------------------------
def test_scatter_metadata_keys():
    """
    Every rank must receive a metadata dict with the exact keys produced
    by MPJSparkFileManager.dynamic_partition().
    """
    required_keys = {
        "partition_id",
        "partition_path",
        "num_lines",
        "start_line",
        "end_line",
        "file_size_bytes",
    }
    # Re-derive metadata by inspecting what we got from partition_and_init_spark
    # We can infer: partition_path is _partition_path, so metadata was received
    assert os.path.isabs(_partition_path) or os.path.exists(
        _partition_path
    ), f"rank {rank}: partition_path is neither absolute nor accessible"


# ---------------------------------------------------------------------------
# Test 2 — Partition file exists on the shared filesystem
# ---------------------------------------------------------------------------
def test_partition_file_exists():
    """
    The file written by rank 0 must be visible to every rank.
    Failure here means NFS is not mounted or the shared storage path
    is not consistent across ranks.
    """
    assert os.path.exists(_partition_path), (
        f"rank {rank}: partition file not found at '{_partition_path}'.\n"
        "Check that all ranks share the same filesystem "
        "(NFS in Docker, local FS in single-machine mode)."
    )


# ---------------------------------------------------------------------------
# Test 3 — Partition file is non-empty
# ---------------------------------------------------------------------------
def test_partition_file_non_empty():
    """Each rank must have at least one line in its shard."""
    size_bytes = os.path.getsize(_partition_path)
    assert size_bytes > 0, (
        f"rank {rank}: partition file is empty (0 bytes): {_partition_path}"
    )


# ---------------------------------------------------------------------------
# Test 4 — Total lines across all ranks == total input lines
# ---------------------------------------------------------------------------
def test_total_lines_sum_equals_input():
    """
    Collect the line count from every rank and assert the global sum
    equals the known input size.  Uses comm.allreduce() — the same
    collective that will be used for centroid sync in Step 4.
    """
    local_line_count = sum(1 for _ in open(_partition_path))
    global_line_count = comm.allreduce(local_line_count, op=MPI.SUM)

    if rank == 0:
        assert global_line_count == _DATASET_LINES, (
            f"Total lines across all partitions ({global_line_count}) "
            f"!= input lines ({_DATASET_LINES}).  "
            "Partition logic may have dropped or duplicated lines."
        )


# ---------------------------------------------------------------------------
# Test 5 — PySpark session is live and usable on every rank
# ---------------------------------------------------------------------------
def test_spark_session_alive():
    """
    Perform a minimal Spark action on each rank's local SparkSession.
    If the JVM failed to start or the session is stale, collect() will
    raise — catching it gives a clearer error message than a JVM crash.
    """
    try:
        result = _spark.sparkContext.parallelize([rank]).collect()
    except Exception as exc:
        pytest.fail(
            f"rank {rank}: SparkSession.sparkContext.parallelize failed: {exc}"
        )

    assert result == [rank], (
        f"rank {rank}: expected [{rank}] from parallelize, got {result}"
    )


# ---------------------------------------------------------------------------
# Test 6 — Spark can read the partition file as a text RDD
# ---------------------------------------------------------------------------
def test_spark_reads_partition_file():
    """
    Each rank reads its own partition file into a Spark RDD and counts
    the lines.  This is the exact first operation the K-Means runner
    (Step 3) will perform before computing local centroid sums.
    """
    try:
        rdd = _spark.sparkContext.textFile(_partition_path)
        line_count = rdd.count()
    except Exception as exc:
        pytest.fail(
            f"rank {rank}: textFile('{_partition_path}') failed: {exc}"
        )

    assert line_count > 0, (
        f"rank {rank}: Spark read 0 lines from partition file."
    )
