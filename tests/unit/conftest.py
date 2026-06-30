# =============================================================
# tests/unit/conftest.py
# Shared pytest fixtures for Phase 1 & Phase 2 unit tests.
# =============================================================
from multiprocessing import Queue

import pytest
from pyspark.sql import SparkSession


# ── SparkSession (one per test session, local mode) ───────────
@pytest.fixture(scope="session")
def spark():
    """
    Provides a single SparkSession in local[2] mode for the
    entire test session. Avoids expensive start/stop per test.
    """
    session = (
        SparkSession.builder.master("local[2]")
        .appName("mpj-spark-unit-tests")
        .config("spark.ui.enabled", "false")
        .config("spark.sql.shuffle.partitions", "2")
        .getOrCreate()
    )
    yield session
    session.stop()


# ── Gossip / Queue helpers ────────────────────────────────────
@pytest.fixture
def gossip_queue():
    """Fresh multiprocessing.Queue for each test."""
    return Queue()


def push_worker_state(queue, worker_id, centres, row_count, wcss=0.5):
    """
    Helper (not a fixture) — pushes a worker centroid state dict
    onto a queue in the exact format GossipAggregator expects.

    Parameters
    ----------
    queue      : multiprocessing.Queue
    worker_id  : int
    centres    : list[list[float]]
    row_count  : int
    wcss       : float  (default 0.5 — irrelevant for aggregation tests)
    """
    queue.put(
        {
            "worker_id": worker_id,
            "centres": centres,
            "row_count": row_count,
            "wcss": wcss,
        }
    )


# ── Temporary input file factory ─────────────────────────────
@pytest.fixture
def tmp_text_file(tmp_path):
    """
    Returns a factory that writes a list of lines to a temp file
    and returns its path. Used by file_manager tests.

    Usage:
        path = tmp_text_file(["hello world", "foo bar"], "input.txt")
    """

    def _factory(lines, filename="input.txt"):
        p = tmp_path / filename
        p.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return str(p)

    return _factory
