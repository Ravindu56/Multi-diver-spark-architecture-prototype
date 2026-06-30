"""
tests/phase3/test_wordcount_mpi_vs_baseline.py

Issue #14 — P3-08: WordCount regression test.

Validates that the MPI multi-driver WordCount (N=5) produces the same
top-N word frequency output as the single-driver Spark baseline on an
identical fixed input file.

Guard
-----
  @pytest.mark.skipif(mpi_size < 2, ...)
  The test is skipped automatically in CI (single-rank / no MPI env).
  Run manually in a full MPI environment:

    mpirun --oversubscribe -n 5 pytest tests/phase3/test_wordcount_mpi_vs_baseline.py -v

Acceptance criteria (Issue #14)
--------------------------------
  1. Test script lives at tests/phase3/test_wordcount_mpi_vs_baseline.py
  2. Runs single-driver WordCount baseline on a fixed input file
  3. Runs MPI multi-driver WordCount (mpirun -n 5) on the same file
  4. Asserts top-N word counts match exactly
  5. Guarded with @pytest.mark.skipif(mpi_size < 2) — skipped in CI
"""

from __future__ import annotations

import collections
import subprocess
import sys
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# MPI size detection — must be importable even without mpi4py
# ---------------------------------------------------------------------------
try:
    from mpi4py import MPI

    _mpi_size = MPI.COMM_WORLD.Get_size()
except ImportError:
    _mpi_size = 1

_NEEDS_MPI = pytest.mark.skipif(
    _mpi_size < 2,
    reason="WordCount MPI regression test requires mpirun -n 5 (mpi_size >= 2). "
    "Skipped in single-rank CI. Run: mpirun --oversubscribe -n 5 pytest "
    "tests/phase3/test_wordcount_mpi_vs_baseline.py -v",
)

# ---------------------------------------------------------------------------
# Project root
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Fixed deterministic input — same text every run
# ---------------------------------------------------------------------------
_WORDCOUNT_INPUT = """
apple banana cherry apple banana apple
cherry date elderberry fig cherry date
banana apple fig grape fig grape grape
apple cherry banana date elderberry apple
honeydew fig banana apple cherry cherry
date grape honeydew honeydew fig apple
elderberry banana cherry apple date date
""".strip()

_TOP_N = 5  # number of most-frequent words to compare


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_input_file(tmp_dir: Path) -> Path:
    """Write the fixed input to a temp file and return its path."""
    p = tmp_dir / "wordcount_input.txt"
    p.write_text(_WORDCOUNT_INPUT)
    return p


def _count_words_baseline(input_path: Path) -> dict[str, int]:
    """
    Run single-driver Spark WordCount baseline.
    Returns full word → count dict.
    """
    from pyspark.sql import SparkSession

    spark = (
        SparkSession.builder.master("local[2]")
        .appName("WordCount-baseline-test")
        .config("spark.ui.enabled", "false")
        .config("spark.driver.host", "127.0.0.1")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    try:
        rdd = spark.sparkContext.textFile(str(input_path))
        counts = (
            rdd.flatMap(lambda line: line.strip().split())
            .filter(bool)
            .map(lambda w: (w.lower(), 1))
            .reduceByKey(lambda a, b: a + b)
            .collect()
        )
        return dict(counts)
    finally:
        spark.stop()


def _parse_mpi_stdout(stdout: str) -> dict[str, int]:
    """
    Parse the word-count output lines emitted by the MPI WordCount app.
    Expected format (produced by mpj_spark.applications.wordcount):
        word: count
    Lines not matching this pattern are ignored.
    """
    result: dict[str, int] = {}
    for line in stdout.splitlines():
        line = line.strip()
        if ":" in line:
            parts = line.split(":", 1)
            word = parts[0].strip().lower()
            try:
                count = int(parts[1].strip())
                if word:
                    result[word] = result.get(word, 0) + count
            except ValueError:
                pass
    return result


def _top_n(counts: dict[str, int], n: int) -> dict[str, int]:
    """Return the top-n most-frequent words as a dict."""
    return dict(collections.Counter(counts).most_common(n))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestWordCountMpiVsBaseline:
    """MPI multi-driver WordCount must produce identical top-N results as single-driver baseline."""

    @_NEEDS_MPI
    def test_top_n_counts_match(self, tmp_path: Path) -> None:
        """
        Core regression test.
        1. Write fixed input.
        2. Run single-driver baseline — collect top-N.
        3. Run MPI multi-driver WordCount — collect top-N from stdout.
        4. Assert top-N word counts are identical.
        """
        input_file = _write_input_file(tmp_path)

        # -- Baseline ----------------------------------------------------------
        baseline_counts = _count_words_baseline(input_file)
        assert baseline_counts, "Baseline produced empty word counts"
        baseline_top = _top_n(baseline_counts, _TOP_N)

        # -- MPI multi-driver --------------------------------------------------
        cmd = [
            "mpirun",
            "--oversubscribe",
            "-n",
            "5",
            sys.executable,
            "-m",
            "mpj_spark.core.main_mpi",
            "--input",
            str(input_file),
            "--app",
            "wordcount",
        ]
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=str(_PROJECT_ROOT),
            timeout=120,
        )
        assert result.returncode == 0, (
            f"MPI WordCount exited with code {result.returncode}.\n"
            f"STDOUT:\n{result.stdout[-2000:]}\n"
            f"STDERR:\n{result.stderr[-2000:]}"
        )

        mpi_counts = _parse_mpi_stdout(result.stdout)
        assert mpi_counts, (
            "MPI WordCount produced no parseable word counts in stdout.\n"
            f"STDOUT:\n{result.stdout[-2000:]}"
        )
        mpi_top = _top_n(mpi_counts, _TOP_N)

        # -- Assert exact match -----------------------------------------------
        assert baseline_top == mpi_top, (
            f"Top-{_TOP_N} word counts MISMATCH.\n"
            f"  Baseline : {baseline_top}\n"
            f"  MPI      : {mpi_top}\n"
            "Architecture is NOT regression-free — investigate Allreduce aggregation."
        )

    @_NEEDS_MPI
    def test_all_words_present_in_mpi_output(self, tmp_path: Path) -> None:
        """
        Every word in the baseline output must also appear in the MPI output.
        Counts may differ by worker partition rounding, but no word should be lost.
        """
        input_file = _write_input_file(tmp_path)
        baseline_counts = _count_words_baseline(input_file)

        cmd = [
            "mpirun",
            "--oversubscribe",
            "-n",
            "5",
            sys.executable,
            "-m",
            "mpj_spark.core.main_mpi",
            "--input",
            str(input_file),
            "--app",
            "wordcount",
        ]
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=str(_PROJECT_ROOT),
            timeout=120,
        )
        assert result.returncode == 0
        mpi_counts = _parse_mpi_stdout(result.stdout)

        missing = set(baseline_counts.keys()) - set(mpi_counts.keys())
        assert not missing, f"Words present in baseline but missing from MPI output: {missing}"

    def test_skipped_in_single_rank(self) -> None:
        """
        Smoke-test: this test always passes in CI (documents the skip guard).
        The actual MPI tests are guarded by @_NEEDS_MPI and skip when mpi_size < 2.
        """
        # This test documents that the suite is CI-safe.
        # The real assertions live in test_top_n_counts_match and
        # test_all_words_present_in_mpi_output, both guarded by @_NEEDS_MPI.
        assert _mpi_size >= 1, "mpi_size must be at least 1 (single process)"
