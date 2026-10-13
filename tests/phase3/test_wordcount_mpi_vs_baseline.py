"""
tests/phase3/test_wordcount_mpi_vs_baseline.py

Issue #14 - P3-08: WordCount regression test.

Validates that the MPI multi-driver WordCount (mpirun -n 5) reports the same
top-N word frequencies as a single-driver Spark baseline on one fixed input.

How it runs
-----------
Run it with PLAIN pytest.  The test starts its own `mpirun` subprocess, so
it must NOT be launched under mpirun (that would nest mpirun inside every
rank).  It is skipped automatically when:
  - `mpirun` is not on PATH (e.g. GitHub Actions without OpenMPI),
  - mpi4py is not installed,
  - pytest itself is already running inside an MPI job, or
  - SKIP_MPI_E2E=1 is set (opt-out for quick local runs).

    python -m pytest tests/phase3/test_wordcount_mpi_vs_baseline.py -v

Output format parsed
--------------------
The root prints the merged result as a block:

    Top-20 words:
      apple                             9
      cherry                            7

(padded columns, thousands separators, no colon).  Only that block is
parsed.  The root prints at most 20 words, so the fixture vocabulary must
stay under 20 words; a test asserts this.
"""

from __future__ import annotations

import collections
import importlib.util
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

_ROOT_PRINT_LIMIT = 20
_MPI_RANKS = 5
_TOP_N = 5

_WORDCOUNT_INPUT = """
apple banana cherry apple banana apple
cherry date elderberry fig cherry date
banana apple fig grape fig grape grape
apple cherry banana date elderberry apple
honeydew fig banana apple cherry cherry
date grape honeydew honeydew fig apple
elderberry banana cherry apple date date
""".strip()

_ROW_RE = re.compile(r"^\s+(\S+)\s+([\d,]+)\s*$")


def _skip_reason() -> str | None:
    if os.environ.get("SKIP_MPI_E2E") == "1":
        return "SKIP_MPI_E2E=1"
    if shutil.which("mpirun") is None:
        return "mpirun not found on PATH (OpenMPI not installed)"
    if importlib.util.find_spec("mpi4py") is None:
        return "mpi4py is not installed"
    if os.environ.get("OMPI_COMM_WORLD_SIZE") or os.environ.get("PMI_SIZE"):
        return "already inside an MPI job; run this file with plain pytest (it starts mpirun)"
    return None


_NEEDS_MPIRUN = pytest.mark.skipif(
    _skip_reason() is not None,
    reason=f"WordCount MPI end-to-end test skipped: {_skip_reason()}",
)


def _parse_top_block(stdout: str) -> dict[str, int]:
    """Parse the root's 'Top-20 words:' block into word -> count."""
    result: dict[str, int] = {}
    in_block = False
    for line in stdout.splitlines():
        if not in_block:
            if line.strip().startswith("Top-20 words"):
                in_block = True
            continue
        if not line.strip():
            continue
        match = _ROW_RE.match(line)
        if match is None:
            break
        result[match.group(1).lower()] = int(match.group(2).replace(",", ""))
    return result


def _top_n(counts: dict[str, int], n: int) -> dict[str, int]:
    return dict(collections.Counter(counts).most_common(n))


def _count_words_baseline(input_path: Path) -> dict[str, int]:
    """Single-driver Spark WordCount baseline: full word -> count dict."""
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


@pytest.fixture(scope="module")
def wordcount_runs(tmp_path_factory):
    """Run the baseline and one MPI job once; share the results."""
    input_file = tmp_path_factory.mktemp("wc") / "wordcount_input.txt"
    input_file.write_text(_WORDCOUNT_INPUT)

    baseline = _count_words_baseline(input_file)

    env = {k: v for k, v in os.environ.items() if not k.startswith(("OMPI_", "PMIX_", "PMI_"))}
    cmd = [
        "mpirun",
        "--oversubscribe",
        "-n",
        str(_MPI_RANKS),
        sys.executable,
        "-m",
        "mpj_spark.core.main_mpi",
        "--input",
        str(input_file),
        "--app",
        "wordcount",
        "--cores",
        "1",
    ]
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=str(_PROJECT_ROOT),
        timeout=300,
        env=env,
    )
    return baseline, proc


# ---------------------------------------------------------------------------
# Always-run unit tests (no MPI needed)
# ---------------------------------------------------------------------------


def test_parse_top_block_reads_root_format():
    out = (
        "noise line: 5\n"
        "\n  Top-20 words:\n"
        "    apple                                 9\n"
        "    cherry                            1,234\n"
        "\n"
        "    banana                                6\n"
        "  Total rows : 99\n"
        "    ignored                              77\n"
    )
    assert _parse_top_block(out) == {"apple": 9, "cherry": 1234, "banana": 6}


def test_parse_top_block_without_block_is_empty():
    assert _parse_top_block("word: 5\nother: 3\n") == {}


def test_fixture_vocabulary_fits_root_print_limit():
    vocab = set(_WORDCOUNT_INPUT.lower().split())
    assert len(vocab) <= _ROOT_PRINT_LIMIT


# ---------------------------------------------------------------------------
# End-to-end (needs mpirun; plain pytest)
# ---------------------------------------------------------------------------


class TestWordCountMpiVsBaseline:
    """MPI multi-driver WordCount must match the single-driver baseline."""

    @_NEEDS_MPIRUN
    def test_top_n_counts_match(self, wordcount_runs) -> None:
        baseline, proc = wordcount_runs
        assert baseline, "Baseline produced empty word counts"
        assert proc.returncode == 0, (
            f"MPI WordCount exited with code {proc.returncode}.\n"
            f"STDOUT:\n{proc.stdout[-2000:]}\nSTDERR:\n{proc.stderr[-2000:]}"
        )
        mpi_counts = _parse_top_block(proc.stdout)
        assert (
            mpi_counts
        ), f"No 'Top-20 words' block found in MPI output.\nSTDOUT:\n{proc.stdout[-2000:]}"
        baseline_top = _top_n(baseline, _TOP_N)
        mpi_top = _top_n(mpi_counts, _TOP_N)
        assert (
            baseline_top == mpi_top
        ), f"Top-{_TOP_N} MISMATCH.\n  Baseline : {baseline_top}\n  MPI      : {mpi_top}"

    @_NEEDS_MPIRUN
    def test_all_words_present_in_mpi_output(self, wordcount_runs) -> None:
        baseline, proc = wordcount_runs
        assert proc.returncode == 0
        mpi_counts = _parse_top_block(proc.stdout)
        missing = set(baseline) - set(mpi_counts)
        assert not missing, f"In baseline but missing from MPI output: {missing}"
        mismatched = {
            w: (baseline[w], mpi_counts[w]) for w in baseline if baseline[w] != mpi_counts.get(w)
        }
        assert not mismatched, f"Count mismatches (baseline, mpi): {mismatched}"
