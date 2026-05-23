# ================================================================
# mpj_spark/core/main_mpi.py  -  Package-Internal MPI Entry Point
# MPJ-SPARK Multi-Driver Architecture  (mpi4py + OpenMPI)
# University of Jaffna  -  2022/E/033 & 2022/E/090
#
# PURPOSE
# -------
# This is the *modular* MPI entry point that lives inside the package
# hierarchy so the framework can be launched as a proper Python module:
#
#   mpirun -n <1+N> python -m mpj_spark.core.main_mpi [options]
#
# It is intentionally a thin rank-dispatch shim — zero application
# logic lives here.  All MPI transport adapters (MpiQueue, MpiEvent,
# MpiRootFanoutQueue) and all coordination logic live in:
#
#   mpj_spark_mpi.py          -> root_main() / worker_main()
#   mpj_spark/core/root_process.py  -> run_root()
#   mpj_spark/workers/worker_process.py -> worker_process()
#
# RELATIONSHIP TO mpj_spark_mpi.py
# ---------------------------------
# mpj_spark_mpi.py  (repo root)  - flat-root launcher; kept for
#                                   backwards-compatibility and for
#                                   Phase 2 → Phase 3 parity testing.
# mpj_spark/core/main_mpi.py     - package-internal launcher; the
#                                   canonical Phase 3+ entry point.
# Both launchers delegate to the *same* root_main() / worker_main()
# functions, so they are always functionally identical.
#
# LAUNCH
# ------
#   # Single-node (dev / lab / parity test)
#   mpirun --oversubscribe -np 3 python -m mpj_spark.core.main_mpi
#
#   # Multi-node (Docker Swarm / Phase 4+)
#   mpirun --hostfile hostfile.txt -np 3 python -m mpj_spark.core.main_mpi \
#       --input /shared/data/dataset.csv --app kmeans --kmeans-k 5
#
# RANK ROLES
# ----------
#   Rank 0  = root coordinator  (MPI_COMM_WORLD replaces multiprocessing.Process)
#   Rank 1+ = independent Spark driver workers
#
# P3-02 ACCEPTANCE CRITERION
# ---------------------------
#   MPI_COMM_WORLD replaces multiprocessing.Process; root is rank 0.
#   Verified by: assert rank == 0 in root path; no multiprocessing.Process
#   instantiation anywhere in this module.
# ================================================================

import os
import sys

# ── JVM / PySpark environment setup (must happen before any Spark import) ──
os.environ.setdefault(
    "JAVA_TOOL_OPTIONS",
    "--add-opens=java.base/sun.nio.ch=ALL-UNNAMED "
    "--add-opens=java.base/java.nio=ALL-UNNAMED "
    "--add-opens=java.base/java.lang=ALL-UNNAMED "
    "--add-opens=java.base/java.util=ALL-UNNAMED "
    "-Djava.security.manager=allow",
)
os.environ["PYSPARK_PYTHON"]        = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

from mpi4py import MPI  # noqa: E402  (must follow env setup)

# ── MPI communicator — initialised once at module load ────────────────────
comm = MPI.COMM_WORLD
rank = comm.Get_rank()   # 0 = root, 1..N-1 = workers
size = comm.Get_size()   # total ranks = 1 root + N workers


# ================================================================
# RANK DISPATCH
# ================================================================

def _build_arg_parser():
    """
    Construct the argument parser shared by both launchers.
    Kept as a factory function so the parser can be imported and
    reused by test helpers without triggering a full MPI run.
    """
    import argparse
    p = argparse.ArgumentParser(
        prog="python -m mpj_spark.core.main_mpi",
        description=(
            "MPJ-SPARK Phase 3 — mpi4py multi-driver entry point.\n"
            "Launch via: mpirun --oversubscribe -np <1+N> "
            "python -m mpj_spark.core.main_mpi [options]"
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Dataset ───────────────────────────────────────────────────
    p.add_argument(
        "--input", default="./test_dataset.txt",
        help="Path to input dataset file.",
    )
    p.add_argument(
        "--generate", type=int, default=50, metavar="MB",
        help="Generate a synthetic dataset of this size (MB) if --input is not found.",
    )

    # ── Parallelism ───────────────────────────────────────────────
    p.add_argument(
        "--workers", type=int, default=None,
        help="Number of worker ranks to use. Defaults to MPI size - 1.",
    )
    p.add_argument(
        "--cores", type=int, default=None,
        help="Override Spark local[N] core count per worker.",
    )

    # ── Application ──────────────────────────────────────────────
    p.add_argument(
        "--app", default="wordcount",
        choices=["wordcount", "kmeans", "logreg"],
        help="Workload to run on each Spark driver.",
    )
    p.add_argument("--compare",     action="store_true",
                   help="Run a single-driver baseline and print a speedup comparison table.")
    p.add_argument("--no-prewarm",  action="store_true",
                   help="Skip JVM pre-warm phase (faster cold start, higher variance).")

    # ── K-Means options ───────────────────────────────────────────
    p.add_argument("--kmeans-k",        type=int,   default=3)
    p.add_argument("--kmeans-iter",     type=int,   default=20)
    p.add_argument("--baseline-threads",type=int,   default=None)
    p.add_argument("--gossip",          action="store_true")
    p.add_argument("--gossip-threshold",type=float, default=1e-3)
    p.add_argument("--gossip-max-rounds",type=int,  default=10)
    p.add_argument("--gossip-fanout",   type=int,   default=2)
    p.add_argument("--global-seed",     action="store_true")
    p.add_argument("--reassign",        action="store_true")

    # ── Logistic Regression options ───────────────────────────────
    p.add_argument("--logreg-iter",      type=int,   default=10)
    p.add_argument("--logreg-reg-param", type=float, default=0.01)
    p.add_argument("--logreg-features",  type=int,   default=10)

    # ── Output ───────────────────────────────────────────────────
    p.add_argument("--results-dir", default="results",
                   help="Directory where profiling CSVs and results are written.")

    return p


def _run_root(args):
    """
    Root coordinator path (rank 0).

    Validates rank, then delegates to root_main() in mpj_spark_mpi.py
    which injects MPI transport adapters into run_root().

    P3-02: rank 0 is the root; MPI_COMM_WORLD provides process identity.
    No multiprocessing.Process is created here.
    """
    assert rank == 0, (
        f"_run_root() called on rank {rank} — must only be called by rank 0."
    )
    assert size >= 2, (
        f"Need at least 2 MPI ranks (got {size}). "
        "Launch with: mpirun -np <1+N> python -m mpj_spark.core.main_mpi"
    )

    # Resolve worker count: CLI arg or MPI-derived
    if args.workers is not None and args.workers != size - 1:
        print(
            f"[main_mpi] WARNING: --workers={args.workers} ignored; "
            f"using MPI-derived worker count = {size - 1} "
            f"(change -np in your mpirun command to adjust parallelism)."
        )
    args.workers = size - 1

    print(
        f"\n[main_mpi] MPI_COMM_WORLD size={size}  "
        f"root=rank-0  workers=ranks-1..{size-1}"
    )

    # Auto-generate dataset if the input file does not exist
    if not os.path.exists(args.input):
        print(
            f"[main_mpi] Input file '{args.input}' not found. "
            f"Auto-generating {args.generate} MB synthetic dataset..."
        )
        # Import from repo-root prototype script (kept for data generation)
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
        from mpj_spark_prototype_v2 import generate_test_dataset  # noqa
        args.input = generate_test_dataset(args.input, args.generate)

    # Delegate to transport-layer root_main()
    from mpj_spark_mpi import root_main  # noqa (repo root, on sys.path)
    root_main(args)


def _run_worker():
    """
    Worker path (ranks 1..N).

    Workers have no CLI args to parse.  They go straight to work by
    delegating to worker_main() in mpj_spark_mpi.py, which injects MPI
    transport adapters into worker_process().

    P3-02: each worker rank is an independent Spark driver process;
    no multiprocessing.Process is created here.
    """
    assert rank >= 1, (
        f"_run_worker() called on rank {rank} — must only be called by rank >= 1."
    )

    from mpj_spark_mpi import worker_main  # noqa (repo root, on sys.path)
    worker_main()


# ================================================================
# MODULE ENTRY POINT
# ================================================================

if __name__ == "__main__":
    # Ensure the repo root is on sys.path so mpj_spark_mpi.py is importable
    # when launched as `python -m mpj_spark.core.main_mpi`.
    _repo_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..")
    )
    if _repo_root not in sys.path:
        sys.path.insert(0, _repo_root)

    if rank == 0:
        # ── Root: parse CLI and run ───────────────────────────────
        # Only rank 0 parses arguments.  Worker ranks must NOT call
        # argparse — doing so would cause a deadlock if argument
        # parsing raises SystemExit on any rank while others are
        # already blocked on comm.recv().
        parser = _build_arg_parser()
        args   = parser.parse_args()
        _run_root(args)
    else:
        # ── Workers: no CLI parsing, go straight to work ─────────
        _run_worker()
