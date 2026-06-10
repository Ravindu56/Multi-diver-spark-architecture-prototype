# ================================================================
# mpj_spark/core/main_mpi.py  -  Package-Internal MPI Entry Point
# MPJ-SPARK Multi-Driver Architecture  (mpi4py + OpenMPI)
# University of Jaffna  -  2022/E/033 & 2022/E/090
#
# PURPOSE
# -------
# Thin rank-dispatch shim.  Zero application logic lives here.
# All coordination logic lives in:
#
#   mpj_spark/core/root_mpi.py       -> run_root_mpi()  [rank 0]
#   mpj_spark/workers/worker_mpi.py  -> run_worker_mpi() [ranks 1..N]
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
#   Rank 0  = root coordinator  (calls run_root_mpi)
#   Rank 1+ = Spark driver workers  (call run_worker_mpi)
#
# BACKWARDS COMPATIBILITY
# -----------------------
# mpj_spark_mpi.py (repo root) is the Phase-2 / Phase-3 parity launcher.
# It delegates through MpiQueue / MpiEvent adapters into run_root() and
# worker_process().  That path is preserved for regression testing but is
# NO LONGER used by this canonical Phase-3+ entry point.
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
os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

from mpi4py import MPI  # noqa: E402  (must follow env setup)

# ── MPI communicator — initialised once at module load ────────────────────
comm = MPI.COMM_WORLD
rank = comm.Get_rank()  # 0 = root, 1..N-1 = workers
size = comm.Get_size()  # total ranks = 1 root + N workers


# ================================================================
# ARGUMENT PARSER  (rank 0 only)
# ================================================================


def _build_arg_parser():
    """
    Construct the CLI argument parser.
    Kept as a factory so test helpers can inspect it without
    triggering a full MPI run.
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
        "--input",
        default="./test_dataset.txt",
        help="Path to input dataset file.",
    )
    p.add_argument(
        "--generate",
        type=int,
        default=50,
        metavar="MB",
        help="Auto-generate a synthetic dataset of this size (MB) if --input not found.",
    )

    # ── Parallelism ───────────────────────────────────────────────
    p.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Informational only — actual worker count is always MPI size - 1.",
    )
    p.add_argument(
        "--cores",
        type=int,
        default=None,
        help="Override Spark local[N] core count per worker.",
    )

    # ── Application ──────────────────────────────────────────────
    p.add_argument(
        "--app",
        default="wordcount",
        choices=["wordcount", "kmeans", "logreg"],
        help="Workload to run on each Spark driver.",
    )
    p.add_argument(
        "--compare",
        action="store_true",
        help="Run a single-driver baseline and print a speedup table.",
    )
    p.add_argument(
        "--no-prewarm",
        action="store_true",
        help="Skip JVM pre-warm barrier (faster cold start, higher variance).",
    )

    # ── K-Means options ───────────────────────────────────────────
    p.add_argument("--kmeans-k", type=int, default=3)
    p.add_argument("--kmeans-iter", type=int, default=20)
    p.add_argument("--baseline-threads", type=int, default=None)
    p.add_argument("--gossip", action="store_true")
    p.add_argument("--gossip-threshold", type=float, default=1e-3)
    p.add_argument("--gossip-max-rounds", type=int, default=10)
    p.add_argument("--gossip-fanout", type=int, default=2)
    p.add_argument("--global-seed", action="store_true")
    p.add_argument("--reassign", action="store_true")

    # ── Logistic Regression options ───────────────────────────────
    p.add_argument("--logreg-iter", type=int, default=10)
    p.add_argument("--logreg-reg-param", type=float, default=0.01)
    p.add_argument("--logreg-features", type=int, default=10)

    # ── Output ───────────────────────────────────────────────────
    p.add_argument(
        "--results-dir",
        default="results",
        help="Directory for profiling CSVs and result files.",
    )

    return p


# ================================================================
# RANK 0  —  Root coordinator
# ================================================================


def _run_root(args):
    """
    Root coordinator path (rank 0).
    Delegates directly to run_root_mpi() in mpj_spark/core/root_mpi.py.

    P3-02: rank 0 is the root; no multiprocessing.Process is created.
    """
    assert rank == 0, f"_run_root() called on rank {rank} — must only be called by rank 0."
    assert size >= 2, (
        f"Need at least 2 MPI ranks (got {size}). "
        "Launch with: mpirun -np <1+N> python -m mpj_spark.core.main_mpi"
    )

    # Warn if --workers conflicts with the MPI-derived count
    if args.workers is not None and args.workers != size - 1:
        print(
            f"[main_mpi] WARNING: --workers={args.workers} ignored; "
            f"actual worker count = {size - 1} "
            f"(change -np in your mpirun command to adjust parallelism)."
        )
    num_workers = size - 1

    print(f"\n[main_mpi] MPI_COMM_WORLD size={size}  " f"root=rank-0  workers=ranks-1..{size - 1}")

    # Auto-generate dataset if the input file does not exist (rank 0 only)
    if not os.path.exists(args.input):
        print(
            f"[main_mpi] Input file '{args.input}' not found. "
            f"Auto-generating {args.generate} MB synthetic dataset ..."
        )
        _repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        if _repo_root not in sys.path:
            sys.path.insert(0, _repo_root)
        from mpj_spark_prototype_v2 import generate_test_dataset  # noqa

        args.input = generate_test_dataset(args.input, args.generate)

    # ── Delegate to the Phase-3 MPI root coordinator ──────────────
    from mpj_spark.core.root_mpi import run_root_mpi

    run_root_mpi(
        comm=comm,
        input_file=args.input,
        num_workers=num_workers,
        compare=args.compare,
        prewarm=not args.no_prewarm,
        cores_override=args.cores,
        app=args.app,
        kmeans_k=args.kmeans_k,
        kmeans_iter=args.kmeans_iter,
        baseline_threads=args.baseline_threads,
        use_gossip=args.gossip,
        gossip_threshold=args.gossip_threshold,
        gossip_max_rounds=args.gossip_max_rounds,
        gossip_fanout=args.gossip_fanout,
        use_global_seed=args.global_seed,
        use_reassign=args.reassign,
        logreg_iter=args.logreg_iter,
        logreg_reg_param=args.logreg_reg_param,
        logreg_features=args.logreg_features,
        results_dir=args.results_dir,
    )


# ================================================================
# RANKS 1..N  —  Spark driver workers
# ================================================================


def _run_worker():
    """
    Worker path (ranks 1..N).
    Workers do not parse CLI args.  They delegate directly to
    run_worker_mpi() in mpj_spark/workers/worker_mpi.py.

    P3-02: each worker rank is an independent Spark driver process;
    no multiprocessing.Process is created here.
    """
    assert rank >= 1, f"_run_worker() called on rank {rank} — must only be called by rank >= 1."

    from mpj_spark.workers.worker_mpi import run_worker_mpi

    run_worker_mpi(comm)


# ================================================================
# MODULE ENTRY POINT
# ================================================================

if __name__ == "__main__":
    # Ensure the repo root is on sys.path (needed for data-generation
    # helper and backwards-compat parity imports).
    _repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if _repo_root not in sys.path:
        sys.path.insert(0, _repo_root)

    if rank == 0:
        # ── Root: parse CLI, then run ─────────────────────────────
        # IMPORTANT: Only rank 0 calls argparse.parse_args().
        # Worker ranks must NOT call argparse — a SystemExit on any
        # rank while others are blocked on comm.recv() causes a deadlock.
        parser = _build_arg_parser()
        args = parser.parse_args()
        _run_root(args)
    else:
        # ── Workers: no CLI parsing, receive config from root ─────
        _run_worker()
