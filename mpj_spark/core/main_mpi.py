# ================================================================
# mpj_spark/core/main_mpi.py  -  Package-Internal MPI Entry Point
# MPJ-SPARK Multi-Driver Architecture  (mpi4py + OpenMPI)
# University of Jaffna  -  2022/E/033 & 2022/E/090
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

from mpi4py import MPI  # noqa: E402

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


def _build_arg_parser():
    import argparse

    from mpj_spark.core.sync_modes import MODE_PS_SYNC_FEDAVG_MPI

    p = argparse.ArgumentParser(
        prog="python -m mpj_spark.core.main_mpi",
        description=(
            "MPJ-SPARK Phase 3 — mpi4py multi-driver entry point.\n"
            "Launch via: mpirun --oversubscribe -np <1+N> "
            "python -m mpj_spark.core.main_mpi [options]"
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument("--input", default="./test_dataset.txt", help="Path to input dataset file.")
    p.add_argument(
        "--generate",
        type=int,
        default=50,
        metavar="MB",
        help="Auto-generate synthetic dataset size (MB).",
    )
    p.add_argument(
        "--workers", type=int, default=None, help="Informational worker count (size - 1)."
    )
    p.add_argument(
        "--cores", type=int, default=None, help="Override Spark local[N] core count per worker."
    )
    p.add_argument(
        "--app", default="wordcount", choices=["wordcount", "kmeans", "logreg"], help="Workload."
    )
    p.add_argument("--compare", action="store_true", help="Run single-driver baseline.")
    p.add_argument("--no-prewarm", action="store_true", help="Skip JVM pre-warm barrier.")

    p.add_argument(
        "--sync-mode",
        type=str,
        default=MODE_PS_SYNC_FEDAVG_MPI,
        help="Cross-driver synchronization mode (ps_sync_fedavg_mpi | ps_sync_fedavg_queue | allreduce_mpi | ps_async | hybrid_ps_allreduce | gossip | none)",
    )

    p.add_argument(
        "--gossip-fanout",
        type=int,
        default=None,
        help="Ring distance contacted per gossip round (default: inherit the run_root_mpi gossip_fanout default).",
    )

    p.add_argument("--kmeans-k", type=int, default=3)
    p.add_argument("--kmeans-iter", type=int, default=20)
    p.add_argument("--baseline-threads", type=int, default=None)
    p.add_argument("--gossip", action="store_true")
    p.add_argument("--gossip-threshold", type=float, default=1e-3)
    p.add_argument("--gossip-max-rounds", type=int, default=10)
    p.add_argument("--gossip-fanout", type=int, default=2)
    p.add_argument("--global-seed", action="store_true")
    p.add_argument("--reassign", action="store_true")

    p.add_argument("--logreg-iter", type=int, default=10)
    p.add_argument("--logreg-reg-param", type=float, default=0.01)
    p.add_argument("--logreg-features", type=int, default=10)
    p.add_argument("--results-dir", default="results", help="Directory for profiling CSVs.")

    return p


def _run_root(args):
    assert rank == 0, f"_run_root() called on rank {rank}"
    assert size >= 2, f"Need at least 2 MPI ranks (got {size})."

    num_workers = size - 1

    from mpj_spark.core.root_mpi import run_root_mpi
    from mpj_spark.core.sync_modes import normalize_sync_mode

    sync_mode = normalize_sync_mode(args.sync_mode)

    run_root_mpi(
        comm=comm,
        input_file=args.input,
        num_workers=num_workers,
        compare=args.compare,
        prewarm=not args.no_prewarm,
        cores_override=args.cores,
        app=args.app,
        sync_mode=sync_mode,
        **({"gossip_fanout": args.gossip_fanout} if args.gossip_fanout is not None else {}),
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


def _run_worker():
    assert rank >= 1, f"_run_worker() called on rank {rank}"
    from mpj_spark.workers.worker_mpi import run_worker_mpi

    run_worker_mpi(comm)


if __name__ == "__main__":
    _repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if _repo_root not in sys.path:
        sys.path.insert(0, _repo_root)

    if rank == 0:
        parser = _build_arg_parser()
        args = parser.parse_args()
        _run_root(args)
    else:
        _run_worker()
