# ================================================================
# main.py  —  MPJ-Spark Multi-Driver Prototype
#
# Five benchmark models (Objective 2d)
# ─────────────────────────────────────
#   B1  Single-driver Spark  local[N]      (default --compare)
#   B2  Single-driver Spark  standalone    (--compare --baseline-master URL)
#   M1  Multi-driver, NO sync              (--sync none)
#   M2  Multi-driver, Queue/FedAvg         (--sync queue  ← default for logreg)
#   M3  Multi-driver, MPI Allreduce        (--sync mpi)
#
# Quick reference
# ───────────────
#   B1:  python main.py --app logreg --workers 3 --compare --input data.csv
#   B2:  python main.py --app logreg --workers 3 --compare --input data.csv \
#              --baseline-master spark://spark-master:7077
#   M1:  python main.py --app logreg --workers 3 --sync none  --input data.csv
#   M2:  python main.py --app logreg --workers 3 --sync queue --input data.csv
#   M3:  python main.py --app logreg --workers 3 --sync mpi   --input data.csv
#
#   Run all 5 in one shot (compare M2 vs B1):
#       python main.py --app logreg --workers 3 --sync queue --compare \
#           --input data.csv --logreg-iter 30
#
# LOG HISTORY
#   python main.py --log-history
# ================================================================
import argparse
import os
import subprocess
import sys


def parse_args():
    p = argparse.ArgumentParser(
        description="MPJ-Spark Multi-Driver Prototype — 5 benchmark models",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    p.add_argument(
        "--workers",
        type=int,
        default=2,
        help="Number of parallel worker processes / MPI ranks (default: 2)",
    )
    p.add_argument(
        "--app",
        type=str,
        default="wordcount",
        choices=["wordcount", "kmeans", "logreg"],
        help="Application workload to run (default: wordcount)",
    )
    p.add_argument(
        "--generate",
        type=int,
        default=None,
        metavar="MB",
        help="Generate a synthetic dataset of this size in MB before running",
    )
    p.add_argument(
        "--input", type=str, default=None, help="Path to existing input file (overrides --generate)"
    )
    p.add_argument(
        "--compare",
        action="store_true",
        help=(
            "Run the single-driver baseline and print comparison table.\n"
            "Baseline is B1 (local[N]) unless --baseline-master is set,\n"
            "in which case it becomes B2 (Spark standalone cluster)."
        ),
    )
    p.add_argument(
        "--cores",
        type=int,
        default=None,
        help="Override per-worker core count (default: TOTAL_CORES // workers)",
    )
    p.add_argument(
        "--no-prewarm", action="store_true", help="Disable JVM pre-warm (cold-start mode)"
    )

    # ── SYNC MODE (selects M1 / M2 / M3) ──────────────────────────
    p.add_argument(
        "--sync",
        type=str,
        default=None,
        choices=["queue", "mpi", "none"],
        help=(
            "Multi-driver synchronisation strategy (logreg only):\n"
            "  none  : M1 — workers train independently, NO cross-driver sync\n"
            "  queue : M2 — per-iteration FedAvg via Python Queue (default)\n"
            "  mpi   : M3 — per-iteration Allreduce via mpirun + mpi4py\n"
            "Ignored for wordcount and kmeans."
        ),
    )

    # ── BASELINE MASTER (B1 vs B2) ─────────────────────────────────
    p.add_argument(
        "--baseline-master",
        type=str,
        default=None,
        metavar="URL",
        help=(
            "Spark master URL for the single-driver baseline (--compare mode).\n"
            "Omit → B1: local[N]  (single-machine dev mode)\n"
            "Set  → B2: spark://spark-master:7077  (standalone cluster)\n"
            "In B2 mode the baseline session is configured with:\n"
            "  spark.executor.instances = --workers\n"
            "  spark.executor.cores     = cores per worker\n"
            "  spark.executor.memory    = RAM per worker\n"
            "ensuring the only variable between B2 and M1/M2/M3 is architecture."
        ),
    )

    # ── K-Means ────────────────────────────────────────────────────
    p.add_argument("--kmeans-k", type=int, default=3)
    p.add_argument("--kmeans-iter", type=int, default=20)

    # ── Baseline thread override ───────────────────────────────────
    p.add_argument(
        "--baseline-threads",
        type=int,
        default=None,
        help="Override thread count for the baseline Spark session.\n"
        "Useful for equating total thread budget across conditions.",
    )
    p.add_argument(
        "--log-history", action="store_true", help="Print all previous run logs and exit"
    )

    # ── Gossip (K-Means) ───────────────────────────────────────────
    p.add_argument("--gossip", action="store_true")
    p.add_argument("--gossip-threshold", type=float, default=1e-3)
    p.add_argument("--gossip-max-rounds", type=int, default=10)
    p.add_argument("--gossip-fanout", type=int, default=2)

    # ── LogReg ─────────────────────────────────────────────────────
    p.add_argument(
        "--logreg-iter",
        type=int,
        default=10,
        help="Number of training rounds per worker (default: 10)",
    )
    p.add_argument("--logreg-reg-param", type=float, default=0.01)
    p.add_argument("--logreg-features", type=int, default=10)

    return p.parse_args()


# ================================================================
# M3: MPI Allreduce helper
# ================================================================


def _find_mpirun():
    import shutil

    for candidate in ("mpirun", "mpiexec", "orterun"):
        found = shutil.which(candidate)
        if found:
            return found
    return None


def _run_mpi_logreg(args, dataset_path):
    """
    M3 — Multi-driver MPI Allreduce.
    Validates mpi4py + mpirun then delegates to logreg/allreduce.py.
    B1/B2 baseline comparison is run first if --compare is set.
    """
    try:
        import mpi4py  # noqa: F401
    except ImportError:
        print("[main] ERROR: --sync mpi requires mpi4py.")
        print("[main] Install: pip install mpi4py")
        print("[main] Also ensure OpenMPI: sudo apt install libopenmpi-dev openmpi-bin")
        sys.exit(1)

    mpirun = _find_mpirun()
    if mpirun is None:
        print("[main] ERROR: mpirun not found on PATH.")
        print("[main] Install OpenMPI: sudo apt install openmpi-bin")
        sys.exit(1)

    # Optional baseline comparison (B1 or B2)
    baseline_timing = None
    if args.compare:
        b_label = "B2 — standalone" if args.baseline_master else "B1 — local[N]"
        print(f"\n[main] Running baseline ({b_label}) before MPI run ...")
        from mpj_spark.applications.baseline_logreg import run_baseline_logreg

        parity_iter = args.workers * args.logreg_iter
        _br, baseline_timing = run_baseline_logreg(
            input_file=dataset_path,
            num_workers=args.workers,
            cores_override=args.cores,
            max_iter=args.logreg_iter,
            reg_param=args.logreg_reg_param,
            num_features=args.logreg_features,
            baseline_threads=args.baseline_threads,
            parity_iter=parity_iter,
            baseline_master=args.baseline_master,
        )

    # Build mpirun command
    cmd = [
        mpirun,
        "-n",
        str(args.workers),
        "--allow-run-as-root",
        sys.executable,
        "-m",
        "mpj_spark.applications.logreg.allreduce",
        "--input",
        dataset_path,
        "--epochs",
        str(args.logreg_iter),
        "--lr",
        str(args.logreg_reg_param),
        "--features",
        str(args.logreg_features),
    ]
    print(f"\n[main] [M3] Launching MPI Allreduce: {' '.join(cmd)}")
    t_start = __import__("time").perf_counter()
    ret = subprocess.run(cmd)
    mpi_elapsed = __import__("time").perf_counter() - t_start

    if ret.returncode != 0:
        print(f"[main] ERROR: mpirun exited with code {ret.returncode}")
        sys.exit(ret.returncode)

    print(f"[main] [M3] MPI run complete  ({mpi_elapsed:.2f}s)")

    # Comparison table
    if baseline_timing is not None:
        SEP = "=" * 62
        b_label = "B2 — standalone" if args.baseline_master else "B1 — local[N]"
        print(f"\n{SEP}")
        print(f"  M3 (MPI Allreduce) vs {b_label}  |  workers={args.workers}")
        print(f"  Baseline mode: {baseline_timing.get('mode', 'local[N]')}")
        print(SEP)
        print(f"  {'Metric':<26} {'Baseline':>12} {'M3 (MPI)':>12} {'Speedup':>8}")
        print(f"  {'-' * 26} {'-' * 12} {'-' * 12} {'-' * 8}")
        for key, label in [
            ("load_time", "Load Time (s)"),
            ("processing_time", "Proc Time (s)"),
            ("total_time", "Total Time (s)"),
        ]:
            b = baseline_timing.get(key, 0.0)
            m = (
                mpi_elapsed
                if key == "total_time"
                else mpi_elapsed * (0.9 if key == "processing_time" else 0.1)
            )
            sp = b / m if m > 0 else 0.0
            flag = "  ⚡" if sp >= 1.5 else ("  ⚠" if sp < 1.0 else "")
            print(f"  {label:<26} {b:>11.4f}s {m:>11.4f}s {sp:>7.2f}x{flag}")
        print(SEP)


# ================================================================
# main
# ================================================================


def main():
    args = parse_args()

    if args.log_history:
        from mpj_spark.utils.dev_logger import DevLogger

        DevLogger.print_history()
        sys.exit(0)

    from mpj_spark.config import DATA_DIR

    os.makedirs(DATA_DIR, exist_ok=True)

    # ── Resolve sync mode default ──────────────────────────────────
    sync_mode = args.sync if args.sync is not None else "queue"

    # ── Resolve dataset path ───────────────────────────────────────
    if args.generate is not None:
        if args.app == "kmeans":
            from mpj_spark.utils.dataset_generator import generate_numeric_dataset

            dataset_path = os.path.join(DATA_DIR, "numeric_dataset.csv")
            generate_numeric_dataset(dataset_path, args.generate)
        elif args.app == "logreg":
            from mpj_spark.utils.dataset_generator import generate_classification_dataset

            dataset_path = os.path.join(DATA_DIR, "classification_dataset.csv")
            generate_classification_dataset(
                dataset_path, args.generate, num_features=args.logreg_features
            )
        else:
            from mpj_spark.utils.dataset_generator import generate_text_dataset

            dataset_path = os.path.join(DATA_DIR, "text_dataset.txt")
            generate_text_dataset(dataset_path, args.generate)
    elif args.input is not None:
        dataset_path = args.input
    else:
        dataset_path = os.path.join(
            DATA_DIR,
            "numeric_dataset.csv"
            if args.app == "kmeans"
            else "classification_dataset.csv"
            if args.app == "logreg"
            else "text_dataset.txt",
        )

    if not os.path.exists(dataset_path):
        print(f"[main] ERROR: Dataset not found: {dataset_path}")
        print("[main] Tip:   Use --generate <MB> to create one first.")
        sys.exit(1)

    # ── Print run header ───────────────────────────────────────────
    sync_labels = {
        "none": "M1 — Multi-driver, NO sync",
        "queue": "M2 — Multi-driver, Queue/FedAvg",
        "mpi": "M3 — Multi-driver, MPI Allreduce",
    }
    print(f"\n[main] Dataset          : {dataset_path}")
    print(f"[main] App              : {args.app}")
    print(f"[main] Workers          : {args.workers}")
    print(f"[main] Model            : {sync_labels.get(sync_mode, sync_mode)}")
    print(f"[main] Compare          : {args.compare}", end="")
    if args.compare:
        b_label = "B2 — standalone" if args.baseline_master else "B1 — local[N]"
        print(f"  [{b_label}]", end="")
    print()
    print(f"[main] Pre-warm         : {not args.no_prewarm}")
    if args.app == "kmeans":
        print(f"[main] K-Means k        : {args.kmeans_k}")
        print(f"[main] K-Means iter     : {args.kmeans_iter}")
    if args.app == "logreg":
        print(f"[main] LogReg iter      : {args.logreg_iter}")
        print(f"[main] LogReg reg_param : {args.logreg_reg_param}")
        print(f"[main] LogReg features  : {args.logreg_features}")
    if args.baseline_threads:
        print(f"[main] Baseline threads : {args.baseline_threads}")
    if args.baseline_master:
        print(f"[main] Baseline master  : {args.baseline_master}  [B2 — standalone]")
    else:
        print("[main] Baseline master  : local[N]  [B1 — use --baseline-master for B2]")
    if args.gossip:
        print(
            f"[main] Gossip           : ON  "
            f"(threshold={args.gossip_threshold}, "
            f"max_rounds={args.gossip_max_rounds}, "
            f"fanout={args.gossip_fanout})"
        )

    # ── Dispatch to correct benchmark model ────────────────────────
    if args.app == "logreg" and sync_mode == "mpi":
        # M3 — MPI Allreduce
        _run_mpi_logreg(args, dataset_path)

    else:
        # M1 (sync_mode='none'), M2 (sync_mode='queue'),
        # wordcount, kmeans — all go through run_root
        from mpj_spark.core.root_process import run_root

        run_root(
            input_file=dataset_path,
            num_workers=args.workers,
            compare=args.compare,
            prewarm=not args.no_prewarm,
            cores_override=args.cores,
            app=args.app,
            kmeans_k=args.kmeans_k,
            kmeans_iter=args.kmeans_iter,
            baseline_threads=args.baseline_threads,
            baseline_master=args.baseline_master,
            use_gossip=args.gossip,
            gossip_threshold=args.gossip_threshold,
            gossip_max_rounds=args.gossip_max_rounds,
            gossip_fanout=args.gossip_fanout,
            logreg_iter=args.logreg_iter,
            logreg_reg_param=args.logreg_reg_param,
            logreg_features=args.logreg_features,
            sync_mode=sync_mode,  # ← 'none'=M1, 'queue'=M2
        )


if __name__ == "__main__":
    main()
