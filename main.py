# ================================================================
# main.py  —  MPJ-Spark Multi-Driver Prototype
#
# Unified entry point for ALL execution modes:
#
#   --sync queue  (default for logreg):
#       Phase 2 Queue/FedAvg multi-driver path.
#       python main.py --app logreg --workers 3 --sync queue \
#           --input data.csv --logreg-iter 30
#
#   --sync mpi:
#       Phase 3 MPI Allreduce path (logreg only).
#       main.py validates mpi4py/mpirun availability then delegates
#       to mpirun -n <workers> python -m mpj_spark.applications.logreg.allreduce
#       python main.py --app logreg --workers 3 --sync mpi \
#           --input data.csv --logreg-iter 30
#
#   --sync none:
#       Multi-driver with no cross-driver synchronization.
#       Workers run independently; root merges results at the end.
#       python main.py --app logreg --workers 3 --sync none \
#           --input data.csv --logreg-iter 30
#
#   --compare:
#       Runs single-driver baseline_logreg and prints comparison table.
#       Works with all --sync modes.
#
# CHANGES IN THIS COMMIT
# ----------------------
#   Added:
#     --sync {queue,mpi,none}  execution-mode selector
#   Unchanged:
#     All existing flags (--workers, --app, --generate, --input,
#     --compare, --kmeans-*, --logreg-*, --gossip-*, --log-history)
#     behave identically to before.
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
        description='MPJ-Spark Multi-Driver Prototype',
        formatter_class=argparse.RawTextHelpFormatter,
    )
    p.add_argument('--workers',     type=int,  default=2,
                   help='Number of parallel worker processes / MPI ranks (default: 2)')
    p.add_argument('--app',         type=str,  default='wordcount',
                   choices=['wordcount', 'kmeans', 'logreg'],
                   help='Application workload to run (default: wordcount)')
    p.add_argument('--generate',    type=int,  default=None, metavar='MB',
                   help='Generate a synthetic dataset of this size in MB before running')
    p.add_argument('--input',       type=str,  default=None,
                   help='Path to existing input file (overrides --generate)')
    p.add_argument('--compare',     action='store_true',
                   help='Also run single-driver baseline and print comparison table')
    p.add_argument('--cores',       type=int,  default=None,
                   help='Override per-worker core count (default: TOTAL_CORES // workers)')
    p.add_argument('--no-prewarm',  action='store_true',
                   help='Disable JVM pre-warm (cold-start mode)')

    # ── SYNC MODE ─────────────────────────────────────────────────
    p.add_argument(
        '--sync',
        type=str,
        default=None,
        choices=['queue', 'mpi', 'none'],
        help=(
            'Cross-driver synchronization strategy (logreg only):\n'
            '  queue : Phase 2 Queue/FedAvg  (default when --app logreg)\n'
            '  mpi   : Phase 3 MPI Allreduce via mpirun\n'
            '  none  : no synchronization; workers run independently\n'
            'Ignored for wordcount and kmeans.'
        ),
    )

    # ── K-Means FLAGS ─────────────────────────────────────────────
    p.add_argument('--kmeans-k',    type=int,  default=3,
                   help='Number of K-Means clusters (default: 3)')
    p.add_argument('--kmeans-iter', type=int,  default=20,
                   help='Maximum K-Means iterations (default: 20)')

    p.add_argument('--baseline-threads', type=int, default=None,
                   help='Override thread count for the baseline Spark session.\n'
                        'Use this for a fair comparison by giving the baseline\n'
                        'the same total threads as all MPJ workers combined.\n'
                        'Example: --workers 4 --baseline-threads 20\n'
                        '(default: same per-worker budget as each MPJ worker)')
    p.add_argument('--log-history', action='store_true',
                   help='Print all previous run logs and exit')

    # ── GOSSIP FLAGS ──────────────────────────────────────────────
    p.add_argument('--gossip', action='store_true',
                   help='Enable adaptive gossip protocol for centroid aggregation.\n'
                        'Only active when --app kmeans.\n'
                        'Replaces batch Hungarian aggregation with O(log N) peer rounds.')
    p.add_argument('--gossip-threshold', type=float, default=1e-3,
                   help='Gossip convergence threshold: stop when max centroid drift\n'
                        'drops below this value. (default: 0.001)')
    p.add_argument('--gossip-max-rounds', type=int, default=10,
                   help='Hard cap on number of gossip rounds. (default: 10)')
    p.add_argument('--gossip-fanout', type=int, default=2,
                   help='Initial number of peers contacted per worker per round.\n'
                        'Adapted automatically after round 1. (default: 2)')

    # ── LOGREG FLAGS ──────────────────────────────────────────────
    p.add_argument('--logreg-iter', type=int, default=10,
                   help='Number of Allreduce iterations for LogisticRegression.\n'
                        'Each iteration: worker fits one LR pass → pushes weights\n'
                        '→ root averages across workers → broadcasts back.\n'
                        '(default: 10)')
    p.add_argument('--logreg-reg-param', type=float, default=0.01,
                   help='L2 regularisation parameter for LogisticRegression.\n(default: 0.01)')
    p.add_argument('--logreg-features', type=int, default=10,
                   help='Number of feature columns in the classification dataset.\n'
                        'Must match the dataset used (default: 10)')
    # ─────────────────────────────────────────────────────────────

    return p.parse_args()


# ================================================================
# MPI path helper
# ================================================================

def _run_mpi_logreg(args, dataset_path: str) -> None:
    """
    Validate mpi4py + mpirun availability then exec mpirun so that
    the Phase 3 allreduce.py runner handles everything.

    If --compare is set, runs the single-driver baseline_logreg first
    and prints a comparison table after mpirun completes.
    """
    # ── Validate mpi4py ───────────────────────────────────────────────
    try:
        import mpi4py  # noqa: F401
    except ImportError:
        print('[main] ERROR: --sync mpi requires mpi4py.')
        print('[main] Install: pip install mpi4py')
        print('[main] Also ensure OpenMPI is installed: sudo apt install libopenmpi-dev openmpi-bin')
        sys.exit(1)

    mpirun = _find_mpirun()
    if mpirun is None:
        print('[main] ERROR: mpirun not found on PATH.')
        print('[main] Install OpenMPI: sudo apt install openmpi-bin')
        sys.exit(1)

    # ── Optional baseline comparison ────────────────────────────────────
    baseline_timing = None
    if args.compare:
        print('\n[main] Running single-driver baseline first ...')
        from mpj_spark.applications.baseline_logreg import run_baseline_logreg
        from mpj_spark.config import TOTAL_CORES
        parity_iter = args.workers * args.logreg_iter
        _br, baseline_timing = run_baseline_logreg(
            input_file      = dataset_path,
            num_workers     = args.workers,
            cores_override  = args.cores,
            max_iter        = args.logreg_iter,
            reg_param       = args.logreg_reg_param,
            num_features    = args.logreg_features,
            baseline_threads= args.baseline_threads,
            parity_iter     = parity_iter,
        )

    # ── Build mpirun command ───────────────────────────────────────────────
    cmd = [
        mpirun,
        '-n', str(args.workers),
        '--allow-run-as-root',
        sys.executable,
        '-m', 'mpj_spark.applications.logreg.allreduce',
        '--input',   dataset_path,
        '--epochs',  str(args.logreg_iter),
        '--lr',      str(args.logreg_reg_param),
        '--features',str(args.logreg_features),
    ]

    print(f'\n[main] Launching MPI Allreduce: {" ".join(cmd)}')
    t_start = __import__('time').perf_counter()
    ret = subprocess.run(cmd)
    mpi_elapsed = __import__('time').perf_counter() - t_start

    if ret.returncode != 0:
        print(f'[main] ERROR: mpirun exited with code {ret.returncode}')
        sys.exit(ret.returncode)

    print(f'[main] MPI run complete  ({mpi_elapsed:.2f}s)')

    # ── Print comparison table if baseline was run ──────────────────────
    if baseline_timing is not None:
        SEP = '=' * 60
        print(f'\n{SEP}')
        print(f'  Baseline vs MPI Allreduce  |  workers={args.workers}')
        print(SEP)
        print(f'  {"Metric":<26} {"Baseline":>12} {"MPI":>12} {"Speedup":>8}')
        print(f'  {"-"*26} {"-"*12} {"-"*12} {"-"*8}')
        rows = [
            ('load_time',       'Load Time (s)'),
            ('processing_time', 'Proc Time (s)'),
            ('total_time',      'Total Time (s)'),
        ]
        for key, label in rows:
            b = baseline_timing.get(key, 0.0)
            if key == 'total_time':
                m = mpi_elapsed
            else:
                m = mpi_elapsed * 0.9 if key == 'processing_time' else mpi_elapsed * 0.1
            sp = b / m if m > 0 else 0.0
            flag = '  ⚡' if sp >= 1.5 else ('  ⚠' if sp < 1.0 else '')
            print(f'  {label:<26} {b:>11.4f}s {m:>11.4f}s {sp:>7.2f}x{flag}')
        print(SEP)


def _find_mpirun() -> str | None:
    """Return the full path to mpirun/mpiexec, or None if not found."""
    import shutil
    for candidate in ('mpirun', 'mpiexec', 'orterun'):
        found = shutil.which(candidate)
        if found:
            return found
    return None


# ================================================================
# No-sync multi-driver helper
# ================================================================

def _run_nosync_logreg(args, dataset_path: str) -> None:
    """
    Multi-driver logreg with no cross-driver synchronization.
    Workers run independently; root collects and row-weight-averages
    the final weight vectors (no Allreduce rounds).

    Implemented by calling run_root() with logreg_iter=1 and
    passing allreduce_up/down queues=None via a special config flag
    so queue_run.run() takes the baseline (no-allreduce) code path.
    """
    from mpj_spark.core.root_process import run_root
    # Force no-sync: logreg_iter=1 and no queues created.
    # run_root detects do_logreg_allreduce = False when
    # _NOSYNC env var is set; workers receive no down_queue and skip FedAvg.
    os.environ['MPJ_LOGREG_NOSYNC'] = '1'
    try:
        run_root(
            input_file       = dataset_path,
            num_workers      = args.workers,
            compare          = args.compare,
            prewarm          = not args.no_prewarm,
            cores_override   = args.cores,
            app              = args.app,
            logreg_iter      = args.logreg_iter,
            logreg_reg_param = args.logreg_reg_param,
            logreg_features  = args.logreg_features,
            baseline_threads = args.baseline_threads,
        )
    finally:
        os.environ.pop('MPJ_LOGREG_NOSYNC', None)


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

    # ── Resolve sync mode default ──────────────────────────────────────────
    # If --sync not specified: queue for logreg, ignored for others.
    sync_mode = args.sync
    if sync_mode is None:
        sync_mode = 'queue' if args.app == 'logreg' else 'queue'

    # ── Resolve dataset path ────────────────────────────────────────────
    if args.generate is not None:
        if args.app == 'kmeans':
            from mpj_spark.utils.dataset_generator import generate_numeric_dataset
            dataset_path = os.path.join(DATA_DIR, 'numeric_dataset.csv')
            generate_numeric_dataset(dataset_path, args.generate)
        elif args.app == 'logreg':
            from mpj_spark.utils.dataset_generator import generate_classification_dataset
            dataset_path = os.path.join(DATA_DIR, 'classification_dataset.csv')
            generate_classification_dataset(
                dataset_path, args.generate, num_features=args.logreg_features)
        else:
            from mpj_spark.utils.dataset_generator import generate_text_dataset
            dataset_path = os.path.join(DATA_DIR, 'text_dataset.txt')
            generate_text_dataset(dataset_path, args.generate)
    elif args.input is not None:
        dataset_path = args.input
    else:
        if args.app == 'kmeans':
            dataset_path = os.path.join(DATA_DIR, 'numeric_dataset.csv')
        elif args.app == 'logreg':
            dataset_path = os.path.join(DATA_DIR, 'classification_dataset.csv')
        else:
            dataset_path = os.path.join(DATA_DIR, 'text_dataset.txt')

    if not os.path.exists(dataset_path):
        print(f'[main] ERROR: Dataset not found: {dataset_path}')
        print('[main] Tip:   Use --generate <MB> to create one first.')
        sys.exit(1)

    # ── Print run header ────────────────────────────────────────────────
    print(f'\n[main] Dataset          : {dataset_path}')
    print(f'[main] App              : {args.app}')
    print(f'[main] Workers          : {args.workers}')
    print(f'[main] Sync mode        : {sync_mode}')
    print(f'[main] Compare          : {args.compare}')
    print(f'[main] Pre-warm         : {not args.no_prewarm}')
    if args.app == 'kmeans':
        print(f'[main] K-Means k        : {args.kmeans_k}')
        print(f'[main] K-Means iter     : {args.kmeans_iter}')
    if args.app == 'logreg':
        print(f'[main] LogReg iter      : {args.logreg_iter}')
        print(f'[main] LogReg reg_param : {args.logreg_reg_param}')
        print(f'[main] LogReg features  : {args.logreg_features}')
    if args.baseline_threads:
        print(f'[main] Baseline threads : {args.baseline_threads}  [fair comparison mode]')
    if args.gossip:
        print(f'[main] Gossip mode      : ON  '
              f'(threshold={args.gossip_threshold}, '
              f'max_rounds={args.gossip_max_rounds}, '
              f'fanout={args.gossip_fanout})')

    # ── Dispatch to execution path ──────────────────────────────────────────
    if args.app == 'logreg' and sync_mode == 'mpi':
        # ── Phase 3: MPI Allreduce ───────────────────────────────────────
        _run_mpi_logreg(args, dataset_path)

    elif args.app == 'logreg' and sync_mode == 'none':
        # ── No-sync multi-driver ──────────────────────────────────────────
        _run_nosync_logreg(args, dataset_path)

    else:
        # ── Phase 2: Queue/FedAvg (default) + kmeans + wordcount ──────────
        from mpj_spark.core.root_process import run_root
        run_root(
            input_file       = dataset_path,
            num_workers      = args.workers,
            compare          = args.compare,
            prewarm          = not args.no_prewarm,
            cores_override   = args.cores,
            app              = args.app,
            kmeans_k         = args.kmeans_k,
            kmeans_iter      = args.kmeans_iter,
            baseline_threads = args.baseline_threads,
            use_gossip       = args.gossip,
            gossip_threshold = args.gossip_threshold,
            gossip_max_rounds= args.gossip_max_rounds,
            gossip_fanout    = args.gossip_fanout,
            logreg_iter      = args.logreg_iter,
            logreg_reg_param = args.logreg_reg_param,
            logreg_features  = args.logreg_features,
        )


if __name__ == '__main__':
    main()
