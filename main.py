# ================================================================
# main.py  —  MPJ-Spark Multi-Driver Prototype
#
# Changes from feature/ml-kmeans-workload:
#
#   GOSSIP EXTENSION (feature/adaptive-gossip-aggregation):
#     Added three new CLI flags:
#       --gossip              enable adaptive gossip aggregation
#       --gossip-threshold F  convergence drift criterion (default 0.001)
#       --gossip-max-rounds N hard cap on rounds (default 10)
#       --gossip-fanout N     initial peer fan-out (default 2, then adaptive)
#
#   LOGREG EXTENSION (feature/ml-logreg-workload):
#     Added three new CLI flags:
#       --logreg-iter N       per-Allreduce-iteration count (default 10)
#       --logreg-reg-param F  L2 regularisation (default 0.01)
#       --logreg-features N   number of feature columns in dataset (default 10)
#     Dataset generator:
#       --generate with --app logreg emits a labelled binary classification
#       CSV via generate_classification_dataset()
#
# Usage examples:
# ---------------
# Standard (unchanged):
#   python main.py --app kmeans --workers 4 --generate 200 --compare \
#                  --kmeans-k 5 --kmeans-iter 30
#
# Gossip mode:
#   python main.py --app kmeans --workers 4 --generate 200 --gossip \
#                  --kmeans-k 5 --kmeans-iter 30
#
# LogReg (basic):
#   python main.py --app logreg --workers 4 --generate 100 \
#                  --logreg-iter 10 --logreg-reg-param 0.01 --logreg-features 10
#
# LogReg with comparison:
#   python main.py --app logreg --workers 4 --generate 200 --compare \
#                  --logreg-iter 15 --logreg-reg-param 0.001 --logreg-features 20
#
# Log history:
#   python main.py --log-history
# ================================================================
import argparse
import os
import sys


def parse_args():
    p = argparse.ArgumentParser(
        description='MPJ-Spark Multi-Driver Prototype',
        formatter_class=argparse.RawTextHelpFormatter,
    )
    p.add_argument('--workers',     type=int,  default=2,
                   help='Number of parallel worker processes (default: 2)')
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
                   help='L2 regularisation parameter for LogisticRegression.\n'
                        '(default: 0.01)')
    p.add_argument('--logreg-features', type=int, default=10,
                   help='Number of feature columns in the classification dataset.\n'
                        'Must match the dataset used (default: 10)')
    # ─────────────────────────────────────────────────────────────

    return p.parse_args()


def main():
    args = parse_args()

    if args.log_history:
        from mpj_spark.utils.dev_logger import DevLogger
        DevLogger.print_history()
        sys.exit(0)

    from mpj_spark.config import DATA_DIR
    os.makedirs(DATA_DIR, exist_ok=True)

    # ── Resolve dataset path ──────────────────────────────────────────
    if args.generate is not None:
        if args.app == 'kmeans':
            from mpj_spark.utils.dataset_generator import generate_numeric_dataset
            dataset_path = os.path.join(DATA_DIR, 'numeric_dataset.csv')
            generate_numeric_dataset(dataset_path, args.generate)
        elif args.app == 'logreg':
            from mpj_spark.utils.dataset_generator import generate_classification_dataset
            dataset_path = os.path.join(DATA_DIR, 'classification_dataset.csv')
            generate_classification_dataset(
                dataset_path, args.generate,
                num_features=args.logreg_features,
            )
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
        print(f"[main] ERROR: Dataset not found: {dataset_path}")
        print("[main] Tip:   Use --generate <MB> to create one first.")
        sys.exit(1)

    print(f'\n[main] Dataset          : {dataset_path}')
    print(f'[main] App              : {args.app}')
    print(f'[main] Workers          : {args.workers}')
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

    # ── Run Root process ──────────────────────────────────────────────
    from mpj_spark.core.root_process import run_root

    run_root(
        input_file        = dataset_path,
        num_workers       = args.workers,
        compare           = args.compare,
        prewarm           = not args.no_prewarm,
        cores_override    = args.cores,
        app               = args.app,
        kmeans_k          = args.kmeans_k,
        kmeans_iter       = args.kmeans_iter,
        baseline_threads  = args.baseline_threads,
        use_gossip        = args.gossip,
        gossip_threshold  = args.gossip_threshold,
        gossip_max_rounds = args.gossip_max_rounds,
        gossip_fanout     = args.gossip_fanout,
        logreg_iter       = args.logreg_iter,
        logreg_reg_param  = args.logreg_reg_param,
        logreg_features   = args.logreg_features,
    )


if __name__ == '__main__':
    main()
