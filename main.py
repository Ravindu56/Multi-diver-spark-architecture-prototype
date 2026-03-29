# ================================================================
# main.py  —  MPJ-Spark Multi-Driver Prototype
#
# Usage
# -----
# WordCount (existing):
#   python3 main.py --workers 2 --generate 500 --compare
#
# K-Means (new):
#   python3 main.py --app kmeans --workers 2 --generate 500 --compare
#   python3 main.py --app kmeans --workers 4 --kmeans-k 5 --kmeans-iter 50
#
# Log history:
#   python3 main.py --log-history
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
                   choices=['wordcount', 'kmeans'],
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
    p.add_argument('--kmeans-k',    type=int,  default=3,
                   help='Number of K-Means clusters (default: 3)')
    p.add_argument('--kmeans-iter', type=int,  default=20,
                   help='Maximum K-Means iterations (default: 20)')
    p.add_argument('--log-history', action='store_true',
                   help='Print all previous run logs and exit')
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
        else:
            from mpj_spark.utils.dataset_generator import generate_text_dataset
            dataset_path = os.path.join(DATA_DIR, 'text_dataset.txt')
            generate_text_dataset(dataset_path, args.generate)
    elif args.input is not None:
        dataset_path = args.input
    else:
        dataset_path = os.path.join(
            DATA_DIR,
            'numeric_dataset.csv' if args.app == 'kmeans' else 'text_dataset.txt'
        )

    if not os.path.exists(dataset_path):
        print(f"[main] ERROR: Dataset not found: {dataset_path}")
        print(f"[main] Tip:   Use --generate <MB> to create one first.")
        sys.exit(1)

    print(f'\n[main] Dataset      : {dataset_path}')
    print(f'[main] App          : {args.app}')
    print(f'[main] Workers      : {args.workers}')
    print(f'[main] Compare      : {args.compare}')
    print(f'[main] Pre-warm     : {not args.no_prewarm}')
    if args.app == 'kmeans':
        print(f'[main] K-Means k    : {args.kmeans_k}')
        print(f'[main] K-Means iter : {args.kmeans_iter}')

    # ── Run Root process ──────────────────────────────────────────────
    from mpj_spark.core.root_process import run_root

    run_root(
        input_file     = dataset_path,
        num_workers    = args.workers,
        compare        = args.compare,
        prewarm        = not args.no_prewarm,
        cores_override = args.cores,
        app            = args.app,
        kmeans_k       = args.kmeans_k,
        kmeans_iter    = args.kmeans_iter,
    )


if __name__ == '__main__':
    main()