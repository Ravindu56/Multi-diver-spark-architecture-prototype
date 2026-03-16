#!/usr/bin/env python3
# ============================================================
# main.py — CLI entry point for MPJ-SPARK prototype
# ============================================================
import os
os.environ['JAVA_TOOL_OPTIONS'] = '-Djava.security.manager=allow'

import argparse
from mpj_spark.core.root_process          import mpj_root_process
from mpj_spark.applications.baseline_spark import run_baseline
from mpj_spark.benchmarks.reporter        import print_comparison
from mpj_spark.utils.dataset_generator    import generate_test_dataset
from mpj_spark.config                     import (
    DEFAULT_DATASET_PATH, DEFAULT_DATASET_SIZE_MB, DEFAULT_NUM_WORKERS
)


def parse_args():
    p = argparse.ArgumentParser(
        description='MPJ-SPARK Multi-Driver Prototype  v2.0'
    )
    p.add_argument('--workers',  type=int, default=DEFAULT_NUM_WORKERS,
                   help=f'Number of MPJ workers (default: {DEFAULT_NUM_WORKERS})')
    p.add_argument('--input',    type=str, default=None,
                   help='Path to an existing input text file')
    p.add_argument('--generate', type=int, default=DEFAULT_DATASET_SIZE_MB,
                   help=f'Generate a synthetic dataset of N MB (default: {DEFAULT_DATASET_SIZE_MB})')
    p.add_argument('--compare',  action='store_true',
                   help='Run standard single-driver Spark baseline for comparison')
    p.add_argument('--app',      type=str, default='wordcount',
                   choices=['wordcount'],
                   help='Application to run (default: wordcount)')
    return p.parse_args()


def main():
    args = parse_args()

    # ── Resolve input file ─────────────────────────────────────────────
    if args.input and os.path.exists(args.input):
        input_file = args.input
    else:
        input_file = generate_test_dataset(DEFAULT_DATASET_PATH, args.generate)

    # ── Multi-Driver run ───────────────────────────────────────────────
    _, multi_timing = mpj_root_process(input_file, args.workers, app=args.app)

    # ── Baseline comparison ────────────────────────────────────────────
    if args.compare:
        _, std_timing = run_baseline(input_file)
        print_comparison(multi_timing, std_timing)


if __name__ == '__main__':
    main()
