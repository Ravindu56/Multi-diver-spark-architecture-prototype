#!/usr/bin/env python3
# ============================================================
# main.py — CLI entry point for MPJ-SPARK prototype
# ============================================================
import os
os.environ['JAVA_TOOL_OPTIONS'] = '-Djava.security.manager=allow'

import argparse
from mpj_spark.core.root_process           import mpj_root_process
from mpj_spark.applications.baseline_spark import run_baseline
from mpj_spark.benchmarks.reporter         import print_comparison
from mpj_spark.benchmarks.dev_logger        import DevLogger
from mpj_spark.utils.dataset_generator     import generate_test_dataset
from mpj_spark.config                      import (
    DEFAULT_DATASET_PATH, DEFAULT_DATASET_SIZE_MB,
    DEFAULT_NUM_WORKERS, TOTAL_CORES,
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
    p.add_argument('--no-prewarm', dest='prewarm', action='store_false',
                   help='Disable JVM pre-warm barrier (cold-start mode)')
    p.add_argument('--cores',    type=int, default=None,
                   help=(
                       'Override cores per worker (and baseline). '
                       f'Default: auto = {TOTAL_CORES} total ÷ --workers. '
                       'Use 0 to restore unconstrained local[*] behaviour.'
                   ))
    p.add_argument('--no-log',   dest='log', action='store_false',
                   help='Disable dev run logging (skip writing to logs/dev/)')
    p.add_argument('--log-history', action='store_true',
                   help='Print summary table of all past dev runs and exit')
    p.set_defaults(prewarm=True, log=True)
    return p.parse_args()


def main():
    args = parse_args()

    # ── Print run history and exit ───────────────────────────────────
    if args.log_history:
        DevLogger().print_summary_table()
        return

    cores_per_entity = None
    if args.cores is not None:
        cores_per_entity = args.cores if args.cores > 0 else None

    cores_display = (
        cores_per_entity if cores_per_entity
        else max(1, TOTAL_CORES // args.workers)
    )
    print(f'[CONFIG] Machine cores: {TOTAL_CORES}  |  '
          f'Workers: {args.workers}  |  '
          f'Cores/entity: {cores_display}  |  '
          f'JVM mode: {"pre-warmed" if args.prewarm else "cold-start"}')

    # ── Resolve input file ─────────────────────────────────────────────
    if args.input and os.path.exists(args.input):
        input_file = args.input
    else:
        input_file = generate_test_dataset(DEFAULT_DATASET_PATH, args.generate)

    # ── Multi-Driver run ───────────────────────────────────────────────
    _, multi_timing = mpj_root_process(
        input_file,
        args.workers,
        app=args.app,
        prewarm=args.prewarm,
        cores_per_worker=cores_per_entity,
    )

    # ── Baseline comparison ────────────────────────────────────────────
    std_timing = None
    if args.compare:
        _, std_timing = run_baseline(
            input_file,
            num_workers=args.workers,
            cores_override=cores_per_entity,
        )
        print_comparison(multi_timing, std_timing)

    # ── Dev logging ─────────────────────────────────────────────────
    if args.log:
        logger = DevLogger()
        run_id = logger.log_run(
            run_config={
                'workers':          args.workers,
                'generate':         args.generate,
                'input_file':       input_file,
                'app':              args.app,
                'prewarm':          args.prewarm,
                'cores_per_entity': cores_display,
            },
            multi_timing=multi_timing,
            std_timing=std_timing,
        )
        print(f'\n[LOG] Run saved → {logger.text_path}  (run_id: {run_id})')


if __name__ == '__main__':
    main()
