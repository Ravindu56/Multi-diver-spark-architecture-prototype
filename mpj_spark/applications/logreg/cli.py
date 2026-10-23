# =============================================================================
# mpj_spark/applications/logreg/cli.py
#
# Command-line entry point for the multi-driver LogReg Allreduce runner, so it
# can be launched under mpirun the same way as the K-Means runner:
#
#   mpirun -n 3 python -m mpj_spark.applications.logreg.cli \
#       --input /data/input/logreg_f10_100mb.csv --epochs 30 --lr 0.01 \
#       --output /data/results/p5-04/lr-A-f10-r1
#
# Every rank computes on its own partition (rank 0 included), matching the
# K-Means Allreduce runner. run_logreg_allreduce() is unchanged.
# =============================================================================

from __future__ import annotations

import argparse
import logging
import os
import sys


def main(argv: list[str] | None = None) -> int:
    from mpi4py import MPI

    from mpj_spark.applications.logreg.allreduce import run_logreg_allreduce

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    parser = argparse.ArgumentParser(
        prog="python -m mpj_spark.applications.logreg.cli",
        description="Multi-driver logistic regression with synchronous MPI Allreduce gradient sync.",
    )
    parser.add_argument("--input", required=True, help="Input CSV on shared storage.")
    parser.add_argument("--epochs", type=int, default=30, help="Maximum epochs (default: 30).")
    parser.add_argument("--lr", type=float, default=0.01, help="Initial learning rate.")
    parser.add_argument("--reg-param", type=float, default=0.01, help="L2 coefficient.")
    parser.add_argument("--tol", type=float, default=1e-4, help="Loss-delta tolerance.")
    parser.add_argument("--cores", type=int, default=None, help="Spark local[N] per rank.")
    parser.add_argument("--output", default="./logreg_results", help="Metrics output directory.")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level for all ranks (default: INFO).",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format=f"%(asctime)s [rank {rank}] %(levelname)s %(name)s - %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
    )
    os.makedirs(args.output, exist_ok=True)

    if rank == 0:
        print(
            f"LogReg Allreduce: ranks={size} epochs={args.epochs} lr={args.lr} "
            f"reg={args.reg_param} tol={args.tol} input={args.input} output={args.output}",
            flush=True,
        )

    result = run_logreg_allreduce(
        comm=comm,
        rank=rank,
        size=size,
        input_file=args.input,
        max_epochs=args.epochs,
        learning_rate=args.lr,
        reg_param=args.reg_param,
        tol=args.tol,
        cores_override=args.cores,
        metrics_output_dir=args.output,
    )

    if rank == 0:
        summary = result["run_summary"]
        print(
            f"Run complete: epochs_run={summary['epochs_run']} "
            f"converged={summary['converged']} total_time_s={summary['total_time_s']}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
