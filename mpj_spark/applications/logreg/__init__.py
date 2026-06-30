# mpj_spark/applications/logreg/__init__.py
#
# Public API for the logreg package.
# Exposes three entry points via lazy __getattr__ to avoid circular
# imports and the Python 3.12 double-import RuntimeWarning:
#
#   logreg.run(...)                  → Queue/FedAvg path (Phase 2)
#                                       worker_process.py calls this for
#                                       both baseline and multi-driver runs.
#
#   logreg.run_logreg_allreduce(...) → MPI Allreduce path (Phase 3)
#                                       called directly by main.py when
#                                       --mpi flag is active.
#
#   logreg.partition_and_init_spark  → MPI helper (Phase 3), eagerly
#                                       imported because it carries no
#                                       heavy dependencies.
#
# Sub-modules:
#   queue_run.py      — run()                  Queue/FedAvg (Phase 2)
#   allreduce.py      — run_logreg_allreduce() MPI Allreduce (Phase 3)
#   partition.py      — partition_and_init_spark
#   local_gradient.py — internal Spark gradient helpers
#   metrics.py        — per-epoch metric collection

from mpj_spark.applications.logreg.partition import partition_and_init_spark

__all__ = ["partition_and_init_spark", "run", "run_logreg_allreduce"]


def __getattr__(name: str):
    if name == "run":
        # Phase 2 Queue-based path — used by worker_process.py for both
        # baseline (no queues) and multi-driver FedAvg (with queues).
        from mpj_spark.applications.logreg.queue_run import run

        return run
    if name == "run_logreg_allreduce":
        # Phase 3 MPI Allreduce path.
        from mpj_spark.applications.logreg.allreduce import run_logreg_allreduce

        return run_logreg_allreduce
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
