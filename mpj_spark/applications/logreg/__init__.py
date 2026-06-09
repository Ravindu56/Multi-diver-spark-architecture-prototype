# mpj_spark/applications/logreg/__init__.py
#
# Public API for the logreg MPI-Allreduce subpackage.
# Steps are implemented as separate modules mirroring the kmeans/ layout:
#
#   partition.py      — Step 1 (MPI init rule) + Step 2 (scatter)       [done]
#   local_gradient.py — Steps 3 & 4 (cores formula + gradient RDD)      [done]
#   allreduce.py      — Steps 5 & 6 (Allreduce sync + full runner)      [done]
#   metrics.py        — Step 6b (per-epoch metric collection)           [done]
#
# NOTE: run_logreg_allreduce is NOT eagerly imported here.
# Eager import of allreduce.py at package load time caused a Python 3.12
# RuntimeWarning when the module is executed via `python -m ...` because
# the module ends up in sys.modules twice (once as a package member, once
# as __main__).  Use a lazy __getattr__ instead so the import only fires
# when the caller explicitly requests the symbol.

from mpj_spark.applications.logreg.partition import partition_and_init_spark

__all__ = ["partition_and_init_spark", "run_logreg_allreduce"]


def __getattr__(name: str):
    if name == "run_logreg_allreduce":
        from mpj_spark.applications.logreg.allreduce import run_logreg_allreduce

        return run_logreg_allreduce
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
