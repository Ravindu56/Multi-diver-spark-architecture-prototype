# mpj_spark/applications/logreg/__init__.py
#
# Public API for the logreg MPI-Allreduce subpackage.
# Steps are implemented as separate modules mirroring the kmeans/ layout:
#
#   partition.py      — Step 1 (MPI init rule) + Step 2 (scatter)
#   allreduce.py      — Steps 3–4 (gradient compute + Allreduce sync)   [pending]
#   convergence.py    — Step 5 (loss broadcast + stop flag)              [pending]
#   metrics.py        — Step 6 (per-epoch metric collection)             [pending]

from mpj_spark.applications.logreg.partition import partition_and_init_spark

__all__ = ["partition_and_init_spark"]
