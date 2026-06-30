# mpj_spark/applications/kmeans/__init__.py
# Public API for the K-Means Allreduce subpackage (Phase 3, Issue #8)
#
# NOTE: allreduce.run_kmeans_allreduce is intentionally NOT exported here.
# It is a CLI entry-point invoked via `python -m ...allreduce`.
# Importing it here would place allreduce in sys.modules before the -m
# runner executes it as __main__, causing a RuntimeWarning.
# Import it directly when needed:
#   from mpj_spark.applications.kmeans.allreduce import run_kmeans_allreduce

from mpj_spark.applications.kmeans.convergence import check_and_broadcast
from mpj_spark.applications.kmeans.metrics import KMeansMetricsCollector

__all__ = [
    "check_and_broadcast",
    "KMeansMetricsCollector",
]
