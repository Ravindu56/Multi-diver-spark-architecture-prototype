"""mpj_spark/applications/kmeans/common.py
Shared K-Means building blocks for every cross-driver sync mode.

Mirrors logreg/warm_common.py: one local step, one WCSS definition and one
shift definition, so FedAvg, gossip, async-PS and hybrid runs are compared on
identical local work and differ only in how centroids are synchronised.

Pure numpy; pyspark is touched only through the RDD object that is passed in.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from mpj_spark.applications.kmeans.local_iteration import compute_local_stats

KMEANS_METRIC_SOLVER = "lloyd"


def unpack_stats(stats) -> tuple[np.ndarray, np.ndarray, float]:
    """Normalise compute_local_stats() output to (sums, counts, wcss)."""
    if isinstance(stats, dict):
        sums = np.asarray(stats.get("cluster_sums", stats.get("sums")), dtype=np.float64)
        counts = np.asarray(stats.get("cluster_counts", stats.get("counts")), dtype=np.float64)
        return sums, counts, float(stats.get("local_wcss", stats.get("wcss", 0.0)))
    sums = np.asarray(stats[0], dtype=np.float64)
    counts = np.asarray(stats[1], dtype=np.float64)
    return sums, counts, float(stats[2]) if len(stats) >= 3 else 0.0


def local_wcss(points_rdd, centroids: np.ndarray) -> float:
    """Sum of squared distances from each local point to its nearest centroid."""
    c = np.asarray(centroids, dtype=np.float64)

    def _sq_dist(p):
        return float(np.min(np.sum((c - np.asarray(p, dtype=np.float64)) ** 2, axis=1)))

    return float(points_rdd.map(_sq_dist).sum())


def local_lloyd(
    points_rdd,
    centroids: np.ndarray,
    epochs: int,
    stats_fn: Callable = compute_local_stats,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Run `epochs` Lloyd iterations on this worker's partition.

    Returns (centroids, cluster_sums, cluster_counts, local_wcss) where the
    sums/counts come from the LAST epoch and local_wcss is measured on the
    returned centroids.  Empty clusters keep their previous position.
    The input array is not modified.
    """
    if epochs < 1:
        raise ValueError(f"local_lloyd: epochs must be >= 1 (got {epochs})")
    cents = np.array(centroids, dtype=np.float64, copy=True)
    k = cents.shape[0]
    sums = np.zeros_like(cents)
    counts = np.zeros(k, dtype=np.float64)
    for _ in range(epochs):
        sums, counts, _ = unpack_stats(stats_fn(points_rdd, cents))
        for j in range(k):
            if counts[j] > 0:
                cents[j] = sums[j] / counts[j]
    return cents, sums, counts, local_wcss(points_rdd, cents)


def centroid_shift(prev: np.ndarray, new: np.ndarray) -> float:
    """Frobenius shift between two centroid sets."""
    return float(np.linalg.norm(np.asarray(new) - np.asarray(prev)))
