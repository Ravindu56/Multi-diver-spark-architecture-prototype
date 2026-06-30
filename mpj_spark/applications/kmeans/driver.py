# =============================================================================
# mpj_spark/applications/kmeans/driver.py
#
# Thin facade exposing run_kmeans_driver() — the interface expected by
# scripts/validate_parity.py (Issue #10).
#
# Wraps run_kmeans_allreduce() and normalises its return dict into the
# parity-check contract:
#   {'centres': list[list[float]], 'wcss': float}
# =============================================================================
from __future__ import annotations


def run_kmeans_driver(
    rank: int,
    size: int,
    comm,
    dataset_path: str,
    k: int = 3,
    max_iter: int = 20,
    tol: float = 1e-4,
    seed: int = 42,
    metrics_output_dir: str = "./kmeans_results",
) -> dict:
    """
    Parity-check facade for the multi-driver K-Means Allreduce runner.

    Parameters
    ----------
    rank, size, comm : MPI rank, world size, and communicator.
    dataset_path     : Shared-storage path to the input CSV (all ranks must
                       be able to read this path).
    k                : Number of clusters.
    max_iter         : Maximum iterations.
    tol              : Centroid-shift convergence tolerance.
    seed             : Random seed for centroid initialisation.
    metrics_output_dir : Directory for per-rank metrics CSV/JSON output.

    Returns  (all ranks return the same dict after root Bcast)
    -------
    {
        'centres' : list[list[float]]   # global centroids  (k x d)
        'wcss'    : float               # final global WCSS / inertia
    }
    """
    from mpj_spark.applications.kmeans.allreduce import run_kmeans_allreduce

    raw = run_kmeans_allreduce(
        comm=comm,
        rank=rank,
        size=size,
        input_file=dataset_path,
        k=k,
        max_iter=max_iter,
        tol=tol,
        seed=seed,
        metrics_output_dir=metrics_output_dir,
    )

    # Normalise to parity-check contract.
    # run_kmeans_allreduce returns 'global_centroids' (list[list[float]])
    # and embeds wcss inside the run_summary dict.
    result = {
        "centres": raw.get("global_centroids", []),
        "wcss": float(
            raw.get("run_summary", {}).get("final_wcss", 0.0)
            or _extract_last_wcss(raw)
        ),
    }

    # Broadcast from root so every rank returns identical data
    # (validate_parity.py only reads on rank 0, but broadcasting is safer).
    result = comm.bcast(result, root=0)
    return result


def _extract_last_wcss(raw: dict) -> float:
    """Fallback: pull WCSS from the last row of the per-iteration metrics."""
    metrics = raw.get("metrics", [])
    if metrics:
        last = metrics[-1]
        # KMeansMetricsCollector columns: iteration, spark_s, sync_s, iter_s,
        # centroid_shift, global_wcss
        return float(last.get("global_wcss", 0.0))
    return 0.0
