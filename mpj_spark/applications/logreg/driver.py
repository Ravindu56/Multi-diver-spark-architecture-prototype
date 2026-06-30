# =============================================================================
# mpj_spark/applications/logreg/driver.py
#
# Thin facade exposing run_logreg_driver() — the interface expected by
# scripts/validate_parity.py (Issue #10).
#
# Wraps run_logreg_allreduce() and normalises its return dict into the
# parity-check contract:
#   {'weights': list[float], 'intercept': float}
# =============================================================================
from __future__ import annotations


def run_logreg_driver(
    rank: int,
    size: int,
    comm,
    dataset_path: str,
    max_iter: int = 20,
    learning_rate: float = 0.01,
    tol: float = 1e-4,
    seed: int = 42,
    metrics_output_dir: str = "./logreg_results",
) -> dict:
    """
    Parity-check facade for the multi-driver Logistic Regression Allreduce runner.

    Parameters
    ----------
    rank, size, comm : MPI rank, world size, and communicator.
    dataset_path     : Shared-storage path to the input CSV.
    max_iter         : Maximum gradient-descent iterations.
    learning_rate    : SGD learning rate.
    tol              : Gradient-norm convergence tolerance.
    seed             : Random seed for weight initialisation.
    metrics_output_dir : Directory for per-rank metrics CSV/JSON output.

    Returns  (all ranks return the same dict after root Bcast)
    -------
    {
        'weights'   : list[float]   # global weight vector
        'intercept' : float         # global bias / intercept
    }
    """
    from mpj_spark.applications.logreg.allreduce import run_logreg_allreduce

    raw = run_logreg_allreduce(
        comm=comm,
        rank=rank,
        size=size,
        input_file=dataset_path,
        max_iter=max_iter,
        learning_rate=learning_rate,
        tol=tol,
        seed=seed,
        metrics_output_dir=metrics_output_dir,
    )

    # Normalise to parity-check contract.
    # run_logreg_allreduce returns 'global_weights' (list[float])
    # and 'global_intercept' (float).
    result = {
        "weights": raw.get("global_weights", []),
        "intercept": float(raw.get("global_intercept", 0.0)),
    }

    # Broadcast from root so every rank has identical data.
    result = comm.bcast(result, root=0)
    return result
