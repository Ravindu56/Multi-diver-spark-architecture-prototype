# =============================================================================
# mpj_spark/applications/logreg/driver.py
#
# Thin facade exposing run_logreg_driver() — the interface expected by
# scripts/validate_parity.py (Issue #10).
#
# Parity-check contract:
#   {'weights': list[float], 'intercept': float}
#
# run_logreg_allreduce() kwarg mapping (corrected):
#   max_iter  →  max_epochs   (actual param name)
#   input_file is correct as-is
#   seed= removed   (allreduce.py has no seed param; w is zero-initialised)
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
    metrics_output_dir: str = "./logreg_results",
) -> dict:
    """
    Parity-check facade for the multi-driver Logistic Regression Allreduce runner.

    Parameters
    ----------
    rank, size, comm : MPI rank, world size, and communicator.
    dataset_path     : Shared-storage path to the input CSV.
    max_iter         : Maximum gradient-descent epochs (maps to max_epochs).
    learning_rate    : Initial SGD learning rate (cosine-decayed internally).
    tol              : Loss-delta convergence tolerance.
    metrics_output_dir : Directory for per-rank metrics CSV/JSON output.

    Returns  (all ranks return the same dict after root Bcast)
    -------
    {
        'weights'   : list[float]   # final global weight vector
        'intercept' : float         # 0.0  (bias folded into weight vector)
    }
    """
    from mpj_spark.applications.logreg.allreduce import run_logreg_allreduce

    raw = run_logreg_allreduce(
        comm=comm,
        rank=rank,
        size=size,
        input_file=dataset_path,
        max_epochs=max_iter,  # correct kwarg name
        learning_rate=learning_rate,
        tol=tol,
        metrics_output_dir=metrics_output_dir,
    )

    result = {
        "weights": raw["weights"],
        "intercept": float(raw["intercept"]),
    }

    # Broadcast from root so every rank has identical data.
    result = comm.bcast(result, root=0)
    return result
