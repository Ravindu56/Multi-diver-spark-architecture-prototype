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
    rank: int | None = None,
    size: int | None = None,
    comm=None,
    dataset_path: str | None = None,
    partition_path: str | None = None,
    worker_id: int | None = None,
    num_workers: int | None = None,
    max_iter: int = 20,
    learning_rate: float = 0.01,
    reg_param: float = 0.01,
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
    reg_param        : L2 regularization coefficient (default 0.01).
    tol              : Loss-delta convergence tolerance.
    metrics_output_dir : Directory for per-rank metrics CSV/JSON output.

    Returns  (all ranks return the same dict after root Bcast)
    -------
    {
        'weights'   : list[float]   # final global weight vector
        'intercept' : float         # 0.0  (bias folded into weight vector)
    }
    """
    actual_dataset_path = dataset_path or partition_path
    actual_rank = rank if rank is not None else worker_id if worker_id is not None else 0
    actual_size = size if size is not None else num_workers if num_workers is not None else 1

    if comm is None or actual_dataset_path is None:
        from mpj_spark.applications.logreg import nosync_run

        raw = nosync_run.run(
            partition_path=actual_dataset_path,
            max_iter=max_iter,
            reg_param=reg_param,
            num_features=10,
            worker_id=actual_rank,
            num_workers=actual_size,
            results_dir=metrics_output_dir,
        )
        return {
            "weights": raw.get("weight_vector", []),
            "intercept": float(raw.get("intercept", 0.0)),
        }

    from mpj_spark.applications.logreg.allreduce import run_logreg_allreduce

    raw = run_logreg_allreduce(
        comm=comm,
        rank=actual_rank,
        size=actual_size,
        input_file=actual_dataset_path,
        max_epochs=max_iter,
        learning_rate=learning_rate,
        tol=tol,
        metrics_output_dir=metrics_output_dir,
    )

    result = {
        "weights": raw["weights"],
        "intercept": float(raw["intercept"]),
    }

    if hasattr(comm, "bcast"):
        result = comm.bcast(result, root=0)
    return result
