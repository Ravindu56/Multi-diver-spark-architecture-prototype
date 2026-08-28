"""mpj_spark/core/hybrid_ps.py
Phase 3 - P3-10: root-side scalar Parameter Server for the hybrid
synchronization mode (Issue #62).

In hybrid_ps_allreduce the DENSE weight vector is exchanged worker-to-
worker via comm.Allreduce over the worker sub-communicator; this root
coordinator only serves the SCALAR channel over COMM_WORLD
(TAG_ALLREDUCE_UP=30 / TAG_ALLREDUCE_DOWN=31): per round it collects
each worker's intercept (plus its static row count), computes the
row-weighted mean, and replies to every worker.

Synchronous (bulk-synchronous rounds) - in contrast to the non-blocking
async PS in async_ps.py (P3-09).  Import-safe on CI runners without a
system MPI library: mpi4py is never imported here; _serve() only needs
an object implementing recv(source, tag) / send(msg, dest, tag), so the
core is unit-testable with a scripted fake communicator.
"""

from __future__ import annotations

import csv
import os
import time
from typing import Any

TAG_ALLREDUCE_UP = 30
TAG_ALLREDUCE_DOWN = 31


def row_weighted_mean(values: list[float], weights: list[float]) -> float:
    """Row-weighted mean of scalars; plain mean when weights sum to zero."""
    if not values:
        return 0.0
    total = float(sum(weights))
    if total <= 0.0:
        return float(sum(values) / len(values))
    return float(sum(w * v for v, w in zip(values, weights, strict=False)) / total)


def _serve(
    comm,
    num_workers: int,
    num_iterations: int,
    tag_up: int = TAG_ALLREDUCE_UP,
    tag_down: int = TAG_ALLREDUCE_DOWN,
) -> dict[str, Any]:
    """Core synchronous scalar-PS loop.

    Per round: recv one intercept payload from every worker rank (1..N),
    compute the row-weighted mean, reply to every worker.  Workers block
    on the reply before their next round, so per-round tag reuse is safe
    (same barrier argument as run_logreg_allreduce_mpi).
    """
    worker_ranks = list(range(1, num_workers + 1))
    records: list[dict[str, Any]] = []
    final_intercept = 0.0

    print(
        f"  [LogReg Hybrid PS] Serving scalars - {num_workers} workers x "
        f"{num_iterations} rounds (dense weights ride the Allreduce channel)"
    )
    t_start = time.perf_counter()

    for iteration in range(num_iterations):
        msgs = [comm.recv(source=r, tag=tag_up) for r in worker_ranks]
        intercepts = [float(m["intercept"]) for m in msgs]
        rows = [float(m.get("row_count", 0)) for m in msgs]

        final_intercept = row_weighted_mean(intercepts, rows)

        payload = {
            "type": "avg_intercept",
            "iteration": iteration,
            "intercept": final_intercept,
        }
        for r in worker_ranks:
            comm.send(payload, dest=r, tag=tag_down)

        records.append(
            {
                "iteration": iteration + 1,
                "intercept": round(final_intercept, 8),
                "elapsed_s": round(time.perf_counter() - t_start, 6),
            }
        )
        print(
            f"  [LogReg Hybrid PS] round {iteration + 1}/{num_iterations}  "
            f"intercept={final_intercept:.6f}"
        )

    wall_time = time.perf_counter() - t_start
    print(f"  [LogReg Hybrid PS] Complete - {num_iterations} rounds in {wall_time:.3f}s")

    return {
        "weight_vector": None,  # dense weights live on the Allreduce channel
        "intercept": float(final_intercept),
        "iterations_done": num_iterations,
        "wall_time_s": wall_time,
        "round_records": records,
    }


def _write_scalar_csv(records: list[dict[str, Any]], results_dir: str) -> str:
    os.makedirs(results_dir, exist_ok=True)
    path = os.path.join(results_dir, "logreg_hybrid_scalar_ps.csv")
    fieldnames = ["iteration", "intercept", "elapsed_s"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    return path


def run_logreg_hybrid_scalar_ps(
    comm,
    num_workers: int,
    num_iterations: int,
    results_dir: str = "results",
    tag_up: int = TAG_ALLREDUCE_UP,
    tag_down: int = TAG_ALLREDUCE_DOWN,
) -> dict[str, Any]:
    """Root-side hybrid scalar-PS entry point (called in a daemon thread by root_mpi).

    The aggregation path reads only 'intercept' from this result; the
    dense weight vector is taken from worker_results[0] (identical
    across workers after the Allreduce collective).
    """
    result = _serve(
        comm,
        num_workers=num_workers,
        num_iterations=num_iterations,
        tag_up=tag_up,
        tag_down=tag_down,
    )
    csv_path = _write_scalar_csv(result.pop("round_records"), results_dir)
    print(f"  [LogReg Hybrid PS] Scalar round log : {csv_path}")
    return result
