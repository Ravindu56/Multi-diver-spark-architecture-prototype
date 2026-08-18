"""mpj_spark/core/async_ps.py
Phase 3 - P3-09: Asynchronous Parameter Server (root side) for LogReg.

Root coordinator for sync_mode="ps_async" (Issue #61).

Unlike the synchronous FedAvg coordinators (Queue-based M2 or the P3-08
native-MPI gather/bcast), the async PS never waits for all workers:
each worker update is applied to the global model the moment it arrives
(comm.Iprobe / comm.recv on MPI.ANY_SOURCE, TAG_ALLREDUCE_UP=30) and the
refreshed global model is returned to that worker immediately
(TAG_ALLREDUCE_DOWN=31).  Workers therefore progress at their own pace —
fast workers are never blocked behind stragglers.

Mixing rule (FedAsync-style, Xie et al. 2020):

    w_global <- (1 - alpha_eff) * w_global + alpha_eff * w_worker
    alpha_eff = server_lr / (1 + staleness)   when staleness_damping=True
    alpha_eff = server_lr                     otherwise

where staleness = global_version - base_version carried by the update.

Import-safe on CI runners without a system MPI library: mpi4py is
imported lazily inside run_logreg_async_ps(); the testable core
(_serve) receives the ANY_SOURCE sentinel as a plain parameter.
"""

from __future__ import annotations

import csv
import os
import time
from typing import Any

import numpy as np

TAG_ALLREDUCE_UP = 30
TAG_ALLREDUCE_DOWN = 31


def blend(global_w: np.ndarray, global_b: float,
          worker_w: np.ndarray, worker_b: float, alpha: float):
    """FedAsync mixing step: g <- (1 - alpha) * g + alpha * w."""
    new_w = (1.0 - alpha) * np.asarray(global_w, dtype=np.float64) + alpha * np.asarray(
        worker_w, dtype=np.float64
    )
    new_b = (1.0 - alpha) * float(global_b) + alpha * float(worker_b)
    return new_w, new_b


def effective_alpha(server_lr: float, staleness: int, staleness_damping: bool = True) -> float:
    """Staleness-damped server learning rate (FedAsync inverse damping)."""
    if not staleness_damping:
        return float(server_lr)
    return float(server_lr) / (1.0 + max(0, int(staleness)))


def _serve(
    comm,
    num_workers: int,
    num_features: int,
    num_iterations: int,
    any_source: Any,
    server_lr: float = 0.5,
    staleness_damping: bool = True,
    tag_up: int = TAG_ALLREDUCE_UP,
    tag_down: int = TAG_ALLREDUCE_DOWN,
    poll_interval_s: float = 0.001,
    timeout_s: float = 900.0,
) -> dict[str, Any]:
    """Core async-PS serving loop.  Testable with any Queue-like fake comm
    implementing Iprobe(source, tag) / recv(source, tag) / send(msg, dest, tag).
    """
    worker_ranks = list(range(1, num_workers + 1))
    updates_done = dict.fromkeys(worker_ranks, 0)

    global_w = np.zeros(num_features, dtype=np.float64)
    global_b = 0.0
    global_version = 0
    records: list[dict[str, Any]] = []

    print(
        f"  [LogReg Async PS] Serving - {num_workers} workers x "
        f"{num_iterations} rounds  (server_lr={server_lr}, "
        f"staleness_damping={'on' if staleness_damping else 'off'})"
    )
    t_start = time.perf_counter()

    while not all(updates_done[r] >= num_iterations for r in worker_ranks):
        if time.perf_counter() - t_start > timeout_s:
            raise TimeoutError(
                f"[LogReg Async PS] Timed out after {timeout_s:.0f}s - "
                f"updates per worker: {updates_done} "
                f"(expected {num_iterations} each)."
            )
        if not comm.Iprobe(source=any_source, tag=tag_up):
            time.sleep(poll_interval_s)
            continue

        msg = comm.recv(source=any_source, tag=tag_up)
        rank = int(msg["rank"])
        if rank not in updates_done:
            raise ValueError(f"[LogReg Async PS] Update from unexpected rank {rank}")

        base_version = int(msg.get("base_version", 0))
        staleness = max(0, global_version - base_version)
        worker_w = np.asarray(msg["weights"], dtype=np.float64)
        worker_b = float(msg["intercept"])

        if global_version == 0:
            # First update bootstraps the global model (alpha = 1).
            global_w = worker_w.copy()
            global_b = worker_b
            alpha = 1.0
        else:
            alpha = effective_alpha(server_lr, staleness, staleness_damping)
            global_w, global_b = blend(global_w, global_b, worker_w, worker_b, alpha)

        global_version += 1
        updates_done[rank] += 1

        reply = {
            "weights": global_w.tolist(),
            "intercept": float(global_b),
            "global_version": global_version,
            "staleness": staleness,
        }
        comm.send(reply, dest=rank, tag=tag_down)

        weight_norm = float(np.linalg.norm(global_w))
        records.append(
            {
                "global_version": global_version,
                "worker_rank": rank,
                "worker_round": int(msg.get("worker_round", updates_done[rank] - 1)),
                "row_count": int(msg.get("row_count", 0)),
                "staleness": staleness,
                "alpha_eff": round(alpha, 6),
                "weight_norm": round(weight_norm, 8),
                "elapsed_s": round(time.perf_counter() - t_start, 6),
            }
        )
        print(
            f"  [LogReg Async PS] v{global_version} <- worker {rank} "
            f"(round {msg.get('worker_round', '?')})  staleness={staleness}  "
            f"alpha={alpha:.3f}  |w|={weight_norm:.4f}"
        )

    wall_time = time.perf_counter() - t_start
    staleness_vals = [r["staleness"] for r in records]
    mean_staleness = float(sum(staleness_vals) / len(staleness_vals)) if staleness_vals else 0.0
    max_staleness = int(max(staleness_vals)) if staleness_vals else 0
    print(
        f"  [LogReg Async PS] Complete - {global_version} updates in {wall_time:.3f}s  "
        f"mean staleness={mean_staleness:.2f}  max={max_staleness}"
    )

    return {
        "weight_vector": global_w.tolist(),
        "intercept": float(global_b),
        "global_version": global_version,
        "mean_staleness": mean_staleness,
        "max_staleness": max_staleness,
        "wall_time_s": wall_time,
        "staleness_records": records,
    }


def _write_staleness_csv(records: list[dict[str, Any]], results_dir: str) -> str:
    os.makedirs(results_dir, exist_ok=True)
    path = os.path.join(results_dir, "logreg_async_ps_staleness.csv")
    fieldnames = [
        "global_version",
        "worker_rank",
        "worker_round",
        "row_count",
        "staleness",
        "alpha_eff",
        "weight_norm",
        "elapsed_s",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    return path


def run_logreg_async_ps(
    comm,
    num_workers: int,
    num_iterations: int,
    num_features: int,
    server_lr: float = 0.5,
    staleness_damping: bool = True,
    results_dir: str = "results",
    tag_up: int = TAG_ALLREDUCE_UP,
    tag_down: int = TAG_ALLREDUCE_DOWN,
) -> dict[str, Any]:
    """Root-side async PS entry point (called in a daemon thread by root_mpi).

    Returns the same result shape as run_logreg_allreduce_mpi() so the
    existing aggregate_logreg_results(allreduce_result=...) path works
    unchanged.
    """
    from mpi4py import MPI  # lazy: module must import without libmpi (CI)

    result = _serve(
        comm,
        num_workers=num_workers,
        num_features=num_features,
        num_iterations=num_iterations,
        any_source=MPI.ANY_SOURCE,
        server_lr=server_lr,
        staleness_damping=staleness_damping,
        tag_up=tag_up,
        tag_down=tag_down,
    )
    csv_path = _write_staleness_csv(result.pop("staleness_records"), results_dir)
    print(f"  [LogReg Async PS] Staleness log : {csv_path}")
    return result
