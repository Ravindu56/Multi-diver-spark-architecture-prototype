"""mpj_spark/core/sync_modes.py
Central Synchronization Mode Registry & Protocol Specifications.

Defines standard identifiers, validation rules, and metadata for all
cross-driver parameter synchronization strategies across Phases 2-3:
- NONE                  ("none"): M1 independent training, post-hoc aggregation
- PS_SYNC_FEDAVG_QUEUE  ("ps_sync_fedavg_queue"): M2 Queue-based periodic FedAvg
- PS_SYNC_FEDAVG_MPI    ("ps_sync_fedavg_mpi"): M2-MPI mpi4py periodic FedAvg
- ALLREDUCE_MPI         ("allreduce_mpi"): M3 per-iteration Allreduce collective
- PS_ASYNC              ("ps_async"): P3-09 asynchronous parameter server (MPI P2P)
- HYBRID_PS_ALLREDUCE   ("hybrid_ps_allreduce"): P3-10 hybrid split — dense
  weights via Allreduce collective, scalars via root parameter server
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

# Canonical mode strings
MODE_NONE = "none"
MODE_PS_SYNC_FEDAVG_QUEUE = "ps_sync_fedavg_queue"
MODE_PS_SYNC_FEDAVG_MPI = "ps_sync_fedavg_mpi"
MODE_ALLREDUCE_MPI = "allreduce_mpi"
MODE_PS_ASYNC = "ps_async"
MODE_HYBRID_PS_ALLREDUCE = "hybrid_ps_allreduce"

# Legacy aliases accepted during CLI normalization
_ALIASES: dict[str, str] = {
    "none": MODE_NONE,
    "no_sync": MODE_NONE,
    "queue": MODE_PS_SYNC_FEDAVG_QUEUE,
    "fedavg_queue": MODE_PS_SYNC_FEDAVG_QUEUE,
    "ps_sync_fedavg_queue": MODE_PS_SYNC_FEDAVG_QUEUE,
    "mpi_fedavg": MODE_PS_SYNC_FEDAVG_MPI,
    "fedavg_mpi": MODE_PS_SYNC_FEDAVG_MPI,
    "ps_sync_fedavg_mpi": MODE_PS_SYNC_FEDAVG_MPI,
    "mpi": MODE_ALLREDUCE_MPI,
    "allreduce": MODE_ALLREDUCE_MPI,
    "allreduce_mpi": MODE_ALLREDUCE_MPI,
    "async": MODE_PS_ASYNC,
    "async_ps": MODE_PS_ASYNC,
    "ps_async": MODE_PS_ASYNC,
    "hybrid": MODE_HYBRID_PS_ALLREDUCE,
    "hybrid_ps_allreduce": MODE_HYBRID_PS_ALLREDUCE,
}

SyncMode = Literal[
    "none",
    "ps_sync_fedavg_queue",
    "ps_sync_fedavg_mpi",
    "allreduce_mpi",
    "ps_async",
    "hybrid_ps_allreduce",
]


@dataclass(frozen=True)
class SyncModeDescriptor:
    """Metadata describing a synchronization strategy."""

    name: str
    transport: str
    description: str
    is_periodic: bool
    requires_mpi: bool


REGISTRY: dict[str, SyncModeDescriptor] = {
    MODE_NONE: SyncModeDescriptor(
        name=MODE_NONE,
        transport="local",
        description="M1 - Independent local training, post-hoc aggregation",
        is_periodic=False,
        requires_mpi=False,
    ),
    MODE_PS_SYNC_FEDAVG_QUEUE: SyncModeDescriptor(
        name=MODE_PS_SYNC_FEDAVG_QUEUE,
        transport="multiprocessing.Queue",
        description="M2 - Periodic FedAvg over multiprocessing Queues",
        is_periodic=True,
        requires_mpi=False,
    ),
    MODE_PS_SYNC_FEDAVG_MPI: SyncModeDescriptor(
        name=MODE_PS_SYNC_FEDAVG_MPI,
        transport="mpi4py (gather/bcast)",
        description="P3-08 - Periodic FedAvg over native MPI collectives",
        is_periodic=True,
        requires_mpi=True,
    ),
    MODE_ALLREDUCE_MPI: SyncModeDescriptor(
        name=MODE_ALLREDUCE_MPI,
        transport="mpi4py (Allreduce)",
        description="M3 - Synchronous per-iteration Allreduce collective",
        is_periodic=False,
        requires_mpi=True,
    ),
    MODE_PS_ASYNC: SyncModeDescriptor(
        name=MODE_PS_ASYNC,
        transport="mpi4py (P2P send/recv)",
        description="P3-09 - Asynchronous parameter server, FedAsync-style mixing",
        is_periodic=True,
        requires_mpi=True,
    ),
    MODE_HYBRID_PS_ALLREDUCE: SyncModeDescriptor(
        name=MODE_HYBRID_PS_ALLREDUCE,
        transport="mpi4py (Allreduce + P2P)",
        description="P3-10 - Hybrid split: dense weights via Allreduce, scalars via root PS",
        is_periodic=True,
        requires_mpi=True,
    ),
}


def normalize_sync_mode(mode: str | None, default: str = MODE_PS_SYNC_FEDAVG_MPI) -> str:
    """Normalize user input or legacy mode string to canonical SyncMode name."""
    if mode is None:
        return default
    cleaned = mode.strip().lower()
    if cleaned in _ALIASES:
        return _ALIASES[cleaned]
    valid = sorted(list(REGISTRY.keys()))
    raise ValueError(f"Unknown sync_mode '{mode}'. Must be one of: {valid}")


def get_descriptor(mode: str) -> SyncModeDescriptor:
    """Retrieve metadata descriptor for a canonical sync mode."""
    canonical = normalize_sync_mode(mode)
    return REGISTRY[canonical]


def is_mpi_required(mode: str) -> bool:
    """Check if the given sync mode requires an active MPI communicator."""
    return get_descriptor(mode).requires_mpi
