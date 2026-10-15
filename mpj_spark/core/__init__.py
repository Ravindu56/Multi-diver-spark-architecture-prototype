"""Core coordination package (lazy exports to avoid circular imports).

root_process imports mpj_spark.workers.worker_process, and worker_process
imports mpj_spark.core.sync_modes. Importing either package eagerly at
package-init time creates a cycle; PEP 562 lazy loading breaks it.
"""

__all__ = ["run_root"]


def __getattr__(name):
    if name == "run_root":
        from .root_process import run_root

        return run_root
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
