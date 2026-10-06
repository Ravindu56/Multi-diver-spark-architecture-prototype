"""Worker package (lazy exports to avoid circular imports with core)."""

__all__ = ["_tag", "run_worker_core", "worker_process"]


def __getattr__(name):
    if name in __all__:
        from . import worker_process as _wp

        return getattr(_wp, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
