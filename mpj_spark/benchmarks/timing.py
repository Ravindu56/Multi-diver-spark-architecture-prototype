# ============================================================
# benchmarks/timing.py
# TimingCollector — lightweight stopwatch for T_Load / T_Proc metrics
# Paper Reference: Section VI.C — Timing analysis
# ============================================================
import time


class TimingCollector:
    """
    Named stopwatch manager.

    Usage:
        tc = TimingCollector()
        tc.start('load')
        ...work...
        tc.stop('load')
        print(tc.elapsed('load'))   # seconds as float
    """

    def __init__(self):
        self._starts:  dict = {}
        self._elapsed: dict = {}

    def start(self, name: str):
        self._starts[name] = time.time()

    def stop(self, name: str):
        if name not in self._starts:
            raise KeyError(f'Timer {name!r} was never started.')
        self._elapsed[name] = time.time() - self._starts[name]

    def elapsed(self, name: str) -> float:
        return self._elapsed.get(name, 0.0)

    def summary(self, worker_timings: list = None) -> dict:
        """
        Return a timing dict with standardised keys for print_comparison()
        and external callers.

        Keys returned
        -------------
        load_time       — T_Load (partitioning phase)
        processing_time — average per-worker T_Proc (actual computation).
                          This is what should be compared against standard
                          Spark's processing time, NOT the wall-clock parallel
                          time which unfairly includes JVM init overhead.
        total_time      — wall-clock total execution time
        parallel_time   — wall-clock time workers ran in parallel (internal)
        avg_init_time   — average per-worker JVM driver init time
        agg_time        — aggregation time at root
        """
        valid = [
            wt for wt in (worker_timings or [])
            if 'error' not in wt
        ]

        avg_proc = (
            sum(wt['processing'] for wt in valid) / len(valid)
            if valid else self._elapsed.get('parallel', 0.0)
        )
        avg_init = (
            sum(wt['driver_init'] for wt in valid) / len(valid)
            if valid else 0.0
        )

        return {
            # ── primary comparison keys (used by print_comparison) ──
            'load_time':       self._elapsed.get('load',     0.0),
            'processing_time': avg_proc,          # avg per-worker T_Proc
            'total_time':      self._elapsed.get('total',    0.0),
            # ── additional detail keys ──
            'parallel_time':   self._elapsed.get('parallel', 0.0),
            'avg_init_time':   avg_init,
            'agg_time':        self._elapsed.get('agg',      0.0),
        }
