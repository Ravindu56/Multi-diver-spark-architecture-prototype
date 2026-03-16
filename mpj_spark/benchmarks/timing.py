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

    def summary(self) -> dict:
        return dict(self._elapsed)
