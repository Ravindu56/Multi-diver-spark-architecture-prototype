# ============================================================
# benchmarks/reporter.py
# Console reporting for results and timing comparison tables
# ============================================================
from mpj_spark.benchmarks.timing import TimingCollector


def print_results(sorted_results: list, top_n: int = 20):
    """Print final aggregated word counts."""
    total_occurrences = sum(v for _, v in sorted_results)
    print('\n' + '=' * 70)
    print('  RESULTS')
    print('=' * 70)
    print(f'  Total unique items    : {len(sorted_results):,}')
    print(f'  Total occurrences     : {total_occurrences:,}')
    print(f'\n  Top {top_n} items:')
    for key, cnt in sorted_results[:top_n]:
        print(f'    {key:20s} -> {cnt:,}')


def print_timing(tc: TimingCollector, worker_timings: list = None):
    """Print timing breakdown (paper metrics format)."""
    total = tc.elapsed('total') or 1.0

    def pct(key):
        return (tc.elapsed(key) / total) * 100

    avg_init = avg_proc = 0.0
    valid = [wt for wt in (worker_timings or []) if 'error' not in wt]
    if valid:
        avg_init = sum(wt['driver_init'] for wt in valid) / len(valid)
        avg_proc = sum(wt['processing']  for wt in valid) / len(valid)

    print('\n' + '=' * 70)
    print('  TIMING ANALYSIS  (Paper Metrics)')
    print('=' * 70)
    print(f"  Load Time      (T_Load) : {tc.elapsed('load'):>8.4f} s  "
          f"({pct('load'):>5.1f}% of total)")
    print(f"  Driver Init    (T_Init) : {avg_init:>8.4f} s  "
          f"(avg per worker)")
    print(f"  Processing     (T_Proc) : {avg_proc:>8.4f} s  "
          f"(avg per worker)")
    print(f"  Aggregation    (T_Agg)  : {tc.elapsed('agg'):>8.4f} s  "
          f"({pct('agg'):>5.1f}% of total)")
    print(f"  Wall-clock parallel     : {tc.elapsed('parallel'):>8.4f} s")
    print(f"  Total Execution Time    : {total:>8.4f} s")

    if valid:
        print(f'\n  {"Worker":<8} {"Driver Init":>12} {"Processing":>12} {"Total":>10}')
        print(f'  {"-"*44}')
        for wt in sorted(valid, key=lambda x: x['worker_id']):
            print(f"  Worker {wt['worker_id']:<3} "
                  f"{wt['driver_init']:>10.2f} s  "
                  f"{wt['processing']:>10.2f} s  "
                  f"{wt['total']:>8.2f} s")


def print_comparison(multi_timing: dict, std_timing: dict):
    """Print side-by-side comparison table."""
    print('\n' + '=' * 70)
    print('  COMPARISON: Multi-Driver  vs  Standard Spark')
    print('=' * 70)
    print(f"  {'Metric':<26} {'Multi-Driver':>14} {'Std Spark':>12} {'Speedup':>10}")
    print(f"  {'-' * 64}")

    def _row(label, mk, sk):
        mv = multi_timing.get(mk, 0)
        sv = std_timing.get(sk, 0)
        speedup = sv / max(mv, 0.0001)
        flag = '✓ faster' if speedup >= 1.0 else '✗ slower'
        print(f"  {label:<26} {mv:>14.4f} {sv:>12.4f} "
              f"{speedup:>8.2f}x  {flag}")

    _row('Load Time (sec)',       'load_time',       'load_time')
    _row('Processing Time (sec)', 'processing_time', 'processing_time')
    _row('Total Time (sec)',      'total_time',       'total_time')
