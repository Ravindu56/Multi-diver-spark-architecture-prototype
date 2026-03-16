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
    total    = tc.elapsed('total')    or 1.0
    parallel = tc.elapsed('parallel') or 0.0
    load     = tc.elapsed('load')     or 0.0
    agg      = tc.elapsed('agg')      or 0.0

    def pct(val):
        return (val / total) * 100

    avg_init = avg_proc = 0.0
    valid = [wt for wt in (worker_timings or []) if 'error' not in wt]
    if valid:
        avg_init = sum(wt['driver_init'] for wt in valid) / len(valid)
        avg_proc = sum(wt['processing']  for wt in valid) / len(valid)

    print('\n' + '=' * 70)
    print('  TIMING ANALYSIS  (Paper Metrics)')
    print('=' * 70)
    print(f'  Load Time      (T_Load) : {load:>8.4f} s  ({pct(load):>5.1f}% of total)')
    print(f'  Driver Init    (T_Init) : {avg_init:>8.4f} s  (avg per worker)')
    print(f'  Processing     (T_Proc) : {avg_proc:>8.4f} s  (avg per worker)')
    print(f'  Aggregation    (T_Agg)  : {agg:>8.4f} s  ({pct(agg):>5.1f}% of total)')
    print(f'  Wall-clock parallel     : {parallel:>8.4f} s')
    print(f'  Total Execution Time    : {total:>8.4f} s')

    if valid:
        print(f'\n  {"Worker":<8} {"Driver Init":>12} {"Processing":>12} {"Total":>10}')
        print(f'  {"-"*44}')
        for wt in sorted(valid, key=lambda x: x['worker_id']):
            print(f"  Worker {wt['worker_id']:<3} "
                  f"{wt['driver_init']:>10.2f} s  "
                  f"{wt['processing']:>10.2f} s  "
                  f"{wt['total']:>8.2f} s")


def print_comparison(multi_timing: dict, std_timing: dict):
    """
    Print side-by-side comparison table.

    multi_timing keys expected:
      load_time       — T_Load (partitioning)
      processing_time — avg per-worker T_Proc (NOT wall-clock parallel)
      total_time      — wall-clock total

    std_timing keys expected:
      load_time, processing_time, total_time
    """
    print('\n' + '=' * 70)
    print('  COMPARISON: Multi-Driver  vs  Standard Spark')
    print('=' * 70)
    print(f"  {'Metric':<30} {'Multi-Driver':>14} {'Std Spark':>12} {'Speedup':>10}")
    print(f"  {'-' * 68}")

    def _row(label, key, note=''):
        mv      = multi_timing.get(key, 0.0)
        sv      = std_timing.get(key,   0.0)
        speedup = sv / max(mv, 0.0001)
        flag    = '\u2713 faster' if speedup >= 1.0 else '\u2717 slower'
        suffix  = f'  [{note}]' if note else ''
        print(f'  {label:<30} {mv:>14.4f} {sv:>12.4f} '
              f'{speedup:>8.2f}x  {flag}{suffix}')

    _row('Load Time (sec)',              'load_time')
    _row('Avg Worker Proc Time (sec)',   'processing_time',
         note='avg per-worker T_Proc')
    _row('Total Wall-clock (sec)',       'total_time')

    # ── Extra context row: show wall-clock parallel explicitly ──
    p_time = multi_timing.get('parallel_time', 0.0)
    if p_time:
        print(f"  {'Wall-clock Parallel (sec)':<30} {p_time:>14.4f} "
              f"{'—':>12}   {'—':>8}     [incl. JVM init]")
