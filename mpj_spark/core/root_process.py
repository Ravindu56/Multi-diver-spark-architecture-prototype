# ================================================================
# mpj_spark/core/root_process.py
# ================================================================
import math
import os
import time
from multiprocessing import Queue, Process, Event

from mpj_spark.core.file_manager import MPJSparkFileManager
from mpj_spark.core.key_value    import KeyValueStructure
from mpj_spark.workers.worker_process import worker_process
from mpj_spark.utils.dev_logger  import DevLogger

SEP  = '=' * 70
DASH = '─' * 70


def _hdr(title):
    print(f'\n{SEP}\n  {title}\n{SEP}')

def _phase(n, label):
    print(f'\n── Phase {n}: {label}')

def _ok(msg):
    print(f'  ✓  {msg}')

def _info(msg):
    print(f'     {msg}')


# ──────────────────────────────────────────────────────────────────
# Isolated seeding subprocess
# ──────────────────────────────────────────────────────────────────

def _seeding_worker(
    result_q: Queue,
    input_file: str,
    k: int,
    total_cores: int,
    sample_fraction: float,
    seed: int,
):
    try:
        from pyspark.sql import SparkSession
        from pyspark.ml.clustering import KMeans
        from pyspark.ml.feature import VectorAssembler

        spark = (
            SparkSession.builder
            .appName('MPJ-Root-Seeding')
            .master(f'local[{min(total_cores, 8)}]')
            .config('spark.ui.enabled', 'false')
            .config('spark.sql.shuffle.partitions', '8')
            .config('spark.driver.memory', '2g')
            .getOrCreate()
        )
        spark.sparkContext.setLogLevel('ERROR')

        df_raw  = spark.read.csv(input_file, inferSchema=True, header=False)
        df_samp = df_raw.sample(fraction=sample_fraction, seed=seed).dropna()

        assembler = VectorAssembler(
            inputCols=df_raw.columns, outputCol='features', handleInvalid='skip')
        df_vec = assembler.transform(df_samp).select('features')

        model   = KMeans(k=k, maxIter=20, seed=seed,
                         featuresCol='features', initMode='k-means||').fit(df_vec)
        centres = [c.tolist() for c in model.clusterCenters()]

        spark.stop()
        result_q.put({'status': 'ok', 'centres': centres})

    except Exception as exc:
        result_q.put({'status': 'error', 'msg': str(exc)})


def compute_global_seed_centres(
    input_file: str,
    k: int,
    total_cores: int,
    sample_fraction: float = 0.05,
    seed: int = 42,
) -> list:
    print(f'  [Seeding] Sampling {sample_fraction*100:.0f}% of dataset '
          f'for global seed centroids (isolated subprocess) ...')
    t0 = time.perf_counter()

    result_q = Queue()
    p = Process(
        target=_seeding_worker,
        args=(result_q, input_file, k, total_cores, sample_fraction, seed),
        daemon=False,
    )
    p.start()
    p.join()

    if p.exitcode != 0:
        raise RuntimeError(
            f'[Seeding] subprocess exited with code {p.exitcode}.')

    result = result_q.get_nowait()
    if result['status'] != 'ok':
        raise RuntimeError(f'[Seeding] seeding worker failed: {result["msg"]}')

    centres = result['centres']
    elapsed = time.perf_counter() - t0

    print(f'  [Seeding] {k} seed centroids computed in {elapsed:.3f}s')
    for i, c in enumerate(centres):
        preview = ', '.join(f'{v:.3f}' for v in c[:4])
        print(f'  [Seeding] Seed C{i}: [{preview}...]')

    return centres


# ──────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────

def dynamic_partition(input_path, num_partitions, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    manager = MPJSparkFileManager(shared_storage_path=output_dir)
    raw     = manager.dynamic_partition(input_path, num_partitions)
    paths   = []
    for item in raw:
        if isinstance(item, dict):
            p = item.get('path') or item.get('file_path') or item.get('partition_path')
            if p is None:
                p = list(item.values())[0]
            paths.append(str(p))
        else:
            paths.append(str(item))
    return paths


def align_centres_hungarian(reference, candidate):
    import numpy as np
    from scipy.optimize import linear_sum_assignment
    ref  = np.array(reference)
    cand = np.array(candidate)
    diff = ref[:, np.newaxis, :] - cand[np.newaxis, :, :]
    cost = np.linalg.norm(diff, axis=2)
    _, col_ind = linear_sum_assignment(cost)
    return [candidate[i] for i in col_ind.tolist()], col_ind.tolist()


def aggregate_kmeans_results(worker_results):
    import numpy as np
    total_rows        = sum(r['row_count'] for r in worker_results)
    k                 = worker_results[0]['k']
    num_dims          = len(worker_results[0]['centres'][0])
    reference_centres = worker_results[0]['centres']
    aligned_results   = [worker_results[0]]

    for w_idx, r in enumerate(worker_results[1:], start=1):
        aligned_centres, perm = align_centres_hungarian(reference_centres, r['centres'])
        _info(f'Worker {w_idx} centroid alignment (Hungarian): {perm}')
        aligned_results.append({**r, 'centres': aligned_centres})

    merged = []
    for c_idx in range(k):
        ws = np.zeros(num_dims)
        for r in aligned_results:
            ws += (r['row_count'] / total_rows) * np.array(r['centres'][c_idx])
        merged.append(ws.tolist())

    total_wcss = sum(r['wcss'] for r in worker_results)
    print(f'\n  K-Means aggregation complete')
    _info(f'Total rows : {total_rows:,}')
    _info(f'Total WCSS : {total_wcss:.4f}')
    _info(f'Workers    : {len(worker_results)}')
    print('  Global centres:')
    for i, c in enumerate(merged):
        preview = ', '.join(f'{v:.3f}' for v in c[:4])
        _info(f'  C{i}: [{preview}{"..." if len(c) > 4 else ""}]')
    return {'centres': merged, 'total_wcss': total_wcss,
            'total_rows': total_rows, 'num_workers': len(worker_results)}


# ──────────────────────────────────────────────────────────────────
# LogReg Allreduce coordinator  (root side)  — TWO-QUEUE design
# ──────────────────────────────────────────────────────────────────

def run_logreg_allreduce(
    up_queue:       Queue,   # workers → root  (weight vectors)
    down_queue:     Queue,   # root → workers  (averaged weights)
    num_workers:    int,
    num_iterations: int,
    num_features:   int,
) -> dict:
    """
    Root-side coordinator for per-iteration weight-vector Allreduce.

    TWO-QUEUE DESIGN
    ────────────────
    up_queue   — workers push {'type':'weights', ...} here.
    down_queue — root pushes {'type':'avg_weights', ...} here;
                 each worker reads exactly one message per iteration.

    Because writes and reads are on separate queues, there is no
    possibility of the coordinator reading back its own broadcast
    messages, eliminating the re-queue livelock of the single-queue
    design.

    Protocol per iteration:
      1. Collect N 'weights' messages from up_queue (any order).
      2. Compute FedAvg: weighted mean of weight vectors.
      3. Put N 'avg_weights' messages onto down_queue.
    """
    import numpy as np

    print(f'  [LogReg Allreduce] Starting — '
          f'{num_workers} workers × {num_iterations} iterations')

    final_weights   = None
    final_intercept = 0.0

    for iteration in range(num_iterations):
        t_iter = time.perf_counter()

        # Collect weight vectors from all workers via UP queue
        msgs = []
        while len(msgs) < num_workers:
            msg = up_queue.get(timeout=300)
            # UP queue only ever receives 'weights' messages — no re-queue needed
            msgs.append(msg)

        # FedAvg: weighted mean proportional to partition size
        total_rows    = sum(m['row_count'] for m in msgs)
        avg_w         = np.zeros(num_features)
        avg_intercept = 0.0
        for m in msgs:
            frac           = m['row_count'] / total_rows
            avg_w         += frac * np.array(m['weights'])
            avg_intercept += frac * m['intercept']

        final_weights   = avg_w.tolist()
        final_intercept = float(avg_intercept)

        # Broadcast averaged weights to all workers via DOWN queue
        for _ in range(num_workers):
            down_queue.put({
                'type'     : 'avg_weights',
                'iteration': iteration,
                'weights'  : final_weights,
                'intercept': final_intercept,
            })

        iter_time   = time.perf_counter() - t_iter
        weight_norm = float(np.linalg.norm(avg_w))
        print(f'  [LogReg Allreduce] iter {iteration+1}/{num_iterations}  '
              f'({iter_time:.3f}s)  |w|={weight_norm:.4f}')

    print(f'  [LogReg Allreduce] Complete')
    return {
        'weight_vector'  : final_weights,
        'intercept'      : final_intercept,
        'iterations_done': num_iterations,
    }


def aggregate_logreg_results(worker_results, allreduce_result=None):
    """
    Aggregate per-worker LogisticRegression results.
    """
    import numpy as np

    total_rows   = sum(r['row_count'] for r in worker_results)
    avg_accuracy = sum(
        r['train_accuracy'] * r['row_count'] / total_rows
        for r in worker_results
    )

    if allreduce_result is not None:
        final_weights   = allreduce_result['weight_vector']
        final_intercept = allreduce_result['intercept']
        agg_mode        = 'Allreduce (FedAvg)'
    else:
        num_features  = len(worker_results[0]['weight_vector'])
        avg_w         = np.zeros(num_features)
        avg_intercept = 0.0
        for r in worker_results:
            frac           = r['row_count'] / total_rows
            avg_w         += frac * np.array(r['weight_vector'])
            avg_intercept += frac * r['intercept']
        final_weights   = avg_w.tolist()
        final_intercept = float(avg_intercept)
        agg_mode        = 'Row-weighted mean (no Allreduce)'

    weight_norm = float(sum(w ** 2 for w in final_weights) ** 0.5)

    print(f'\n  LogReg aggregation complete  [{agg_mode}]')
    _info(f'Total rows       : {total_rows:,}')
    _info(f'Weighted accuracy: {avg_accuracy:.4f}')
    _info(f'Final |w|        : {weight_norm:.4f}')
    _info(f'Final intercept  : {final_intercept:.4f}')
    w_preview = ', '.join(f'{v:.4f}' for v in final_weights[:5])
    _info(f'Weight preview   : [{w_preview}{"..." if len(final_weights) > 5 else ""}]')

    return {
        'weight_vector'  : final_weights,
        'intercept'      : final_intercept,
        'avg_accuracy'   : avg_accuracy,
        'total_rows'     : total_rows,
        'num_workers'    : len(worker_results),
        'weight_norm'    : weight_norm,
        'agg_mode'       : agg_mode,
    }


def reassign_pass_root(
    processes_alive: list,
    gossip_centres: list,
    reassign_queue: Queue,
    num_workers: int,
    k: int,
    dims: int,
) -> list:
    import numpy as np

    print(f'  [Reassign] Broadcasting gossip-final centroids to {num_workers} workers ...')
    for _ in range(num_workers):
        reassign_queue.put({'type': 'reassign', 'centres': gossip_centres})

    all_sums   = np.zeros((k, dims))
    all_counts = np.zeros(k, dtype=np.int64)
    total_rows = 0
    received   = 0

    while received < num_workers:
        msg = reassign_queue.get(timeout=180)
        if msg.get('type') == 'stats':
            for j in range(k):
                all_sums[j]   += np.array(msg['cluster_sums'][j])
                all_counts[j] += msg['cluster_counts'][j]
            total_rows += msg['row_count']
            received   += 1

    corrected = []
    for j in range(k):
        if all_counts[j] > 0:
            corrected.append((all_sums[j] / all_counts[j]).tolist())
        else:
            corrected.append(gossip_centres[j])

    print(f'  [Reassign] Recomputed {k} exact global centroids from {total_rows:,} rows')
    for i, c in enumerate(corrected):
        preview = ', '.join(f'{v:.3f}' for v in c[:4])
        print(f'  [Reassign] C{i}: [{preview}...]')
    return corrected


# ──────────────────────────────────────────────────────────────────
# Output helpers
# ──────────────────────────────────────────────────────────────────

def _print_comparison(
    multi_timing, baseline_timing, num_workers, app, baseline_threads=None
):
    note = f'  [baseline-threads={baseline_threads}]' if baseline_threads else ''
    print(f'\n{SEP}')
    print(f'  Multi-Driver vs Baseline  |  app={app}  |  workers={num_workers}{note}')
    print(SEP)
    print(f'  {"Metric":<28} {"Multi-Driver":>13} {"Baseline":>13} {"Speedup":>9}')
    print(f'  {"-"*28} {"-"*13} {"-"*13} {"-"*9}')

    rows = [
        ('load_time',       'Load Time (s)'),
        ('processing_time', 'Proc Time — fit only (s)'),
    ]

    reassign_val = multi_timing.get('reassign_time')
    if reassign_val is not None:
        rows.append(('_reassign', 'Re-assign Pass (s)'))

    rows.append(('total_time', 'Total Time (s)'))

    for key, label in rows:
        if key == '_reassign':
            m  = reassign_val
            print(f'  {label:<28} {m:>12.4f}s {"N/A":>13} {"—":>9}')
            continue
        m  = multi_timing[key]
        b  = baseline_timing.get(key, 0.0)
        sp = b / m if m > 0 else 0.0
        flag = '  ⚡' if sp >= 1.5 else ('  ⚠' if sp < 1.0 else '')
        print(f'  {label:<28} {m:>12.4f}s {b:>12.4f}s {sp:>8.2f}x{flag}')

    print(SEP)


def _print_timing_summary(load_time, avg_proc, agg_time, t_wall,
                          prewarm_init=None, gossip_info=None,
                          seed_time=None, reassign_time=None):
    print(f'\n{DASH}')
    print('  Timing Summary')
    print(DASH)
    if seed_time is not None:
        print(f'  {"Global Seed Sampling":<28} {seed_time:>8.4f} s')
    print(f'  {"Partition / Load":<28} {load_time:>8.4f} s')
    print(f'  {"Avg Worker Proc":<28} {avg_proc:>8.4f} s')
    print(f'  {"Gossip / Allreduce Agg":<28} {agg_time:>8.4f} s')
    if reassign_time is not None:
        print(f'  {"Re-assignment Pass":<28} {reassign_time:>8.4f} s')
    if prewarm_init is not None:
        print(f'  {"Avg Worker JVM Init (excl.)":<28} {prewarm_init:>8.4f} s')
    if gossip_info:
        rounds = gossip_info.get('rounds_run') or gossip_info.get('iterations_done', '—')
        converged = gossip_info.get('converged', '—')
        print(f'  {"Allreduce Rounds / Iters":<28} {str(rounds):>8}')
        if converged != '—':
            print(f'  {"Converged":<28} {str(converged):>8}')
    print(f'  {"-"*40}')
    print(f'  {"Total Wall-clock":<28} {t_wall:>8.4f} s')
    print(DASH)


# ──────────────────────────────────────────────────────────────────
# Main entry point
# ──────────────────────────────────────────────────────────────────

def run_root(
    input_file,
    num_workers=2,
    compare=False,
    prewarm=True,
    cores_override=None,
    app='wordcount',
    kmeans_k=3,
    kmeans_iter=20,
    baseline_threads=None,
    use_gossip=False,
    gossip_threshold=1e-3,
    gossip_max_rounds=10,
    gossip_fanout=2,
    use_global_seed=True,
    use_reassign=True,
    logreg_iter=10,
    logreg_reg_param=0.01,
    logreg_features=10,
):
    from mpj_spark.config import TOTAL_CORES, DATA_DIR
    logger = DevLogger(worker_id='root')

    do_seed             = use_global_seed and use_gossip and app == 'kmeans'
    do_reassign         = use_reassign    and use_gossip and app == 'kmeans'
    do_logreg_allreduce = (app == 'logreg')

    agg_mode = (
        f'Adaptive Gossip  (threshold={gossip_threshold}, '
        f'max_rounds={gossip_max_rounds}, fanout={gossip_fanout})'
        if (use_gossip and app == 'kmeans')
        else 'Batch Hungarian' if app == 'kmeans'
        else f'Allreduce (FedAvg, {logreg_iter} iters)' if app == 'logreg'
        else 'N/A'
    )
    correctness_flags = ''
    if use_gossip and app == 'kmeans':
        correctness_flags = (
            f'  Global Seed     : {"ON" if do_seed else "OFF"}\n'
            f'  Re-assign Pass  : {"ON" if do_reassign else "OFF"}'
        )
    if app == 'logreg':
        title_extra = (
            f'  iter={logreg_iter}  reg_param={logreg_reg_param}  features={logreg_features}\n'
            f'  Aggregation : {agg_mode}'
        )
    elif app == 'kmeans':
        title_extra = (
            f'  k={kmeans_k}  max_iter={kmeans_iter}\n'
            f'  Aggregation : {agg_mode}\n'
            f'{correctness_flags}'
        )
    else:
        title_extra = ''

    _hdr(
        f'MPJ-Spark Multi-Driver  |  app={app}  |  workers={num_workers}\n'
        f'{title_extra}'
    )

    cores = max(1, cores_override) if cores_override else max(1, math.ceil(TOTAL_CORES / num_workers))
    print(f'  Core budget : local[{cores}]  ({TOTAL_CORES} total ÷ {num_workers} workers)')

    seed_centres = None
    seed_time    = None

    if do_seed:
        _phase('1b', 'Computing global seed centroids (Option 1 — isolated subprocess)')
        t_seed = time.perf_counter()
        seed_centres = compute_global_seed_centres(
            input_file=input_file,
            k=kmeans_k,
            total_cores=TOTAL_CORES,
            sample_fraction=0.05,
            seed=42,
        )
        seed_time = time.perf_counter() - t_seed
        _ok(f'Global seed centroids ready  ({seed_time:.3f}s)  '
            f'[seeding JVM fully terminated before workers fork]')

    worker_cfg = {
        'app'             : app,
        'cores_override'  : cores,
        'kmeans_k'        : kmeans_k,
        'kmeans_max_iter' : kmeans_iter,
        'num_workers'     : num_workers,
        'seed_centres'    : seed_centres,
        'logreg_iter'     : logreg_iter,
        'logreg_reg_param': logreg_reg_param,
        'logreg_features' : logreg_features,
    }

    # ── Phase 1: Partition ─────────────────────────────────────────
    _phase(1, 'Partitioning')
    t_load_start    = time.perf_counter()
    partition_paths = dynamic_partition(input_file, num_workers, DATA_DIR)
    load_time       = time.perf_counter() - t_load_start
    _ok(f'Split into {num_workers} partitions  ({load_time:.3f}s)')

    # ── Phase 2: Launch workers ────────────────────────────────────
    _phase(2, f'Launching {num_workers} workers')
    result_queue   = Queue()
    timing_queue   = Queue()
    go_signals     = [Event() for _ in range(num_workers)]
    ready_signals  = [Event() for _ in range(num_workers)]
    processes      = []

    # k-means gossip uses a single shared queue (unchanged)
    gossip_queue   = Queue() if (use_gossip and app == 'kmeans') else None
    reassign_queue = Queue() if do_reassign else None

    # LogReg Allreduce: TWO separate queues — up (workers→root) and down (root→workers)
    # gossip_queue is repurposed as the UP queue when app == 'logreg'
    allreduce_up_queue   = None
    allreduce_down_queue = None
    if do_logreg_allreduce:
        allreduce_up_queue   = Queue()   # workers push weight vectors here
        allreduce_down_queue = Queue()   # root pushes averaged weights here

    for i in range(num_workers):
        p = Process(
            target=worker_process,
            args=(i, partition_paths[i], result_queue,
                  go_signals[i], ready_signals[i], timing_queue, worker_cfg,
                  # gossip_queue slot: k-means uses gossip_queue;
                  # logreg repurposes it as allreduce_up_queue
                  gossip_queue if app == 'kmeans' else allreduce_up_queue,
                  reassign_queue),
            kwargs={'allreduce_down_queue': allreduce_down_queue},
            daemon=True)
        p.start()
        processes.append(p)
        _info(f'Worker {i} started  (PID {p.pid})')

    print('  Waiting for JVM barrier ...')
    for i, sig in enumerate(ready_signals):
        sig.wait()
        _ok(f'Worker {i} JVM ready')

    # ── Phase 3: Fire ──────────────────────────────────────────────
    _phase(3, 'Firing all workers simultaneously')
    t_proc_start = time.perf_counter()
    for sig in go_signals:
        sig.set()

    # ── Phase 3b: LogReg Allreduce coordinator (background thread) ─
    allreduce_result = None
    allreduce_thread = None

    if do_logreg_allreduce:
        import threading
        _allreduce_container = []

        def _allreduce_thread_fn():
            res = run_logreg_allreduce(
                up_queue       = allreduce_up_queue,
                down_queue     = allreduce_down_queue,
                num_workers    = num_workers,
                num_iterations = logreg_iter,
                num_features   = logreg_features,
            )
            _allreduce_container.append(res)

        allreduce_thread = threading.Thread(target=_allreduce_thread_fn, daemon=True)
        allreduce_thread.start()
        print(f'  [LogReg Allreduce] Coordinator started in background thread')

    # ── Phase 4: Collect ───────────────────────────────────────────
    _phase(4, 'Collecting results')
    worker_results = []
    worker_timings = []
    errors         = []
    for _ in range(num_workers):
        res = result_queue.get(timeout=600)
        if res['status'] == 'success':
            worker_results.append(res['result'])
        else:
            errors.append(res)
            print(f"  ✗  Worker {res['worker_id']} FAILED: {res.get('error')}")
    for _ in range(num_workers):
        worker_timings.append(timing_queue.get(timeout=60))
    proc_time = time.perf_counter() - t_proc_start
    if errors:
        for p in processes:
            p.join(timeout=5)
        print(f'  {len(errors)} worker(s) failed. Aborting.')
        return
    _ok(f'All {num_workers} workers completed')

    if allreduce_thread is not None:
        allreduce_thread.join(timeout=60)
        if _allreduce_container:
            allreduce_result = _allreduce_container[0]

    # ── Phase 5: Aggregate ─────────────────────────────────────────
    _phase(5, 'Aggregating results')
    t_agg_start = time.perf_counter()
    gossip_info = None
    agg         = None

    if app == 'wordcount':
        kv = KeyValueStructure()
        for r in worker_results:
            kv.merge(KeyValueStructure.from_serializable(r))
        top = kv.get_top_n(20)
        print('\n  Top-20 words:')
        for word, count in top:
            print(f'    {word:<22} {count:>12,}')

    elif app == 'kmeans':
        if use_gossip and gossip_queue is not None:
            print('  Using Adaptive Gossip aggregation ...')
            from mpj_spark.core.gossip_aggregator import GossipAggregator
            gagg = GossipAggregator(
                num_workers=num_workers,
                convergence_threshold=gossip_threshold,
                max_rounds=gossip_max_rounds,
                initial_fanout=gossip_fanout,
                verbose=True,
            )
            agg = gagg.aggregate(
                gossip_queue,
                timeout_per_worker=120.0,
                seed_centres=seed_centres,
            )
            gossip_info = agg
            _ok(f'Gossip done  rounds={agg["rounds_run"]}  converged={agg["converged"]}')
        else:
            agg = aggregate_kmeans_results(worker_results)

        _info(f'Total rows : {agg["total_rows"]:,}')
        _info(f'Total WCSS : {agg["total_wcss"]:.4f}')

    elif app == 'logreg':
        agg = aggregate_logreg_results(worker_results, allreduce_result=allreduce_result)
        gossip_info = {
            'iterations_done': agg['iterations_done'] if 'iterations_done' in agg
                               else logreg_iter,
        }
        _info(f'Total rows       : {agg["total_rows"]:,}')
        _info(f'Weighted accuracy: {agg["avg_accuracy"]:.4f}')
        _info(f'Agg mode         : {agg["agg_mode"]}')

    agg_time      = time.perf_counter() - t_agg_start
    reassign_time = None

    if do_reassign and agg is not None and app == 'kmeans':
        _phase('5b', 'Re-assignment pass (Option 2) — exact global centroid correction')
        t_reassign     = time.perf_counter()
        gossip_centres = agg['centres']
        k_val          = len(gossip_centres)
        d_val          = len(gossip_centres[0])

        corrected_centres = reassign_pass_root(
            processes_alive=processes,
            gossip_centres=gossip_centres,
            reassign_queue=reassign_queue,
            num_workers=num_workers,
            k=k_val,
            dims=d_val,
        )
        reassign_time  = time.perf_counter() - t_reassign
        agg['centres'] = corrected_centres

        print(f'\n{DASH}')
        print('  Final Corrected Centres (post re-assignment):')
        for i, c in enumerate(corrected_centres):
            preview = ', '.join(f'{v:.3f}' for v in c[:4])
            _info(f'C{i}: [{preview}{"..." if d_val > 4 else ""}]')
        print(DASH)
        _ok(f'Re-assignment done  ({reassign_time:.3f}s)')

    for p in processes:
        p.join(timeout=30)

    t_wall   = load_time + proc_time + agg_time + (reassign_time or 0.0)
    avg_proc = sum(t['processing_time'] for t in worker_timings) / num_workers
    avg_init = sum(t['init_time'] for t in worker_timings) / num_workers if prewarm else None

    _print_timing_summary(
        load_time, avg_proc, agg_time, t_wall,
        prewarm_init=avg_init,
        gossip_info=gossip_info,
        seed_time=seed_time,
        reassign_time=reassign_time,
    )

    logger.log_run(app=app, num_workers=num_workers, cores=cores,
                   load_time=load_time, proc_time=avg_proc,
                   agg_time=agg_time, total_time=t_wall)

    if compare:
        _phase('B', f'Running {app} baseline for comparison')
        if app == 'wordcount':
            from mpj_spark.applications.baseline_spark import run_baseline
            _, baseline_timing = run_baseline(input_file, num_workers, cores_override)
        elif app == 'kmeans':
            from mpj_spark.applications.baseline_kmeans import run_baseline_kmeans
            _, baseline_timing = run_baseline_kmeans(
                input_file, num_workers, cores_override,
                kmeans_k, kmeans_iter, baseline_threads=baseline_threads)
        elif app == 'logreg':
            from mpj_spark.applications.baseline_logreg import run_baseline_logreg
            _, baseline_timing = run_baseline_logreg(
                input_file, num_workers, cores_override,
                logreg_iter, logreg_reg_param, logreg_features,
                baseline_threads=baseline_threads)

        multi_timing = {
            'load_time'       : load_time,
            'processing_time' : avg_proc,
            'reassign_time'   : reassign_time,
            'total_time'      : t_wall,
        }
        _print_comparison(multi_timing, baseline_timing, num_workers, app,
                          baseline_threads=baseline_threads)


mpj_root_process = run_root
