# ================================================================
# mpj_spark/workers/worker_process.py
#
# Single worker subprocess — launched by root_process.run_root().
#
# Dispatch table (logreg):
#   sync_mode='queue'  →  logreg.queue_run.run()   [M2 — FedAvg]
#   sync_mode='none'   →  logreg.nosync_run.run()  [M1 — no sync]
#
# All other apps (wordcount, kmeans) are unaffected.
# ================================================================
import math
import time


def worker_process(
    worker_id,
    partition_path,
    result_queue,
    go_signal,
    ready_signal,
    timing_queue,
    worker_cfg,
    allreduce_up_queue=None,
    reassign_queue=None,
    allreduce_down_queue=None,
):
    from mpj_spark.config import TOTAL_CORES

    app              = worker_cfg.get('app', 'wordcount')
    cores            = worker_cfg.get('cores_override', max(1, TOTAL_CORES // 2))
    num_workers      = worker_cfg.get('num_workers', 1)
    results_dir      = worker_cfg.get('results_dir', 'results')
    # sync_mode drives M1 vs M2 dispatch for logreg
    sync_mode        = worker_cfg.get('sync_mode', 'queue')

    try:
        from pyspark.sql import SparkSession
        t_init_start = time.perf_counter()
        spark = (
            SparkSession.builder
            .appName(f'MPJ-Worker-{worker_id}')
            .master(f'local[{cores}]')
            .config('spark.ui.enabled', 'false')
            .config('spark.sql.shuffle.partitions', str(cores * 2))
            .config('spark.memory.fraction', '0.8')
            .config('spark.memory.storageFraction', '0.2')
            .getOrCreate()
        )
        spark.sparkContext.setLogLevel('ERROR')
        init_time = time.perf_counter() - t_init_start

        ready_signal.set()
        go_signal.wait()

        t_proc_start = time.perf_counter()

        # ── dispatch ───────────────────────────────────────────────────
        if app == 'wordcount':
            from mpj_spark.applications.wordcount import run_wordcount
            result = run_wordcount(partition_path, spark)

        elif app == 'kmeans':
            from mpj_spark.applications.kmeans import run_kmeans
            result = run_kmeans(
                partition_path,
                k=worker_cfg.get('kmeans_k', 3),
                max_iter=worker_cfg.get('kmeans_max_iter', 20),
                worker_id=worker_id,
                gossip_queue=allreduce_up_queue,
                reassign_queue=reassign_queue,
                seed_centres=worker_cfg.get('seed_centres'),
                num_workers=num_workers,
            )

        elif app == 'logreg':
            logreg_kwargs = dict(
                partition_path=partition_path,
                max_iter=worker_cfg.get('logreg_iter', 10),
                reg_param=worker_cfg.get('logreg_reg_param', 0.01),
                num_features=worker_cfg.get('logreg_features', 10),
                worker_id=worker_id,
                num_workers=num_workers,
                results_dir=results_dir,
            )

            if sync_mode == 'none':
                # M1 — Multi-driver, NO synchronisation
                from mpj_spark.applications.logreg import nosync_run
                result = nosync_run.run(**logreg_kwargs)

            else:
                # M2 — Multi-driver, Queue/FedAvg (default)
                from mpj_spark.applications.logreg import queue_run
                result = queue_run.run(
                    **logreg_kwargs,
                    allreduce_up_queue=allreduce_up_queue,
                    allreduce_down_queue=allreduce_down_queue,
                )

        else:
            raise ValueError(f'Unknown app: {app!r}')

        proc_time = time.perf_counter() - t_proc_start

        result_queue.put({'status': 'success', 'worker_id': worker_id, 'result': result})
        timing_queue.put({
            'worker_id':       worker_id,
            'init_time':       init_time,
            'processing_time': proc_time,
        })

    except Exception as exc:
        import traceback
        result_queue.put({
            'status':    'error',
            'worker_id': worker_id,
            'error':     str(exc),
            'traceback': traceback.format_exc(),
        })
        timing_queue.put({
            'worker_id':       worker_id,
            'init_time':       0.0,
            'processing_time': 0.0,
        })
    finally:
        try:
            from pyspark.sql import SparkSession
            active = SparkSession.getActiveSession()
            if active:
                active.stop()
        except Exception:
            pass
