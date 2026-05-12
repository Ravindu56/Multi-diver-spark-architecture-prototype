# ================================================================
# mpj_spark/workers/worker_process.py
#
# Changes from feature/ml-kmeans-workload:
#
#   GOSSIP EXTENSION (feature/adaptive-gossip-aggregation):
#     - Added optional `gossip_queue` parameter.
#     - After KMeans completes, worker pushes its raw centroid state
#       into gossip_queue BEFORE putting final result into result_queue.
#     - This feeds the GossipAggregator in root_process.py.
#     - wordcount and non-gossip kmeans paths are completely unchanged.
#
#   LOGGING REFACTOR:
#     - [Worker N] phase labels now use INIT/LOAD/PROC/DONE/ERROR/STOP
#     - DevLogger.log_worker_timing() writes silently to file only
#       (no [DevLogger] stdout noise)
# ================================================================
import time
import traceback
from multiprocessing import Queue

from mpj_spark.workers.spark_session import build_spark_session
from mpj_spark.utils.dev_logger import DevLogger

_W = '[Worker {id}]'


def _tag(worker_id, phase):
    return f'[Worker {worker_id}][{phase}]'


def worker_process(
    worker_id:      int,
    partition_path: str,
    result_queue:   Queue,
    go_signal,
    ready_signal,
    timing_queue:   Queue,
    worker_config:  dict = None,
    gossip_queue:   Queue = None,
):
    if worker_config is None:
        worker_config = {}

    app_name       = worker_config.get('app',            'wordcount')
    cores_override = worker_config.get('cores_override',  None)
    kmeans_k       = int(worker_config.get('kmeans_k',        3))
    kmeans_iter    = int(worker_config.get('kmeans_max_iter', 20))
    num_workers    = worker_config.get('num_workers', 1)

    logger = DevLogger(worker_id=worker_id)

    try:
        # ── INIT ──────────────────────────────────────────────────────
        print(f'{_tag(worker_id, "INIT")} Starting SparkSession (app={app_name}) ...')
        t_init_start = time.perf_counter()
        spark = build_spark_session(
            app_name=f'MPJ-Worker-{worker_id}-{app_name}',
            cores_override=cores_override,
            num_workers=num_workers,
        )
        init_time = time.perf_counter() - t_init_start
        print(f'{_tag(worker_id, "INIT")} SparkSession ready  ({init_time:.3f}s)')

        # ── BARRIER ───────────────────────────────────────────────────
        ready_signal.set()
        print(f'{_tag(worker_id, "WAIT")} Waiting for go-signal ...')
        go_signal.wait()

        # ── LOAD ──────────────────────────────────────────────────────
        print(f'{_tag(worker_id, "LOAD")} Loading partition ...')
        t_load_start = time.perf_counter()

        if app_name == 'wordcount':
            text_rdd  = spark.sparkContext.textFile(partition_path)
            text_rdd.cache()
            row_count = text_rdd.count()
            load_time = time.perf_counter() - t_load_start
            print(f'{_tag(worker_id, "LOAD")} {row_count:,} rows  ({load_time:.3f}s)')
        else:
            load_time = 0.0

        # ── PROC ──────────────────────────────────────────────────────
        print(f'{_tag(worker_id, "PROC")} Running {app_name} ...')
        t_proc_start = time.perf_counter()

        if app_name == 'wordcount':
            from mpj_spark.applications import wordcount
            app_result = wordcount.run(text_rdd)

        elif app_name == 'kmeans':
            from mpj_spark.applications import kmeans
            app_result = kmeans.run(
                partition_path,
                k=kmeans_k,
                max_iter=kmeans_iter,
            )
            if gossip_queue is not None:
                gossip_queue.put({
                    'worker_id' : worker_id,
                    'centres'   : app_result['centres'],
                    'wcss'      : app_result['wcss'],
                    'row_count' : app_result['row_count'],
                })
                print(f'{_tag(worker_id, "PROC")} Centroid state → gossip_queue')

        else:
            raise ValueError(f"Unknown app '{app_name}'. Valid: 'wordcount', 'kmeans'")

        proc_time = time.perf_counter() - t_proc_start
        print(f'{_tag(worker_id, "DONE")} {app_name} complete  ({proc_time:.3f}s)')

        # ── QUEUE RESULTS ────────────────────────────────────────────
        result_queue.put({'worker_id': worker_id, 'result': app_result, 'status': 'success'})
        timing_queue.put({
            'worker_id'      : worker_id,
            'init_time'      : init_time,
            'load_time'      : load_time,
            'processing_time': proc_time,
            'total_time'     : init_time + load_time + proc_time,
        })
        # Silent file write — no console output
        logger.log_worker_timing(worker_id=worker_id, init_time=init_time,
                                 load_time=load_time, proc_time=proc_time)

    except Exception as exc:
        print(f'{_tag(worker_id, "ERROR")} {exc}')
        traceback.print_exc()
        result_queue.put({'worker_id': worker_id, 'result': None,
                          'status': 'error', 'error': str(exc)})
        timing_queue.put({'worker_id': worker_id, 'init_time': 0.0,
                          'load_time': 0.0, 'processing_time': 0.0, 'total_time': 0.0})

    finally:
        try:
            spark.stop()
            print(f'{_tag(worker_id, "STOP")} SparkSession stopped.')
        except Exception:
            pass
