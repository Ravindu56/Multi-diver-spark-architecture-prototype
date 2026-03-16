# ============================================================
# workers/worker_process.py
# MPJ Worker Process — each owns an independent Spark Driver
# Paper Reference: Section IV.A + Algorithm 1
# ============================================================
import time
from multiprocessing import Queue

from mpj_spark.core.key_value        import KeyValueStructure
from mpj_spark.workers.spark_session  import build_spark_session


def mpj_worker_process(worker_id:          int,
                       partition_metadata:  dict,
                       result_queue:        Queue,
                       timing_queue:        Queue,
                       app:                 str   = 'wordcount',
                       ready_queue:         Queue = None,
                       go_queue:            Queue = None,
                       cores_per_worker:    int   = None):
    """
    MPJ Worker Process (Paper §IV.A) with JVM pre-warm and fair-core support.

    Parameters
    ----------
    worker_id        : unique worker identifier
    partition_metadata: partition path + line counts from Root
    result_queue     : send computed results back to Root
    timing_queue     : send timing breakdown back to Root
    app              : application key ('wordcount', ...)
    ready_queue      : signal Root when JVM is warmed (None = legacy)
    go_queue         : receive start signal from Root (None = legacy)
    cores_per_worker : cores for local[N]; None = auto via build_spark_session
    """
    try:
        # ── PHASE A: JVM INIT (not timed as computation) ──────────────────
        t_spawn = time.time()

        spark = build_spark_session(
            app_name=f'MPJ-Worker-{worker_id}',
            cores_override=cores_per_worker,
        )
        sc = spark.sparkContext

        t_init_done   = time.time()
        jvm_init_time = t_init_done - t_spawn

        if ready_queue is not None:
            ready_queue.put({'worker_id': worker_id, 'init_time': jvm_init_time})
            go_queue.get()   # block until Root fires go-signal

        # ── PHASE B: COMPUTATION (pure T_Proc) ─────────────────────────
        t_compute_start = time.time()

        partition_path = partition_metadata['partition_path']
        text_rdd = sc.textFile(partition_path)
        results  = _run_application(sc, text_rdd, app)

        t_compute_end = time.time()
        compute_time  = t_compute_end - t_compute_start

        kv = KeyValueStructure().from_rdd_collect(results)

        result_queue.put({
            'worker_id':       worker_id,
            'results':         kv.to_serializable(),
            'num_items':       len(kv.data),
            'partition_lines': partition_metadata['num_lines'],
        })
        timing_queue.put({
            'worker_id':   worker_id,
            'driver_init': jvm_init_time,
            'processing':  compute_time,
            'total':       t_compute_end - t_spawn,
        })

        spark.stop()

    except Exception as exc:
        result_queue.put({'worker_id': worker_id, 'error': str(exc)})
        timing_queue.put({'worker_id': worker_id, 'error': str(exc)})


def _run_application(sc, text_rdd, app: str):
    if app == 'wordcount':
        from mpj_spark.applications.wordcount import run as wc_run
        return wc_run(text_rdd)
    raise ValueError(f'Unknown application: {app!r}. Available: wordcount')
