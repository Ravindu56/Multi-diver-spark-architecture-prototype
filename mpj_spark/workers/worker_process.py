# ============================================================
# workers/worker_process.py
# MPJ Worker Process — each owns an independent Spark Driver
# Paper Reference: Section IV.A + Algorithm 1
# ============================================================
# Pre-warm design
# ---------------
# Each worker runs in TWO phases:
#
#   Phase A — JVM INIT (NOT timed as computation)
#     1. Build SparkSession (cold JVM start, ~3-4 s)
#     2. Signal root via ready_queue: "I am warmed up"
#     3. Block on go_queue waiting for root's start signal
#
#   Phase B — COMPUTATION (timed as T_Proc)
#     4. Read partition from shared storage
#     5. Execute application pipeline
#     6. Convert RDD → KeyValueStructure
#     7. Send results + timings to root via result_queue / timing_queue
#
# Root waits until ALL workers signal ready, THEN fires the parallel
# timer and sends go-signals to all workers simultaneously.
# This mirrors the HPC behaviour where Spark drivers are already
# running on live nodes before a job is dispatched.
# ============================================================
import time
from multiprocessing import Queue

from mpj_spark.core.key_value        import KeyValueStructure
from mpj_spark.workers.spark_session  import build_spark_session


def mpj_worker_process(worker_id:         int,
                       partition_metadata: dict,
                       result_queue:       Queue,
                       timing_queue:       Queue,
                       app:                str   = 'wordcount',
                       ready_queue:        Queue = None,
                       go_queue:           Queue = None):
    """
    MPJ Worker Process (Paper §IV.A) with JVM pre-warm support.

    Parameters
    ----------
    worker_id         : int   — unique worker identifier
    partition_metadata: dict  — partition path + line counts from Root
    result_queue      : Queue — send computed results back to Root
    timing_queue      : Queue — send timing breakdown back to Root
    app               : str   — application key ('wordcount', ...)
    ready_queue       : Queue — signal Root when JVM is warmed up
                                (None → legacy mode, no pre-warm)
    go_queue          : Queue — receive start signal from Root
                                (None → legacy mode, no pre-warm)
    """
    try:
        # ────────────────────────────────────────────────────
        # PHASE A — JVM INIT  (excluded from computation timer)
        # ────────────────────────────────────────────────────
        t_spawn = time.time()

        # Step 1 — build independent SparkSession (cold JVM start)
        spark = build_spark_session(f'MPJ-Worker-{worker_id}')
        sc    = spark.sparkContext

        t_init_done = time.time()
        jvm_init_time = t_init_done - t_spawn

        if ready_queue is not None:
            # Step 2 — notify Root: JVM ready, waiting for go-signal
            ready_queue.put({'worker_id': worker_id, 'init_time': jvm_init_time})

            # Step 3 — block until Root fires start signal
            go_queue.get()   # blocks here; Root puts a token when all workers ready

        # ────────────────────────────────────────────────────
        # PHASE B — COMPUTATION  (this is what gets timed as T_Proc)
        # ────────────────────────────────────────────────────
        t_compute_start = time.time()

        # Step 4 — read partition from shared storage
        partition_path = partition_metadata['partition_path']
        text_rdd = sc.textFile(partition_path)

        # Step 5 — execute application pipeline
        results = _run_application(sc, text_rdd, app)

        t_compute_end = time.time()
        compute_time  = t_compute_end - t_compute_start

        # Step 6 — convert RDD results to KeyValue structure
        kv = KeyValueStructure().from_rdd_collect(results)

        # Step 7 — send results back to Root (simulates MPJ Send)
        result_queue.put({
            'worker_id':       worker_id,
            'results':         kv.to_serializable(),
            'num_items':       len(kv.data),
            'partition_lines': partition_metadata['num_lines'],
        })
        timing_queue.put({
            'worker_id':   worker_id,
            'driver_init': jvm_init_time,
            'processing':  compute_time,           # pure computation only
            'total':       t_compute_end - t_spawn, # spawn → done
        })

        spark.stop()

    except Exception as exc:
        result_queue.put({'worker_id': worker_id, 'error': str(exc)})
        timing_queue.put({'worker_id': worker_id, 'error': str(exc)})


# ------------------------------------------------------------------
def _run_application(sc, text_rdd, app: str):
    """
    Dispatch to the correct application module.
    Add new application keys here as the research expands.
    """
    if app == 'wordcount':
        from mpj_spark.applications.wordcount import run as wc_run
        return wc_run(text_rdd)
    else:
        raise ValueError(f'Unknown application: {app!r}. '
                         f'Available: wordcount')
