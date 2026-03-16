# ============================================================
# workers/worker_process.py
# MPJ Worker Process — each owns an independent Spark Driver
# Paper Reference: Section IV.A + Algorithm 1
# ============================================================
import time
from multiprocessing import Queue

from mpj_spark.core.key_value       import KeyValueStructure
from mpj_spark.workers.spark_session import build_spark_session


def mpj_worker_process(worker_id: int,
                       partition_metadata: dict,
                       result_queue: Queue,
                       timing_queue: Queue,
                       app: str = 'wordcount'):
    """
    MPJ Worker Process (Paper §IV.A).

    Steps
    -----
    1. Receive partition metadata from Root (via args)
    2. Spin up an INDEPENDENT SparkSession (isolated Spark driver)
    3. Read partition from shared storage using metadata path
    4. Execute the requested application (app parameter)
    5. Convert RDD results → KeyValueStructure
    6. Send results + timings back via Queues (simulates MPJ Send)
    """
    try:
        t_start = time.time()

        # Step 2 — independent SparkSession per worker
        spark = build_spark_session(f'MPJ-Worker-{worker_id}')
        sc    = spark.sparkContext
        t_init = time.time()

        # Step 3 — read partition
        partition_path = partition_metadata['partition_path']
        text_rdd = sc.textFile(partition_path)

        # Step 4 — execute application
        results = _run_application(sc, text_rdd, app)
        t_proc  = time.time()

        # Step 5 — convert to KeyValue
        kv = KeyValueStructure().from_rdd_collect(results)

        # Step 6 — send back
        result_queue.put({
            'worker_id': worker_id,
            'results':   kv.to_serializable(),
            'num_items': len(kv.data),
            'partition_lines': partition_metadata['num_lines'],
        })
        timing_queue.put({
            'worker_id':      worker_id,
            'driver_init':    t_init - t_start,
            'processing':     t_proc  - t_init,
            'total':          t_proc  - t_start,
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
