# ================================================================
# mpj_spark/workers/worker_process.py
#
# CHANGE from previous version:
#   In Phase 3/4, kmeans now receives partition_path (str) directly,
#   not a pre-built text_rdd. This is required for Fix 1 so the
#   native spark.read.csv() can open the file itself inside the JVM.
#
#   wordcount still uses text_rdd — no change there.
# ================================================================
import time
import traceback
from multiprocessing import Queue

from mpj_spark.workers.spark_session import build_spark_session
from mpj_spark.utils.dev_logger import DevLogger


def worker_process(
    worker_id:      int,
    partition_path: str,
    result_queue:   Queue,
    go_signal,
    ready_signal,
    timing_queue:   Queue,
    worker_config:  dict = None,
):
    if worker_config is None:
        worker_config = {}

    app_name       = worker_config.get('app',            'wordcount')
    cores_override = worker_config.get('cores_override',  None)
    kmeans_k       = int(worker_config.get('kmeans_k',        3))
    kmeans_iter    = int(worker_config.get('kmeans_max_iter', 20))

    logger = DevLogger(worker_id=worker_id)

    try:
        # ── Phase 1: Build SparkSession ───────────────────────────────
        print(f"[Worker {worker_id}] Starting SparkSession (app={app_name}) ...")
        t_init_start = time.perf_counter()

        num_workers = worker_config.get('num_workers', 1)
        spark = build_spark_session(
            app_name=f'MPJ-Worker-{worker_id}-{app_name}',
            cores_override=cores_override,
            num_workers=num_workers,
        )

        t_init_end = time.perf_counter()
        init_time  = t_init_end - t_init_start
        print(f"[Worker {worker_id}] SparkSession ready in {init_time:.3f}s")

        # ── Phase 2: Barrier ─────────────────────────────────────────
        ready_signal.set()
        print(f"[Worker {worker_id}] Waiting for go-signal ...")
        go_signal.wait()
        print(f"[Worker {worker_id}] Go! Loading partition ...")

        # ── Phase 3: Load (wordcount only — kmeans loads inside app) ─
        t_load_start = time.perf_counter()

        if app_name == 'wordcount':
            text_rdd  = spark.sparkContext.textFile(partition_path)
            text_rdd.cache()
            row_count = text_rdd.count()
            t_load_end = time.perf_counter()
            load_time  = t_load_end - t_load_start
            print(f"[Worker {worker_id}] Loaded {row_count:,} rows in {load_time:.3f}s")
        else:
            # For kmeans: file path is passed directly to the app.
            # spark.read.csv() inside kmeans.run() handles loading natively.
            t_load_end = t_load_start  # load time accounted inside proc
            load_time  = 0.0

        # ── Phase 4: Application dispatch ────────────────────────────
        t_proc_start = time.perf_counter()

        if app_name == 'wordcount':
            from mpj_spark.applications import wordcount
            app_result = wordcount.run(text_rdd)

        elif app_name == 'kmeans':
            from mpj_spark.applications import kmeans
            # FIX 1: pass partition_path (str) — not text_rdd
            app_result = kmeans.run(
                partition_path,
                k=kmeans_k,
                max_iter=kmeans_iter,
            )

        else:
            raise ValueError(
                f"[Worker {worker_id}] Unknown app '{app_name}'. "
                f"Valid choices: 'wordcount', 'kmeans'"
            )

        t_proc_end = time.perf_counter()
        proc_time  = t_proc_end - t_proc_start
        print(f"[Worker {worker_id}] Processing done in {proc_time:.3f}s")

        # ── Phase 5: Put results ──────────────────────────────────────
        result_queue.put({
            'worker_id': worker_id,
            'result'   : app_result,
            'status'   : 'success',
        })
        timing_queue.put({
            'worker_id'      : worker_id,
            'init_time'      : init_time,
            'load_time'      : load_time,
            'processing_time': proc_time,
            'total_time'     : init_time + load_time + proc_time,
        })
        logger.log_worker_timing(
            worker_id=worker_id,
            init_time=init_time,
            load_time=load_time,
            proc_time=proc_time,
        )

    except Exception as exc:
        print(f"[Worker {worker_id}] ERROR: {exc}")
        traceback.print_exc()
        result_queue.put({
            'worker_id': worker_id,
            'result'   : None,
            'status'   : 'error',
            'error'    : str(exc),
        })
        timing_queue.put({
            'worker_id'      : worker_id,
            'init_time'      : 0.0,
            'load_time'      : 0.0,
            'processing_time': 0.0,
            'total_time'     : 0.0,
        })

    finally:
        try:
            spark.stop()
            print(f"[Worker {worker_id}] SparkSession stopped.")
        except Exception:
            pass