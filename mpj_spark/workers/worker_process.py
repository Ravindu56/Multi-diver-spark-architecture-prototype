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
import time


def _tag(worker_id, phase):
    return f"[W{worker_id}|{phase}]"


def run_worker_core(
    worker_id,
    partition_path,
    spark,
    worker_config,
    up_queue=None,
    down_queue=None,
    reassign_adapter=None,
    comm=None,
):
    app = worker_config.get("app", "wordcount")
    num_workers = worker_config.get("num_workers", 1)
    results_dir = worker_config.get("results_dir", "results")
    sync_mode = worker_config.get("sync_mode", "queue")

    t_load_start = time.perf_counter()
    load_time = time.perf_counter() - t_load_start

    t_proc_start = time.perf_counter()

    if app == "wordcount":
        from mpj_spark.applications.wordcount import run_wordcount

        result = run_wordcount(partition_path, spark)

    elif app == "kmeans":
        try:
            from mpj_spark.applications.kmeans.allreduce import run_kmeans_allreduce

            use_allreduce = True
        except ImportError:
            use_allreduce = False
            run_kmeans_allreduce = None

        if use_allreduce:
            result = run_kmeans_allreduce(
                comm=comm,
                rank=worker_id,
                size=num_workers,
                input_file=partition_path,
                k=worker_config.get("kmeans_k", 3),
                max_iter=worker_config.get("kmeans_max_iter", 20),
                tol=1e-4,
                seed=42,
                cores_override=worker_config.get("cores_override"),
                metrics_output_dir=results_dir,
            )
        else:
            from mpj_spark.applications.kmeans.driver import run_kmeans_driver

            class _DummyComm:
                def bcast(self, val, root=0):
                    return val

            result = run_kmeans_driver(
                rank=worker_id,
                size=num_workers,
                comm=comm or _DummyComm(),
                dataset_path=partition_path,
                k=worker_config.get("kmeans_k", 3),
                max_iter=worker_config.get("kmeans_max_iter", 20),
                tol=1e-4,
                seed=worker_config.get("seed_centres", 42) or 42,
                metrics_output_dir=results_dir,
            )

    elif app == "logreg":
        logreg_kwargs = dict(
            partition_path=partition_path,
            max_iter=worker_config.get("logreg_iter", 10),
            reg_param=worker_config.get("logreg_reg_param", 0.01),
            num_features=worker_config.get("logreg_features", 10),
            worker_id=worker_id,
            num_workers=num_workers,
            results_dir=results_dir,
        )

        if sync_mode == "none":
            from mpj_spark.applications.logreg import nosync_run

            result = nosync_run.run(**logreg_kwargs)
        else:
            from mpj_spark.applications.logreg import queue_run

            result = queue_run.run(
                **logreg_kwargs,
                allreduce_up_queue=up_queue,
                allreduce_down_queue=down_queue,
            )

    else:
        raise ValueError(f"Unknown app: {app!r}")

    proc_time = time.perf_counter() - t_proc_start

    return {
        "status": "success",
        "result": result,
        "timing": {
            "worker_id": worker_id,
            "load_time": load_time,
            "processing_time": proc_time,
        },
    }


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

    cores = worker_cfg.get("cores_override", max(1, TOTAL_CORES // 2))
    num_workers = worker_cfg.get("num_workers", 1)

    try:
        from pyspark.sql import SparkSession

        t_init_start = time.perf_counter()
        spark = (
            SparkSession.builder.appName(f"MPJ-Worker-{worker_id}")
            .master(f"local[{cores}]")
            .config("spark.ui.enabled", "false")
            .config("spark.sql.shuffle.partitions", str(cores * 2))
            .config("spark.memory.fraction", "0.8")
            .config("spark.memory.storageFraction", "0.2")
            .getOrCreate()
        )
        spark.sparkContext.setLogLevel("ERROR")
        init_time = time.perf_counter() - t_init_start

        ready_signal.set()
        go_signal.wait()

        outcome = run_worker_core(
            worker_id=worker_id,
            partition_path=partition_path,
            spark=spark,
            worker_config={**worker_cfg, "num_workers": num_workers},
            up_queue=allreduce_up_queue,
            down_queue=allreduce_down_queue,
            reassign_adapter=reassign_queue,
            comm=None,  # Placeholder for MPI communicator if needed
        )

        result_queue.put(
            {
                "status": outcome["status"],
                "worker_id": worker_id,
                "result": outcome["result"],
            }
        )
        timing_queue.put(
            {
                "worker_id": worker_id,
                "init_time": init_time,
                "load_time": outcome["timing"].get("load_time", 0.0),
                "processing_time": outcome["timing"]["processing_time"],
            }
        )

    except Exception as exc:
        import traceback

        result_queue.put(
            {
                "status": "error",
                "worker_id": worker_id,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
        )
        timing_queue.put(
            {
                "worker_id": worker_id,
                "init_time": 0.0,
                "load_time": 0.0,
                "processing_time": 0.0,
            }
        )
    finally:
        try:
            from pyspark.sql import SparkSession

            active = SparkSession.getActiveSession()
            if active:
                active.stop()
        except Exception:
            pass
