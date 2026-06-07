# ================================================================
# mpj_spark/workers/worker_process.py
# ================================================================
# Transport-agnostic worker core + Phase-2 multiprocessing wrapper.
#
# PUBLIC API
# ----------
# run_worker_core(worker_id, partition_path, spark, worker_config,
#                up_queue, down_queue, reassign_adapter)
#     Pure Spark / application logic.  Zero multiprocessing or MPI
#     imports.  Called by both worker_process() (Phase 2) and
#     run_worker_mpi() (Phase 3).
#
# worker_process(worker_id, partition_path, result_queue, go_signal,
#                ready_signal, timing_queue, worker_config,
#                gossip_queue, reassign_queue, allreduce_down_queue)
#     Phase-2 multiprocessing wrapper.  Handles SparkSession init and
#     the multiprocessing Event barrier, then delegates to
#     run_worker_core().
#
# _reassign_pass(spark, partition_path, global_centres, worker_id)
#     Pure Spark/NumPy re-assignment helper.  No transport dependency.
#     Imported directly by worker_mpi.py to avoid duplication.
# ================================================================
import time
import traceback
from multiprocessing import Queue


from mpj_spark.workers.spark_session import build_spark_session
from mpj_spark.utils.dev_logger import DevLogger


def _tag(worker_id, phase):
    return f"[Worker {worker_id}][{phase}]"


# ================================================================
# _reassign_pass  —  pure Spark/NumPy, no transport dependency
# ================================================================


def _reassign_pass(
    spark, partition_path: str, global_centres: list, worker_id: int
) -> dict:
    """
    Option 2 re-assignment pass.

    Loads the partition, assigns every point to the nearest centroid
    in `global_centres` (no model training), then returns per-cluster
    (weighted_sum, count) for exact centroid recomputation at root.
    """
    import numpy as _np
    from pyspark.ml.feature import VectorAssembler
    from pyspark.sql import functions as F
    from pyspark.sql.types import IntegerType, DoubleType
    from pyspark.sql.functions import udf, col

    centres_bc = spark.sparkContext.broadcast(_np.array(global_centres))
    k = len(global_centres)
    dims = len(global_centres[0])

    df_raw = spark.read.csv(partition_path, inferSchema=True, header=False)
    feature_cols = df_raw.columns
    df = df_raw.dropna()

    assembler = VectorAssembler(
        inputCols=feature_cols, outputCol="features", handleInvalid="skip"
    )
    df_vec = assembler.transform(df).select("features")

    def assign_cluster(features):
        arr = _np.array(features.toArray())
        dists = _np.linalg.norm(centres_bc.value - arr, axis=1)
        return int(_np.argmin(dists))

    assign_udf = udf(assign_cluster, IntegerType())
    df_assigned = df_vec.withColumn("cluster", assign_udf(col("features")))

    def expand_to_cols(df_a, dims_):
        for d in range(dims_):
            df_a = df_a.withColumn(
                f"f{d}",
                udf(lambda v, _d=d: float(v[_d]), DoubleType())(col("features")),
            )
        return df_a.drop("features")

    df_expanded = expand_to_cols(df_assigned, dims)
    agg_exprs = [F.sum(f"f{d}").alias(f"s{d}") for d in range(dims)] + [
        F.count("*").alias("cnt")
    ]
    df_agg = df_expanded.groupBy("cluster").agg(*agg_exprs).collect()

    cluster_sums = [[0.0] * dims for _ in range(k)]
    cluster_counts = [0] * k
    total_rows = 0
    for row in df_agg:
        c = row["cluster"]
        cnt = row["cnt"]
        cluster_counts[c] = cnt
        cluster_sums[c] = [row[f"s{d}"] for d in range(dims)]
        total_rows += cnt

    centres_bc.unpersist()
    print(
        f"{_tag(worker_id, 'REASSIGN')} Done — {total_rows:,} rows assigned to {k} clusters"
    )
    return {
        "cluster_sums": cluster_sums,
        "cluster_counts": cluster_counts,
        "row_count": total_rows,
    }


# ================================================================
# run_worker_core  —  transport-agnostic Spark logic
# ================================================================


def run_worker_core(
    worker_id: int,
    partition_path: str,
    spark,  # active SparkSession, already initialised
    worker_config: dict,
    up_queue=None,  # Queue-like: worker → root  (gossip / allreduce-UP)
    down_queue=None,  # Queue-like: root → worker  (allreduce-DOWN)
    reassign_adapter=None,  # Queue-like or None: for kmeans re-assignment pass
) -> dict:
    """
    Transport-agnostic worker Spark logic.

    Executes load → proc → emit result/timing → optional reassign pass
    for the assigned application.  Has NO knowledge of whether the
    transport layer is multiprocessing.Queue, MPI, or anything else —
    callers pass Queue-compatible objects (anything with .put() / .get()).

    Parameters
    ----------
    worker_id        : 0-indexed worker identifier.
    partition_path   : Path to the local data partition file.
    spark            : An already-initialised SparkSession.
    worker_config    : Dict of application parameters (app, k, iter, etc.).
    up_queue         : Queue-like used by kmeans (centroid gossip) and
                       logreg (allreduce-UP weights). May be None.
    down_queue       : Queue-like used by logreg (allreduce-DOWN averaged
                       weights, root → worker). May be None.
    reassign_adapter : Queue-like used for the optional K-Means re-assignment
                       pass (recv global centroids, send cluster stats).
                       May be None — pass None to skip the reassign pass.

    Returns
    -------
    dict with keys:
        result   — application result dict  (or None on error)
        timing   — {'worker_id', 'init_time', 'load_time',
                     'processing_time', 'total_time'}
        status   — 'success' or 'error'
        error    — str (present only on error)
    """
    app_name = worker_config.get("app", "wordcount")
    kmeans_k = int(worker_config.get("kmeans_k", 3))
    kmeans_iter = int(worker_config.get("kmeans_max_iter", 20))
    num_workers = worker_config.get("num_workers", 1)
    seed_centres = worker_config.get("seed_centres", None)
    logreg_iter = int(worker_config.get("logreg_iter", 10))
    logreg_reg_param = float(worker_config.get("logreg_reg_param", 0.01))
    logreg_features = int(worker_config.get("logreg_features", 10))

    logger = DevLogger(worker_id=worker_id)
    load_time = 0.0
    proc_time = 0.0

    try:
        # ── LOAD ────────────────────────────────────────────────────────
        print(f"{_tag(worker_id, 'LOAD')} Loading partition ...")
        t_load_start = time.perf_counter()

        if app_name == "wordcount":
            text_rdd = spark.sparkContext.textFile(partition_path)
            text_rdd.cache()
            row_count = text_rdd.count()
            load_time = time.perf_counter() - t_load_start
            print(f"{_tag(worker_id, 'LOAD')} {row_count:,} rows  ({load_time:.3f}s)")
        # kmeans and logreg load inside their run() calls

        # ── PROC ────────────────────────────────────────────────────────
        print(f"{_tag(worker_id, 'PROC')} Running {app_name} ...")
        t_proc_start = time.perf_counter()

        if app_name == "wordcount":
            from mpj_spark.applications import wordcount

            app_result = wordcount.run(text_rdd)

        elif app_name == "kmeans":
            from mpj_spark.applications import kmeans

            app_result = kmeans.run(
                partition_path,
                k=kmeans_k,
                max_iter=kmeans_iter,
                seed_centres=seed_centres,
            )
            if up_queue is not None:
                up_queue.put(
                    {
                        "worker_id": worker_id,
                        "centres": app_result["centres"],
                        "wcss": app_result["wcss"],
                        "row_count": app_result["row_count"],
                    }
                )
                print(f"{_tag(worker_id, 'PROC')} Centroid state → up_queue")

        elif app_name == "logreg":
            from mpj_spark.applications import logreg

            app_result = logreg.run(
                partition_path,
                max_iter=logreg_iter,
                reg_param=logreg_reg_param,
                num_features=logreg_features,
                worker_id=worker_id,
                allreduce_up_queue=up_queue,
                allreduce_down_queue=down_queue,
                num_workers=num_workers,
            )
            print(
                f"{_tag(worker_id, 'PROC')} LogReg done  "
                f"acc={app_result['train_accuracy']:.4f}  "
                f"iters={app_result['iterations_done']}"
            )

        else:
            raise ValueError(
                f"Unknown app '{app_name}'. Valid: 'wordcount', 'kmeans', 'logreg'"
            )

        proc_time = time.perf_counter() - t_proc_start
        print(f"{_tag(worker_id, 'DONE')} {app_name} complete  ({proc_time:.3f}s)")

        # ── TIMING ────────────────────────────────────────────────────
        # init_time is not known here (SparkSession was built by caller);
        # callers must inject it into the returned timing dict if needed.
        timing = {
            "worker_id": worker_id,
            "init_time": 0.0,  # filled in by caller (knows init_time)
            "load_time": load_time,
            "processing_time": proc_time,
            "total_time": load_time + proc_time,
        }
        logger.log_worker_timing(
            worker_id=worker_id,
            init_time=0.0,
            load_time=load_time,
            proc_time=proc_time,
        )

        # ── REASSIGN PASS (kmeans only, optional) ────────────────────
        if reassign_adapter is not None and app_name == "kmeans":
            print(f"{_tag(worker_id, 'REASSIGN')} Waiting for global centroids ...")
            msg = reassign_adapter.get(timeout=300)
            if msg.get("type") == "reassign":
                global_centres = msg["centres"]
                reassign_stats = _reassign_pass(
                    spark, partition_path, global_centres, worker_id
                )
                reassign_adapter.put(
                    {
                        "type": "stats",
                        "worker_id": worker_id,
                        "cluster_sums": reassign_stats["cluster_sums"],
                        "cluster_counts": reassign_stats["cluster_counts"],
                        "row_count": reassign_stats["row_count"],
                    }
                )

        return {"result": app_result, "timing": timing, "status": "success"}

    except Exception as exc:
        print(f"{_tag(worker_id, 'ERROR')} {exc}")
        traceback.print_exc()
        return {
            "result": None,
            "timing": {
                "worker_id": worker_id,
                "init_time": 0.0,
                "load_time": 0.0,
                "processing_time": 0.0,
                "total_time": 0.0,
            },
            "status": "error",
            "error": str(exc),
        }


# ================================================================
# worker_process  —  Phase-2 multiprocessing wrapper (unchanged API)
# ================================================================


def worker_process(
    worker_id: int,
    partition_path: str,
    result_queue: Queue,
    go_signal,
    ready_signal,
    timing_queue: Queue,
    worker_config: dict = None,
    gossip_queue: Queue = None,  # kmeans gossip  OR  logreg allreduce-UP
    reassign_queue: Queue = None,
    allreduce_down_queue: Queue = None,  # logreg allreduce-DOWN  (root → workers)
):
    """
    Phase-2 multiprocessing wrapper.  Public API is unchanged.

    Handles SparkSession init and the multiprocessing Event barrier
    (ready_signal / go_signal), then delegates all Spark logic to
    run_worker_core().
    """
    if worker_config is None:
        worker_config = {}

    app_name = worker_config.get("app", "wordcount")
    cores_override = worker_config.get("cores_override", None)
    num_workers = worker_config.get("num_workers", 1)
    logger = DevLogger(worker_id=worker_id)

    try:
        # ── INIT ────────────────────────────────────────────────────────
        print(f"{_tag(worker_id, 'INIT')} Starting SparkSession (app={app_name}) ...")
        t_init_start = time.perf_counter()
        spark = build_spark_session(
            app_name=f"MPJ-Worker-{worker_id}-{app_name}",
            cores_override=cores_override,
            num_workers=num_workers,
        )
        init_time = time.perf_counter() - t_init_start
        print(f"{_tag(worker_id, 'INIT')} SparkSession ready  ({init_time:.3f}s)")

        # ── BARRIER ───────────────────────────────────────────────────
        ready_signal.set()
        print(f"{_tag(worker_id, 'WAIT')} Waiting for go-signal ...")
        go_signal.wait()

        # ── DELEGATE to transport-agnostic core ──────────────────────
        outcome = run_worker_core(
            worker_id=worker_id,
            partition_path=partition_path,
            spark=spark,
            worker_config=worker_config,
            up_queue=gossip_queue,
            down_queue=allreduce_down_queue,
            reassign_adapter=reassign_queue,
        )

        # Patch init_time into the timing dict (core doesn't know it)
        outcome["timing"]["init_time"] = init_time
        outcome["timing"]["total_time"] = (
            init_time
            + outcome["timing"]["load_time"]
            + outcome["timing"]["processing_time"]
        )

        # Re-log with correct init_time
        logger.log_worker_timing(
            worker_id=worker_id,
            init_time=init_time,
            load_time=outcome["timing"]["load_time"],
            proc_time=outcome["timing"]["processing_time"],
        )

        # ── EMIT via multiprocessing Queues ───────────────────────────
        result_queue.put(
            {
                "worker_id": worker_id,
                "result": outcome["result"],
                "status": outcome["status"],
                **({"error": outcome["error"]} if outcome["status"] == "error" else {}),
            }
        )
        timing_queue.put(outcome["timing"])

    except Exception as exc:
        print(f"{_tag(worker_id, 'ERROR')} {exc}")
        traceback.print_exc()
        result_queue.put(
            {
                "worker_id": worker_id,
                "result": None,
                "status": "error",
                "error": str(exc),
            }
        )
        timing_queue.put(
            {
                "worker_id": worker_id,
                "init_time": 0.0,
                "load_time": 0.0,
                "processing_time": 0.0,
                "total_time": 0.0,
            }
        )

    finally:
        try:
            spark.stop()
            print(f"{_tag(worker_id, 'STOP')} SparkSession stopped.")
        except Exception:
            pass
