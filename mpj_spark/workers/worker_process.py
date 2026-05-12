# ================================================================
# mpj_spark/workers/worker_process.py
#
# Changes from feature/adaptive-gossip-aggregation (previous):
#
#   OPTION 1 — Global Seeding:
#     - worker_config may now carry 'seed_centres' (list[list[float]])
#       set by root after its pre-partitioning sampling step.
#     - seed_centres is forwarded into kmeans.run() so every worker
#       warm-starts from the SAME initial centroid positions.
#
#   OPTION 2 — Re-assignment Pass:
#     - Added optional `reassign_queue` parameter.
#     - After gossip completes, root sends final global centroids via
#       reassign_queue.  Worker reads them, runs a lightweight assign-
#       only pass (no model.fit), computes per-cluster (sum, count),
#       and puts the stats back into reassign_queue for root to
#       recompute exact weighted global centroids.
# ================================================================
import time
import traceback
from multiprocessing import Queue

import numpy as np

from mpj_spark.workers.spark_session import build_spark_session
from mpj_spark.utils.dev_logger import DevLogger


def _tag(worker_id, phase):
    return f'[Worker {worker_id}][{phase}]'


def _reassign_pass(spark, partition_path: str, global_centres: list,
                   worker_id: int) -> dict:
    """
    Option 2 re-assignment pass.

    Loads the partition, assigns every point to the nearest centroid
    in `global_centres` (no model training), then returns per-cluster
    (weighted_sum, count) for exact centroid recomputation at root.

    Parameters
    ----------
    spark           : active SparkSession
    partition_path  : path to this worker's CSV partition
    global_centres  : list[list[float]] — gossip-final global centroids
    worker_id       : int

    Returns
    -------
    dict with keys:
        cluster_sums   : list[list[float]]  — sum of feature vectors per cluster
        cluster_counts : list[int]          — number of points per cluster
        row_count      : int                — total rows processed
    """
    import numpy as _np
    from pyspark.ml.feature import VectorAssembler
    from pyspark.sql import functions as F
    from pyspark.sql.types import IntegerType

    centres_bc = spark.sparkContext.broadcast(_np.array(global_centres))
    k          = len(global_centres)
    dims       = len(global_centres[0])

    df_raw       = spark.read.csv(partition_path, inferSchema=True, header=False)
    feature_cols = df_raw.columns
    df           = df_raw.dropna()

    assembler = VectorAssembler(
        inputCols=feature_cols, outputCol='features', handleInvalid='skip')
    df_vec = assembler.transform(df).select('features')

    # Assign cluster label: argmin L2 distance to each broadcast centroid
    def assign_cluster(features):
        arr  = _np.array(features.toArray())
        dists = _np.linalg.norm(centres_bc.value - arr, axis=1)
        return int(_np.argmin(dists))

    from pyspark.sql.functions import udf
    assign_udf = udf(assign_cluster, IntegerType())
    df_assigned = df_vec.withColumn('cluster', assign_udf(F.col('features')))

    # Collect cluster sums and counts via groupBy
    # Expand DenseVector to individual columns for aggregation
    from pyspark.sql.functions import col
    from pyspark.sql.types import DoubleType

    # Build a DataFrame with one column per feature dimension
    def expand_to_cols(df_a, dims_):
        for d in range(dims_):
            df_a = df_a.withColumn(
                f'f{d}',
                udf(lambda v, _d=d: float(v[_d]), DoubleType())(col('features'))
            )
        return df_a.drop('features')

    df_expanded = expand_to_cols(df_assigned, dims)
    agg_exprs   = [F.sum(f'f{d}').alias(f's{d}') for d in range(dims)] + \
                  [F.count('*').alias('cnt')]
    df_agg      = df_expanded.groupBy('cluster').agg(*agg_exprs).collect()

    cluster_sums   = [[0.0] * dims for _ in range(k)]
    cluster_counts = [0] * k
    total_rows     = 0
    for row in df_agg:
        c   = row['cluster']
        cnt = row['cnt']
        cluster_counts[c]  = cnt
        cluster_sums[c]    = [row[f's{d}'] for d in range(dims)]
        total_rows        += cnt

    centres_bc.unpersist()
    print(f'{_tag(worker_id, "REASSIGN")} Done — {total_rows:,} rows assigned to {k} clusters')
    return {
        'cluster_sums'  : cluster_sums,
        'cluster_counts': cluster_counts,
        'row_count'     : total_rows,
    }


def worker_process(
    worker_id:      int,
    partition_path: str,
    result_queue:   Queue,
    go_signal,
    ready_signal,
    timing_queue:   Queue,
    worker_config:  dict = None,
    gossip_queue:   Queue = None,
    reassign_queue: Queue = None,
):
    if worker_config is None:
        worker_config = {}

    app_name       = worker_config.get('app',             'wordcount')
    cores_override = worker_config.get('cores_override',   None)
    kmeans_k       = int(worker_config.get('kmeans_k',         3))
    kmeans_iter    = int(worker_config.get('kmeans_max_iter',  20))
    num_workers    = worker_config.get('num_workers',  1)
    # Option 1 — global seed centroids broadcast by root (None = use k-means||)
    seed_centres   = worker_config.get('seed_centres',     None)

    logger = DevLogger(worker_id=worker_id)

    try:
        # ── INIT ───────────────────────────────────────────────────────
        print(f'{_tag(worker_id, "INIT")} Starting SparkSession (app={app_name}) ...')
        t_init_start = time.perf_counter()
        spark = build_spark_session(
            app_name=f'MPJ-Worker-{worker_id}-{app_name}',
            cores_override=cores_override,
            num_workers=num_workers,
        )
        init_time = time.perf_counter() - t_init_start
        print(f'{_tag(worker_id, "INIT")} SparkSession ready  ({init_time:.3f}s)')

        # ── BARRIER ─────────────────────────────────────────────────
        ready_signal.set()
        print(f'{_tag(worker_id, "WAIT")} Waiting for go-signal ...')
        go_signal.wait()

        # ── LOAD ──────────────────────────────────────────────────
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

        # ── PROC ──────────────────────────────────────────────────
        print(f'{_tag(worker_id, "PROC")} Running {app_name} ...')
        t_proc_start = time.perf_counter()

        if app_name == 'wordcount':
            from mpj_spark.applications import wordcount
            app_result = wordcount.run(text_rdd)

        elif app_name == 'kmeans':
            from mpj_spark.applications import kmeans
            # Option 1: pass broadcast seed_centres (None = k-means|| fallback)
            app_result = kmeans.run(
                partition_path,
                k=kmeans_k,
                max_iter=kmeans_iter,
                seed_centres=seed_centres,
            )
            if gossip_queue is not None:
                gossip_queue.put({
                    'worker_id' : worker_id,
                    'centres'   : app_result['centres'],
                    'wcss'      : app_result['wcss'],
                    'row_count' : app_result['row_count'],
                })
                print(f'{_tag(worker_id, "PROC")} Centroid state → gossip_queue')

            # Option 2: Re-assignment pass — wait for gossip-final centroids
            if reassign_queue is not None:
                print(f'{_tag(worker_id, "REASSIGN")} Waiting for global centroids ...')
                msg = reassign_queue.get(timeout=120)
                if msg.get('type') == 'reassign':
                    global_centres = msg['centres']
                    reassign_stats = _reassign_pass(
                        spark, partition_path, global_centres, worker_id)
                    reassign_queue.put({
                        'type'          : 'stats',
                        'worker_id'     : worker_id,
                        'cluster_sums'  : reassign_stats['cluster_sums'],
                        'cluster_counts': reassign_stats['cluster_counts'],
                        'row_count'     : reassign_stats['row_count'],
                    })

        else:
            raise ValueError(f"Unknown app '{app_name}'. Valid: 'wordcount', 'kmeans'")

        proc_time = time.perf_counter() - t_proc_start
        print(f'{_tag(worker_id, "DONE")} {app_name} complete  ({proc_time:.3f}s)')

        # ── QUEUE RESULTS ─────────────────────────────────────────
        result_queue.put({'worker_id': worker_id, 'result': app_result, 'status': 'success'})
        timing_queue.put({
            'worker_id'      : worker_id,
            'init_time'      : init_time,
            'load_time'      : load_time,
            'processing_time': proc_time,
            'total_time'     : init_time + load_time + proc_time,
        })
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
