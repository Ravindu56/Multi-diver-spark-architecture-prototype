# ================================================================
# mpj_spark/applications/logreg/queue_run.py
#
# Queue-based (Phase 2) Logistic Regression MLlib pipeline.
# Originally logreg.py — moved inside the logreg/ package so that
# the package and the flat module no longer shadow each other.
#
# Called by worker_process.py as:
#     from mpj_spark.applications import logreg
#     app_result = logreg.run(...)
# which resolves via logreg/__init__.py.__getattr__("run").
#
# ALLREDUCE STRATEGY (simulated Queue-based, TWO-QUEUE design)
# ─────────────────────────────────────────────────────────────
# Uses TWO separate queues to avoid deadlock:
#   allreduce_up_queue   — this worker → root  (local weight vectors)
#   allreduce_down_queue — root → this worker  (averaged weights)
#
# Per iteration (warm-start via LinearOffset):
#   1. Broadcast current_weights (w_prev) into a Spark UDF.
#   2. Compute residual df: subtract w_prev·x + b_prev from the logit
#      by encoding the prior as a constant offset column.
#   3. lr.fit(df_residual) learns only Δw, Δb (the correction).
#   4. Real weights: w_new = w_prev + Δw  (intercept: b_new = b_prev + Δb)
#   5. Push w_new to root via allreduce_up_queue.
#   6. Block on allreduce_down_queue for global averaged weights.
#   7. Set current_weights = avg_weights for next iteration.
#
# WHY LinearOffset warm-start?
# ─────────────────────────────────────────────────────────────
# Spark MLlib's LogisticRegression has no setInitialWeights() / warm-
# start API in the public Python interface. Calling lr.fit(df_vec) from
# scratch every iteration always initialises weights to zero, so the
# coordinator's averaged weights are silently discarded. The LinearOffset
# trick sidesteps this: it folds the prior weight vector into the data
# transformation so the optimizer always starts from zero-residual, which
# is equivalent to warm-starting from the prior.
#
# BASELINE path (no queues): lr.fit(df_vec) from zero each iteration —
# identical behaviour to before; no warm-start overhead for the baseline.
#
# PER-ITERATION METRICS (Objective 2a)
# ─────────────────────────────────────────────────────────────
# Each iteration accumulates a record:
#   worker_id, iteration (1-based), iter_time_s, weight_norm (global
#   after Allreduce), weight_delta (||w_t - w_{t-1}||_2), local_weight_norm
#   (before Allreduce), intercept, row_count
#
# Written to results/worker_<id>_logreg_iter_metrics.csv before return.
# Root merges all per-worker CSVs into results/logreg_iter_metrics.csv
# (see root_process.aggregate_logreg_results).
#
# NOTE ON CSV READING
# ─────────────────────────────────────────────────────────────
# file_manager.dynamic_partition() streams the source CSV line-by-line
# using round-robin assignment. The source CSV has a header row
# (f0,f1,...,label) which lands in whichever partition gets line 0.
# Fix: explicit StructType schema (all DoubleType) so Spark never
# infers StringType; filter WHERE f0 IS NULL to drop the stray header.
# ================================================================

import csv
import math
import os
import time

import numpy as np
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import VectorAssembler
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, IntegerType, StructField, StructType


def _build_schema(num_features: int) -> StructType:
    fields = [StructField(f'f{i}', DoubleType(), nullable=True)
              for i in range(num_features)]
    fields.append(StructField('label', DoubleType(), nullable=True))
    return StructType(fields)


def _weight_norm(weights) -> float:
    return math.sqrt(sum(w * w for w in weights))


def _write_worker_metrics(worker_id: int, records: list, results_dir: str) -> str:
    """
    Write per-iteration records for this worker to a CSV file.
    Returns the file path written.

    Columns:
        worker_id, iteration, iter_time_s, weight_norm, weight_delta,
        local_weight_norm, intercept, row_count
    """
    os.makedirs(results_dir, exist_ok=True)
    path = os.path.join(results_dir, f'worker_{worker_id}_logreg_iter_metrics.csv')
    fieldnames = [
        'worker_id', 'iteration', 'iter_time_s',
        'weight_norm', 'weight_delta', 'local_weight_norm',
        'intercept', 'row_count',
    ]
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    return path


ITER_METRICS_FIELDS = [
    'worker_id', 'iteration', 'iter_time_s',
    'weight_norm', 'weight_delta', 'local_weight_norm',
    'intercept', 'row_count',
]


def _make_residual_df(spark, df_vec, w_prev: list, b_prev: float, feature_cols: list):
    """
    Build a residual DataFrame that encodes the LinearOffset warm-start.

    The prior weight vector (w_prev, b_prev) is folded into the data as
    a constant logit offset so that lr.fit(df_residual) learns only the
    correction (Δw, Δb) relative to the current model, not from scratch.

    Implementation:
      - Broadcast w_prev as a DenseVector and dot it with each row's
        feature vector to produce a scalar offset per sample.
      - Append 'offset' column to df_vec (Spark MLlib supports
        LogisticRegression(offsetCol='offset') since Spark 3.0).
      - lr is then configured with offsetCol='offset', which shifts the
        linear predictor: η = w·x + b + offset.
      - After fit, the real weights are: w = w_prev + Δw, b = b_prev + Δb.

    Returns (df_residual, offset_col_added: bool).
    If b_prev and w_prev are both zero (first iteration or baseline),
    returns (df_vec, False) — no offset column, no overhead.
    """
    if w_prev is None or (all(v == 0.0 for v in w_prev) and b_prev == 0.0):
        return df_vec, False

    w_bc = spark.sparkContext.broadcast(w_prev)
    b_bc = spark.sparkContext.broadcast(b_prev)

    from pyspark.ml.linalg import Vectors
    from pyspark.sql.functions import udf
    from pyspark.sql.types import DoubleType as DT

    @udf(returnType=DT())
    def compute_offset(features):
        w = w_bc.value
        arr = features.toArray()
        return float(sum(w[i] * arr[i] for i in range(len(w)))) + b_bc.value

    df_residual = df_vec.withColumn('_offset', compute_offset(F.col('features'))).cache()
    return df_residual, True


def run(
    partition_path: str,
    max_iter: int = 10,
    reg_param: float = 0.01,
    num_features: int = 10,
    seed: int = 42,
    worker_id: int = 0,
    allreduce_up_queue=None,    # worker → root
    allreduce_down_queue=None,  # root  → worker
    # Legacy single-queue param kept for backwards compat — ignored when
    # up/down queues are provided.
    allreduce_queue=None,
    num_workers: int = 1,
    results_dir: str = 'results',
) -> dict:
    """
    Queue-based Logistic Regression pipeline on a labelled binary-classification CSV.
    Used by worker_process.py for both the baseline path (no queues) and
    the Phase 2 Queue/FedAvg multi-driver path (with queues).

    For the Phase 3 MPI Allreduce path use logreg.run_logreg_allreduce().

    Partition format (headerless lines produced by MPJSparkFileManager,
    possibly containing one stray header line from the source CSV):
        <f0>,<f1>,...,<f{num_features-1}>,<label>
    where label ∈ {0, 1}.

    Returns a dict that includes 'iter_metrics': list[dict] — one record
    per iteration, used by root_process.aggregate_logreg_results() to
    write results/logreg_iter_metrics.csv (Objective 2a dataset).
    """
    spark = SparkSession.getActiveSession()
    if spark is None:
        raise RuntimeError('[LogReg] No active SparkSession found in worker.')

    # Resolve queue handles: prefer explicit up/down; fall back to legacy
    # single-queue mode (up == down) for any caller that hasn't migrated.
    _up   = allreduce_up_queue   if allreduce_up_queue   is not None else allreduce_queue
    _down = allreduce_down_queue if allreduce_down_queue is not None else allreduce_queue
    use_allreduce = _up is not None and _down is not None

    # ─ 1. Load CSV with explicit schema ──────────────────────────────────
    schema = _build_schema(num_features)
    df_raw = spark.read.csv(partition_path, schema=schema, header=False)

    # Drop stray header row (parses as NULLs) and cast label to int
    df = (df_raw
          .filter(F.col('f0').isNotNull())
          .dropna()
          .withColumn('label', F.col('label').cast(IntegerType())))

    feature_cols = [f'f{i}' for i in range(num_features)]
    row_count    = df.count()

    if row_count == 0:
        raise RuntimeError(
            f'[LogReg Worker {worker_id}] Partition is empty after cleaning. '
            f'Check --logreg-features matches the dataset (expected {num_features} features).')

    print(f'[LogReg Worker {worker_id}] reg_param={reg_param} | '
          f'max_iter={max_iter} | rows={row_count:,} | '
          f'features={len(feature_cols)} | allreduce={use_allreduce}')

    # ─ 2. Assemble feature vector ─────────────────────────────────────────
    assembler = VectorAssembler(
        inputCols=feature_cols, outputCol='features', handleInvalid='skip')
    df_vec = assembler.transform(df).select('features', 'label').cache()

    # ─ 3. Iterative Allreduce training (warm-start via LinearOffset) ──────
    # current_weights / current_intercept: the global averaged model state.
    # On iteration 0 both are None/0.0 → first lr.fit() starts from zero
    # (same as before). From iteration 1 onward the prior is folded into
    # df_residual so lr.fit() corrects only the delta.
    current_weights   = None
    current_intercept = 0.0
    iterations_done   = 0
    iter_metrics      = []   # Objective 2a: per-iteration convergence records
    prev_weights_vec  = np.zeros(num_features)  # for true L2 weight_delta

    for iteration in range(max_iter):
        t_iter = time.perf_counter()

        # ── Warm-start: build residual DataFrame using prior weights ─────
        # Baseline path (use_allreduce=False) always passes w=None so
        # df_fit = df_vec (no overhead, identical to original behaviour).
        if use_allreduce and current_weights is not None:
            df_fit, _has_offset = _make_residual_df(
                spark, df_vec, current_weights, current_intercept, feature_cols)
        else:
            df_fit, _has_offset = df_vec, False

        # ── Fit one LR pass on the residual ─────────────────────────────
        lr_kwargs = dict(
            featuresCol='features',
            labelCol='label',
            maxIter=1,
            regParam=reg_param,
            elasticNetParam=0.0,
            family='binomial',
            fitIntercept=True,
            standardization=True,
        )
        if _has_offset:
            lr_kwargs['offsetCol'] = '_offset'

        lr    = LogisticRegression(**lr_kwargs)
        model = lr.fit(df_fit)

        delta_w   = model.coefficients.toArray().tolist()
        delta_b   = float(model.intercept)
        local_norm = _weight_norm(delta_w)

        # ── Accumulate delta onto current model ──────────────────────────
        # Baseline path: current_weights stays None until after Allreduce
        # assignment below, so the addition guard is needed.
        if use_allreduce and current_weights is not None:
            local_weights   = [current_weights[i] + delta_w[i] for i in range(num_features)]
            local_intercept = current_intercept + delta_b
        else:
            local_weights   = delta_w
            local_intercept = delta_b

        # Unpersist residual df to avoid OOM over many iterations
        if _has_offset:
            df_fit.unpersist()

        # ── Allreduce (Queue / FedAvg path) ──────────────────────────────
        if use_allreduce:
            _up.put({
                'type'     : 'weights',
                'worker_id': worker_id,
                'iteration': iteration,
                'weights'  : local_weights,
                'intercept': local_intercept,
                'row_count': row_count,
            })

            # Block until root returns globally averaged weights
            msg = _down.get(timeout=180)
            if msg.get('type') == 'avg_weights':
                current_weights   = msg['weights']      # ← THIS is the fix:
                current_intercept = msg['intercept']    #   averaged weights
            else:                                        #   used as warm-start
                current_weights   = local_weights       #   for next iteration
                current_intercept = local_intercept
        else:
            # Baseline (no queues): accept local result as-is
            current_weights   = local_weights
            current_intercept = local_intercept

        iterations_done += 1
        iter_time   = time.perf_counter() - t_iter
        global_norm = _weight_norm(current_weights)

        # True L2 norm of weight vector difference (not scalar norm diff)
        cur_vec      = np.array(current_weights)
        weight_delta = float(np.linalg.norm(cur_vec - prev_weights_vec))
        prev_weights_vec = cur_vec.copy()

        # ── Accumulate per-iteration record (Objective 2a) ───────────────
        iter_metrics.append({
            'worker_id'        : worker_id,
            'iteration'        : iteration + 1,          # 1-based
            'iter_time_s'      : round(iter_time, 6),
            'weight_norm'      : round(global_norm, 8),  # after Allreduce
            'weight_delta'     : round(weight_delta, 8), # ||w_t - w_{t-1}||_2
            'local_weight_norm': round(local_norm, 8),   # before Allreduce
            'intercept'        : round(current_intercept, 8),
            'row_count'        : row_count,
        })

        print(f'[LogReg Worker {worker_id}] iter {iteration+1}/{max_iter}  '
              f'({iter_time:.3f}s)  '
              f'|w|={global_norm:.4f}  '
              f'\u0394={weight_delta:.6f}')

    # ─ 4. Final accuracy using converged weights ──────────────────────────
    # FIXED: evaluate on the converged model, not a fresh re-fit from zero.
    # We score the converged current_weights directly against the cached
    # df_vec by computing predictions with a UDF, avoiding a stale re-fit.
    if current_weights is not None:
        w_final = np.array(current_weights)
        b_final = current_intercept

        from pyspark.sql.functions import udf as _udf
        from pyspark.sql.types import IntegerType as _IT

        w_bc = spark.sparkContext.broadcast(w_final.tolist())
        b_bc = spark.sparkContext.broadcast(b_final)

        @_udf(returnType=_IT())
        def predict(features):
            import math as _m
            w = w_bc.value
            arr = features.toArray()
            logit = sum(w[i] * arr[i] for i in range(len(w))) + b_bc.value
            prob  = 1.0 / (1.0 + _m.exp(-logit))
            return int(prob >= 0.5)

        df_pred      = df_vec.withColumn('prediction', predict(F.col('features')))
        correct      = df_pred.filter(F.col('prediction') == F.col('label')).count()
        train_accuracy = correct / row_count if row_count > 0 else 0.0
        weight_vector  = current_weights
        intercept_final = current_intercept

        w_bc.unpersist()
    else:
        # Fallback: should only happen if max_iter == 0
        lr_fb = LogisticRegression(
            featuresCol='features', labelCol='label',
            maxIter=1, regParam=reg_param,
            elasticNetParam=0.0, family='binomial', fitIntercept=True)
        model_fb        = lr_fb.fit(df_vec)
        train_accuracy  = float(model_fb.summary.accuracy)
        weight_vector   = model_fb.coefficients.toArray().tolist()
        intercept_final = float(model_fb.intercept)

    print(f'[LogReg Worker {worker_id}] Final train accuracy: {train_accuracy:.4f}')
    print(f'[LogReg Worker {worker_id}] Weight norm: '
          f'{_weight_norm(weight_vector):.4f}')

    df_vec.unpersist()

    # ─ 5. Write per-worker metrics CSV ───────────────────────────────────
    worker_csv_path = _write_worker_metrics(worker_id, iter_metrics, results_dir)
    print(f'[LogReg Worker {worker_id}] Iter metrics \u2192 {worker_csv_path} '
          f'({len(iter_metrics)} rows)')

    return {
        'weight_vector'  : weight_vector,
        'intercept'      : intercept_final,
        'train_accuracy' : train_accuracy,
        'row_count'      : row_count,
        'iterations_done': iterations_done,
        'partition_path' : partition_path,
        'iter_metrics'   : iter_metrics,   # Objective 2a — consumed by root
    }
