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
# Per iteration (bias-column warm-start):
#   1. Append 'prior_logit' = w_prev·x + b_prev as an extra column (iter 1+)
#   2. Assemble feature vector including 'prior_logit'
#   3. lr.fit(df_vec) learns corrective residual (Δw, Δb)
#   4. Real weights after correction: only Δw on original features matter;
#      the prior_logit coefficient is discarded (it was a scaffold).
#   5. For Allreduce: push FULL local_weights (original feature dims only)
#   6. Block on down_queue for globally averaged weights.
#   7. Set current_weights = avg_weights for next iteration.
#
# WHY bias-column warm-start (not offsetCol)?
# ─────────────────────────────────────────────────────────────
# pyspark.ml.classification.LogisticRegression does NOT support offsetCol
# (it exists only in LinearRegression / GeneralizedLinearRegression).
# The bias-column approach achieves the same warm-starting effect by
# including the prior logit as a feature: the optimizer only has to learn
# a correction to the prior, not the full model from scratch.
#
# Accuracy tracking: uses converged current_weights (original feature dims)
# directly via prediction UDF — NOT a re-fit from scratch.
#
# PER-ITERATION METRICS (Objective 2a)
# ─────────────────────────────────────────────────────────────
# Each iteration accumulates a record:
#   worker_id, iteration (1-based), iter_time_s, weight_norm (global
#   after Allreduce), weight_delta (||w_t - w_{t-1}||_2), local_weight_norm
#   (before Allreduce), intercept, row_count
#
# Written to results/worker_<id>_logreg_iter_metrics.csv before return.
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


# ================================================================
# Schema helpers
# ================================================================

def _build_schema(num_features: int) -> StructType:
    fields = [StructField(f'f{i}', DoubleType(), nullable=True)
              for i in range(num_features)]
    fields.append(StructField('label', DoubleType(), nullable=True))
    return StructType(fields)


def _weight_norm(weights) -> float:
    return math.sqrt(sum(w * w for w in weights))


# ================================================================
# Per-worker metrics CSV
# ================================================================

def _write_worker_metrics(worker_id: int, records: list, results_dir: str) -> str:
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


# ================================================================
# Bias-column warm-start helper
# ================================================================

def _append_prior_logit_col(df, w_prev: list, b_prev: float, feature_cols: list):
    """
    Append a 'prior_logit' column = w_prev · x + b_prev to df.

    This column encodes the current global model's prediction logit
    for each sample. When included as an extra feature in VectorAssembler,
    the LogisticRegression optimizer only needs to learn the residual
    correction, which is equivalent to warm-starting from w_prev.

    This approach works in ALL PySpark / Spark versions because it uses
    only standard column arithmetic — no offsetCol, no private APIs.

    Returns the augmented DataFrame (with 'prior_logit' column cached).
    """
    # Build the dot-product expression: sum(w_i * f_i) + b
    dot_expr = F.lit(b_prev)
    for i, col_name in enumerate(feature_cols):
        dot_expr = dot_expr + F.lit(float(w_prev[i])) * F.col(col_name)

    return df.withColumn('prior_logit', dot_expr.cast(DoubleType()))


# ================================================================
# Main run() entry point
# ================================================================

def run(
    partition_path: str,
    max_iter: int = 10,
    reg_param: float = 0.01,
    num_features: int = 10,
    seed: int = 42,
    worker_id: int = 0,
    allreduce_up_queue=None,    # worker → root
    allreduce_down_queue=None,  # root  → worker
    allreduce_queue=None,       # legacy single-queue (ignored when up/down provided)
    num_workers: int = 1,
    results_dir: str = 'results',
) -> dict:
    """
    Queue-based Logistic Regression pipeline on a labelled binary-classification CSV.

    Warm-start strategy: bias-column (prior_logit) — compatible with all Spark versions.
    For the Phase 3 MPI Allreduce path use logreg.run_logreg_allreduce().

    Returns dict with 'iter_metrics' list consumed by root_process
    to build results/logreg_iter_metrics.csv (Objective 2a dataset).
    """
    spark = SparkSession.getActiveSession()
    if spark is None:
        raise RuntimeError('[LogReg] No active SparkSession found in worker.')

    # Resolve queue handles
    _up   = allreduce_up_queue   if allreduce_up_queue   is not None else allreduce_queue
    _down = allreduce_down_queue if allreduce_down_queue is not None else allreduce_queue
    use_allreduce = _up is not None and _down is not None

    # ─ 1. Load CSV with explicit schema ──────────────────────────────────
    schema = _build_schema(num_features)
    df_raw = spark.read.csv(partition_path, schema=schema, header=False)

    # Drop stray header row (parsed as NULLs) and cast label to int
    df_base = (df_raw
               .filter(F.col('f0').isNotNull())
               .dropna()
               .withColumn('label', F.col('label').cast(IntegerType()))
               .cache())

    feature_cols = [f'f{i}' for i in range(num_features)]
    row_count    = df_base.count()

    if row_count == 0:
        raise RuntimeError(
            f'[LogReg Worker {worker_id}] Partition is empty after cleaning. '
            f'Check --logreg-features matches the dataset '
            f'(expected {num_features} features).')

    print(f'[LogReg Worker {worker_id}] reg_param={reg_param} | '
          f'max_iter={max_iter} | rows={row_count:,} | '
          f'features={len(feature_cols)} | allreduce={use_allreduce}')

    # ─ 2. Iterative training with bias-column warm-start ──────────────────
    # current_weights / current_intercept hold the global averaged model.
    # On iteration 0: train from zero (no prior logit column).
    # On iteration 1+: append prior_logit column so LR learns residual.
    current_weights   = None          # None = "not yet trained"
    current_intercept = 0.0
    prev_weights_vec  = np.zeros(num_features)
    iter_metrics      = []
    iterations_done   = 0

    for iteration in range(max_iter):
        t_iter = time.perf_counter()

        # ── Build training DataFrame for this iteration ────────────────────
        if use_allreduce and current_weights is not None:
            # Warm-start: inject prior logit as extra feature
            df_aug    = _append_prior_logit_col(
                df_base, current_weights, current_intercept, feature_cols)
            assem_cols = feature_cols + ['prior_logit']
        else:
            # First iteration or baseline (no-allreduce): plain feature matrix
            df_aug    = df_base
            assem_cols = feature_cols

        assembler = VectorAssembler(
            inputCols=assem_cols, outputCol='features', handleInvalid='skip')
        df_vec = assembler.transform(df_aug).select('features', 'label').cache()

        # ── Fit one LR pass ────────────────────────────────────────────────
        lr = LogisticRegression(
            featuresCol='features',
            labelCol='label',
            maxIter=1,
            regParam=reg_param,
            elasticNetParam=0.0,
            family='binomial',
            fitIntercept=True,
            standardization=True,
        )
        model = lr.fit(df_vec)
        df_vec.unpersist()

        # Extract coefficients for the ORIGINAL feature dimensions only
        # (last coefficient is for prior_logit on iter 1+; discard it)
        all_coeffs = model.coefficients.toArray().tolist()
        delta_w    = all_coeffs[:num_features]
        delta_b    = float(model.intercept)
        local_norm = _weight_norm(delta_w)

        # Accumulate correction onto current global model
        if use_allreduce and current_weights is not None:
            local_weights   = [current_weights[i] + delta_w[i]
                               for i in range(num_features)]
            local_intercept = current_intercept + delta_b
        else:
            local_weights   = delta_w
            local_intercept = delta_b

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

            msg = _down.get(timeout=180)
            if msg.get('type') == 'avg_weights':
                current_weights   = msg['weights']   # ← warm-start for next iter
                current_intercept = msg['intercept']
            else:
                current_weights   = local_weights
                current_intercept = local_intercept
        else:
            current_weights   = local_weights
            current_intercept = local_intercept

        iterations_done += 1
        iter_time    = time.perf_counter() - t_iter
        global_norm  = _weight_norm(current_weights)

        # True L2 weight vector distance (detects oscillation / divergence)
        cur_vec      = np.array(current_weights)
        weight_delta = float(np.linalg.norm(cur_vec - prev_weights_vec))
        prev_weights_vec = cur_vec.copy()

        iter_metrics.append({
            'worker_id'        : worker_id,
            'iteration'        : iteration + 1,
            'iter_time_s'      : round(iter_time, 6),
            'weight_norm'      : round(global_norm, 8),
            'weight_delta'     : round(weight_delta, 8),
            'local_weight_norm': round(local_norm, 8),
            'intercept'        : round(current_intercept, 8),
            'row_count'        : row_count,
        })

        print(f'[LogReg Worker {worker_id}] iter {iteration+1}/{max_iter}  '
              f'({iter_time:.3f}s)  '
              f'|w|={global_norm:.4f}  '
              f'\u0394={weight_delta:.6f}')

    # ─ 3. Final accuracy using converged weights ──────────────────────────
    # Score the converged model directly via UDF — no re-fit from scratch.
    if current_weights is not None:
        w_final = current_weights
        b_final = current_intercept

        from pyspark.sql.functions import udf as _udf
        from pyspark.sql.types import IntegerType as _IT

        # Need df_vec on original feature_cols for evaluation
        assembler_eval = VectorAssembler(
            inputCols=feature_cols, outputCol='features_eval', handleInvalid='skip')
        df_eval = assembler_eval.transform(df_base).select('features_eval', 'label').cache()

        w_list = w_final
        b_val  = b_final

        @_udf(returnType=_IT())
        def predict(features):
            import math as _m
            w   = w_list
            arr = features.toArray()
            logit = sum(w[i] * arr[i] for i in range(len(w))) + b_val
            return int(1.0 / (1.0 + _m.exp(-logit)) >= 0.5)

        correct        = df_eval.withColumn('pred', predict(F.col('features_eval'))) \
                                .filter(F.col('pred') == F.col('label')).count()
        train_accuracy = correct / row_count if row_count > 0 else 0.0
        weight_vector  = w_final
        intercept_final = b_final
        df_eval.unpersist()
    else:
        lr_fb = LogisticRegression(
            featuresCol='features', labelCol='label',
            maxIter=1, regParam=reg_param,
            elasticNetParam=0.0, family='binomial', fitIntercept=True)
        assembler_fb = VectorAssembler(
            inputCols=feature_cols, outputCol='features', handleInvalid='skip')
        df_fb  = assembler_fb.transform(df_base).select('features', 'label')
        model_fb        = lr_fb.fit(df_fb)
        train_accuracy  = float(model_fb.summary.accuracy)
        weight_vector   = model_fb.coefficients.toArray().tolist()
        intercept_final = float(model_fb.intercept)

    print(f'[LogReg Worker {worker_id}] Final train accuracy : {train_accuracy:.4f}')
    print(f'[LogReg Worker {worker_id}] Weight norm          : '
          f'{_weight_norm(weight_vector):.4f}')

    df_base.unpersist()

    # ─ 4. Write per-worker metrics CSV ───────────────────────────────────
    worker_csv_path = _write_worker_metrics(worker_id, iter_metrics, results_dir)
    print(f'[LogReg Worker {worker_id}] Iter metrics → {worker_csv_path} '
          f'({len(iter_metrics)} rows)')

    return {
        'weight_vector'  : weight_vector,
        'intercept'      : intercept_final,
        'train_accuracy' : train_accuracy,
        'row_count'      : row_count,
        'iterations_done': iterations_done,
        'partition_path' : partition_path,
        'iter_metrics'   : iter_metrics,
    }
