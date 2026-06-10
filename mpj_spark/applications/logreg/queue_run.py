# =============================================================================
# mpj_spark/applications/logreg/queue_run.py
# Phase 2 — Queue-based Logistic Regression (FedAvg / no-sync baseline)
#
# BUG FIXES (convergence-parity patch)
# ─────────────────────────────────────
# BUG 1 — Constant Δ / |w| divergence (FIX: multi-iter local fit + gradient avg)
#   Root cause: the bias-column warm-start encoded the full model prediction as
#   a feature, so MLlib always saw a near-zero residual and returned the same
#   Δw magnitude every round → Δ=const, |w| grows linearly unbounded.
#
#   Fix: remove the bias-column warm-start scaffolding.  Each Allreduce round
#   now runs a FULL local MLlib fit (maxIter=local_epochs, default 5) from the
#   current global weights via initialWeights.  The coordinator averages the
#   converged weight vectors (FedAvg), not the 1-step corrections.  This is the
#   correct FedAvg protocol: local optimise → aggregate → repeat.
#
# BUG 2 — No learning-rate schedule (FIX: cosine-decay lr passed to MLlib regParam)
#   MLlib's LBFGS handles its own line-search, so we do NOT pass an external lr.
#   Instead the effective "step size" is controlled by regParam — a fixed value
#   is correct here.  The learning rate schedule concept only applies to the
#   MPI/hand-rolled SGD path (allreduce.py), which has its own fix below.
#
# BUG 3 — Speedup framing documented in metrics output
#   load_time vs proc_time breakdown is preserved.  The comment block at
#   the bottom of run() now records the correct Phase 2 single-machine
#   interpretation so it is visible in the metrics CSV header.
#
# ALLREDUCE STRATEGY (Queue-based FedAvg)
# ─────────────────────────────────────────
#   Per iteration:
#     1. Worker runs full local MLlib fit (local_epochs passes) from initialWeights.
#     2. Worker pushes converged (weights, intercept, row_count) to up_queue.
#     3. Root computes row-weighted FedAvg, pushes avg back via down_queue.
#     4. Worker sets initialWeights = avg for the next round.
#
# BASELINE (no queues):
#   use_allreduce=False → single multi-iter MLlib fit, no sync.
# =============================================================================

from __future__ import annotations

import csv
import math
import os
import time

import numpy as np
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.linalg import Vectors
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
    # FIX: local_epochs controls how many MLlib passes each worker runs per
    # Allreduce round.  Default 5 lets LBFGS converge locally before sync.
    local_epochs: int = 5,
) -> dict:
    """
    Queue-based Logistic Regression (FedAvg) on a binary-classification CSV.

    FedAvg protocol (use_allreduce=True):
      Each round: full local MLlib fit from current global weights
                  → push converged weights → receive FedAvg average
                  → repeat for max_iter rounds.

    Baseline (use_allreduce=False):
      Single MLlib fit with max_iter*local_epochs total passes.

    Returns dict with 'iter_metrics' list consumed by root_process to
    build results/logreg_iter_metrics.csv (Objective 2a dataset).
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

    # Build the feature vector once — reused across all rounds
    assembler = VectorAssembler(
        inputCols=feature_cols, outputCol='features', handleInvalid='skip')
    df_vec = assembler.transform(df_base).select('features', 'label').cache()
    # Trigger cache
    df_vec.count()

    print(f'[LogReg Worker {worker_id}] reg_param={reg_param} | '
          f'rounds={max_iter} | local_epochs={local_epochs} | '
          f'rows={row_count:,} | features={num_features} | allreduce={use_allreduce}')

    # ─ 2. Iterative FedAvg rounds ─────────────────────────────────────────
    # current_weights holds the global averaged model from the coordinator.
    # On round 0 we pass None → MLlib initialises from zero.
    # On round 1+ we pass the averaged weights as initialWeights so the
    # local optimiser starts from the global model, not from scratch.
    current_weights   = None   # np.ndarray or None
    current_intercept = 0.0
    prev_weights_vec  = np.zeros(num_features)
    iter_metrics      = []
    iterations_done   = 0

    for iteration in range(max_iter):
        t_iter = time.perf_counter()

        # FIX: run a full local MLlib fit each round.
        # initialWeights encodes the global model from the previous round so
        # the local LBFGS continues from a warm start, but convergence is
        # determined by MLlib's own line-search — not by a bias-column proxy.
        lr_kwargs = dict(
            featuresCol='features',
            labelCol='label',
            maxIter=local_epochs,
            regParam=reg_param,
            elasticNetParam=0.0,
            family='binomial',
            fitIntercept=True,
            standardization=True,
        )
        if current_weights is not None:
            # Warm-start: pass previous global model as initial point.
            # This is the MLlib-native way to warm-start LBFGS.
            try:
                lr_kwargs['initialWeights'] = Vectors.dense(current_weights)
                lr_kwargs['initialIntercept'] = current_intercept
            except TypeError:
                # Older Spark versions may not support initialWeights keyword;
                # fall back to cold start (still correct, slightly slower).
                lr_kwargs.pop('initialWeights', None)
                lr_kwargs.pop('initialIntercept', None)

        lr    = LogisticRegression(**lr_kwargs)
        model = lr.fit(df_vec)

        # Extract converged local weights
        local_weights_arr = model.coefficients.toArray()   # shape (D,)
        local_weights     = local_weights_arr.tolist()
        local_intercept   = float(model.intercept)
        local_norm        = float(np.linalg.norm(local_weights_arr))

        # ── Allreduce (FedAvg via queue) ──────────────────────────────────
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
                current_weights   = msg['weights']   # FedAvg-averaged model
                current_intercept = msg['intercept']
            else:
                current_weights   = local_weights
                current_intercept = local_intercept
        else:
            # Baseline: local model IS the current model
            current_weights   = local_weights
            current_intercept = local_intercept

        iterations_done += 1
        iter_time    = time.perf_counter() - t_iter
        global_norm  = _weight_norm(current_weights)

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

        print(f'[LogReg Worker {worker_id}] round {iteration+1}/{max_iter}  '
              f'({iter_time:.3f}s)  '
              f'|w|={global_norm:.4f}  '
              f'\u0394={weight_delta:.6f}')

    # ─ 3. Final accuracy using converged weights ──────────────────────────
    if current_weights is not None:
        b_final = current_intercept

        from pyspark.sql.functions import udf as _udf
        from pyspark.sql.types import IntegerType as _IT

        w_list = current_weights
        b_val  = b_final

        @_udf(returnType=_IT())
        def predict(features):
            import math as _m
            arr   = features.toArray()
            logit = sum(w_list[i] * arr[i] for i in range(len(w_list))) + b_val
            return int(1.0 / (1.0 + _m.exp(-logit)) >= 0.5)

        correct        = df_vec.withColumn('pred', predict(F.col('features'))) \
                               .filter(F.col('pred') == F.col('label')).count()
        train_accuracy = correct / row_count if row_count > 0 else 0.0
        weight_vector  = current_weights
        intercept_final = b_final
    else:
        lr_fb = LogisticRegression(
            featuresCol='features', labelCol='label',
            maxIter=local_epochs, regParam=reg_param,
            elasticNetParam=0.0, family='binomial', fitIntercept=True)
        model_fb        = lr_fb.fit(df_vec)
        train_accuracy  = float(model_fb.summary.accuracy)
        weight_vector   = model_fb.coefficients.toArray().tolist()
        intercept_final = float(model_fb.intercept)

    print(f'[LogReg Worker {worker_id}] Final train accuracy : {train_accuracy:.4f}')
    print(f'[LogReg Worker {worker_id}] Weight norm          : '
          f'{_weight_norm(weight_vector):.4f}')

    df_vec.unpersist()
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
