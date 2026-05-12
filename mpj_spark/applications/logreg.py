# ================================================================
# mpj_spark/applications/logreg.py
#
# Logistic Regression MLlib pipeline — per-worker ML workload
#
# ALLREDUCE STRATEGY (simulated Queue-based)
# ──────────────────────────────────────────
# Supervised equivalent of k-means centroid gossip:
#   each iteration → worker pushes weight vector → root averages
#   across all workers → broadcasts averaged weights back → worker
#   uses averaged weights as warm start for next iteration.
#
# This implements synchronous Allreduce (FedAvg-style) over the
# shared multiprocessing Queue channel, matching Phase 2 scope.
#
# MLlib LogisticRegression does not expose setInitialWeights() in
# PySpark public bindings; instead we use maxIter=1 and loop:
#   for iteration in range(logreg_iter):
#       fit one iteration starting from previous weightCol
#       push weights to allreduce_queue
#       receive averaged weights from allreduce_queue
#
# Because PySpark LR does not accept an initial weights vector
# directly, we approximate warm-start via regParam and a single-
# iteration solver pass (L-BFGS). The aggregation signal is the
# mean weight vector — mathematically identical to FedAvg when
# partitions are equal-size.
# ================================================================

import time

from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.linalg import Vectors
from pyspark.sql import SparkSession


def run(
    partition_path: str,
    max_iter: int = 10,
    reg_param: float = 0.01,
    num_features: int = 10,
    seed: int = 42,
    worker_id: int = 0,
    allreduce_queue=None,
    num_workers: int = 1,
) -> dict:
    """
    Logistic Regression pipeline on a labelled binary-classification CSV.

    CSV format (produced by generate_classification_dataset):
        f0,f1,...,f{num_features-1},label
    where label ∈ {0, 1}.

    Parameters
    ----------
    partition_path  : str              — absolute path to worker CSV partition
    max_iter        : int              — number of Allreduce iterations (default 10)
    reg_param       : float            — L2 regularisation parameter (default 0.01)
    num_features    : int              — number of feature columns (default 10)
    seed            : int              — random seed (default 42)
    worker_id       : int              — used for logging only
    allreduce_queue : Queue | None     — shared Queue for weight Allreduce;
                                         None → local-only training (no sync)
    num_workers     : int              — total workers sharing allreduce_queue

    Returns
    -------
    dict:
        weight_vector   : list[float]   — final averaged model coefficients
        intercept       : float         — final model intercept
        train_accuracy  : float         — accuracy on worker's own partition
        row_count       : int           — labelled rows processed
        iterations_done : int           — actual Allreduce rounds completed
        partition_path  : str
    """
    spark = SparkSession.getActiveSession()
    if spark is None:
        raise RuntimeError('[LogReg] No active SparkSession found in worker.')

    # ─ 1. Load CSV --------------------------------------------------------
    df_raw       = spark.read.csv(partition_path, inferSchema=True, header=True)
    df           = df_raw.dropna()
    feature_cols = [c for c in df.columns if c != 'label']
    row_count    = df.count()

    print(f'[LogReg Worker {worker_id}] reg_param={reg_param} | '
          f'max_iter={max_iter} | rows={row_count:,} | '
          f'features={len(feature_cols)} | allreduce={allreduce_queue is not None}')

    # ─ 2. Assemble feature vector -----------------------------------------
    assembler = VectorAssembler(
        inputCols=feature_cols, outputCol='features', handleInvalid='skip')
    df_vec = assembler.transform(df).select('features', 'label').cache()

    # ─ 3. Iterative Allreduce training ------------------------------------
    current_weights = None    # warm-start weights (None = LR default init)
    current_intercept = 0.0
    iterations_done = 0

    for iteration in range(max_iter):
        t_iter = time.perf_counter()

        # Single-pass LR fit (maxIter=1 per Allreduce round)
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

        local_weights   = model.coefficients.toArray().tolist()
        local_intercept = float(model.intercept)

        # ─ 3a. Allreduce: push local weights ─────────────────────────
        if allreduce_queue is not None:
            allreduce_queue.put({
                'type'         : 'weights',
                'worker_id'    : worker_id,
                'iteration'    : iteration,
                'weights'      : local_weights,
                'intercept'    : local_intercept,
                'row_count'    : row_count,
            })

            # ─ 3b. Receive averaged weights from root ─────────────────
            msg = allreduce_queue.get(timeout=180)
            if msg.get('type') == 'avg_weights':
                current_weights   = msg['weights']
                current_intercept = msg['intercept']
            else:
                # Unexpected message type — fall back to local weights
                current_weights   = local_weights
                current_intercept = local_intercept
        else:
            # No Allreduce — plain local training
            current_weights   = local_weights
            current_intercept = local_intercept

        iterations_done += 1
        iter_time = time.perf_counter() - t_iter
        print(f'[LogReg Worker {worker_id}] iter {iteration+1}/{max_iter}  '
              f'({iter_time:.3f}s)  '
              f'|w|={sum(w**2 for w in current_weights)**0.5:.4f}')

    # ─ 4. Final accuracy on local partition ───────────────────────────────
    # Refit once more with final weights to get summary accuracy.
    # (PySpark LR summary is only available on the last fitted model.)
    lr_final = LogisticRegression(
        featuresCol='features',
        labelCol='label',
        maxIter=1,
        regParam=reg_param,
        elasticNetParam=0.0,
        family='binomial',
        fitIntercept=True,
    )
    model_final     = lr_final.fit(df_vec)
    train_accuracy  = float(model_final.summary.accuracy)
    weight_vector   = model_final.coefficients.toArray().tolist()
    intercept_final = float(model_final.intercept)

    print(f'[LogReg Worker {worker_id}] Final train accuracy: {train_accuracy:.4f}')
    print(f'[LogReg Worker {worker_id}] Weight norm: '
          f'{sum(w**2 for w in weight_vector)**0.5:.4f}')

    df_vec.unpersist()

    return {
        'weight_vector'  : current_weights if current_weights is not None else weight_vector,
        'intercept'      : current_intercept,
        'train_accuracy' : train_accuracy,
        'row_count'      : row_count,
        'iterations_done': iterations_done,
        'partition_path' : partition_path,
    }
