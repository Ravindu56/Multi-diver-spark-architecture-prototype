# ================================================================
# mpj_spark/applications/logreg.py
#
# Logistic Regression MLlib pipeline — per-worker ML workload
#
# ALLREDUCE STRATEGY (simulated Queue-based, TWO-QUEUE design)
# ─────────────────────────────────────────────────────────────
# Uses TWO separate queues to avoid deadlock:
#   allreduce_up_queue   — this worker → root  (local weight vectors)
#   allreduce_down_queue — root → this worker  (averaged weights)
#
# Per iteration:
#   1. Worker fits one LR pass (maxIter=1)
#   2. Pushes {'type':'weights', ...} onto allreduce_up_queue
#   3. Blocks on allreduce_down_queue.get() for averaged weights
#   4. Uses averaged weights as starting point for next iteration
#
# NOTE ON CSV READING
# ───────────────────
# file_manager.dynamic_partition() streams the source CSV line-by-line
# using round-robin assignment. The source CSV has a header row
# (f0,f1,...,label) which lands in whichever partition gets line 0.
# Fix: explicit StructType schema (all DoubleType) so Spark never
# infers StringType; filter WHERE f0 IS NULL to drop the stray header.
# ================================================================

import time

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
) -> dict:
    """
    Logistic Regression pipeline on a labelled binary-classification CSV.

    Partition format (headerless lines produced by MPJSparkFileManager,
    possibly containing one stray header line from the source CSV):
        <f0>,<f1>,...,<f{num_features-1}>,<label>
    where label ∈ {0, 1}.
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

    # ─ 3. Iterative Allreduce training ────────────────────────────────────
    current_weights   = None
    current_intercept = 0.0
    iterations_done   = 0

    for iteration in range(max_iter):
        t_iter = time.perf_counter()

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

        if use_allreduce:
            # Push to root via UP queue
            _up.put({
                'type'     : 'weights',
                'worker_id': worker_id,
                'iteration': iteration,
                'weights'  : local_weights,
                'intercept': local_intercept,
                'row_count': row_count,
            })

            # Block on DOWN queue — root writes exactly one msg per worker per iter
            msg = _down.get(timeout=180)
            if msg.get('type') == 'avg_weights':
                current_weights   = msg['weights']
                current_intercept = msg['intercept']
            else:
                current_weights   = local_weights
                current_intercept = local_intercept
        else:
            current_weights   = local_weights
            current_intercept = local_intercept

        iterations_done += 1
        iter_time = time.perf_counter() - t_iter
        print(f'[LogReg Worker {worker_id}] iter {iteration+1}/{max_iter}  '
              f'({iter_time:.3f}s)  '
              f'|w|={sum(w**2 for w in current_weights)**0.5:.4f}')

    # ─ 4. Final accuracy on local partition ──────────────────────────────
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
