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
# NOTE ON CSV READING
# ───────────────────
# file_manager.dynamic_partition() streams the source CSV line-by-line
# using round-robin assignment.  The source CSV has a header row
# (f0,f1,...,label) which lands in whichever partition gets line 0.
# To handle this robustly across all partitions:
#   1. Provide an explicit StructType schema (all DoubleType) so Spark
#      never infers StringType from a stray header value.
#   2. Read with header=False and schema= (inferSchema disabled).
#      When the header string "f0" is parsed as Double it becomes NULL.
#   3. Filter WHERE f0 IS NOT NULL to silently drop the one stray row.
#   4. Cast the label column to IntegerType for MLlib compatibility.
# ================================================================

import time

from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import VectorAssembler
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, IntegerType, StructField, StructType


def _build_schema(num_features: int) -> StructType:
    """Return an explicit schema: f0..f{n-1} as Double, label as Double.
    Label is Double here so the CSV parser never sees a type mismatch;
    we cast to Integer after loading."""
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
    allreduce_queue=None,
    num_workers: int = 1,
) -> dict:
    """
    Logistic Regression pipeline on a labelled binary-classification CSV.

    Partition format (headerless lines produced by MPJSparkFileManager,
    possibly containing one stray header line from the source CSV):
        <f0>,<f1>,...,<f{num_features-1}>,<label>
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

    # ─ 1. Load CSV with explicit schema ──────────────────────────────────
    # Use an explicit StructType so Spark never infers StringType when a
    # stray header row is present. Header strings (e.g. "f0") parse as
    # NULL under DoubleType; we filter them out immediately after load.
    schema = _build_schema(num_features)

    df_raw = spark.read.csv(
        partition_path,
        schema=schema,       # explicit — no inferSchema
        header=False,        # partitions are headerless (or have one stray row)
    )

    # Drop the stray header row (parsed as NULLs) and any other nulls
    df = df_raw.filter(F.col('f0').isNotNull()) \
               .dropna() \
               .withColumn('label', F.col('label').cast(IntegerType()))

    feature_cols = [f'f{i}' for i in range(num_features)]
    row_count    = df.count()

    if row_count == 0:
        raise RuntimeError(
            f'[LogReg Worker {worker_id}] Partition is empty after cleaning. '
            f'Check --logreg-features matches the dataset (expected {num_features} features).'
        )

    print(f'[LogReg Worker {worker_id}] reg_param={reg_param} | '
          f'max_iter={max_iter} | rows={row_count:,} | '
          f'features={len(feature_cols)} | allreduce={allreduce_queue is not None}')

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

        # ─ 3a. Allreduce: push local weights ──────────────────────────
        if allreduce_queue is not None:
            allreduce_queue.put({
                'type'     : 'weights',
                'worker_id': worker_id,
                'iteration': iteration,
                'weights'  : local_weights,
                'intercept': local_intercept,
                'row_count': row_count,
            })

            # ─ 3b. Receive averaged weights from root ─────────────────
            msg = allreduce_queue.get(timeout=180)
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

    # ─ 4. Final accuracy on local partition ───────────────────────────────
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
