# ============================================================
# applications/wordcount.py
# WordCount RDD pipeline — current benchmark application
# ============================================================

from __future__ import annotations

from pathlib import Path


def run(text_rdd):
    """
    Standard WordCount RDD pipeline.

    Parameters
    ----------
    text_rdd : pyspark.rdd.RDD
        Text lines already loaded from an assigned data partition.

    Returns
    -------
    list[tuple[str, int]]
        Word-count tuples produced by this driver.
    """
    return (
        text_rdd.flatMap(lambda line: line.lower().split())
        .filter(lambda word: len(word) > 0)
        .map(lambda word: (word, 1))
        .reduceByKey(lambda a, b: a + b)
        .collect()
    )


def run_wordcount(partition_path: str | Path, spark):
    """
    Worker-facing WordCount entry point.

    Loads one partition allocated to a Spark driver, applies the standard
    WordCount RDD pipeline, and returns the local word-count result.

    Parameters
    ----------
    partition_path : str | Path
        Path to the worker's assigned text partition.
    spark : pyspark.sql.SparkSession
        Worker-local independent SparkSession.

    Returns
    -------
    list[tuple[str, int]]
        Local word-count result for root-side aggregation.
    """
    partition_path = str(partition_path)

    if not partition_path:
        raise ValueError("WordCount partition path must not be empty.")

    text_rdd = spark.sparkContext.textFile(partition_path)
    return run(text_rdd)



def run_wordcount_rdd(text_rdd, worker_config=None):
    """Phase 3 worker-dispatch adapter.

    worker_process.py calls run_wordcount(rdd, worker_config); the
    Phase 2 run() only needs the RDD. WordCount has no tunable
    parameters, so worker_config is accepted and ignored.
    """
    return run(text_rdd)
