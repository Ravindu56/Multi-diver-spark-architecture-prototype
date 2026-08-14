# ============================================================
# applications/wordcount.py
# WordCount RDD pipeline — current benchmark application
# Paper Reference: Section VI.A — Application used for evaluation
# ============================================================


def run(text_rdd):
    """
    Standard WordCount pipeline.

    Pipeline: textFile → flatMap → filter → map → reduceByKey

    Parameters
    ----------
    text_rdd : pyspark.rdd.RDD  — text lines loaded from partition

    Returns
    -------
    list of (word: str, count: int) tuples
    """
    return (
        text_rdd.flatMap(lambda line: line.lower().split())
        .filter(lambda word: len(word) > 0)
        .map(lambda word: (word, 1))
        .reduceByKey(lambda a, b: a + b)
        .collect()
    )



def run_wordcount(text_rdd, worker_config=None):
    """Phase 3 worker-dispatch adapter.

    worker_process.py calls run_wordcount(rdd, worker_config); the
    Phase 2 run() only needs the RDD. WordCount has no tunable
    parameters, so worker_config is accepted and ignored.
    """
    return run(text_rdd)