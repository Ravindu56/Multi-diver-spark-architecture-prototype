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
