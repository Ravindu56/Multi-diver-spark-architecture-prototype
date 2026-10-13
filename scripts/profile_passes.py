"""Time the Spark passes that dominate K-Means and LogReg rounds: per-row vs block cache.

Run from the repository root with the project venv active (no MPI needed):

    python scripts/profile_passes.py --input ./shared_storage/logreg_100mb.csv --rows 540000

It loads the first --rows lines (one worker's share of the 100 MB file at 2 workers),
then reports the median seconds per pass (one warm-up, then --reps timed runs) for

  * a trivial Spark job (pure scheduling overhead),
  * a count() scan (pure per-row un-pickling cost),
  * the K-Means stats and WCSS passes, per-row path vs cached-block path,
  * the LogReg loss+gradient pass, per-row path vs cached-block path,

and checks that the block results equal the per-row results.  If the code under test
predates the block cache, only the per-row numbers are printed.
"""

from __future__ import annotations

import argparse
import itertools
import os
import statistics
import sys
import tempfile
import time

os.environ.setdefault(
    "JAVA_TOOL_OPTIONS",
    "--add-opens=java.base/sun.nio.ch=ALL-UNNAMED "
    "--add-opens=java.base/java.nio=ALL-UNNAMED "
    "--add-opens=java.base/java.lang=ALL-UNNAMED "
    "--add-opens=java.base/java.util=ALL-UNNAMED "
    "-Djava.security.manager=allow",
)
os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np  # noqa: E402


def _timed(fn, reps):
    fn()  # warm-up (JIT, caches, worker start)
    times = []
    out = None
    for _ in range(reps):
        t0 = time.perf_counter()
        out = fn()
        times.append(time.perf_counter() - t0)
    return statistics.median(times), out


def _head(src, n_rows):
    fd, path = tempfile.mkstemp(suffix=".csv", prefix="profile_")
    os.close(fd)
    with open(src, encoding="utf-8") as fin, open(path, "w", encoding="utf-8") as fout:
        fout.writelines(itertools.islice(fin, n_rows))
    return path


def _row(label, seconds, ref=None):
    ratio = f"  ({ref / seconds:4.1f}x faster)" if ref and seconds > 0 else ""
    print(f"  {label:<44}{seconds:8.3f} s{ratio}")


def profile_kmeans(spark, path, reps, k):
    from mpj_spark.applications.kmeans import local_iteration as li

    print("\nK-Means")
    rows_rdd = li.load_partition_rdd(spark, path)
    blocks = getattr(rows_rdd, getattr(li, "_BLOCKS_ATTR", "_mpj_blocks"), None)
    init = li.init_centroids(rows_rdd, k=k, seed=42)

    t_overhead, _ = _timed(lambda: spark.sparkContext.parallelize(range(2), 2).count(), reps)
    t_scan, _ = _timed(rows_rdd.count, reps)
    _row("trivial Spark job (scheduling overhead)", t_overhead)
    _row("per-row scan: rows_rdd.count()", t_scan)

    attr = getattr(li, "_BLOCKS_ATTR", "_mpj_blocks")
    if blocks is not None:
        t_bscan, _ = _timed(blocks.count, reps)
        _row("block scan : blocks.count()", t_bscan, t_scan)
        delattr(rows_rdd, attr)  # force the per-row path for the reference timings
    t_stats_row, res_row = _timed(lambda: li.compute_local_stats(rows_rdd, init), reps)
    t_wcss_row, w_row = _timed(lambda: li.compute_local_wcss(rows_rdd, init), reps)
    _row("compute_local_stats  (per-row path)", t_stats_row)
    _row("compute_local_wcss   (per-row path)", t_wcss_row)

    if blocks is not None:
        setattr(rows_rdd, attr, blocks)
        t_stats_blk, res_blk = _timed(lambda: li.compute_local_stats(rows_rdd, init), reps)
        t_wcss_blk, w_blk = _timed(lambda: li.compute_local_wcss(rows_rdd, init), reps)
        _row("compute_local_stats  (block path)", t_stats_blk, t_stats_row)
        _row("compute_local_wcss   (block path)", t_wcss_blk, t_wcss_row)
        same = np.array_equal(res_row[1], res_blk[1]) and np.allclose(
            res_row[0], res_blk[0], rtol=1e-10, atol=1e-8
        )
        same_w = abs(w_row - w_blk) <= 1e-9 * max(1.0, abs(w_row))
        print(f"  results identical: counts+sums={same}  wcss={same_w}")
        est_row = 5 * t_stats_row + t_wcss_row
        est_blk = 5 * t_stats_blk + t_wcss_blk
        print(
            f"  estimated Spark time per round (5 epochs + WCSS): "
            f"{est_row:.2f}s -> {est_blk:.2f}s"
        )
    else:
        print("  (no block cache found: running the per-row path only)")


def profile_logreg(spark, path, reps, num_features):
    from mpj_spark.applications.logreg.warm_common import load_points
    from mpj_spark.core import warm_start_lr as ws

    print("\nLogReg")
    _, points, n_rows = load_points(spark, path, num_features, "profile")
    reg = 0.01
    rng = np.random.default_rng(0)
    theta = rng.normal(0, 0.1, num_features + 1)

    def row_loss_grad(th):
        bc = th.copy()

        def per_partition(it):
            rows = list(it)
            if not rows:
                return
            X = np.asarray([r[0] for r in rows], dtype=np.float64)
            y = np.asarray([r[1] for r in rows], dtype=np.float64)
            yield ws.partition_stats(X, y, bc)

        zero = (0.0, np.zeros(num_features + 1), 0)
        s, g, n = points.mapPartitions(per_partition).fold(
            zero, lambda a, c: (a[0] + c[0], a[1] + c[1], a[2] + c[2])
        )
        return ws.regularised(s, g, n if n else n_rows, th, reg)

    t_scan, _ = _timed(points.count, reps)
    t_row, (f_row, g_row) = _timed(lambda: row_loss_grad(theta), reps)
    _row("per-row scan: points.count()", t_scan)
    _row("loss+grad pass (per-row, reference)", t_row)

    fg = ws.spark_loss_grad(points, num_features, reg, n_rows)  # block cache built here
    t_blk, (f_blk, g_blk) = _timed(lambda: fg(theta), reps)
    _row("loss+grad pass (spark_loss_grad)", t_blk, t_row)
    same = abs(f_row - f_blk) <= 1e-10 * max(1.0, abs(f_row)) and np.allclose(
        g_row, g_blk, rtol=1e-9, atol=1e-12
    )
    print(f"  results identical: loss+gradient={same}")
    print(f"  estimated Spark time per ~7-pass round: {7 * t_row:.2f}s -> {7 * t_blk:.2f}s")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--input", default="./shared_storage/logreg_100mb.csv")
    ap.add_argument("--rows", type=int, default=540_000)
    ap.add_argument("--cores", type=int, default=2)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--features", type=int, default=10)
    ap.add_argument("--only", choices=["kmeans", "logreg"], default=None)
    args = ap.parse_args()

    from mpj_spark.workers.spark_session import build_spark_session

    path = _head(args.input, args.rows)
    spark = build_spark_session(
        app_name="profile-passes", cores_override=args.cores, num_workers=2
    )
    try:
        print(f"rows={args.rows:,} cores={args.cores} reps={args.reps}")
        if args.only in (None, "kmeans"):
            profile_kmeans(spark, path, args.reps, args.k)
        if args.only in (None, "logreg"):
            profile_logreg(spark, path, args.reps, args.features)
    finally:
        spark.stop()
        os.remove(path)


if __name__ == "__main__":
    main()
