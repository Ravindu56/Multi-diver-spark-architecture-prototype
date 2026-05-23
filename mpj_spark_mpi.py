# ============================================================
# MPJ-SPARK Multi-Driver Architecture — Phase 3 (MPI Layer)
# WordCount with mpi4py + OpenMPI replacing multiprocessing
# University of Jaffna — 2022/E/033 & 2022/E/090
# Phase 3 — Obj 1b, 1c | Ref: Issue #7
# ============================================================

import os
import sys

os.environ["JAVA_TOOL_OPTIONS"] = (
    "--add-opens=java.base/sun.nio.ch=ALL-UNNAMED "
    "--add-opens=java.base/java.nio=ALL-UNNAMED "
    "--add-opens=java.base/java.lang=ALL-UNNAMED "
    "--add-opens=java.base/java.util=ALL-UNNAMED "
    "-Djava.security.manager=allow"
)
os.environ["PYSPARK_PYTHON"]        = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable


import time
import shutil
import argparse
from collections import defaultdict
from mpi4py import MPI
from pyspark.sql import SparkSession


# ── MPI Communicator (initialised once at program start) ───────────────────
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()   # total MPI processes = 1 root + N workers


# ============================================================
# FILE MANAGER (unchanged from v2 — reused as-is)
# ============================================================
class MPJSparkFileManager:
    def __init__(self, shared_storage_path="./shared_storage"):
        self.shared_storage_path = shared_storage_path
        os.makedirs(shared_storage_path, exist_ok=True)
        self.partitions_dir = os.path.join(shared_storage_path, "partitions")
        os.makedirs(self.partitions_dir, exist_ok=True)

    def dynamic_partition(self, input_file_path, num_workers):
        file_size = os.path.getsize(input_file_path)
        partition_paths = [
            os.path.join(self.partitions_dir, f"partition_{i}.txt")
            for i in range(num_workers)
        ]
        writers = [open(p, 'w', encoding='utf-8', buffering=1024*1024)
                   for p in partition_paths]
        line_counts = [0] * num_workers
        try:
            with open(input_file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f):
                    idx = line_num % num_workers
                    writers[idx].write(line)
                    line_counts[idx] += 1
        finally:
            for w in writers:
                w.close()

        partition_metadata_list = []
        for i in range(num_workers):
            partition_metadata_list.append({
                "partition_id":      i,
                "partition_path":    partition_paths[i],
                "num_lines":         line_counts[i],
                "file_size_bytes":   os.path.getsize(partition_paths[i]),
                "total_input_bytes": file_size,
            })
        return partition_metadata_list

    def cleanup(self):
        if os.path.exists(self.partitions_dir):
            shutil.rmtree(self.partitions_dir)
            os.makedirs(self.partitions_dir, exist_ok=True)


# ============================================================
# RANK 0 — ROOT COORDINATOR
# ============================================================
def root_process(input_file_path):
    """
    Root (rank 0) responsibilities:
      Phase 1 — Partition input file
      Phase 2 — Send partition metadata to each worker rank via MPI Send
      Phase 3 — Receive results from each worker rank via MPI Recv
      Phase 4 — Aggregate final word counts using Root's Spark driver
    """
    num_workers = size - 1   # ranks 1..(size-1)

    print("=" * 70)
    print("  MPJ-SPARK Multi-Driver WordCount — Phase 3 (mpi4py)")
    print(f"  MPI size: {size} | Workers: {num_workers} | Root: rank 0")
    print("=" * 70)

    total_start = time.time()

    # ── Phase 1: Partition ────────────────────────────────────────────────
    print("\n[ROOT] Phase 1: Partitioning input file...")
    file_manager = MPJSparkFileManager()
    load_start = time.time()
    partition_metadata_list = file_manager.dynamic_partition(
        input_file_path, num_workers
    )
    load_time = time.time() - load_start
    print(f"[ROOT] {num_workers} partitions created in {load_time:.3f}s")

    # ── Phase 2: Send metadata to workers via MPI ─────────────────────────
    # Replaces: Process(target=worker_fn, args=(i, partition_metadata[i], ...))
    print(f"\n[ROOT] Phase 2: Sending partition metadata to {num_workers} workers...")
    process_start = time.time()

    for worker_rank in range(1, size):
        # worker_rank i handles partition i-1
        metadata = partition_metadata_list[worker_rank - 1]
        comm.send(metadata, dest=worker_rank, tag=10)
        print(f"  [ROOT] Sent metadata to rank {worker_rank} "
              f"→ partition_{worker_rank - 1}.txt "
              f"({metadata['num_lines']:,} lines)")

    # ── Phase 3: Receive results from workers ────────────────────────────
    # Replaces: result_queue.get() × num_workers
    print(f"\n[ROOT] Phase 3: Receiving results from {num_workers} workers...")
    all_results    = []
    worker_timings = []

    for worker_rank in range(1, size):
        result = comm.recv(source=worker_rank, tag=20)
        timing = comm.recv(source=worker_rank, tag=21)

        if "error" in result:
            print(f"  [ERROR] Rank {worker_rank}: {result['error']}")
        else:
            all_results.append(result)
            worker_timings.append(timing)
            print(f"  [ROOT] Received {result['num_words']:,} unique words "
                  f"from rank {worker_rank}")

    wall_clock_proc = time.time() - process_start

    if not all_results:
        print("[ABORT] All workers failed.")
        file_manager.cleanup()
        return

    # ── Phase 4: Final Aggregation ────────────────────────────────────────
    print("\n[ROOT] Phase 4: Aggregating results...")
    agg_start = time.time()
    final_word_counts = defaultdict(int)
    for worker_result in all_results:
        for word, count in worker_result["results"]:
            final_word_counts[word] += count
    agg_time = time.time() - agg_start

    sorted_results = sorted(
        final_word_counts.items(), key=lambda x: x[1], reverse=True
    )
    total_time = time.time() - total_start

    # ── Print Results ─────────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("  RESULTS")
    print(f"{'=' * 70}")
    print(f"  Total unique words:     {len(sorted_results):,}")
    print(f"  Total word occurrences: {sum(v for _, v in sorted_results):,}")
    print(f"\n  Top 20 words:")
    for word, count in sorted_results[:20]:
        print(f"    {word:25s} -> {count:,}")

    # ── Timing Report ─────────────────────────────────────────────────────
    avg_driver_init = sum(t["driver_init_time"] for t in worker_timings) / len(worker_timings)
    avg_proc        = sum(t["processing_time"]  for t in worker_timings) / len(worker_timings)

    print(f"\n{'=' * 70}")
    print("  TIMING (Phase 3 — MPI Layer)")
    print(f"{'=' * 70}")
    print(f"  Load Time      (T_Load):  {load_time:8.4f} s")
    print(f"  Driver Init    (T_Init):  {avg_driver_init:8.4f} s  [avg per worker]")
    print(f"  Processing     (T_Proc):  {avg_proc:8.4f} s  [avg per worker]")
    print(f"  Aggregation    (T_Agg):   {agg_time:8.4f} s")
    print(f"  Wall-clock parallel:      {wall_clock_proc:8.4f} s")
    print(f"  Total Execution Time:     {total_time:8.4f} s")
    print(f"\n  Per-Worker Timings:")
    print(f"  {'Rank':<8} {'Driver Init':<15} {'Processing':<15} {'Total':<12}")
    print(f"  {'-'*50}")
    for t in sorted(worker_timings, key=lambda x: x["rank"]):
        print(f"  rank {t['rank']}   "
              f"{t['driver_init_time']:>8.2f} s      "
              f"{t['processing_time']:>8.2f} s      "
              f"{t['total_worker_time']:>6.2f} s")

    file_manager.cleanup()


# ============================================================
# RANKS 1..N — SPARK DRIVER WORKERS
# ============================================================
def worker_process():
    """
    Each worker rank:
      1. Receives partition metadata from root (MPI Recv, tag=10)
      2. Creates its own independent PySpark driver
      3. Runs WordCount on its assigned partition
      4. Sends results back to root (MPI Send, tag=20)
      5. Sends timing data to root (MPI Send, tag=21)
    """
    # ── Step 1: Receive metadata from root ───────────────────────────────
    # Replaces: partition_metadata received as Process() argument
    partition_metadata = comm.recv(source=0, tag=10)

    worker_start = time.time()
    total_cores  = os.cpu_count() or 4
    num_workers  = size - 1
    cores_per_worker = max(1, total_cores // num_workers)
    partition_lines  = partition_metadata["num_lines"]
    shuffle_parts    = max(2, partition_lines // 500_000)

    try:
        # ── Step 2: Create independent Spark driver ───────────────────────
        spark = (SparkSession.builder
            .master(f"local[{cores_per_worker}]")
            .appName(f"MPJ-MPI-Worker-rank{rank}-SparkDriver")
            .config("spark.ui.enabled",             "false")
            .config("spark.driver.host",            "localhost")
            .config("spark.sql.shuffle.partitions", str(shuffle_parts))
            .config("spark.default.parallelism",    str(cores_per_worker * 2))
            .config("spark.driver.extraJavaOptions",
                    "-Djava.security.manager=allow")
            .config("spark.pyspark.python",         sys.executable)
            .getOrCreate())

        sc = spark.sparkContext
        sc.setLogLevel("ERROR")
        driver_init_time = time.time()

        # ── Step 3: WordCount on assigned partition ───────────────────────
        text_rdd = sc.textFile(partition_metadata["partition_path"])
        results  = (text_rdd
                    .flatMap(lambda line: line.lower().split())
                    .filter(lambda word: len(word) > 1)
                    .map(lambda word: (word, 1))
                    .reduceByKey(lambda a, b: a + b)
                    .collect())
        processing_done = time.time()
        spark.stop()

        # ── Step 4 & 5: Send results and timings to root ──────────────────
        # Replaces: result_queue.put(...) and timing_queue.put(...)
        comm.send({
            "rank":      rank,
            "results":   [(str(k), int(v)) for k, v in results],
            "num_words": len(results),
        }, dest=0, tag=20)

        comm.send({
            "rank":             rank,
            "driver_init_time": driver_init_time - worker_start,
            "processing_time":  processing_done  - driver_init_time,
            "total_worker_time": time.time()     - worker_start,
        }, dest=0, tag=21)

    except Exception as e:
        comm.send({"rank": rank, "error": str(e)}, dest=0, tag=20)
        comm.send({"rank": rank, "error": str(e)}, dest=0, tag=21)


# ============================================================
# MAIN ENTRY POINT — rank-based dispatch
# ============================================================
if __name__ == "__main__":
    if rank == 0:
        parser = argparse.ArgumentParser(
            description="MPJ-SPARK Phase 3 — mpi4py WordCount"
        )
        parser.add_argument("--input",    type=str, default="./test_dataset.txt")
        parser.add_argument("--generate", type=int, default=50)
        args = parser.parse_args()

        input_file = args.input
        if not os.path.exists(input_file):
            # Import generator from v2 prototype
            from mpj_spark_prototype_v2 import generate_test_dataset
            input_file = generate_test_dataset(input_file, args.generate)

        root_process(input_file)
    else:
        worker_process()