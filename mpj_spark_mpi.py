# ============================================================
# mpj_spark_mpi.py — Phase 3 MPI Entry Point
# MPJ-SPARK Multi-Driver Architecture (mpi4py + OpenMPI)
# University of Jaffna — 2022/E/033 & 2022/E/090
#
# This file is the MPI rank-dispatch entry point ONLY.
# All business logic lives in the mpj_spark package:
#   mpj_spark/core/file_manager.py   — MPJSparkFileManager
#   mpj_spark/core/key_value.py      — KeyValueStructure
#   mpj_spark/workers/spark_session.py — Spark session factory
#
# Launch with:
#   mpirun --oversubscribe -np <1+N> \
#       /path/to/venv/bin/python3 mpj_spark_mpi.py --generate 50
#
# Rank 0  = Root coordinator (partition → send → recv → aggregate)
# Rank 1+ = Independent Spark driver workers
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
import argparse
from collections import defaultdict
from mpi4py import MPI

# ── Package imports (no inline duplication) ────────────────────────────────
from mpj_spark.core.file_manager import MPJSparkFileManager
from mpj_spark.core.key_value    import KeyValueStructure
from mpj_spark.workers.spark_session import create_spark_session

# ── MPI communicator ───────────────────────────────────────────────────────
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()   # total MPI processes = 1 root + N workers


# ============================================================
# RANK 0 — ROOT COORDINATOR
# ============================================================

def root_process(input_file_path):
    """
    Root coordinator (rank 0).

    Execution phases:
      1  Partition input file via MPJSparkFileManager
      2  Distribute partition metadata to worker ranks via MPI Send (tag=10)
      3  Receive WordCount results from workers via MPI Recv  (tag=20, 21)
      4  Aggregate final word counts with KeyValueStructure
    """
    num_workers = size - 1

    print("=" * 70)
    print("  MPJ-SPARK Multi-Driver WordCount — Phase 3 (mpi4py)")
    print(f"  MPI size: {size} | Workers: {num_workers} | Root: rank 0")
    print("=" * 70)

    total_start = time.time()

    # ── Phase 1: Partition ────────────────────────────────────────────────
    print("\n[ROOT] Phase 1: Partitioning input file...")
    file_manager = MPJSparkFileManager()
    load_start   = time.time()
    partition_metadata_list = file_manager.dynamic_partition(
        input_file_path, num_workers
    )
    load_time = time.time() - load_start
    print(f"[ROOT] {num_workers} partitions created in {load_time:.3f}s")

    # ── Phase 2: Send metadata to workers via MPI ─────────────────────────
    print(f"\n[ROOT] Phase 2: Sending partition metadata to {num_workers} workers...")
    process_start = time.time()

    for worker_rank in range(1, size):
        metadata = partition_metadata_list[worker_rank - 1]
        comm.send(metadata, dest=worker_rank, tag=10)
        print(
            f"  [ROOT] Sent metadata to rank {worker_rank} "
            f"-> partition_{worker_rank - 1}.txt "
            f"({metadata['num_lines']:,} lines)"
        )

    # ── Phase 3: Receive results from workers ─────────────────────────────
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
            print(
                f"  [ROOT] Received {result['num_words']:,} unique words "
                f"from rank {worker_rank}"
            )

    wall_clock_proc = time.time() - process_start

    if not all_results:
        print("[ABORT] All workers failed.")
        file_manager.cleanup()
        return

    # ── Phase 4: Aggregate with KeyValueStructure ─────────────────────────
    print("\n[ROOT] Phase 4: Aggregating results...")
    agg_start = time.time()

    final_kv = KeyValueStructure()
    for worker_result in all_results:
        final_kv.merge(KeyValueStructure.from_serializable(worker_result["results"]))

    agg_time       = time.time() - agg_start
    sorted_results = final_kv.get_top_n(len(final_kv))
    total_time     = time.time() - total_start

    # ── Results ───────────────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("  RESULTS")
    print(f"{'=' * 70}")
    print(f"  Total unique words:     {len(sorted_results):,}")
    print(f"  Total word occurrences: {sum(v for _, v in sorted_results):,}")
    print(f"\n  Top 20 words:")
    for word, count in sorted_results[:20]:
        print(f"    {word:25s} -> {count:,}")

    # ── Timing report ─────────────────────────────────────────────────────
    avg_driver_init = (
        sum(t["driver_init_time"] for t in worker_timings) / len(worker_timings)
    )
    avg_proc = (
        sum(t["processing_time"] for t in worker_timings) / len(worker_timings)
    )

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
    print(f"  {'Rank':<8} {'Driver Init':<16} {'Processing':<16} {'Total':<12}")
    print(f"  {'-' * 52}")
    for t in sorted(worker_timings, key=lambda x: x["rank"]):
        print(
            f"  rank {t['rank']}   "
            f"{t['driver_init_time']:>8.2f} s      "
            f"{t['processing_time']:>8.2f} s      "
            f"{t['total_worker_time']:>6.2f} s"
        )

    file_manager.cleanup()


# ============================================================
# RANKS 1..N — SPARK DRIVER WORKERS
# ============================================================

def worker_process():
    """
    Each worker rank (rank >= 1):
      1. Receive partition metadata from root  (MPI Recv, tag=10)
      2. Create an independent Spark session via create_spark_session()
      3. Run WordCount on the assigned partition
      4. Send results to root                  (MPI Send, tag=20)
      5. Send timing data to root              (MPI Send, tag=21)
    """
    # Step 1: Receive metadata from root
    partition_metadata = comm.recv(source=0, tag=10)

    worker_start     = time.time()
    num_workers      = size - 1
    partition_lines  = partition_metadata["num_lines"]
    shuffle_parts    = max(2, partition_lines // 500_000)

    try:
        # Step 2: Independent Spark session from package factory
        spark = create_spark_session(
            app_name=f"MPJ-MPI-Worker-rank{rank}",
            rank=rank,
            num_workers=num_workers,
            shuffle_partitions=shuffle_parts,
        )
        sc = spark.sparkContext
        sc.setLogLevel("ERROR")
        driver_init_time = time.time()

        # Step 3: WordCount
        text_rdd = sc.textFile(partition_metadata["partition_path"])
        results  = (
            text_rdd
            .flatMap(lambda line: line.lower().split())
            .filter(lambda word: len(word) > 1)
            .map(lambda word: (word, 1))
            .reduceByKey(lambda a, b: a + b)
            .collect()
        )
        processing_done = time.time()
        spark.stop()

        serialized = [(str(k), int(v)) for k, v in results]

        # Step 4: Send results
        comm.send(
            {
                "rank":      rank,
                "results":   serialized,
                "num_words": len(results),
            },
            dest=0,
            tag=20,
        )

        # Step 5: Send timings
        comm.send(
            {
                "rank":              rank,
                "driver_init_time":  driver_init_time - worker_start,
                "processing_time":   processing_done  - driver_init_time,
                "total_worker_time": time.time()      - worker_start,
            },
            dest=0,
            tag=21,
        )

    except Exception as exc:
        comm.send({"rank": rank, "error": str(exc)}, dest=0, tag=20)
        comm.send({"rank": rank, "error": str(exc)}, dest=0, tag=21)


# ============================================================
# ENTRY POINT — rank-based dispatch
# ============================================================

if __name__ == "__main__":
    if rank == 0:
        parser = argparse.ArgumentParser(
            description="MPJ-SPARK Phase 3 — mpi4py WordCount entry point"
        )
        parser.add_argument(
            "--input",
            type=str,
            default="./test_dataset.txt",
            help="Path to input text file",
        )
        parser.add_argument(
            "--generate",
            type=int,
            default=50,
            help="Generate synthetic dataset of this size (MB) if --input not found",
        )
        args = parser.parse_args()

        input_file = args.input
        if not os.path.exists(input_file):
            from mpj_spark_prototype_v2 import generate_test_dataset
            input_file = generate_test_dataset(input_file, args.generate)

        root_process(input_file)
    else:
        worker_process()
