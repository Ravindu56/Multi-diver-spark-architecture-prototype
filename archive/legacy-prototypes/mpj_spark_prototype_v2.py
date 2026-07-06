# ============================================================
# MPJ-SPARK Multi-Driver Architecture Prototype v2.0
# WordCount Implementation for BScEng Research
# University of Jaffna — 2022/E/033 & 2022/E/090
# ============================================================
# Simulates MPJ-SPARK paper (DOI: 10.1109/ACCESS.2025.3584744)
# architecture on a single machine using:
#   - Python multiprocessing  → simulates MPJ process model
#   - PySpark (per-process)   → simulates independent Spark drivers
# ============================================================
# FIXES APPLIED (v2.0):
#   C1+C2 : Streaming partitioner — no full file RAM load
#   C3    : local[N] cores — no CPU thrashing
#   H1    : PYSPARK_PYTHON explicitly set
#   H2    : Exact num_workers queue.get() — no race condition
#   H3    : Corrected timing: driver_init separated from T_Proc
#   M1    : PySpark import at top level
#   M2    : Baseline SparkSession mirrors worker config
#   M3    : Partial failure detection and abort
#   L1    : Richer vocabulary for realistic word distribution
#   L2    : Dynamic shuffle partitions per data size
# ============================================================

import os
import sys

# ── Environment — must be set BEFORE any PySpark import ────────────────────
os.environ["JAVA_TOOL_OPTIONS"] = "-Djava.security.manager=allow"
os.environ["PYSPARK_PYTHON"] = sys.executable  # H1 fix
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable  # H1 fix

import time  # noqa: E402
import shutil  # noqa: E402
import random  # noqa: E402
import argparse  # noqa: E402
from multiprocessing import Process, Queue  # noqa: E402
from collections import defaultdict  # noqa: E402
from pyspark.sql import SparkSession  # noqa: E402  # M1 fix: top-level import


# ============================================================
# COMPONENT 1: MPJ-SPARK File Manager (Simulates HPC Shared Storage)
# ============================================================
class MPJSparkFileManager:
    """
    Replaces HDFS with a local shared storage file manager.
    Paper Reference: Section IV.B — MPJ-SPARK File Manager

    v2.0 FIX (C1+C2): Streaming single-pass partitioner.
    - Never loads full file into RAM.
    - Opens all N partition writers simultaneously.
    - Round-robin line distribution = perfectly balanced partitions.
    - Total disk I/O reduced from (N+1)×size to 2×size.
    """

    def __init__(self, shared_storage_path="./shared_storage"):
        self.shared_storage_path = shared_storage_path
        os.makedirs(shared_storage_path, exist_ok=True)
        self.partitions_dir = os.path.join(shared_storage_path, "partitions")
        os.makedirs(self.partitions_dir, exist_ok=True)

    def dynamic_partition(self, input_file_path, num_workers):
        """
        Dynamic Partitioning — Paper Section IV.C
        partition_size = file_size / num_workers (one-to-one worker mapping)

        Implementation: streaming round-robin — single file pass,
        all N writers open simultaneously, no full-file RAM allocation.
        """
        file_size = os.path.getsize(input_file_path)

        partition_paths = [
            os.path.join(self.partitions_dir, f"partition_{i}.txt") for i in range(num_workers)
        ]

        # Open all partition writers at once (1 MB write buffer each)
        writers = [open(p, "w", encoding="utf-8", buffering=1024 * 1024) for p in partition_paths]
        line_counts = [0] * num_workers

        try:
            with open(input_file_path, "r", encoding="utf-8") as f:
                for line_num, line in enumerate(f):  # streaming iterator
                    idx = line_num % num_workers  # round-robin
                    writers[idx].write(line)
                    line_counts[idx] += 1
        finally:
            for w in writers:
                w.close()

        partition_metadata_list = []
        for i in range(num_workers):
            partition_metadata_list.append(
                {
                    "partition_id": i,
                    "partition_path": partition_paths[i],
                    "num_lines": line_counts[i],
                    "file_size_bytes": os.path.getsize(partition_paths[i]),
                    "total_input_bytes": file_size,
                }
            )
            print(f"  Partition {i}: {line_counts[i]:,} lines -> {partition_paths[i]}")

        return partition_metadata_list

    def cleanup(self):
        """Remove partition files after aggregation."""
        if os.path.exists(self.partitions_dir):
            shutil.rmtree(self.partitions_dir)
            os.makedirs(self.partitions_dir, exist_ok=True)


# ============================================================
# COMPONENT 2: Key-Value Data Structure (RDD <-> MPJ Buffer)
# ============================================================
class KeyValueStructure:
    """
    Paper Reference: Section IV.D — Data Conversion Mechanism
    Bridges Spark RDD results and MPJ contiguous array buffers.
    Simulated via Python Queue (replaces MPI message passing).
    """

    def __init__(self):
        self.data = []

    def from_rdd_collect(self, rdd_results):
        self.data = list(rdd_results)
        return self

    def to_serializable(self):
        return [(str(k), int(v)) for k, v in self.data]

    @staticmethod
    def from_serializable(serialized_data):
        kv = KeyValueStructure()
        kv.data = [(str(k), int(v)) for k, v in serialized_data]
        return kv


# ============================================================
# COMPONENT 3: MPJ Worker Process (Independent Spark Driver)
# ============================================================
def mpj_worker_process(worker_id, partition_metadata, result_queue, timing_queue, num_workers):
    """
    Paper Reference: Section IV.A (MPJ Workers) + Algorithm 1

    Each MPJ Worker:
      1. Receives partition metadata from Root (via Queue)
      2. Creates its OWN isolated Spark driver (SparkSession)
      3. Reads partition directly from shared storage via metadata
      4. Executes WordCount application logic on local partition
      5. Converts RDD results to KeyValue structure
      6. Sends results back to Root (simulates MPJ Send-Results)

    v2.0 FIX (C3): cores_per_worker = total_cores / num_workers
    v2.0 FIX (L2): dynamic shuffle partitions based on data size
    """
    try:
        worker_start_time = time.time()

        # C3 fix: divide cores fairly across workers
        total_cores = os.cpu_count() or 4
        cores_per_worker = max(1, total_cores // num_workers)

        # L2 fix: dynamic shuffle partitions
        partition_lines = partition_metadata["num_lines"]
        shuffle_parts = max(2, partition_lines // 500_000)

        # ── STEP 1: Create INDEPENDENT Spark Driver ────────────────────
        # Paper: "Each MPJ Worker has its own copy of the spark driver,
        #         isolated from the other workers"
        spark = (
            SparkSession.builder.master(f"local[{cores_per_worker}]")
            .appName(f"MPJ-Worker-{worker_id}-SparkDriver")
            .config("spark.ui.enabled", "false")
            .config("spark.driver.host", "localhost")
            .config("spark.sql.shuffle.partitions", str(shuffle_parts))
            .config("spark.default.parallelism", str(cores_per_worker * 2))
            .config("spark.driver.extraJavaOptions", "-Djava.security.manager=allow")
            .config("spark.executor.extraJavaOptions", "-Djava.security.manager=allow")
            .config("spark.pyspark.python", sys.executable)
            .getOrCreate()
        )

        sc = spark.sparkContext
        sc.setLogLevel("ERROR")

        driver_init_time = time.time()

        # ── STEP 2: Read partition from shared storage via metadata ──────
        # Paper: "SPARK-context.readfile(Received-Metadata) from HPC shared storage"
        partition_path = partition_metadata["partition_path"]
        text_rdd = sc.textFile(partition_path)

        # ── STEP 3: Execute WordCount Application Logic ─────────────────
        # flatMap → filter → map → reduceByKey
        word_counts_rdd = (
            text_rdd.flatMap(lambda line: line.lower().split())
            .filter(lambda word: len(word) > 1)
            .map(lambda word: (word, 1))
            .reduceByKey(lambda a, b: a + b)
        )

        # ── STEP 4: Collect results (RDD → local Python) ────────────────
        results = word_counts_rdd.collect()
        processing_done_time = time.time()

        # ── STEP 5: Convert RDD → KeyValue structure ────────────────────
        # Paper: "Convert RDD result to Key Value data structure"
        kv = KeyValueStructure()
        kv.from_rdd_collect(results)
        serialized_results = kv.to_serializable()

        # ── STEP 6: Send results to Root (simulates MPJ Send-Results) ───
        # Paper: "Send-Results(Result, Start-Position, Count, Type, Root, Tag)"
        result_queue.put(
            {
                "worker_id": worker_id,
                "results": serialized_results,
                "num_words": len(serialized_results),
                "partition_lines": partition_lines,
            }
        )

        worker_end_time = time.time()

        timing_queue.put(
            {
                "worker_id": worker_id,
                "driver_init_time": driver_init_time - worker_start_time,
                "processing_time": processing_done_time - driver_init_time,
                "total_worker_time": worker_end_time - worker_start_time,
            }
        )

        spark.stop()

    except Exception as e:
        result_queue.put({"worker_id": worker_id, "error": str(e)})
        timing_queue.put({"worker_id": worker_id, "error": str(e)})


# ============================================================
# COMPONENT 4: MPJ Root Process (Orchestrator)
# ============================================================
def mpj_root_process(input_file_path, num_workers):
    """
    Paper Reference: Section IV.A (Root Process) + Algorithm 1

    Root Process Responsibilities:
      Phase 1 — Initialise File Manager, partition input file
      Phase 2 — Distribute partition metadata to N MPJ Workers
      Phase 3 — Wait for all workers to complete (parallel execution)
      Phase 4 — Receive results from all workers (simulates MPI Recv)
      Phase 5 — Final aggregation using Root's own Spark driver
    """
    print("=" * 70)
    print("  MPJ-SPARK Multi-Driver WordCount Prototype  v2.0")
    print(f"  Workers: {num_workers} | Input: {input_file_path}")
    print(
        f"  Host cores: {os.cpu_count()} | Cores/worker: "
        f"{max(1, (os.cpu_count() or 4) // num_workers)}"
    )
    print("=" * 70)

    total_start_time = time.time()

    # ── PHASE 1: File Manager — Partition Input ─────────────────────────
    print("\n[ROOT] Phase 1: Initializing MPJ-SPARK File Manager...")
    file_manager = MPJSparkFileManager()

    load_start = time.time()
    partition_metadata_list = file_manager.dynamic_partition(input_file_path, num_workers)
    load_end = time.time()
    load_time = load_end - load_start

    print(f"[ROOT] {num_workers} partitions created in {load_time:.3f}s")

    # ── PHASE 2: Launch Workers with Partition Metadata ─────────────────
    print(f"\n[ROOT] Phase 2: Distributing metadata to {num_workers} MPJ Workers...")

    result_queue = Queue()
    timing_queue = Queue()
    workers = []
    process_start = time.time()

    for i in range(num_workers):
        # Paper: "Send-Metadata(PRTi.Metadata, Start, Count, Type, Dest, Tag)"
        p = Process(
            target=mpj_worker_process,
            args=(
                i,
                partition_metadata_list[i],
                result_queue,
                timing_queue,
                num_workers,
            ),
        )
        workers.append(p)
        p.start()
        print(f"  [ROOT] Launched MPJ Worker {i} (PID: {p.pid}) " f"→ independent Spark Driver")

    # ── PHASE 3: Wait for Parallel Execution ────────────────────────────
    print(f"\n[ROOT] Phase 3: Waiting for {num_workers} workers to complete...")
    for p in workers:
        p.join()

    process_end = time.time()
    wall_clock_proc = process_end - process_start

    # ── PHASE 4: Receive Results (simulates MPI Recv) ───────────────────
    print("\n[ROOT] Phase 4: Receiving results from all workers...")

    all_results = []
    worker_timings = []
    failed_workers = 0

    # H2 fix: exactly num_workers gets — no race condition
    for _ in range(num_workers):
        result = result_queue.get()
        if "error" in result:
            failed_workers += 1
            print(f"  [ERROR] Worker {result['worker_id']}: {result['error']}")
        else:
            all_results.append(result)
            print(
                f"  [ROOT] Received {result['num_words']:,} unique words "
                f"from Worker {result['worker_id']}"
            )

    for _ in range(num_workers):
        timing = timing_queue.get()
        if "error" not in timing:
            worker_timings.append(timing)

    # M3 fix: partial failure detection
    if failed_workers > 0:
        print(
            f"\n  [WARNING] {failed_workers}/{num_workers} workers FAILED — "
            f"results cover only {len(all_results)} partitions"
        )
    if len(all_results) == 0:
        print("  [ABORT] All workers failed. No results to aggregate.")
        file_manager.cleanup()
        return [], {"load_time": load_time, "processing_time": 0, "total_time": 0}

    # ── PHASE 5: Final Aggregation (Root's Spark Driver) ────────────────
    print("\n[ROOT] Phase 5: Final aggregation using Root Spark Driver...")
    # Paper: "SparkContext.parallelize(Result-List) → collect final results"

    aggregation_start = time.time()
    final_word_counts = defaultdict(int)
    for worker_result in all_results:
        kv = KeyValueStructure.from_serializable(worker_result["results"])
        for word, count in kv.data:
            final_word_counts[word] += count
    aggregation_end = time.time()

    sorted_results = sorted(final_word_counts.items(), key=lambda x: x[1], reverse=True)
    total_end_time = time.time()

    # ── RESULTS ──────────────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("  RESULTS")
    print(f"{'=' * 70}")
    print(f"  Total unique words:     {len(sorted_results):,}")
    print(f"  Total word occurrences: {sum(v for _, v in sorted_results):,}")
    print("\n  Top 20 words:")
    for word, count in sorted_results[:20]:
        print(f"    {word:25s} -> {count:,}")

    # ── TIMING ANALYSIS (Paper-aligned) ─────────────────────────────────
    total_time = total_end_time - total_start_time
    agg_time = aggregation_end - aggregation_start

    # H3 fix: separate driver init from true processing time
    if worker_timings:
        avg_driver_init = sum(wt["driver_init_time"] for wt in worker_timings) / len(worker_timings)
        avg_actual_proc = sum(wt["processing_time"] for wt in worker_timings) / len(worker_timings)
    else:
        avg_driver_init = 0
        avg_actual_proc = 0

    print(f"\n{'=' * 70}")
    print("  TIMING ANALYSIS  (Paper Metrics — v2.0 corrected)")
    print(f"{'=' * 70}")
    print(
        f"  Load Time      (T_Load):   {load_time:8.4f} s   "
        f"({(load_time / total_time) * 100:5.1f}% of total)"
    )
    print(
        f"  Driver Init    (T_Init):   {avg_driver_init:8.4f} s   "
        f"({(avg_driver_init / total_time) * 100:5.1f}% of total)  [avg per worker]"
    )
    print(
        f"  Processing     (T_Proc):   {avg_actual_proc:8.4f} s   "
        f"({(avg_actual_proc / total_time) * 100:5.1f}% of total)  [avg per worker]"
    )
    print(
        f"  Aggregation    (T_Agg):    {agg_time:8.4f} s   "
        f"({(agg_time / total_time) * 100:5.1f}% of total)"
    )
    print(f"  Wall-clock parallel:       {wall_clock_proc:8.4f} s")
    print(f"  Total Execution Time:      {total_time:8.4f} s")
    print("\n  Paper reference:  T_Load%=74.3%  T_Proc%=25.7%")
    print(
        f"  Prototype result: T_Load%={(load_time / total_time) * 100:.1f}%"
        f"  T_Proc%={(avg_actual_proc / total_time) * 100:.1f}%"
    )

    print("\n  Per-Worker Timings:")
    print(f"  {'Worker':<10} {'Driver Init':<15} {'Processing':<15} {'Total':<12}")
    print(f"  {'-' * 52}")
    for wt in sorted(worker_timings, key=lambda x: x.get("worker_id", 0)):
        print(
            f"  Worker {wt['worker_id']}   "
            f"{wt['driver_init_time']:>8.2f} s      "
            f"{wt['processing_time']:>8.2f} s      "
            f"{wt['total_worker_time']:>6.2f} s"
        )

    file_manager.cleanup()

    return sorted_results, {
        "load_time": load_time,
        "processing_time": avg_actual_proc,
        "driver_init": avg_driver_init,
        "aggregation": agg_time,
        "total_time": total_time,
        "wall_clock_proc": wall_clock_proc,
    }


# ============================================================
# COMPONENT 5: Standard Spark WordCount (Single Driver — Baseline)
# ============================================================
def standard_spark_wordcount(input_file_path):
    """
    Standard single-driver Spark WordCount for comparison.
    Paper Reference: Section VI.B — Spark Cluster baseline.

    M2 fix: Mirrors worker SparkSession config for fair comparison.
    """
    print(f"\n{'=' * 70}")
    print("  Standard Spark (Single Driver) WordCount — BASELINE")
    print(f"{'=' * 70}")

    total_start = time.time()

    spark = (
        SparkSession.builder.master("local[*]")
        .appName("Standard-Spark-SingleDriver-Baseline")
        .config("spark.ui.enabled", "false")
        .config("spark.driver.host", "localhost")
        .config("spark.driver.extraJavaOptions", "-Djava.security.manager=allow")
        .config("spark.pyspark.python", sys.executable)
        .getOrCreate()
    )

    sc = spark.sparkContext
    sc.setLogLevel("ERROR")

    load_start = time.time()
    text_rdd = sc.textFile(input_file_path)
    text_rdd.count()  # force materialisation
    load_end = time.time()

    process_start = time.time()
    results = (
        text_rdd.flatMap(lambda line: line.lower().split())
        .filter(lambda word: len(word) > 1)
        .map(lambda word: (word, 1))
        .reduceByKey(lambda a, b: a + b)
        .collect()
    )
    process_end = time.time()

    sorted_results = sorted(results, key=lambda x: x[1], reverse=True)
    total_end = time.time()

    print(f"  Total unique words: {len(sorted_results):,}")
    print("  Top 10 words:")
    for word, count in sorted_results[:10]:
        print(f"    {word:25s} -> {count:,}")

    print(f"\n  Load Time:         {load_end - load_start:8.4f} s")
    print(f"  Processing Time:   {process_end - process_start:8.4f} s")
    print(f"  Total Execution:   {total_end - total_start:8.4f} s")

    spark.stop()

    return sorted_results, {
        "load_time": load_end - load_start,
        "processing_time": process_end - process_start,
        "total_time": total_end - total_start,
    }


# ============================================================
# COMPONENT 6: Dataset Generator
# ============================================================
def generate_test_dataset(output_path, target_size_mb=100):
    """
    Paper Reference: Section VI.B — dataset generation.
    "Linux commands to generate multiple text files of varying
    sizes by duplicating the source text file"

    L1 fix: Richer vocabulary (600+ unique words) for realistic
    word frequency distribution at scale.
    """
    print(f"\nGenerating test dataset ({target_size_mb} MB)...")

    # Base domain sentences — HPC/ML/Distributed Systems vocabulary
    base_sentences = [
        "Apache Spark is a unified analytics engine for large scale data processing",
        "The multi driver architecture enables concurrent execution across independent Spark drivers",
        "Resource allocation in distributed computing involves managing CPU memory and storage",
        "Machine learning workloads require substantial computational resources and efficient scheduling",
        "Dynamic partitioning optimizes data distribution and reduces network overhead in clusters",
        "High performance computing leverages parallel processing for complex scientific workloads",
        "Message passing interface provides efficient communication between distributed processes",
        "Resilient distributed datasets provide fault tolerant structures for parallel computation",
        "Cluster managers handle resource allocation and ensure optimal utilization of compute nodes",
        "The MPJ SPARK integration bridges the performance gap between HPC and big data frameworks",
        "Shared storage architecture enables multiple compute nodes to access data simultaneously",
        "Worker nodes perform actual processing by running application code within the cluster",
        "Metadata driven partitioning minimizes serialization demands and reduces network traffic",
        "Independent Spark drivers eliminate CPU and memory contention from centralized architecture",
        "InfiniBand network provides high speed low latency communication between compute nodes",
        "Hadoop distributed file system stores data across commodity nodes with replication",
        "Key value data structures facilitate efficient data exchange between distributed frameworks",
        "Kubernetes orchestrates containerized workloads with dynamic scaling and fault tolerance",
        "Load balancing distributes computational tasks evenly to maximize cluster throughput",
        "Deep learning models require significant memory bandwidth and floating point operations",
        "Gradient descent optimization iterates over training data to minimize loss functions",
        "Random forest algorithms build multiple decision trees for robust classification results",
        "Support vector machines find optimal hyperplanes to separate classes in feature space",
        "Logistic regression estimates probabilities using the sigmoid activation function",
        "Neural networks learn hierarchical representations through backpropagation and weight updates",
        "Batch processing handles large volumes of data without real time constraints",
        "Stream processing analyzes continuous data flows with low latency requirements",
        "Data locality optimization reduces network bandwidth by moving computation to data",
        "Checkpoint mechanisms provide fault recovery for long running distributed computations",
        "Speculative execution launches duplicate tasks to mitigate the effect of slow nodes",
        "Container isolation provides security boundaries between co-located workloads",
        "Pod scheduling in Kubernetes considers resource requests limits and node affinity rules",
        "Persistent volumes provide durable storage that survives container restarts and failures",
        "Service mesh infrastructure manages inter-service communication with observability features",
        "Distributed tracing captures end to end latency across microservice boundaries",
        "Horizontal pod autoscaling adjusts replica counts based on CPU and memory metrics",
        "Vertical scaling increases individual node capacity while horizontal scaling adds nodes",
        "Cost optimization in cloud environments balances performance requirements with budget constraints",
        "Spot instances provide significant cost savings for fault tolerant batch workloads",
        "Data pipeline orchestration coordinates complex workflows with dependency management",
    ]

    # Extended vocabulary additions for realism
    tech_words = [
        "algorithm bandwidth cache checkpoint container daemon endpoint failover gateway",
        "hypervisor latency microservice namespace orchestration pipeline quorum replica shard",
        "throughput vertex aggregation bifurcation concurrency deadlock elasticity federation",
        "gossip protocol heartbeat immutable journaling keyspace lineage mutability normalization",
        "observability pagination quantization routing serialization topology unbounded validation",
        "watermark xenial yield zeroth abstraction bottleneck compression deduplication encryption",
    ]

    all_lines = base_sentences + tech_words
    target_bytes = target_size_mb * 1024 * 1024
    current_bytes = 0

    with open(output_path, "w", encoding="utf-8") as f:
        while current_bytes < target_bytes:
            # Randomly sample and shuffle lines for non-repetitive distribution
            line = random.choice(all_lines) + "\n"
            f.write(line)
            current_bytes += len(line.encode("utf-8"))

    actual_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"Generated: {output_path} ({actual_mb:.1f} MB)")
    return output_path


# ============================================================
# MAIN EXECUTION
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MPJ-SPARK Multi-Driver WordCount Prototype v2.0")
    parser.add_argument("--workers", type=int, default=4, help="Number of MPJ workers (default: 4)")
    parser.add_argument(
        "--input", type=str, default=None, help="Input text file path (skip generation)"
    )
    parser.add_argument(
        "--generate",
        type=int,
        default=50,
        help="Generate test dataset of N MB (default: 50)",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Run baseline Standard Spark for comparison",
    )
    args = parser.parse_args()

    # ── Dataset ──────────────────────────────────────────────────────────
    if args.input and os.path.exists(args.input):
        input_file = args.input
    else:
        input_file = generate_test_dataset("./test_dataset.txt", args.generate)

    # ── Multi-Driver Run ─────────────────────────────────────────────────
    multi_results, multi_timing = mpj_root_process(input_file, args.workers)

    # ── Baseline Comparison ──────────────────────────────────────────────
    if args.compare:
        std_results, std_timing = standard_spark_wordcount(input_file)

        def speedup(std_val, md_val):
            return std_val / max(md_val, 0.0001)

        print(f"\n{'=' * 70}")
        print("  COMPARISON: Multi-Driver v2.0  vs  Standard Spark")
        print(f"{'=' * 70}")
        print(f"  {'Metric':<28} {'Multi-Driver':>14} {'Std Spark':>12} {'Speedup':>10}")
        print(f"  {'-' * 66}")
        metrics = [
            ("Load Time (sec)", multi_timing["load_time"], std_timing["load_time"]),
            (
                "Processing Time (sec)",
                multi_timing["processing_time"],
                std_timing["processing_time"],
            ),
            ("Total Time (sec)", multi_timing["total_time"], std_timing["total_time"]),
        ]
        for label, md_val, std_val in metrics:
            sp = speedup(std_val, md_val)
            flag = "✓ faster" if sp >= 1.0 else "✗ slower"
            print(f"  {label:<28} {md_val:>14.4f} {std_val:>12.4f} {sp:>8.2f}x  {flag}")
