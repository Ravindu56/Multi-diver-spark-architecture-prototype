# ============================================================
# MPJ-SPARK Multi-Driver Architecture Prototype
# WordCount Implementation for BScEng Research
# ============================================================
# Simulates the MPJ-SPARK paper's architecture on a single laptop
# using Python multiprocessing (simulates MPJ) + PySpark (Spark drivers)
# ============================================================

import os

os.environ["JAVA_TOOL_OPTIONS"] = "-Djava.security.manager=allow"

import time  # noqa: E402
import math  # noqa: E402
import shutil  # noqa: E402
from multiprocessing import Process, Queue  # noqa: E402
from collections import defaultdict  # noqa: E402


# ============================================================
# COMPONENT 1: MPJ-SPARK File Manager (Simulates HPC Shared Storage)
# ============================================================
class MPJSparkFileManager:
    """
    Replaces HDFS with a local shared storage file manager.
    Handles: read, partition, and write operations.
    Paper Reference: Section IV.B - MPJ-SPARK File Manager
    """

    def __init__(self, shared_storage_path="./shared_storage"):
        self.shared_storage_path = shared_storage_path
        os.makedirs(shared_storage_path, exist_ok=True)
        self.partitions_dir = os.path.join(shared_storage_path, "partitions")
        os.makedirs(self.partitions_dir, exist_ok=True)

    def read_input_file(self, input_file_path):
        """Read input file from shared storage"""
        with open(input_file_path, "r", encoding="utf-8") as f:
            content = f.read()
        file_size = os.path.getsize(input_file_path)
        return content, file_size

    def dynamic_partition(self, input_file_path, num_workers):
        """
        Dynamic Partitioning: Paper Section IV.C
        Partition size = file_size / num_workers
        One-to-one relationship between partitions and MPJ workers
        """
        content, file_size = self.read_input_file(input_file_path)
        lines = content.strip().split("\n")

        # Calculate partition size dynamically
        partition_size = math.ceil(len(lines) / num_workers)

        partition_metadata_list = []

        for i in range(num_workers):
            start_idx = i * partition_size
            end_idx = min((i + 1) * partition_size, len(lines))
            partition_lines = lines[start_idx:end_idx]

            # Write partition to shared storage
            partition_filename = f"partition_{i}.txt"
            partition_path = os.path.join(self.partitions_dir, partition_filename)
            with open(partition_path, "w", encoding="utf-8") as f:
                f.write("\n".join(partition_lines))

            # Return METADATA only (not raw data) - Key paper principle
            metadata = {
                "partition_id": i,
                "partition_path": partition_path,
                "num_lines": len(partition_lines),
                "start_line": start_idx,
                "end_line": end_idx,
            }
            partition_metadata_list.append(metadata)

        return partition_metadata_list

    def cleanup(self):
        """Clean up partition files"""
        if os.path.exists(self.partitions_dir):
            shutil.rmtree(self.partitions_dir)
            os.makedirs(self.partitions_dir, exist_ok=True)


# ============================================================
# COMPONENT 2: Key-Value Data Structure (RDD <-> MPJ Buffer)
# ============================================================
class KeyValueStructure:
    """
    Paper Reference: Section IV.D - Data Conversion
    Converts Spark RDD results into contiguous key-value arrays
    suitable for MPJ message passing (simulated via Queue).
    """

    def __init__(self):
        self.data = []  # List of (key, value) tuples

    def from_rdd_collect(self, rdd_results):
        """Convert collected RDD results to key-value structure"""
        self.data = list(rdd_results)
        return self

    def to_serializable(self):
        """Convert to serializable format for inter-process communication"""
        return [(str(k), int(v)) for k, v in self.data]

    @staticmethod
    def from_serializable(serialized_data):
        """Reconstruct from serialized format"""
        kv = KeyValueStructure()
        kv.data = [(str(k), int(v)) for k, v in serialized_data]
        return kv


# ============================================================
# COMPONENT 3: MPJ Worker Process (Each has its own Spark Driver)
# ============================================================
def mpj_worker_process(worker_id, partition_metadata, result_queue, timing_queue):
    """
    Paper Reference: Section IV.A (MPJ Workers) + Algorithm 1
    Each MPJ Worker:
    1. Receives partition metadata from Root
    2. Crea tes its OWN independent Spark driver (SparkSession)
    3. Reads data from shared storage using metadata
    4. Executes WordCount application logic
    5. Converts RDD results to KeyValue structure
    6. Sends results back to Root via Queue (simulates MPJ Send)
    """
    try:
        from pyspark.sql import SparkSession

        worker_start_time = time.time()

        # === STEP 1: Create INDEPENDENT Spark Driver for this worker ===
        # Paper: "Each MPJ Worker has its own copy of the spark driver"
        # Paper: "isolated from the other workers"
        spark = (
            SparkSession.builder.master("local[*]")
            .appName(f"MPJ-Worker-{worker_id}-SparkDriver")
            .config("spark.ui.enabled", "false")
            .config("spark.driver.host", "localhost")
            .config("spark.sql.shuffle.partitions", "2")
            .config("spark.default.parallelism", "2")
            .config("spark.driver.extraJavaOptions", "-Djava.security.manager=allow")
            .config("spark.executor.extraJavaOptions", "-Djava.security.manager=allow")
            .getOrCreate()
        )

        sc = spark.sparkContext
        sc.setLogLevel("ERROR")

        driver_init_time = time.time()

        # === STEP 2: Read partition from shared storage using metadata ===
        # Paper: "SPARK-context.readfile(Received-Metadata) from HPC shared storage"
        partition_path = partition_metadata["partition_path"]
        text_rdd = sc.textFile(partition_path)

        # === STEP 3: Execute WordCount Application Logic ===
        # flatMap -> map -> reduceByKey (standard WordCount)
        word_counts_rdd = (
            text_rdd.flatMap(lambda line: line.lower().split())
            .filter(lambda word: len(word) > 0)
            .map(lambda word: (word, 1))
            .reduceByKey(lambda a, b: a + b)
        )

        # === STEP 4: Collect results (RDD -> local) ===
        results = word_counts_rdd.collect()

        processing_time = time.time()

        # === STEP 5: Convert RDD to KeyValue data structure ===
        # Paper: "Convert RDD result to Key Value data structure"
        kv_structure = KeyValueStructure()
        kv_structure.from_rdd_collect(results)
        serialized_results = kv_structure.to_serializable()

        # === STEP 6: Send results to Root (simulates MPJ Send) ===
        # Paper: "Method Send-Results(Result, Start-Position, Count, Type, Root Process, Tag)"
        result_queue.put(
            {
                "worker_id": worker_id,
                "results": serialized_results,
                "num_words": len(serialized_results),
                "partition_lines": partition_metadata["num_lines"],
            }
        )

        worker_end_time = time.time()

        timing_queue.put(
            {
                "worker_id": worker_id,
                "driver_init_time": driver_init_time - worker_start_time,
                "processing_time": processing_time - driver_init_time,
                "total_worker_time": worker_end_time - worker_start_time,
            }
        )

        # Cleanup
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
    Root Process:
    1. Initializes MPJ-SPARK File Manager
    2. Partitions input file
    3. Distributes metadata to workers
    4. Receives and aggregates results
    5. Performs final Spark computation
    """
    print("=" * 70)
    print("  MPJ-SPARK Multi-Driver WordCount Prototype")
    print(f"  Workers: {num_workers} | Input: {input_file_path}")
    print("=" * 70)

    total_start_time = time.time()

    # === PHASE 1: File Manager - Partition Input ===
    print("\n[ROOT] Phase 1: Initializing MPJ-SPARK File Manager...")
    file_manager = MPJSparkFileManager()

    load_start = time.time()
    partition_metadata_list = file_manager.dynamic_partition(
        input_file_path, num_workers
    )
    load_end = time.time()
    load_time = load_end - load_start

    print(f"[ROOT] Input partitioned into {num_workers} partitions")
    for meta in partition_metadata_list:
        print(
            f"  Partition {meta['partition_id']}: {meta['num_lines']} lines -> {meta['partition_path']}"
        )

    # === PHASE 2: Distribute Metadata & Launch Workers ===
    print(f"\n[ROOT] Phase 2: Distributing metadata to {num_workers} MPJ Workers...")

    result_queue = Queue()
    timing_queue = Queue()
    workers = []

    process_start = time.time()

    for i in range(num_workers):
        # Paper: "Send-Metadata(PRTi.Metadata, Start-Position, Count, Type, Dest-Worker-Process, Tag)"
        p = Process(
            target=mpj_worker_process,
            args=(i, partition_metadata_list[i], result_queue, timing_queue),
        )
        workers.append(p)
        p.start()
        print(
            f"  [ROOT] Launched MPJ Worker {i} (PID: {p.pid}) with independent Spark Driver"
        )

    # === PHASE 3: Wait for all workers ===
    print(f"\n[ROOT] Phase 3: Waiting for {num_workers} workers to complete...")
    for p in workers:
        p.join()

    process_end = time.time()
    processing_time = process_end - process_start

    # === PHASE 4: Receive and Aggregate Results ===
    print("\n[ROOT] Phase 4: Receiving results from all workers...")

    # Paper: "Receive-Results-from-All-Workers(Collected-Results, ...)"
    all_results = []
    worker_timings = []

    while not result_queue.empty():
        result = result_queue.get()
        if "error" in result:
            print(f"  [ERROR] Worker {result['worker_id']}: {result['error']}")
        else:
            all_results.append(result)
            print(
                f"  [ROOT] Received {result['num_words']} unique words from Worker {result['worker_id']}"
            )

    while not timing_queue.empty():
        worker_timings.append(timing_queue.get())

    # === PHASE 5: Final Aggregation (Root Spark Driver) ===
    print("\n[ROOT] Phase 5: Final aggregation using Root Spark Driver...")

    # Paper: "SparkContext.parallelize(Result-List) -> Collect the final results"
    aggregation_start = time.time()

    # Aggregate word counts from all workers
    final_word_counts = defaultdict(int)
    for worker_result in all_results:
        kv = KeyValueStructure.from_serializable(worker_result["results"])
        for word, count in kv.data:
            final_word_counts[word] += count

    aggregation_end = time.time()

    # Sort by count (descending)
    sorted_results = sorted(final_word_counts.items(), key=lambda x: x[1], reverse=True)

    total_end_time = time.time()

    # === RESULTS ===
    print(f"\n{'=' * 70}")
    print("  RESULTS")
    print(f"{'=' * 70}")
    print(f"  Total unique words: {len(sorted_results)}")
    print(f"  Total word occurrences: {sum(v for _, v in sorted_results)}")
    print("\n  Top 20 words:")
    for word, count in sorted_results[:20]:
        print(f"    {word:20s} -> {count}")

    # === TIMING ANALYSIS ===
    print(f"\n{'=' * 70}")
    print("  TIMING ANALYSIS (Paper Metrics)")
    print(f"{'=' * 70}")
    print(f"  Load Time (T_Load):       {load_time:.4f} sec")
    print(f"  Processing Time (T_Proc):  {processing_time:.4f} sec")
    print(f"  Aggregation Time:          {aggregation_end - aggregation_start:.4f} sec")
    print(f"  Total Execution Time:      {total_end_time - total_start_time:.4f} sec")
    print(
        f"  Load %% of Total:           {(load_time / (total_end_time - total_start_time)) * 100:.1f}%%"
    )
    print(
        f"  Processing %% of Total:     {(processing_time / (total_end_time - total_start_time)) * 100:.1f}%%"
    )

    if worker_timings:
        print("\n  Per-Worker Timings:")
        for wt in sorted(worker_timings, key=lambda x: x.get("worker_id", 0)):
            if "error" not in wt:
                print(
                    f"    Worker {wt['worker_id']}: Driver Init={wt['driver_init_time']:.2f}s, "
                    f"Processing={wt['processing_time']:.2f}s, Total={wt['total_worker_time']:.2f}s"
                )

    # Cleanup
    file_manager.cleanup()

    return sorted_results, {
        "load_time": load_time,
        "processing_time": processing_time,
        "total_time": total_end_time - total_start_time,
    }


# ============================================================
# COMPONENT 5: Standard Spark WordCount (Single Driver - Baseline)
# ============================================================
def standard_spark_wordcount(input_file_path):
    """
    Standard single-driver Spark WordCount for comparison.
    Paper Reference: Section VI.B - Spark Cluster baseline
    """
    from pyspark.sql import SparkSession

    print("\n" + "=" * 70)
    print("  Standard Spark (Single Driver) WordCount - BASELINE")
    print("=" * 70)

    total_start = time.time()

    spark = (
        SparkSession.builder.master("local[*]")
        .appName("Standard-Spark-SingleDriver")
        .config("spark.ui.enabled", "false")
        .getOrCreate()
    )

    sc = spark.sparkContext
    sc.setLogLevel("ERROR")

    load_start = time.time()
    text_rdd = sc.textFile(input_file_path)
    text_rdd.count()  # Force load
    load_end = time.time()

    process_start = time.time()
    word_counts = (
        text_rdd.flatMap(lambda line: line.lower().split())
        .filter(lambda word: len(word) > 0)
        .map(lambda word: (word, 1))
        .reduceByKey(lambda a, b: a + b)
    )

    results = word_counts.collect()
    process_end = time.time()

    sorted_results = sorted(results, key=lambda x: x[1], reverse=True)

    total_end = time.time()

    print(f"  Total unique words: {len(sorted_results)}")
    print("  Top 10 words:")
    for word, count in sorted_results[:10]:
        print(f"    {word:20s} -> {count}")

    print(f"\n  Load Time:          {load_end - load_start:.4f} sec")
    print(f"  Processing Time:    {process_end - process_start:.4f} sec")
    print(f"  Total Execution:    {total_end - total_start:.4f} sec")

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
    Paper Reference: Section VI.B - Generate Testing Dataset
    "We employed Linux commands to generate multiple text files
    of varying sizes by duplicating the source text file"
    """
    print(f"\nGenerating test dataset ({target_size_mb} MB)...")

    # Sample text (will be duplicated to reach target size)
    sample_text = """The quick brown fox jumps over the lazy dog
Apache Spark is a powerful distributed computing system designed for big data analytics
High performance computing environments leverage parallel processing for complex workloads
Machine learning workloads require substantial computational resources and efficient scheduling
Resource allocation in cluster computing involves distributing tasks across available nodes
The multi driver architecture enables independent execution of Spark applications in parallel
Message passing interface provides efficient communication between distributed processes
Big data analytics has transformed how organizations process and analyze large datasets
Cloud computing offers scalable infrastructure for deploying distributed applications
The integration of HPC and big data frameworks bridges the performance gap between platforms
Dynamic partitioning optimizes data distribution and reduces network overhead in clusters
Resilient distributed datasets provide fault tolerant data structures for parallel computation
Worker nodes perform actual processing by running application code within the cluster
The driver program is the main control process responsible for starting the Spark context
Executors are processes initiated on worker nodes to execute tasks assigned by the driver
Cluster managers handle resource allocation and ensure optimal utilization of resources
Hadoop distributed file system provides distributed storage across commodity nodes
Shared storage architecture enables multiple compute nodes to access data simultaneously
Key value data structures facilitate efficient data exchange between different frameworks
The partition size affects network bandwidth during data transfer between computing nodes
"""

    target_bytes = target_size_mb * 1024 * 1024
    current_size = 0

    with open(output_path, "w", encoding="utf-8") as f:
        while current_size < target_bytes:
            f.write(sample_text)
            current_size += len(sample_text.encode("utf-8"))

    actual_size = os.path.getsize(output_path)
    print(f"Generated: {output_path} ({actual_size / (1024 * 1024):.1f} MB)")
    return output_path


# ============================================================
# MAIN EXECUTION
# ============================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="MPJ-SPARK Multi-Driver WordCount Prototype"
    )
    parser.add_argument(
        "--workers", type=int, default=4, help="Number of MPJ workers (default: 4)"
    )
    parser.add_argument("--input", type=str, default=None, help="Input text file path")
    parser.add_argument(
        "--generate",
        type=int,
        default=50,
        help="Generate test dataset of N MB (default: 50)",
    )
    parser.add_argument(
        "--compare", action="store_true", help="Run comparison with standard Spark"
    )

    args = parser.parse_args()

    # Generate or use provided input file
    if args.input and os.path.exists(args.input):
        input_file = args.input
    else:
        input_file = generate_test_dataset("./test_dataset.txt", args.generate)

    # Run Multi-Driver WordCount
    multi_results, multi_timing = mpj_root_process(input_file, args.workers)

    # Run Standard Spark WordCount for comparison
    if args.compare:
        std_results, std_timing = standard_spark_wordcount(input_file)

        print("\n" + "=" * 70)
        print("  COMPARISON: Multi-Driver vs Standard Spark")
        print("=" * 70)
        print(
            f"  {'Metric':<25} {'Multi-Driver':>15} {'Standard Spark':>15} {'Speedup':>10}"
        )
        print(f"  {'-' * 65}")
        print(
            f"  {'Load Time (sec)':<25} {multi_timing['load_time']:>15.4f} {std_timing['load_time']:>15.4f} {std_timing['load_time'] / max(multi_timing['load_time'], 0.001):>9.2f}x"
        )
        print(
            f"  {'Processing Time (sec)':<25} {multi_timing['processing_time']:>15.4f} {std_timing['processing_time']:>15.4f} {std_timing['processing_time'] / max(multi_timing['processing_time'], 0.001):>9.2f}x"
        )
        print(
            f"  {'Total Time (sec)':<25} {multi_timing['total_time']:>15.4f} {std_timing['total_time']:>15.4f} {std_timing['total_time'] / max(multi_timing['total_time'], 0.001):>9.2f}x"
        )
