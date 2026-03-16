# ============================================================
# utils/dataset_generator.py
# Synthetic dataset generator for benchmark experiments
# Paper Reference: Section VI.B — dataset generation method
# ============================================================
import os

# Domain-relevant sentences covering distributed systems vocabulary
_SAMPLE_TEXT = """The quick brown fox jumps over the lazy dog
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
Latency sensitive workloads benefit from in memory computation and fast data access patterns
Fault tolerance mechanisms ensure job completion even when individual nodes fail during execution
"""


def generate_test_dataset(output_path: str, target_size_mb: int = 50) -> str:
    """
    Generate a synthetic text dataset by repeating domain sentences.

    Replicates the paper's Linux file-duplication approach (Section VI.B).

    Parameters
    ----------
    output_path    : str — destination file path
    target_size_mb : int — desired file size in MB

    Returns
    -------
    output_path (str)
    """
    print(f'\nGenerating test dataset ({target_size_mb} MB)...')
    target_bytes  = target_size_mb * 1024 * 1024
    sample_bytes  = len(_SAMPLE_TEXT.encode('utf-8'))
    current_bytes = 0

    with open(output_path, 'w', encoding='utf-8') as fh:
        while current_bytes < target_bytes:
            fh.write(_SAMPLE_TEXT)
            current_bytes += sample_bytes

    actual_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f'Generated: {output_path} ({actual_mb:.1f} MB)')
    return output_path
