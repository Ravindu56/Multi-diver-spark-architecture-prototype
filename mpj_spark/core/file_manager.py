# ============================================================
# core/file_manager.py
# MPJ-SPARK File Manager — simulates HPC shared storage
# Paper Reference: Section IV.B
# ============================================================
import os
import math
import shutil
from mpj_spark.config import SHARED_STORAGE_PATH, PARTITIONS_DIR


class MPJSparkFileManager:
    """
    Replaces HDFS with a local shared-storage file manager.
    Responsibilities: read input, dynamic partition, write partitions,
    return metadata only (key paper principle — Section IV.C).
    """

    def __init__(self, shared_storage_path: str = SHARED_STORAGE_PATH):
        self.shared_storage_path = shared_storage_path
        self.partitions_dir = os.path.join(shared_storage_path, 'partitions')
        os.makedirs(self.partitions_dir, exist_ok=True)

    # ----------------------------------------------------------
    def read_input_file(self, input_file_path: str):
        """Read full input file; return (content, file_size_bytes)."""
        with open(input_file_path, 'r', encoding='utf-8') as fh:
            content = fh.read()
        return content, os.path.getsize(input_file_path)

    # ----------------------------------------------------------
    def dynamic_partition(self, input_file_path: str, num_workers: int) -> list:
        """
        Dynamic Partitioning (Paper §IV.C):
          partition_size = ceil(total_lines / num_workers)
          One-to-one: each partition  →  one MPJ worker.
        Returns a list of partition metadata dicts (NOT raw data).
        """
        content, _ = self.read_input_file(input_file_path)
        lines = content.strip().split('\n')
        partition_size = math.ceil(len(lines) / num_workers)

        metadata_list = []
        for i in range(num_workers):
            start = i * partition_size
            end   = min((i + 1) * partition_size, len(lines))
            chunk = lines[start:end]

            part_path = os.path.join(self.partitions_dir, f'partition_{i}.txt')
            with open(part_path, 'w', encoding='utf-8') as fh:
                fh.write('\n'.join(chunk))

            metadata_list.append({
                'partition_id':  i,
                'partition_path': part_path,
                'num_lines':      len(chunk),
                'start_line':     start,
                'end_line':       end,
            })

        return metadata_list

    # ----------------------------------------------------------
    def cleanup(self):
        """Remove all partition files from shared storage."""
        if os.path.exists(self.partitions_dir):
            shutil.rmtree(self.partitions_dir)
        os.makedirs(self.partitions_dir, exist_ok=True)
