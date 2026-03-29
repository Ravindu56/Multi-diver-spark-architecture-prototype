# mpj_spark/utils/dev_logger.py
# Stub DevLogger for feature/ml-kmeans-workload branch
import json
import os
import time


class DevLogger:
    """Minimal logger stub — logs run metadata to console and optional JSONL file."""

    def __init__(self, worker_id=None):
        self.worker_id = worker_id

    def log_run(self, app='', num_workers=0, cores=0,
                load_time=0.0, proc_time=0.0, agg_time=0.0, total_time=0.0):
        print(f"[DevLogger] Run logged: app={app} workers={num_workers} "
              f"load={load_time:.3f}s proc={proc_time:.3f}s total={total_time:.3f}s")

    def log_worker_timing(self, worker_id=0, init_time=0.0,
                          load_time=0.0, proc_time=0.0):
        print(f"[DevLogger] Worker {worker_id}: "
              f"init={init_time:.3f}s load={load_time:.3f}s proc={proc_time:.3f}s")

    @staticmethod
    def print_history():
        print("[DevLogger] No history log found.")
