# ============================================================
# mpj_spark/utils/dev_logger.py
# Structured persistent logger for MPJ-Spark research runs.
# ============================================================
import json
import os
import socket
import uuid
from datetime import datetime, timezone

LOGS_DIR      = os.path.join(os.path.dirname(__file__), '..', '..', 'logs', 'dev')
RUNS_JSONL    = os.path.join(LOGS_DIR, 'dev_runs.jsonl')
RUNS_TXT      = os.path.join(LOGS_DIR, 'dev_runs.txt')
WORKERS_JSONL = os.path.join(LOGS_DIR, 'worker_timings.jsonl')


def _ensure_dir():
    os.makedirs(LOGS_DIR, exist_ok=True)


class DevLogger:
    """
    Structured append-only logger for MPJ-Spark dev runs.

    Console output: NONE — all logging is written silently to files
    under logs/dev/ so it does not pollute the live run output.

    Files produced:
        logs/dev/dev_runs.jsonl       — one JSON record per run_root() call
        logs/dev/dev_runs.txt         — human-readable summary table (append)
        logs/dev/worker_timings.jsonl — one record per worker per run
    """

    def __init__(self, worker_id=None):
        self.worker_id = worker_id
        self._run_id   = None          # set by log_run(); shared with log_worker_timing()

    # ------------------------------------------------------------------
    # run_root() calls this once after aggregation
    # ------------------------------------------------------------------
    def log_run(
        self,
        app='',
        num_workers=0,
        cores=0,
        load_time=0.0,
        proc_time=0.0,
        agg_time=0.0,
        total_time=0.0,
        extra: dict = None,
    ):
        _ensure_dir()
        self._run_id = str(uuid.uuid4())[:8]
        ts           = datetime.now(timezone.utc).isoformat()
        hostname     = socket.gethostname()

        record = {
            'run_id'     : self._run_id,
            'timestamp'  : ts,
            'hostname'   : hostname,
            'app'        : app,
            'num_workers': num_workers,
            'cores'      : cores,
            'load_time'  : round(load_time,  4),
            'proc_time'  : round(proc_time,  4),
            'agg_time'   : round(agg_time,   4),
            'total_time' : round(total_time, 4),
        }
        if extra:
            record.update(extra)

        # JSONL
        with open(RUNS_JSONL, 'a') as f:
            f.write(json.dumps(record) + '\n')

        # Human-readable append
        line = (
            f"{ts[:19]}  run={self._run_id}  app={app:<12}  "
            f"workers={num_workers}  cores={cores}  "
            f"load={load_time:.3f}s  proc={proc_time:.3f}s  "
            f"agg={agg_time:.4f}s  total={total_time:.3f}s\n"
        )
        with open(RUNS_TXT, 'a') as f:
            f.write(line)

    # ------------------------------------------------------------------
    # worker_process.py calls this per worker — silent file write only
    # ------------------------------------------------------------------
    def log_worker_timing(
        self,
        worker_id=0,
        init_time=0.0,
        load_time=0.0,
        proc_time=0.0,
    ):
        _ensure_dir()
        record = {
            'run_id'    : self._run_id,
            'timestamp' : datetime.now(timezone.utc).isoformat(),
            'worker_id' : worker_id,
            'init_time' : round(init_time, 4),
            'load_time' : round(load_time, 4),
            'proc_time' : round(proc_time, 4),
            'total_time': round(init_time + load_time + proc_time, 4),
        }
        with open(WORKERS_JSONL, 'a') as f:
            f.write(json.dumps(record) + '\n')

    # ------------------------------------------------------------------
    # --log-history  entry point
    # ------------------------------------------------------------------
    @staticmethod
    def print_history():
        _ensure_dir()
        if not os.path.exists(RUNS_JSONL):
            print('  [History] No run history found.')
            return

        with open(RUNS_JSONL) as f:
            records = [json.loads(line) for line in f if line.strip()]

        if not records:
            print('  [History] Log file is empty.')
            return

        SEP = '=' * 102
        HDR = f"{'RUN':<8}  {'TIMESTAMP':<19}  {'APP':<10}  {'W':>3}  {'C':>3}  "\
              f"{'LOAD':>7}  {'PROC':>7}  {'AGG':>7}  {'TOTAL':>8}"
        print(f'\n{SEP}')
        print(f'  Dev Run History  ({len(records)} records)  → {RUNS_JSONL}')
        print(SEP)
        print(f'  {HDR}')
        print(f'  {"-"*98}')
        for r in records:
            ts = r.get('timestamp', '')[:19]
            print(
                f"  {r.get('run_id','?'):<8}  {ts:<19}  "
                f"{r.get('app',''):<10}  "
                f"{r.get('num_workers',0):>3}  "
                f"{r.get('cores',0):>3}  "
                f"{r.get('load_time',0):>6.3f}s  "
                f"{r.get('proc_time',0):>6.3f}s  "
                f"{r.get('agg_time',0):>6.4f}s  "
                f"{r.get('total_time',0):>7.3f}s"
            )
        print(SEP)
