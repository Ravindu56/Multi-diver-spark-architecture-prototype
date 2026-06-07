# ============================================================
# benchmarks/dev_logger.py
# Dev Run Logger — persistent structured logging for test runs
# ============================================================
# Every call to log_run() appends one JSON record to:
#   logs/dev/dev_runs.jsonl          ← machine-readable, one record per line
#   logs/dev/dev_runs.txt            ← human-readable, mirrors console output
#
# Log file is never overwritten — every run is appended so the
# full experiment history is preserved for your paper's results.
#
# Usage (called automatically from main.py after every run):
#   from mpj_spark.benchmarks.dev_logger import DevLogger
#   logger = DevLogger()                     # uses default logs/dev/
#   logger.log_run(args, multi_timing, std_timing)   # std_timing=None if no --compare
#
# Read all past runs:
#   import json
#   with open('logs/dev/dev_runs.jsonl') as f:
#       runs = [json.loads(line) for line in f]
# ============================================================
import os
import json
import datetime
import platform
import multiprocessing

LOG_DIR = os.path.join("logs", "dev")
JSONL_FILE = os.path.join(LOG_DIR, "dev_runs.jsonl")
TEXT_FILE = os.path.join(LOG_DIR, "dev_runs.txt")
DIVIDER = "=" * 70


class DevLogger:
    """
    Appends one structured record per test run to logs/dev/.

    Parameters
    ----------
    log_dir : str — directory for log files (default: logs/dev/)
    """

    def __init__(self, log_dir: str = LOG_DIR):
        self.log_dir = log_dir
        self.jsonl_path = os.path.join(log_dir, "dev_runs.jsonl")
        self.text_path = os.path.join(log_dir, "dev_runs.txt")
        os.makedirs(log_dir, exist_ok=True)

    # ----------------------------------------------------------
    def log_run(
        self, run_config: dict, multi_timing: dict, std_timing: dict = None
    ) -> str:
        """
        Record one complete test run.

        Parameters
        ----------
        run_config   : dict — CLI args / run configuration
                        Expected keys: workers, generate, app, prewarm,
                        cores_per_entity, input_file
        multi_timing : dict — timing summary from mpj_root_process()
                        Keys: load_time, processing_time, total_time,
                              parallel_time, jvm_init_time, avg_init_time
        std_timing   : dict | None — baseline timing from run_baseline();
                        None if --compare was not used

        Returns
        -------
        run_id : str — unique ID for this run (timestamp-based)
        """
        ts = datetime.datetime.now()
        run_id = ts.strftime("%Y%m%d_%H%M%S")

        # ── Build structured record ───────────────────────────────────────
        record = {
            "run_id": run_id,
            "timestamp": ts.isoformat(),
            "machine": {
                "hostname": platform.node(),
                "os": platform.system(),
                "total_cores": multiprocessing.cpu_count(),
            },
            "config": {
                "workers": run_config.get("workers"),
                "dataset_mb": run_config.get("generate"),
                "input_file": run_config.get("input_file"),
                "app": run_config.get("app", "wordcount"),
                "prewarm": run_config.get("prewarm", True),
                "cores_per_entity": run_config.get("cores_per_entity"),
                "jvm_mode": "pre-warmed"
                if run_config.get("prewarm", True)
                else "cold-start",
            },
            "multi_driver": {
                "load_time": round(multi_timing.get("load_time", 0.0), 4),
                "avg_proc_time": round(multi_timing.get("processing_time", 0.0), 4),
                "total_time": round(multi_timing.get("total_time", 0.0), 4),
                "parallel_time": round(multi_timing.get("parallel_time", 0.0), 4),
                "jvm_init_time": round(multi_timing.get("jvm_init_time", 0.0), 4),
                "avg_init_time": round(multi_timing.get("avg_init_time", 0.0), 4),
            },
            "std_spark": None,
            "speedup": None,
        }

        if std_timing:
            record["std_spark"] = {
                "load_time": round(std_timing.get("load_time", 0.0), 4),
                "proc_time": round(std_timing.get("processing_time", 0.0), 4),
                "total_time": round(std_timing.get("total_time", 0.0), 4),
            }

            # Compute speedup ratios — safe division
            def sp(std_val, multi_val):
                return round(std_val / max(multi_val, 0.0001), 4)

            record["speedup"] = {
                "load": sp(std_timing["load_time"], multi_timing["load_time"]),
                "processing": sp(
                    std_timing["processing_time"], multi_timing["processing_time"]
                ),
                "total": sp(std_timing["total_time"], multi_timing["total_time"]),
            }

        # ── Append to JSONL (machine-readable) ──────────────────────────
        with open(self.jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

        # ── Append to TXT (human-readable) ────────────────────────────
        with open(self.text_path, "a", encoding="utf-8") as f:
            f.write(f"\n{DIVIDER}\n")
            f.write(f"  RUN ID  : {run_id}\n")
            f.write(f"  Time    : {ts.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(
                f"  Host    : {record['machine']['hostname']} "
                f"({record['machine']['total_cores']} cores, "
                f"{record['machine']['os']})\n"
            )
            f.write(f"{DIVIDER}\n")
            f.write("  CONFIG\n")
            f.write(f"    Workers        : {record['config']['workers']}\n")
            f.write(f"    Dataset        : {record['config']['dataset_mb']} MB\n")
            f.write(f"    App            : {record['config']['app']}\n")
            f.write(f"    JVM Mode       : {record['config']['jvm_mode']}\n")
            f.write(f"    Cores/entity   : {record['config']['cores_per_entity']}\n")
            f.write(f"{DIVIDER}\n")
            f.write("  MULTI-DRIVER TIMING\n")
            md = record["multi_driver"]
            f.write(f"    T_Load         : {md['load_time']:>8.4f} s\n")
            f.write(f"    T_Proc (avg)   : {md['avg_proc_time']:>8.4f} s\n")
            f.write(f"    T_Init (avg)   : {md['avg_init_time']:>8.4f} s\n")
            f.write(f"    T_Parallel     : {md['parallel_time']:>8.4f} s\n")
            f.write(f"    T_Total        : {md['total_time']:>8.4f} s\n")
            if record["std_spark"]:
                ss = record["std_spark"]
                sp = record["speedup"]
                f.write(f"{DIVIDER}\n")
                f.write("  BASELINE (Std Spark)\n")
                f.write(f"    T_Load         : {ss['load_time']:>8.4f} s\n")
                f.write(f"    T_Proc         : {ss['proc_time']:>8.4f} s\n")
                f.write(f"    T_Total        : {ss['total_time']:>8.4f} s\n")
                f.write(f"{DIVIDER}\n")
                f.write("  SPEEDUP\n")
                f.write(f"    Load           : {sp['load']:>6.2f}x\n")
                f.write(f"    Processing     : {sp['processing']:>6.2f}x\n")
                f.write(f"    Total          : {sp['total']:>6.2f}x\n")
            f.write(f"{DIVIDER}\n")

        return run_id

    # ----------------------------------------------------------
    def list_runs(self) -> list:
        """
        Return all past run records as a list of dicts.
        Returns empty list if no log file exists yet.
        """
        if not os.path.exists(self.jsonl_path):
            return []
        with open(self.jsonl_path, "r", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]

    # ----------------------------------------------------------
    def print_summary_table(self):
        """
        Print a compact summary table of all past runs to console.
        Useful for quick comparison across experiments.
        """
        runs = self.list_runs()
        if not runs:
            print("[DevLogger] No runs recorded yet.")
            return

        print(f"\n{DIVIDER}")
        print(f"  DEV RUN HISTORY  ({len(runs)} runs)  —  {self.jsonl_path}")
        print(f"{DIVIDER}")
        header = (
            f"  {'Run ID':<18} {'W':>3} {'MB':>5} {'Cores':>6} "
            f"{'Mode':<12} {'T_Proc':>8} {'Std_Proc':>9} "
            f"{'Speedup':>8} {'T_Total':>8}"
        )
        print(header)
        print(f"  {'-' * 68}")
        for r in runs:
            cfg = r["config"]
            md = r["multi_driver"]
            sp = r.get("speedup") or {}
            ss = r.get("std_spark") or {}
            print(
                f"  {r['run_id']:<18} "
                f"{cfg.get('workers', '?'):>3} "
                f"{cfg.get('dataset_mb', '?'):>5} "
                f"{str(cfg.get('cores_per_entity', 'auto')):>6} "
                f"{cfg.get('jvm_mode', '?'):<12} "
                f"{md['avg_proc_time']:>8.2f}s "
                f"{ss.get('proc_time', 0.0):>8.2f}s "
                f"{sp.get('processing', 0.0):>8.2f}x "
                f"{md['total_time']:>8.2f}s"
            )
        print(f"{DIVIDER}\n")
