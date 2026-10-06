#!/usr/bin/env python3
"""
resource_benchmark.py -- workload-characterization run driver (Objective 2a).

Runs the mpj_spark benchmark matrix (~90 runs by default) on the HOST with
mpirun, sampling CPU and memory of every Spark driver (MPI rank subtree)
every 0.5 s, and appends results to a CSV. Resumable: re-run the script
after an interruption and it skips (app, size, np, rep) combos already
present in the CSV.

Each MPI rank spawns a PySpark driver which in turn spawns a Java (JVM)
gateway process, so each rank is monitored as a process SUBTREE -- this is
important, because the JVM frequently holds more memory than the python
process itself.

Each run's stdout/stderr streams live to results/logs/<run_id>.log, and a
console heartbeat is printed every 30 s while a run is in progress (a
silent capture looked like a hang; see P3-12 lesson in run_sync_benchmark).
Ctrl+C SIGTERMs the whole mpirun process group so no orphan JVMs survive;
completed runs stay in the CSV and the next invocation resumes.

Usage (from repo root, venv activated):
    python resource_benchmark.py --out results/resource_runs.csv
    python resource_benchmark.py --np 3 --apps kmeans --reps 1      # quick check
    nohup python resource_benchmark.py --out results/resource_runs.csv \
        > results/benchmark_log.txt 2>&1 &                          # overnight
"""
import argparse, csv, itertools, os, signal, socket, subprocess, sys, threading, time
from datetime import datetime

import psutil

# ---------------- configuration (edit freely) ----------------
APPS     = ["wordcount", "kmeans", "logreg"]   # batch + the two iterative ML apps
SIZES    = [50, 100, 200]                      # passed to --generate
NP_LIST  = [1, 3]                              # np=1 -> single-driver baseline (Obj 2d-i)
REPS     = 5
# 3 apps x 3 sizes x 2 np x 5 reps = 90 runs (+>~10 preflight runs = ~100)
SAMPLE_INTERVAL_S = 0.5
RUN_TIMEOUT_S     = 1800                       # hard cap per run; OOM-hung runs get killed
SETTLE_S          = 10                         # cooldown between runs (GC, JVM teardown)
HEARTBEAT_S       = 30                         # console liveness print cadence
# -------------------------------------------------------------

FIELDS = ["run_id", "timestamp", "host", "app", "size", "np", "rep",
          "driver_idx", "pid", "wall_s", "exit_code", "n_samples",
          "cpu_avg_pct", "cpu_peak_pct", "rss_avg_mb", "rss_peak_mb",
          "sys_cpu_avg_pct", "sys_mem_avg_mb"]

def done_keys(csv_path):
    keys = set()
    if os.path.exists(csv_path):
        with open(csv_path, newline="") as f:
            for row in csv.DictReader(f):
                keys.add((row["app"], row["size"], row["np"], row["rep"]))
    return keys

def subtree_stats(root_pid, stop_flag):
    """Sample mpirun's child subtrees (top-level child => one driver rank)."""
    try:
        root = psutil.Process(root_pid)
    except psutil.NoSuchProcess:
        return {}, []
    stats, sys_samples = {}, []
    while not stop_flag[0]:
        try:
            drivers = root.children(recursive=False)
            for i, d in enumerate(drivers):
                try:
                    members = [d] + d.children(recursive=True)
                except (psutil.NoSuchProcess, psutil.ZombieProcess):
                    continue
                cpu = rss = 0.0
                for p in members:
                    try:
                        cpu += p.cpu_percent(interval=None)   # % of one core
                        rss += p.memory_info().rss
                    except (psutil.NoSuchProcess, psutil.ZombieProcess):
                        pass
                rec = stats.setdefault(i, {"pid": d.pid, "cpu": [], "rss": []})
                rec["cpu"].append(cpu)
                rec["rss"].append(rss / 1e6)
        except psutil.NoSuchProcess:
            break
        sys_samples.append((psutil.cpu_percent(interval=None),
                            psutil.virtual_memory().used / 1e6))
        time.sleep(SAMPLE_INTERVAL_S)
    return stats, sys_samples

def run_once(app, size, np_, rep, repo, pybin, log_path):
    basic = ["mpirun", "--oversubscribe", "-np", str(np_), pybin,
             "-m", "mpj_spark.core.main_mpi", "--app", app, "--generate", str(size)]
    t0 = time.perf_counter()
    logf = open(log_path, "w")
    proc = subprocess.Popen(
        basic, cwd=repo, stdout=logf, stderr=subprocess.STDOUT,
        start_new_session=True, text=True)
    stop = [False]
    holder = {}
    def monitor():
        # let processes start, then prime cpu_percent
        time.sleep(2.0)
        try:
            for p in psutil.Process(proc.pid).children(recursive=True):
                try: p.cpu_percent(interval=None)
                except psutil.NoSuchProcess: pass
        except psutil.NoSuchProcess:
            pass
        holder["stats"], holder["sys"] = subtree_stats(proc.pid, stop)
    th = threading.Thread(target=monitor, daemon=True)
    th.start()

    exit_code, out = None, ""
    try:
        last_hb = t0
        while proc.poll() is None:
            time.sleep(1)
            now = time.perf_counter()
            if now - last_hb >= HEARTBEAT_S:
                print(f"    ... run in progress ({now - t0:.0f}s elapsed, "
                      f"log: {log_path})", flush=True)
                last_hb = now
            if now - t0 > RUN_TIMEOUT_S:
                os.killpg(proc.pid, signal.SIGKILL)
                exit_code, out = 124, "TIMEOUT_KILLED"
                break
        if exit_code is None:
            exit_code = proc.wait()
            out = f"(output captured in {log_path})"
    except KeyboardInterrupt:
        print("\n[!] Ctrl+C -- SIGTERM to mpirun process group (kills ranks+JVMs)", flush=True)
        try:
            os.killpg(proc.pid, signal.SIGTERM)
            proc.wait(timeout=10)
        except Exception:
            try: os.killpg(proc.pid, signal.SIGKILL)
            except Exception: pass
        stop[0] = True
        th.join(timeout=5)
        logf.close()
        print(f"[!] Aborted mid-run {(time.perf_counter() - t0):.0f}s in. "
              f"Completed runs are saved; re-run to resume.", flush=True)
        sys.exit(130)

    stop[0] = True
    th.join(timeout=5)
    logf.close()
    wall = time.perf_counter() - t0
    return wall, exit_code, holder.get("stats", {}), holder.get("sys", []), out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=os.getcwd())
    ap.add_argument("--out", default="results/resource_runs.csv")
    ap.add_argument("--apps", nargs="+", default=APPS)
    ap.add_argument("--sizes", type=int, nargs="+", default=SIZES)
    ap.add_argument("--np", type=int, nargs="+", default=NP_LIST, dest="nps")
    ap.add_argument("--reps", type=int, default=REPS)
    ap.add_argument("--pybin", default=sys.executable)
    a = ap.parse_args()

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    os.makedirs("results/logs", exist_ok=True)
    new_file = not os.path.exists(a.out)
    done = done_keys(a.out)

    grid = list(itertools.product(a.apps, a.sizes, a.nps, range(1, a.reps + 1)))
    todo = [g for g in grid if (g[0], str(g[1]), str(g[2]), str(g[3])) not in done]
    print(f"[{datetime.now():%H:%M:%S}] total={len(grid)}  done={len(grid)-len(todo)}  todo={len(todo)}", flush=True)

    f = open(a.out, "a", newline="")
    w = csv.DictWriter(f, fieldnames=FIELDS)
    if new_file:
        w.writeheader()

    elapsed = []
    for k, (app, size, np_, rep) in enumerate(todo, 1):
        run_id = f"{app}_{size}_np{np_}_r{rep}"
        log_path = f"results/logs/{run_id}.log"
        print(f"[{datetime.now():%H:%M:%S}] START {k}/{len(todo)}  {run_id}", flush=True)
        wall, code, stats, sys_s, out = run_once(app, size, np_, rep, a.repo, a.pybin, log_path)
        scpu = (sum(s[0] for s in sys_s) / len(sys_s)) if sys_s else ""
        smem = (sum(s[1] for s in sys_s) / len(sys_s)) if sys_s else ""
        rows = stats or {0: {"pid": "", "cpu": [], "rss": []}}
        for drv, rec in sorted(rows.items()):
            n = len(rec["cpu"]) or 1
            w.writerow({
                "run_id": run_id, "timestamp": datetime.now().isoformat(timespec="seconds"),
                "host": socket.gethostname(), "app": app, "size": size, "np": np_, "rep": rep,
                "driver_idx": drv, "pid": rec["pid"], "wall_s": f"{wall:.2f}", "exit_code": code,
                "n_samples": n if rec["cpu"] else 0,
                "cpu_avg_pct": f"{sum(rec['cpu'])/n:.1f}" if rec["cpu"] else "",
                "cpu_peak_pct": f"{max(rec['cpu']):.1f}" if rec["cpu"] else "",
                "rss_avg_mb": f"{sum(rec['rss'])/n:.0f}" if rec["rss"] else "",
                "rss_peak_mb": f"{max(rec['rss']):.0f}" if rec["rss"] else "",
                "sys_cpu_avg_pct": f"{scpu:.1f}" if scpu != "" else "",
                "sys_mem_avg_mb": f"{smem:.0f}" if smem != "" else ""})
        f.flush()
        elapsed.append(wall)
        eta = (sum(elapsed)/len(elapsed)) * (len(todo) - k)
        print(f"[{datetime.now():%H:%M:%S}] DONE {k}/{len(todo)}  {run_id}  "
              f"wall={wall:.1f}s exit={code}  ETA={eta/3600:.1f}h", flush=True)
        time.sleep(SETTLE_S)
    f.close()
    print(f"ALL DONE. rows in {a.out}")

if __name__ == "__main__":
    main()
