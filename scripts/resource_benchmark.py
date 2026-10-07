#!/usr/bin/env python3
"""
resource_benchmark.py -- workload-characterization run driver (Objective 2a).

Runs the mpj_spark benchmark matrix on the HOST with mpirun, sampling CPU and
memory of every MPI rank subtree every 0.5 s, and appends results to a CSV.
Resumable: re-run after an interruption and it skips (app, size, np, rep)
combos that already finished with exit_code 0 (failed runs are retried).

MPI layout: rank 0 is the root coordinator, ranks 1..N are the Spark drivers.
np=2 -> 1 driver (single-driver baseline), np=3 -> 2 drivers, etc.
CSV rows: driver_idx 0 / role "root" is rank 0 (no Spark); the rest are
drivers.  (Assumes mpirun's children appear in rank order, which holds for
local launches.)  cpu_* columns are % of one core, summed over the rank's
whole process subtree (python + JVM); rss_* are MB over the same subtree.

Inputs: kmeans/logreg read a numeric CSV passed with --input
(<data-dir>/<app>_<size>mb.csv, default data-dir=shared_storage).  The
main_mpi default input is ./test_dataset.txt (prose) and --generate produces
text, so numeric apps must always be given --input explicitly.  wordcount
keeps --generate <size>.  A preflight check verifies every input exists and
starts with numeric rows before any run begins.

Safety rails: --pybin and mpirun are validated up front; a lock file stops two
sweeps writing the same CSV; an old-schema CSV is rotated aside instead of
being appended to; the sweep aborts after 3 consecutive failed runs.

Each run's stdout/stderr streams live to results/logs/<run_id>.log (first line
is the exact mpirun command), and a console heartbeat is printed every 30 s.
Ctrl+C SIGTERMs the whole mpirun process group so no orphan JVMs survive.

Usage (from repo root, venv activated):
    python scripts/resource_benchmark.py --pybin "$(which python)" --out results/resource_runs.csv
    python scripts/resource_benchmark.py --pybin "$(which python)" --np 3 --apps kmeans --sizes 50 --reps 1
    nohup python scripts/resource_benchmark.py --pybin "$(which python)" \
        --out results/resource_runs.csv > results/benchmark_log.txt 2>&1 &
"""
import argparse, csv, fcntl, itertools, os, shutil, signal, socket, subprocess, sys, threading, time
from datetime import datetime

import psutil

# ---------------- configuration (edit freely) ----------------
APPS     = ["wordcount", "kmeans", "logreg"]   # batch + the two iterative ML apps
SIZES    = [50, 100, 200]                      # dataset size label (MB)
NP_LIST  = [2, 3]                              # MPI ranks: np=2 -> 1 driver (baseline)
REPS     = 5
SAMPLE_INTERVAL_S = 0.5
RUN_TIMEOUT_S     = 1800                       # hard cap per run
SETTLE_S          = 10                         # cooldown between runs
HEARTBEAT_S       = 30                         # console liveness print cadence
MAX_CONSEC_FAILS  = 3                          # abort sweep after this many failures in a row
NUMERIC_APPS      = ("kmeans", "logreg")
# -------------------------------------------------------------

FIELDS = ["run_id", "timestamp", "host", "app", "size", "np", "workers", "rep",
          "driver_idx", "role", "pid", "wall_s", "exit_code", "n_samples",
          "cpu_avg_pct", "cpu_peak_pct", "rss_avg_mb", "rss_peak_mb",
          "sys_cpu_avg_pct", "sys_mem_avg_mb"]

def input_path(app, size, data_dir):
    return os.path.abspath(os.path.join(data_dir, f"{app}_{size}mb.csv"))

def build_cmd(app, size, np_, pybin, data_dir):
    base = ["mpirun", "--oversubscribe", "-np", str(np_), pybin,
            "-m", "mpj_spark.core.main_mpi", "--app", app]
    if app in NUMERIC_APPS:
        return base + ["--input", input_path(app, size, data_dir)]
    return base + ["--generate", str(size)]

def check_inputs(apps, sizes, data_dir):
    """Fail fast (1 s) instead of after a 3-minute JVM start on a wrong file."""
    problems = []
    for app in apps:
        if app not in NUMERIC_APPS:
            continue
        for size in sizes:
            p = input_path(app, size, data_dir)
            if not os.path.isfile(p):
                problems.append(f"missing: {p}")
                continue
            try:
                with open(p) as fh:
                    lines = [fh.readline() for _ in range(3)]
                rows = [l for l in lines if l.strip()]
                try:
                    [float(x) for x in rows[0].strip().split(",") if x.strip()]
                    probe = rows[0]
                except ValueError:
                    probe = rows[1]          # first line is a header
                [float(x) for x in probe.strip().split(",") if x.strip()]
            except (ValueError, IndexError):
                problems.append(f"not numeric CSV: {p}")
    if problems:
        print("[!] input preflight failed:", flush=True)
        for pr in problems:
            print("    " + pr, flush=True)
        print("    Stage one numeric file per size, e.g. (verify flags with "
              "`python scripts/generate_datasets.py --help`):", flush=True)
        print("      for s in " + " ".join(str(s) for s in sizes) + "; do\n"
              "        MPJ_KMEANS_DATA=$PWD/" + data_dir + "/kmeans_${s}mb.csv "
              "python scripts/generate_datasets.py --kmeans-only --size-mb $s\n"
              "        MPJ_LOGREG_DATA=$PWD/" + data_dir + "/logreg_${s}mb.csv "
              "python scripts/generate_datasets.py --logreg-only --size-mb $s\n"
              "      done", flush=True)
        sys.exit(2)

def prepare_csv(path):
    """Rotate an old-schema CSV aside so rows never shift under a stale header."""
    if os.path.isfile(path) and os.path.getsize(path) > 0:
        with open(path) as fh:
            header = fh.readline().strip()
        if header != ",".join(FIELDS):
            bak = f"{path}.old-{datetime.now():%Y%m%d_%H%M%S}"
            os.rename(path, bak)
            print(f"[!] {path} has a different schema; moved to {bak}", flush=True)

def done_keys(csv_path):
    """Only runs that finished with exit_code 0 count as done (failures are retried)."""
    keys = set()
    if os.path.exists(csv_path):
        with open(csv_path, newline="") as f:
            for row in csv.DictReader(f):
                if row.get("exit_code") == "0":
                    keys.add((row["app"], row["size"], row["np"], row["rep"]))
    return keys

def subtree_stats(root_pid, stop_flag):
    """Sample mpirun's child subtrees (top-level child => one MPI rank).

    psutil.Process objects are cached across samples: cpu_percent(interval=None)
    measures the change since that object's previous call, so a fresh object
    always reports 0.0.
    """
    try:
        root = psutil.Process(root_pid)
    except psutil.NoSuchProcess:
        return {}, []
    stats, sys_samples = {}, []
    procs = {}                                   # pid -> Process kept across samples
    psutil.cpu_percent(interval=None)            # prime system-wide counter
    while not stop_flag[0]:
        try:
            ranks = root.children(recursive=False)
            for i, d in enumerate(ranks):
                try:
                    members = [d] + d.children(recursive=True)
                except (psutil.NoSuchProcess, psutil.ZombieProcess):
                    continue
                cpu = rss = 0.0
                for p in members:
                    q = procs.get(p.pid)
                    try:
                        if q is None:
                            procs[p.pid] = q = p
                            q.cpu_percent(interval=None)       # prime; first reading is meaningless
                        else:
                            cpu += q.cpu_percent(interval=None)   # % of one core since last sample
                        rss += q.memory_info().rss
                    except (psutil.NoSuchProcess, psutil.ZombieProcess):
                        procs.pop(p.pid, None)
                rec = stats.setdefault(i, {"pid": d.pid, "cpu": [], "rss": []})
                rec["cpu"].append(cpu)
                rec["rss"].append(rss / 1e6)
        except psutil.NoSuchProcess:
            break
        sys_samples.append((psutil.cpu_percent(interval=None),
                            psutil.virtual_memory().used / 1e6))
        time.sleep(SAMPLE_INTERVAL_S)
    return stats, sys_samples

def run_once(app, size, np_, rep, repo, pybin, data_dir, log_path):
    basic = build_cmd(app, size, np_, pybin, data_dir)
    t0 = time.perf_counter()
    logf = open(log_path, "w")
    logf.write("CMD: " + " ".join(basic) + "\n")
    logf.flush()
    proc = subprocess.Popen(
        basic, cwd=repo, stdout=logf, stderr=subprocess.STDOUT,
        start_new_session=True, text=True)
    stop = [False]
    holder = {}
    def monitor():
        time.sleep(1.0)                          # let mpirun spawn its ranks
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
    ap.add_argument("--data-dir", default="shared_storage",
                    help="dir holding <app>_<size>mb.csv for kmeans/logreg")
    a = ap.parse_args()

    if any(n < 2 for n in a.nps):
        sys.exit("np must be >= 2 (rank 0 is the root coordinator; np=2 is the "
                 "single-driver baseline)")
    if not a.pybin or not (os.path.isfile(a.pybin) and os.access(a.pybin, os.X_OK)):
        sys.exit(f"--pybin {a.pybin!r} is not an executable file. Activate the venv "
                 f"(source .venv/bin/activate) so that $(which python) prints a path.")
    if shutil.which("mpirun") is None:
        sys.exit("mpirun not found on PATH")
    check_inputs(a.apps, a.sizes, a.data_dir)

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    os.makedirs("results/logs", exist_ok=True)

    lock_f = open(a.out + ".lock", "w")
    try:
        fcntl.flock(lock_f, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        sys.exit(f"another resource_benchmark.py is already writing to {a.out} "
                 f"(check: pgrep -af resource_benchmark)")

    prepare_csv(a.out)
    new_file = not os.path.exists(a.out)
    done = done_keys(a.out)

    grid = list(itertools.product(a.apps, a.sizes, a.nps, range(1, a.reps + 1)))
    todo = [g for g in grid if (g[0], str(g[1]), str(g[2]), str(g[3])) not in done]
    print(f"[{datetime.now():%H:%M:%S}] total={len(grid)}  done={len(grid)-len(todo)}  todo={len(todo)}", flush=True)

    f = open(a.out, "a", newline="")
    w = csv.DictWriter(f, fieldnames=FIELDS)
    if new_file:
        w.writeheader()

    elapsed, consec_fail = [], 0
    for k, (app, size, np_, rep) in enumerate(todo, 1):
        run_id = f"{app}_{size}_np{np_}_r{rep}"
        log_path = f"results/logs/{run_id}.log"
        print(f"[{datetime.now():%H:%M:%S}] START {k}/{len(todo)}  {run_id}", flush=True)
        wall, code, stats, sys_s, out = run_once(
            app, size, np_, rep, a.repo, a.pybin, a.data_dir, log_path)
        scpu = (sum(s[0] for s in sys_s) / len(sys_s)) if sys_s else ""
        smem = (sum(s[1] for s in sys_s) / len(sys_s)) if sys_s else ""
        rows = stats or {0: {"pid": "", "cpu": [], "rss": []}}
        for drv, rec in sorted(rows.items()):
            n = len(rec["cpu"]) or 1
            role = ("root" if drv == 0 else "driver") if rec["rss"] else ""
            w.writerow({
                "run_id": run_id, "timestamp": datetime.now().isoformat(timespec="seconds"),
                "host": socket.gethostname(), "app": app, "size": size, "np": np_,
                "workers": np_ - 1, "rep": rep,
                "driver_idx": drv, "role": role, "pid": rec["pid"],
                "wall_s": f"{wall:.2f}", "exit_code": code,
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
        consec_fail = consec_fail + 1 if code != 0 else 0
        if consec_fail >= MAX_CONSEC_FAILS:
            print(f"[!] {consec_fail} consecutive failed runs -- aborting sweep. "
                  f"Check {log_path} (first line is the exact mpirun command).", flush=True)
            f.close()
            sys.exit(3)
        time.sleep(SETTLE_S)
    f.close()
    print(f"ALL DONE. rows in {a.out}")

if __name__ == "__main__":
    main()
