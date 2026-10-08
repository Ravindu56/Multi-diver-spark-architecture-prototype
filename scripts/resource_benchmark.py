#!/usr/bin/env python3
"""
resource_benchmark.py -- workload-characterization run driver (Objective 2a).

Runs the mpj_spark benchmark matrix on the HOST with mpirun, sampling CPU and
memory of every MPI rank subtree every 0.5 s, and appends results to a CSV.
Resumable: re-run after an interruption and it skips (app, size, np, cores,
heap, rep) combos that already finished with exit_code 0 (failed runs retried).

MPI layout: rank 0 is the root coordinator, ranks 1..N are the Spark drivers.
np=2 -> 1 driver (single-driver baseline), np=3 -> 2 drivers, etc.
CSV rows: driver_idx 0 / role "root" is rank 0 (no Spark; it busy-polls MPI so it
shows ~100% CPU); the rest are drivers.  cpu_* are % of one core summed over the
rank's whole process subtree (python + JVM); rss_* are MB over the same subtree.
RSS counts shared pages once per process, so it can over-state real use; the
sys_mem_* columns are system-wide (what htop shows) and --pss adds PSS.

Axes: --cores N [N ...] sets cores per driver (env SLOTS_OVERRIDE) and
--heap-mb N [N ...] sets the driver heap (env SPARK_DRIVER_MEMORY); the values
Spark actually used are parsed from the run log into cores_alloc / heap_mb_alloc.
Leave them out for the default policy (total cores / drivers, auto heap).

Timeouts: --timeout S (default 1800; 0 disables it).  A timeout never aborts the
sweep.  After --max-cfg-timeouts (default 2) timeouts in one configuration the
remaining reps of that configuration are skipped, also on later resumes (timeouts
already in the CSV count).  --skip app:size:np excludes a configuration.
Only non-timeout failures (e.g. mpirun cannot start) count toward the
3-consecutive-failures abort.

Inputs: kmeans/logreg read a numeric CSV passed with --input
(<data-dir>/<app>_<size>mb.csv, default data-dir=shared_storage).  wordcount keeps
--generate <size> (its size axis is not meaningful: every size ran identically).
A preflight check verifies every numeric input exists and starts with numbers.

Safety rails: --pybin and mpirun are validated up front; a lock file stops two
sweeps writing the same CSV; a CSV from an older schema is upgraded in place when
the new columns were only appended, otherwise rotated aside.

Each run's output streams to results/logs/<run_id>.log (first line is the exact
mpirun command), with a 30 s console heartbeat.  Ctrl+C SIGTERMs the whole
mpirun process group so no orphan JVMs survive.

Usage (from repo root, venv activated):
    python scripts/resource_benchmark.py --pybin "$(which python)" --out results/resource_runs.csv
    python scripts/resource_benchmark.py --pybin "$(which python)" --apps logreg --sizes 50 \
        --np 3 --cores 1 2 3 4 8 11 --reps 3 --out results/cores_sweep.csv
    nohup python scripts/resource_benchmark.py --pybin "$(which python)" \
        --skip kmeans:200:2 --out results/resource_runs.csv > results/benchmark_log.txt 2>&1 &
"""
import argparse, csv, fcntl, itertools, os, re, shutil, signal, socket, subprocess, sys, threading, time
from datetime import datetime

import psutil

# ---------------- configuration (edit freely) ----------------
APPS     = ["wordcount", "kmeans", "logreg"]   # batch + the two iterative ML apps
SIZES    = [50, 100, 200]                      # dataset size label (MB)
NP_LIST  = [2, 3]                              # MPI ranks: np=2 -> 1 driver (baseline)
REPS     = 5
SAMPLE_INTERVAL_S = 0.5
DEFAULT_TIMEOUT_S = 1800                       # per-run cap; 0 disables
SETTLE_S          = 10                         # cooldown between runs
HEARTBEAT_S       = 30                         # console liveness print cadence
MAX_CONSEC_FAILS  = 3                          # non-timeout failures in a row -> abort
MAX_CFG_TIMEOUTS  = 2                          # timeouts per configuration -> skip the rest
PSS_EVERY         = 4                          # with --pss, sample PSS every Nth sample
NUMERIC_APPS      = ("kmeans", "logreg")
# -------------------------------------------------------------

# New columns are only ever appended so an older CSV can be upgraded in place.
FIELDS = ["run_id", "timestamp", "host", "app", "size", "np", "workers", "rep",
          "driver_idx", "role", "pid", "wall_s", "exit_code", "n_samples",
          "cpu_avg_pct", "cpu_peak_pct", "rss_avg_mb", "rss_peak_mb",
          "sys_cpu_avg_pct", "sys_mem_avg_mb",
          "cores_req", "heap_req_mb", "cores_alloc", "heap_mb_alloc",
          "cpu_p95_pct", "rss_p95_mb", "pss_avg_mb", "pss_peak_mb",
          "sys_mem_peak_mb", "sys_mem_avail_min_mb"]

ALLOC_RE = re.compile(r"\[SparkSession\].*?local\[(\d+)\].*?heap=(\d+) MB")

def mean(vals):
    return sum(vals) / len(vals) if vals else None

def mx(vals):
    return max(vals) if vals else None

def pctl(vals, q):
    if not vals:
        return None
    s = sorted(vals)
    return s[min(len(s) - 1, int(round(q / 100.0 * (len(s) - 1))))]

def fmt(x, nd=0):
    return "" if x is None else f"{x:.{nd}f}"

def input_path(app, size, data_dir):
    return os.path.abspath(os.path.join(data_dir, f"{app}_{size}mb.csv"))

def build_cmd(app, size, np_, pybin, data_dir, exports):
    base = ["mpirun", "--oversubscribe", "-np", str(np_)] + exports + [
        pybin, "-m", "mpj_spark.core.main_mpi", "--app", app]
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
    """Upgrade an older-schema CSV in place when columns were only appended;
    rotate it aside if the schemas are incompatible."""
    if not (os.path.isfile(path) and os.path.getsize(path) > 0):
        return
    with open(path, newline="") as fh:
        rows = list(csv.reader(fh))
    header = rows[0]
    if header == FIELDS:
        return
    if FIELDS[:len(header)] == header:
        pad = [""] * (len(FIELDS) - len(header))
        with open(path, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(FIELDS)
            for r in rows[1:]:
                w.writerow(r + pad)
        print(f"[i] {path}: upgraded to the new schema ({len(FIELDS) - len(header)} "
              f"new columns, blank for existing rows)", flush=True)
        return
    bak = f"{path}.old-{datetime.now():%Y%m%d_%H%M%S}"
    os.rename(path, bak)
    print(f"[!] {path} has an incompatible schema; moved to {bak}", flush=True)

def row_key(row):
    return (row.get("app", ""), row.get("size", ""), row.get("np", ""),
            row.get("cores_req", "") or "", row.get("heap_req_mb", "") or "",
            row.get("rep", ""))

def done_keys(csv_path):
    """Only runs that finished with exit_code 0 count as done (failures are retried)."""
    keys = set()
    if os.path.exists(csv_path):
        with open(csv_path, newline="") as f:
            for row in csv.DictReader(f):
                if row.get("exit_code") == "0":
                    keys.add(row_key(row))
    return keys

def prior_timeouts(csv_path):
    """Per configuration: number of distinct reps already recorded as timed out (124)."""
    seen = {}
    if os.path.exists(csv_path):
        with open(csv_path, newline="") as f:
            for row in csv.DictReader(f):
                if row.get("exit_code") == "124":
                    k = row_key(row)
                    seen.setdefault(k[:5], set()).add(k[5])
    return {cfg: len(reps) for cfg, reps in seen.items()}

def parse_alloc(log_path):
    """Cores and heap Spark actually used, from the first [SparkSession] log line."""
    try:
        with open(log_path, errors="replace") as fh:
            for line in fh:
                m = ALLOC_RE.search(line)
                if m:
                    return m.group(1), m.group(2)
    except OSError:
        pass
    return "", ""

def subtree_stats(root_pid, stop_flag, use_pss):
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
    tick = 0
    while not stop_flag[0]:
        tick += 1
        do_pss = use_pss and tick % PSS_EVERY == 1
        try:
            ranks = root.children(recursive=False)
            for i, d in enumerate(ranks):
                try:
                    members = [d] + d.children(recursive=True)
                except (psutil.NoSuchProcess, psutil.ZombieProcess):
                    continue
                cpu = rss = pss = 0.0
                for p in members:
                    q = procs.get(p.pid)
                    try:
                        if q is None:
                            procs[p.pid] = q = p
                            q.cpu_percent(interval=None)       # prime; first reading is meaningless
                        else:
                            cpu += q.cpu_percent(interval=None)   # % of one core since last sample
                        rss += q.memory_info().rss
                        if do_pss:
                            try:
                                pss += q.memory_full_info().pss
                            except (psutil.AccessDenied, AttributeError):
                                pass
                    except (psutil.NoSuchProcess, psutil.ZombieProcess):
                        procs.pop(p.pid, None)
                rec = stats.setdefault(i, {"pid": d.pid, "cpu": [], "rss": [], "pss": []})
                rec["cpu"].append(cpu)
                rec["rss"].append(rss / 1e6)
                if do_pss:
                    rec["pss"].append(pss / 1e6)
        except psutil.NoSuchProcess:
            break
        vm = psutil.virtual_memory()
        sys_samples.append((psutil.cpu_percent(interval=None), vm.used / 1e6, vm.available / 1e6))
        time.sleep(SAMPLE_INTERVAL_S)
    return stats, sys_samples

def run_once(app, size, np_, cores, heap, repo, pybin, data_dir, log_path, timeout_s, use_pss):
    env = os.environ.copy()
    if cores is not None:
        env["SLOTS_OVERRIDE"] = str(cores)
    if heap is not None:
        env["SPARK_DRIVER_MEMORY"] = f"{heap}m"
    exports = []
    for var in ("SLOTS_OVERRIDE", "SPARK_DRIVER_MEMORY"):
        if var in env:
            exports += ["-x", var]
    basic = build_cmd(app, size, np_, pybin, data_dir, exports)
    t0 = time.perf_counter()
    logf = open(log_path, "w")
    logf.write("CMD: " + " ".join(basic) + "\n")
    logf.flush()
    proc = subprocess.Popen(
        basic, cwd=repo, env=env, stdout=logf, stderr=subprocess.STDOUT,
        start_new_session=True, text=True)
    stop = [False]
    holder = {}
    def monitor():
        time.sleep(1.0)                          # let mpirun spawn its ranks
        holder["stats"], holder["sys"] = subtree_stats(proc.pid, stop, use_pss)
    th = threading.Thread(target=monitor, daemon=True)
    th.start()

    exit_code = None
    try:
        last_hb = t0
        while proc.poll() is None:
            time.sleep(1)
            now = time.perf_counter()
            if now - last_hb >= HEARTBEAT_S:
                print(f"    ... run in progress ({now - t0:.0f}s elapsed, "
                      f"log: {log_path})", flush=True)
                last_hb = now
            if timeout_s and now - t0 > timeout_s:
                os.killpg(proc.pid, signal.SIGKILL)
                exit_code = 124
                break
        if exit_code is None:
            exit_code = proc.wait()
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
    return wall, exit_code, holder.get("stats", {}), holder.get("sys", [])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=os.getcwd())
    ap.add_argument("--out", default="results/resource_runs.csv")
    ap.add_argument("--apps", nargs="+", default=APPS)
    ap.add_argument("--sizes", type=int, nargs="+", default=SIZES)
    ap.add_argument("--np", type=int, nargs="+", default=NP_LIST, dest="nps")
    ap.add_argument("--reps", type=int, default=REPS)
    ap.add_argument("--cores", type=int, nargs="+", default=[],
                    help="cores per driver (SLOTS_OVERRIDE); default = automatic")
    ap.add_argument("--heap-mb", type=int, nargs="+", default=[], dest="heap_mb",
                    help="driver heap in MB (SPARK_DRIVER_MEMORY); default = automatic")
    ap.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT_S,
                    help="per-run timeout in seconds; 0 disables it")
    ap.add_argument("--max-cfg-timeouts", type=int, default=MAX_CFG_TIMEOUTS,
                    dest="max_cfg_timeouts",
                    help="skip a configuration's remaining reps after this many timeouts")
    ap.add_argument("--skip", nargs="+", default=[], metavar="APP:SIZE:NP",
                    help="configurations to exclude, e.g. kmeans:200:2")
    ap.add_argument("--pss", action="store_true",
                    help="also sample PSS (shared pages counted once; slightly slower)")
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
    skip_cfg = set()
    for s in a.skip:
        try:
            sa, ss, sn = s.split(":")
            skip_cfg.add((sa, str(int(ss)), str(int(sn))))
        except ValueError:
            sys.exit(f"--skip expects app:size:np, got {s!r}")
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
    timeouts = prior_timeouts(a.out)

    cores_list = a.cores or [None]
    heap_list = a.heap_mb or [None]
    grid = list(itertools.product(a.apps, a.sizes, a.nps, cores_list, heap_list,
                                  range(1, a.reps + 1)))
    def key_of(g):
        app, size, np_, cores, heap, rep = g
        return (app, str(size), str(np_), "" if cores is None else str(cores),
                "" if heap is None else str(heap), str(rep))
    todo = [g for g in grid if key_of(g) not in done]
    print(f"[{datetime.now():%H:%M:%S}] total={len(grid)}  done={len(grid)-len(todo)}  "
          f"todo={len(todo)}  timeout={'off' if not a.timeout else str(a.timeout) + 's'}", flush=True)

    f = open(a.out, "a", newline="")
    w = csv.DictWriter(f, fieldnames=FIELDS)
    if new_file:
        w.writeheader()

    elapsed, consec_fail, announced = [], 0, set()
    for k, g in enumerate(todo, 1):
        app, size, np_, cores, heap, rep = g
        cfg = key_of(g)[:5]
        if (app, str(size), str(np_)) in skip_cfg:
            if cfg not in announced:
                print(f"[i] skipping {app} size={size} np={np_} (--skip)", flush=True)
                announced.add(cfg)
            continue
        if timeouts.get(cfg, 0) >= a.max_cfg_timeouts:
            if cfg not in announced:
                print(f"[i] skipping remaining reps of {app} size={size} np={np_}: "
                      f"{timeouts[cfg]} timeouts already recorded", flush=True)
                announced.add(cfg)
            continue
        tag = (f"_c{cores}" if cores is not None else "") + (f"_h{heap}" if heap is not None else "")
        run_id = f"{app}_{size}_np{np_}{tag}_r{rep}"
        log_path = f"results/logs/{run_id}.log"
        print(f"[{datetime.now():%H:%M:%S}] START {k}/{len(todo)}  {run_id}", flush=True)
        wall, code, stats, sys_s = run_once(
            app, size, np_, cores, heap, a.repo, a.pybin, a.data_dir, log_path,
            a.timeout, a.pss)
        cores_alloc, heap_alloc = parse_alloc(log_path)
        scpu = mean([s[0] for s in sys_s])
        smem = mean([s[1] for s in sys_s])
        smem_peak = mx([s[1] for s in sys_s])
        savail_min = min([s[2] for s in sys_s]) if sys_s else None
        rows = stats or {0: {"pid": "", "cpu": [], "rss": [], "pss": []}}
        for drv, rec in sorted(rows.items()):
            role = ("root" if drv == 0 else "driver") if rec["rss"] else ""
            w.writerow({
                "run_id": run_id, "timestamp": datetime.now().isoformat(timespec="seconds"),
                "host": socket.gethostname(), "app": app, "size": size, "np": np_,
                "workers": np_ - 1, "rep": rep,
                "driver_idx": drv, "role": role, "pid": rec["pid"],
                "wall_s": f"{wall:.2f}", "exit_code": code,
                "n_samples": len(rec["cpu"]),
                "cpu_avg_pct": fmt(mean(rec["cpu"]), 1),
                "cpu_peak_pct": fmt(mx(rec["cpu"]), 1),
                "rss_avg_mb": fmt(mean(rec["rss"])),
                "rss_peak_mb": fmt(mx(rec["rss"])),
                "sys_cpu_avg_pct": fmt(scpu, 1),
                "sys_mem_avg_mb": fmt(smem),
                "cores_req": "" if cores is None else cores,
                "heap_req_mb": "" if heap is None else heap,
                "cores_alloc": cores_alloc, "heap_mb_alloc": heap_alloc,
                "cpu_p95_pct": fmt(pctl(rec["cpu"], 95), 1),
                "rss_p95_mb": fmt(pctl(rec["rss"], 95)),
                "pss_avg_mb": fmt(mean(rec.get("pss", []))),
                "pss_peak_mb": fmt(mx(rec.get("pss", []))),
                "sys_mem_peak_mb": fmt(smem_peak),
                "sys_mem_avail_min_mb": fmt(savail_min)})
        f.flush()
        elapsed.append(wall)
        eta = (sum(elapsed)/len(elapsed)) * (len(todo) - k)
        print(f"[{datetime.now():%H:%M:%S}] DONE {k}/{len(todo)}  {run_id}  "
              f"wall={wall:.1f}s exit={code}  ETA<={eta/3600:.1f}h", flush=True)
        if code == 124:
            timeouts[cfg] = timeouts.get(cfg, 0) + 1
        consec_fail = 0 if code in (0, 124) else consec_fail + 1
        if consec_fail >= MAX_CONSEC_FAILS:
            print(f"[!] {consec_fail} consecutive failed runs (not timeouts) -- aborting "
                  f"sweep. Check {log_path} (first line is the exact mpirun command).", flush=True)
            f.close()
            sys.exit(3)
        time.sleep(SETTLE_S)
    f.close()
    print(f"ALL DONE. rows in {a.out}")

if __name__ == "__main__":
    main()
