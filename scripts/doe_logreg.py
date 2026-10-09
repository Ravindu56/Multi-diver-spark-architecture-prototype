#!/usr/bin/env python3
"""
doe_logreg.py -- Set B: logistic-regression training runs for the resource-demand
prediction model (Objective 2b).  Same method as doe_kmeans.py (Set A).

Runs a space-filling sample of LogReg configurations on the HOST with mpirun at ONE
fixed reference allocation (cores per driver, numeric-thread caps, heap) so that the
measured CPU/memory describes the workload, not the launch settings.  Per run it
stores per-driver CPU/RSS summaries (avg, peak, p95) and the 0.5 s series under
results/traces/.

Factors: size_mb 25/50/100/150/200, np 2/3/4 (1-3 drivers), iterations 10/30/60,
features 10/20/40.  Combinations whose estimated total memory exceeds
--mem-budget-gb (default 12) are dropped: estimate = drivers x (2.7 + 0.024 x MB per
driver) + 1 GB, fitted to the earlier logreg runs (np=4 with size >= 150 is dropped;
logreg 200 MB with 2 drivers reached 13 GB at the default allocation).

New in Set B:
  * stall watchdog -- if every driver stays under 10% CPU for --stall-s seconds (after
    a 60 s grace) the run is killed (exit code 125) and retried once; a K-Means run
    once hung for an hour this way (root spinning, drivers idle).
  * iters_logged -- the highest iteration/epoch/round number printed in the run log
    (heuristic; blank if the log prints none).  Compare with the requested iterations.

Data: numeric CSV generated per (size, features) into <data-dir>/doe/.  The header
style and label format are copied from --template (default
shared_storage/logreg_data.csv); the label is the last column.  main_mpi's default
input is prose, so the file is always passed with --input.

Preflight: `main_mpi --help` must list --app, --input and the iteration flag
(--iter-flag, default --logreg-iter) and, unless --features-flag none, the features
flag (default --logreg-features).

Usage (repo root, venv active):
    python scripts/doe_logreg.py --pybin "$(which python)" --design-only
    python scripts/doe_logreg.py --pybin "$(which python)" --prepare-only
    nohup python scripts/doe_logreg.py --pybin "$(which python)" \
        > results/doe_logreg_log.txt 2>&1 &
"""
import argparse, csv, fcntl, itertools, os, re, shutil, signal, socket, subprocess, sys, threading, time
from datetime import datetime

import numpy as np
import psutil

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import resource_benchmark as rb  # noqa: E402  (helpers: mean, pctl, fmt, parse_alloc)

SIZES = [25, 50, 100, 150, 200]
NPS   = [2, 3, 4]
ITERS = [10, 30, 60]
FEATS = [10, 20, 40]
FACTORS = [("size_mb", SIZES), ("np", NPS), ("iters", ITERS), ("features", FEATS)]
MAX_CONSEC_FAILS = 3
STALL_CPU_PCT = 10.0
STALL_GRACE_S = 60
ITER_RE = re.compile(r"(?i)\b(?:iter(?:ation)?s?|epoch|round)\D{0,3}(\d+)")

FIELDS = ["design_id", "run_id", "attempt", "timestamp", "host", "app", "size_mb", "np",
          "workers", "iters", "features", "cores", "heap_mb", "cap_threads", "rep",
          "driver_idx", "role", "pid", "wall_s", "exit_code", "n_samples",
          "cpu_avg_pct", "cpu_peak_pct", "cpu_p95_pct",
          "rss_avg_mb", "rss_peak_mb", "rss_p95_mb", "pss_avg_mb", "pss_peak_mb",
          "sys_cpu_avg_pct", "sys_mem_avg_mb", "sys_mem_peak_mb", "sys_mem_avail_min_mb",
          "cores_alloc", "heap_mb_alloc", "iters_logged", "data_mb_actual",
          "data_file", "trace_file"]


def est_total_gb(size_mb, np_):
    workers = np_ - 1
    return workers * (2.7 + 0.024 * size_mb / workers) + 1.0


def full_grid(budget_gb):
    out = []
    for size, np_, it, d in itertools.product(SIZES, NPS, ITERS, FEATS):
        if est_total_gb(size, np_) > budget_gb:
            continue
        out.append(dict(size_mb=size, np=np_, iters=it, features=d))
    return out


def norm(cfg):
    return np.array([levels.index(cfg[name]) / (len(levels) - 1) for name, levels in FACTORS])


def make_design(n, seed, budget_gb):
    """Farthest-point sampling with a per-level cap so every level is covered."""
    grid = full_grid(budget_gb)
    X = np.array([norm(c) for c in grid])
    rng = np.random.default_rng(seed)
    n = min(n, len(grid))
    caps = {name: -(-n // len(levels)) + 1 for name, levels in FACTORS}
    counts = {name: {lv: 0 for lv in levels} for name, levels in FACTORS}

    def add(i):
        for name, _ in FACTORS:
            counts[name][grid[i][name]] += 1

    def allowed(i):
        return all(counts[name][grid[i][name]] < caps[name] for name, _ in FACTORS)

    chosen = [int(rng.integers(len(grid)))]
    add(chosen[0])
    dist = np.linalg.norm(X - X[chosen[0]], axis=1)
    while len(chosen) < n:
        order = np.argsort(-dist, kind="stable")
        nxt = next((int(i) for i in order if i not in chosen and allowed(int(i))), None)
        if nxt is None:
            caps = {name: v + 1 for name, v in caps.items()}   # relax the caps and retry
            continue
        chosen.append(nxt)
        add(nxt)
        dist = np.minimum(dist, np.linalg.norm(X - X[nxt], axis=1))
    order = [int(i) for i in rng.permutation(chosen)]
    return [dict(grid[i], design_id=f"b{j + 1:02d}") for j, i in enumerate(order)]


def data_path(data_dir, size, feat):
    return os.path.abspath(os.path.join(data_dir, "doe", f"logreg_s{size}_f{feat}.csv"))


def template_info(path):
    """Header style and label format of an existing logreg CSV."""
    info = {"header": True, "label_name": "label", "label_float": False}
    try:
        with open(path) as fh:
            first = fh.readline().strip().split(",")
            second = fh.readline().strip().split(",")
    except OSError:
        return info
    try:
        [float(x) for x in first if x != ""]
        info["header"] = False
        probe = first
    except ValueError:
        info["label_name"] = first[-1]
        probe = second
    if probe and "." in probe[-1]:
        info["label_float"] = True
    return info


def ensure_data(path, size_mb, d, info, seed=11):
    target = size_mb * 1_000_000
    if os.path.isfile(path) and abs(os.path.getsize(path) - target) <= 0.03 * target:
        return False
    os.makedirs(os.path.dirname(path), exist_ok=True)
    rng = np.random.default_rng(seed + d)
    n_inf = min(8, d)
    w = rng.normal(0, 0.8, n_inf)
    fmt = ["%.15g"] * d + ["%.1f" if info["label_float"] else "%d"]
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        if info["header"]:
            fh.write(",".join([f"f{i}" for i in range(d)] + [info["label_name"]]) + "\n")
        while True:
            X = rng.normal(0, 1, (500, d))
            p = 1.0 / (1.0 + np.exp(-(X[:, :n_inf] @ w)))
            y = (rng.random(500) < p).astype(float)
            np.savetxt(fh, np.column_stack([X, y]), delimiter=",", fmt=fmt)
            fh.flush()
            if os.path.getsize(tmp) >= target:
                break
    os.replace(tmp, path)
    return True


def check_flags(pybin, need):
    try:
        res = subprocess.run([pybin, "-m", "mpj_spark.core.main_mpi", "--help"],
                             capture_output=True, text=True, timeout=180)
    except (OSError, subprocess.TimeoutExpired) as e:
        sys.exit(f"could not run main_mpi --help: {e}")
    text = res.stdout + res.stderr
    missing = [f for f in need if f not in text]
    if missing:
        print("[!] main_mpi --help does not list: " + " ".join(missing), flush=True)
        print("    relevant help lines:", flush=True)
        for line in text.splitlines():
            if any(s in line.lower() for s in ("logreg", "iter", "epoch", "feature", "input")):
                print("    " + line.strip(), flush=True)
        print("    use --iter-flag / --features-flag (or --features-flag none)", flush=True)
        sys.exit(2)


def sampler(root_pid, stop_flag, use_pss, live):
    """Process-tree sampler (as resource_benchmark.subtree_stats) that also publishes
    the latest per-rank CPU in `live` for the stall watchdog."""
    try:
        root = psutil.Process(root_pid)
    except psutil.NoSuchProcess:
        return {}, []
    stats, sys_samples, procs = {}, [], {}
    psutil.cpu_percent(interval=None)
    tick = 0
    while not stop_flag[0]:
        tick += 1
        do_pss = use_pss and tick % rb.PSS_EVERY == 1
        try:
            for i, d in enumerate(root.children(recursive=False)):
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
                            q.cpu_percent(interval=None)
                        else:
                            cpu += q.cpu_percent(interval=None)
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
                live[i] = cpu
                if do_pss:
                    rec["pss"].append(pss / 1e6)
        except psutil.NoSuchProcess:
            break
        vm = psutil.virtual_memory()
        sys_samples.append((psutil.cpu_percent(interval=None), vm.used / 1e6, vm.available / 1e6))
        time.sleep(rb.SAMPLE_INTERVAL_S)
    return stats, sys_samples


def run_cmd(cmd, env, log_path, timeout_s, use_pss, stall_s):
    t0 = time.perf_counter()
    logf = open(log_path, "w")
    logf.write("CMD: " + " ".join(cmd) + "\n")
    logf.flush()
    proc = subprocess.Popen(cmd, env=env, stdout=logf, stderr=subprocess.STDOUT,
                            start_new_session=True, text=True)
    stop, holder, live = [False], {}, {}

    def monitor():
        time.sleep(1.0)
        holder["stats"], holder["sys"] = sampler(proc.pid, stop, use_pss, live)

    th = threading.Thread(target=monitor, daemon=True)
    th.start()
    code, idle_since = None, None
    try:
        last = t0
        while proc.poll() is None:
            time.sleep(1)
            now = time.perf_counter()
            if now - last >= rb.HEARTBEAT_S:
                print(f"    ... run in progress ({now - t0:.0f}s elapsed)", flush=True)
                last = now
            drivers = [v for i, v in list(live.items()) if i != 0]
            if stall_s and drivers and now - t0 > STALL_GRACE_S and max(drivers) < STALL_CPU_PCT:
                idle_since = idle_since or now
                if now - idle_since >= stall_s:
                    print(f"    [!] drivers idle for {stall_s}s -- killing stalled run", flush=True)
                    os.killpg(proc.pid, signal.SIGKILL)
                    code = 125
                    break
            else:
                idle_since = None
            if timeout_s and now - t0 > timeout_s:
                os.killpg(proc.pid, signal.SIGKILL)
                code = 124
                break
        if code is None:
            code = proc.wait()
    except KeyboardInterrupt:
        print("\n[!] Ctrl+C -- SIGTERM to mpirun process group (kills ranks+JVMs)", flush=True)
        try:
            os.killpg(proc.pid, signal.SIGTERM)
            proc.wait(timeout=10)
        except Exception:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except Exception:
                pass
        stop[0] = True
        th.join(timeout=5)
        logf.close()
        print("[!] Aborted mid-run. Completed runs are saved; re-run to resume.", flush=True)
        sys.exit(130)
    stop[0] = True
    th.join(timeout=5)
    logf.close()
    return time.perf_counter() - t0, code, holder.get("stats", {}), holder.get("sys", [])


def parse_iters(log_path):
    best = None
    try:
        with open(log_path, errors="replace") as fh:
            for i, line in enumerate(fh):
                if i == 0 or "max_iter" in line.lower() or "max-iter" in line.lower():
                    continue
                for m in ITER_RE.finditer(line):
                    v = int(m.group(1))
                    best = v if best is None else max(best, v)
    except OSError:
        pass
    return "" if best is None else best


def read_state(path):
    done, timed_out = set(), set()
    if os.path.exists(path):
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                if row.get("exit_code") == "0":
                    done.add((row["design_id"], row["rep"]))
                elif row.get("exit_code") == "124":
                    timed_out.add(row["design_id"])
    return done, timed_out


def write_trace(path, stats, wall):
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["rank_idx", "t_s", "cpu_pct", "rss_mb"])
        for drv, rec in sorted(stats.items()):
            n = len(rec["cpu"]) or 1
            step = max(wall - 1.0, 0.0) / n
            for i, (c, r) in enumerate(zip(rec["cpu"], rec["rss"])):
                w.writerow([drv, f"{1.0 + i * step:.2f}", f"{c:.1f}", f"{r:.0f}"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=os.getcwd())
    ap.add_argument("--out", default="results/doe_logreg.csv")
    ap.add_argument("--n-configs", type=int, default=40, dest="n_configs")
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--seed", type=int, default=43)
    ap.add_argument("--cores", type=int, default=4, help="reference cores per driver")
    ap.add_argument("--heap-mb", type=int, default=6144, dest="heap_mb")
    ap.add_argument("--no-cap-threads", action="store_true", dest="no_cap")
    ap.add_argument("--mem-budget-gb", type=float, default=12.0, dest="budget")
    ap.add_argument("--timeout", type=int, default=3600)
    ap.add_argument("--stall-s", type=int, default=300, dest="stall_s",
                    help="kill a run whose drivers stay idle this long (0 disables)")
    ap.add_argument("--pss", action="store_true")
    ap.add_argument("--iter-flag", default="--logreg-iter", dest="iter_flag")
    ap.add_argument("--features-flag", default="--logreg-features", dest="features_flag")
    ap.add_argument("--template", default="shared_storage/logreg_data.csv")
    ap.add_argument("--data-dir", default="shared_storage", dest="data_dir")
    ap.add_argument("--pybin", default=sys.executable)
    ap.add_argument("--design-only", action="store_true", dest="design_only")
    ap.add_argument("--prepare-only", action="store_true", dest="prepare_only")
    a = ap.parse_args()

    design = make_design(a.n_configs, a.seed, a.budget)
    files = sorted({(c["size_mb"], c["features"]) for c in design})
    print(f"design: {len(design)} configs x {a.reps} reps = {len(design) * a.reps} runs; "
          f"{len(files)} data files ({sum(s for s, _ in files)} MB); "
          f"memory budget {a.budget} GB", flush=True)
    os.makedirs("results", exist_ok=True)
    with open("results/doe_logreg_design.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["design_id", "size_mb", "np", "iters", "features"])
        w.writeheader()
        for c in design:
            w.writerow({k: c[k] for k in w.fieldnames})
    if a.design_only:
        for c in design:
            print("  ", c, f"est. memory {est_total_gb(c['size_mb'], c['np']):.1f} GB")
        return

    info = template_info(a.template)
    print(f"data format from {a.template}: header={info['header']} label={info['label_name']} "
          f"label_float={info['label_float']}", flush=True)
    for size, d in files:
        p = data_path(a.data_dir, size, d)
        t0 = time.perf_counter()
        made = ensure_data(p, size, d, info)
        print(f"data {'generated' if made else 'ok       '} {p} ({time.perf_counter() - t0:.0f}s)", flush=True)
    if a.prepare_only:
        return

    if not a.pybin or not (os.path.isfile(a.pybin) and os.access(a.pybin, os.X_OK)):
        sys.exit(f"--pybin {a.pybin!r} is not an executable file (activate the venv)")
    if shutil.which("mpirun") is None:
        sys.exit("mpirun not found on PATH")
    need = ["--app", "--input", a.iter_flag] + ([] if a.features_flag == "none" else [a.features_flag])
    check_flags(a.pybin, need)

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    os.makedirs("results/logs", exist_ok=True)
    os.makedirs("results/traces", exist_ok=True)
    lock_f = open(a.out + ".lock", "w")
    try:
        fcntl.flock(lock_f, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        sys.exit(f"another run is already writing to {a.out}")

    done, timed_out = read_state(a.out)
    new_file = not os.path.exists(a.out) or os.path.getsize(a.out) == 0
    f = open(a.out, "a", newline="")
    w = csv.DictWriter(f, fieldnames=FIELDS)
    if new_file:
        w.writeheader()

    todo = [(c, rep) for rep in range(1, a.reps + 1) for c in design
            if (c["design_id"], str(rep)) not in done]
    print(f"[{datetime.now():%H:%M:%S}] todo={len(todo)} of {len(design) * a.reps}  "
          f"reference allocation: cores={a.cores} heap={a.heap_mb}MB "
          f"thread-cap={'off' if a.no_cap else a.cores}  stall={a.stall_s}s", flush=True)

    env = os.environ.copy()
    env["SLOTS_OVERRIDE"] = str(a.cores)
    env["SPARK_DRIVER_MEMORY"] = f"{a.heap_mb}m"
    if not a.no_cap:
        for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
            env[v] = str(a.cores)
    exports = []
    for v in ("SLOTS_OVERRIDE", "SPARK_DRIVER_MEMORY", "OMP_NUM_THREADS",
              "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        if v in env:
            exports += ["-x", v]

    elapsed, consec = [], 0
    for k_, (c, rep) in enumerate(todo, 1):
        did = c["design_id"]
        if did in timed_out:
            continue
        dfile = data_path(a.data_dir, c["size_mb"], c["features"])
        for attempt in (1, 2):
            run_id = (f"B_{did}_s{c['size_mb']}_np{c['np']}_i{c['iters']}"
                      f"_f{c['features']}_r{rep}" + ("" if attempt == 1 else "_retry"))
            log_path = f"results/logs/{run_id}.log"
            trace_path = f"results/traces/{run_id}.csv"
            cmd = (["mpirun", "--oversubscribe", "-np", str(c["np"])] + exports +
                   [a.pybin, "-m", "mpj_spark.core.main_mpi", "--app", "logreg",
                    "--input", dfile, a.iter_flag, str(c["iters"])])
            if a.features_flag != "none":
                cmd += [a.features_flag, str(c["features"])]
            print(f"[{datetime.now():%H:%M:%S}] START {k_}/{len(todo)}  {run_id}", flush=True)
            wall, code, stats, sys_s = run_cmd(cmd, env, log_path, a.timeout, a.pss, a.stall_s)
            cores_alloc, heap_alloc = rb.parse_alloc(log_path)
            iters_logged = parse_iters(log_path)
            if stats:
                write_trace(trace_path, stats, wall)
            scpu = rb.mean([s[0] for s in sys_s])
            smem = rb.mean([s[1] for s in sys_s])
            smem_peak = rb.mx([s[1] for s in sys_s])
            savail = min([s[2] for s in sys_s]) if sys_s else None
            rows = stats or {0: {"pid": "", "cpu": [], "rss": [], "pss": []}}
            for drv, rec in sorted(rows.items()):
                w.writerow({
                    "design_id": did, "run_id": run_id, "attempt": attempt,
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    "host": socket.gethostname(), "app": "logreg", "size_mb": c["size_mb"],
                    "np": c["np"], "workers": c["np"] - 1, "iters": c["iters"],
                    "features": c["features"], "cores": a.cores, "heap_mb": a.heap_mb,
                    "cap_threads": "" if a.no_cap else a.cores, "rep": rep,
                    "driver_idx": drv,
                    "role": ("root" if drv == 0 else "driver") if rec["rss"] else "",
                    "pid": rec["pid"], "wall_s": f"{wall:.2f}", "exit_code": code,
                    "n_samples": len(rec["cpu"]),
                    "cpu_avg_pct": rb.fmt(rb.mean(rec["cpu"]), 1),
                    "cpu_peak_pct": rb.fmt(rb.mx(rec["cpu"]), 1),
                    "cpu_p95_pct": rb.fmt(rb.pctl(rec["cpu"], 95), 1),
                    "rss_avg_mb": rb.fmt(rb.mean(rec["rss"])),
                    "rss_peak_mb": rb.fmt(rb.mx(rec["rss"])),
                    "rss_p95_mb": rb.fmt(rb.pctl(rec["rss"], 95)),
                    "pss_avg_mb": rb.fmt(rb.mean(rec.get("pss", []))),
                    "pss_peak_mb": rb.fmt(rb.mx(rec.get("pss", []))),
                    "sys_cpu_avg_pct": rb.fmt(scpu, 1), "sys_mem_avg_mb": rb.fmt(smem),
                    "sys_mem_peak_mb": rb.fmt(smem_peak), "sys_mem_avail_min_mb": rb.fmt(savail),
                    "cores_alloc": cores_alloc, "heap_mb_alloc": heap_alloc,
                    "iters_logged": iters_logged,
                    "data_mb_actual": f"{os.path.getsize(dfile) / 1e6:.1f}",
                    "data_file": os.path.relpath(dfile),
                    "trace_file": trace_path if stats else ""})
            f.flush()
            elapsed.append(wall)
            eta = (sum(elapsed) / len(elapsed)) * (len(todo) - k_)
            print(f"[{datetime.now():%H:%M:%S}] DONE {k_}/{len(todo)}  {run_id}  wall={wall:.1f}s "
                  f"exit={code}  iters_logged={iters_logged}  ETA<={eta / 3600:.1f}h", flush=True)
            if code == 125 and attempt == 1:
                print("    retrying once after a stall", flush=True)
                time.sleep(rb.SETTLE_S)
                continue
            break
        if code == 124:
            timed_out.add(did)
        consec = 0 if code in (0, 124, 125) else consec + 1
        if consec >= MAX_CONSEC_FAILS:
            print(f"[!] {consec} consecutive failed runs (not timeouts/stalls) -- aborting. "
                  f"Check {log_path}", flush=True)
            f.close()
            sys.exit(3)
        time.sleep(rb.SETTLE_S)
    f.close()
    print(f"ALL DONE. rows in {a.out}")


if __name__ == "__main__":
    main()
