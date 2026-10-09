#!/usr/bin/env python3
"""
doe_kmeans.py -- Set A: K-Means training runs for the resource-demand
prediction model (Objective 2b).

Runs a space-filling sample of K-Means configurations on the HOST with mpirun at
ONE fixed reference allocation (cores per driver, thread caps, heap), so the
measured CPU/memory is a property of the workload and not of how it was launched.
For every run it stores per-driver CPU/RSS summaries (avg, peak, p95) and the
full 0.5 s time series under results/traces/ (for an optional LSTM).

Factors (levels): size_mb 25/50/100/150/200, np 2/3/4 (1-3 drivers),
iterations 10/20/40, k 3/5/10, features 8/16/32.  np=2 with size>=150 is excluded
(a single driver timed out at 200 MB in earlier runs).  The design is chosen by
farthest-point sampling over the normalised factor levels (seeded, with a per-level cap so every level is covered), then
randomised in run order.  All reps of the first pass run before the second pass,
so stopping early still leaves full coverage.

Data: numeric CSV with a header, 8 Gaussian blobs, generated per (size, features)
into <data-dir>/doe/ and reused.  main_mpi's own default input is prose, so the
file is always passed with --input.

Preflight: `main_mpi --help` must list the K and iteration flags (names are
options: --k-flag / --iter-flag, defaults --kmeans-k / --kmeans-iter).

Usage (repo root, venv active):
    python scripts/doe_kmeans.py --pybin "$(which python)" --design-only
    python scripts/doe_kmeans.py --pybin "$(which python)" --prepare-only
    nohup python scripts/doe_kmeans.py --pybin "$(which python)" \
        > results/doe_kmeans_log.txt 2>&1 &
"""
import argparse, csv, fcntl, itertools, os, shutil, signal, socket, subprocess, sys, threading, time
from datetime import datetime

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import resource_benchmark as rb  # noqa: E402  (reuses the process-tree sampler and helpers)

SIZES = [25, 50, 100, 150, 200]
NPS   = [2, 3, 4]
ITERS = [10, 20, 40]
KS    = [3, 5, 10]
FEATS = [8, 16, 32]
N_BLOBS = 8
MAX_CONSEC_FAILS = 3

FIELDS = ["design_id", "run_id", "timestamp", "host", "app", "size_mb", "np", "workers",
          "k", "iters", "features", "cores", "heap_mb", "cap_threads", "rep",
          "driver_idx", "role", "pid", "wall_s", "exit_code", "n_samples",
          "cpu_avg_pct", "cpu_peak_pct", "cpu_p95_pct",
          "rss_avg_mb", "rss_peak_mb", "rss_p95_mb", "pss_avg_mb", "pss_peak_mb",
          "sys_cpu_avg_pct", "sys_mem_avg_mb", "sys_mem_peak_mb", "sys_mem_avail_min_mb",
          "cores_alloc", "heap_mb_alloc", "data_file", "trace_file"]


def full_grid():
    out = []
    for size, np_, it, k, d in itertools.product(SIZES, NPS, ITERS, KS, FEATS):
        if np_ == 2 and size >= 150:
            continue
        out.append(dict(size_mb=size, np=np_, iters=it, k=k, features=d))
    return out


def norm(cfg):
    return np.array([SIZES.index(cfg["size_mb"]) / (len(SIZES) - 1),
                     NPS.index(cfg["np"]) / (len(NPS) - 1),
                     ITERS.index(cfg["iters"]) / (len(ITERS) - 1),
                     KS.index(cfg["k"]) / (len(KS) - 1),
                     FEATS.index(cfg["features"]) / (len(FEATS) - 1)])


def make_design(n, seed):
    """Farthest-point sampling with a per-level cap so every level of every
    factor is covered (plain maximin piles points on the corners)."""
    grid = full_grid()
    X = np.array([norm(c) for c in grid])
    rng = np.random.default_rng(seed)
    n = min(n, len(grid))
    factors = [("size_mb", SIZES), ("np", NPS), ("iters", ITERS), ("k", KS), ("features", FEATS)]
    caps = {name: -(-n // len(levels)) + 1 for name, levels in factors}
    counts = {name: {lv: 0 for lv in levels} for name, levels in factors}

    def add(i):
        for name, _ in factors:
            counts[name][grid[i][name]] += 1

    def allowed(i):
        return all(counts[name][grid[i][name]] < caps[name] for name, _ in factors)

    chosen = [int(rng.integers(len(grid)))]
    add(chosen[0])
    dist = np.linalg.norm(X - X[chosen[0]], axis=1)
    while len(chosen) < n:
        order = np.argsort(-dist, kind="stable")
        nxt = next((int(i) for i in order if i not in chosen and allowed(int(i))), None)
        if nxt is None:
            break
        chosen.append(nxt)
        add(nxt)
        dist = np.minimum(dist, np.linalg.norm(X - X[nxt], axis=1))
    order = [int(i) for i in rng.permutation(chosen)]
    return [dict(grid[i], design_id=f"d{j + 1:02d}") for j, i in enumerate(order)]


def data_path(data_dir, size, feat):
    return os.path.abspath(os.path.join(data_dir, "doe", f"kmeans_s{size}_f{feat}.csv"))


def ensure_data(path, size_mb, d, seed=7):
    target = size_mb * 1_000_000
    if os.path.isfile(path) and abs(os.path.getsize(path) - target) <= 0.03 * target:
        return False
    os.makedirs(os.path.dirname(path), exist_ok=True)
    rng = np.random.default_rng(seed + d)
    centers = rng.uniform(-10, 10, (N_BLOBS, d))
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        fh.write(",".join(f"f{i}" for i in range(d)) + "\n")
        while True:
            lab = rng.integers(0, N_BLOBS, 500)
            X = centers[lab] + rng.normal(0, 2.0, (500, d))
            np.savetxt(fh, X, delimiter=",", fmt="%.15g")
            fh.flush()
            if os.path.getsize(tmp) >= target:
                break
    os.replace(tmp, path)
    return True


def check_flags(pybin, k_flag, iter_flag):
    try:
        res = subprocess.run([pybin, "-m", "mpj_spark.core.main_mpi", "--help"],
                             capture_output=True, text=True, timeout=180)
    except (OSError, subprocess.TimeoutExpired) as e:
        sys.exit(f"could not run main_mpi --help: {e}")
    text = res.stdout + res.stderr
    missing = [f for f in (k_flag, iter_flag, "--input", "--app") if f not in text]
    if missing:
        print("[!] main_mpi --help does not list: " + " ".join(missing), flush=True)
        print("    relevant help lines:", flush=True)
        for line in text.splitlines():
            if any(s in line.lower() for s in ("kmeans", "iter", "--k", "input")):
                print("    " + line.strip(), flush=True)
        print("    pass the right names with --k-flag / --iter-flag", flush=True)
        sys.exit(2)


def run_cmd(cmd, env, log_path, timeout_s, use_pss):
    t0 = time.perf_counter()
    logf = open(log_path, "w")
    logf.write("CMD: " + " ".join(cmd) + "\n")
    logf.flush()
    proc = subprocess.Popen(cmd, env=env, stdout=logf, stderr=subprocess.STDOUT,
                            start_new_session=True, text=True)
    stop, holder = [False], {}

    def monitor():
        time.sleep(1.0)
        holder["stats"], holder["sys"] = rb.subtree_stats(proc.pid, stop, use_pss)

    th = threading.Thread(target=monitor, daemon=True)
    th.start()
    code = None
    try:
        last = t0
        while proc.poll() is None:
            time.sleep(1)
            now = time.perf_counter()
            if now - last >= rb.HEARTBEAT_S:
                print(f"    ... run in progress ({now - t0:.0f}s elapsed)", flush=True)
                last = now
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
    ap.add_argument("--out", default="results/doe_kmeans.csv")
    ap.add_argument("--n-configs", type=int, default=40, dest="n_configs")
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cores", type=int, default=4, help="reference cores per driver")
    ap.add_argument("--heap-mb", type=int, default=3072, dest="heap_mb")
    ap.add_argument("--no-cap-threads", action="store_true", dest="no_cap",
                    help="do not cap OMP/OPENBLAS/MKL threads at --cores")
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--pss", action="store_true")
    ap.add_argument("--k-flag", default="--kmeans-k", dest="k_flag")
    ap.add_argument("--iter-flag", default="--kmeans-iter", dest="iter_flag")
    ap.add_argument("--data-dir", default="shared_storage", dest="data_dir")
    ap.add_argument("--pybin", default=sys.executable)
    ap.add_argument("--design-only", action="store_true", dest="design_only")
    ap.add_argument("--prepare-only", action="store_true", dest="prepare_only")
    a = ap.parse_args()

    design = make_design(a.n_configs, a.seed)
    files = sorted({(c["size_mb"], c["features"]) for c in design})
    total_mb = sum(s for s, _ in files)
    print(f"design: {len(design)} configs x {a.reps} reps = {len(design) * a.reps} runs; "
          f"{len(files)} data files ({total_mb} MB)", flush=True)
    os.makedirs("results", exist_ok=True)
    with open("results/doe_kmeans_design.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["design_id", "size_mb", "np", "iters", "k", "features"])
        w.writeheader()
        for c in design:
            w.writerow({k: c[k] for k in w.fieldnames})
    if a.design_only:
        for c in design:
            print("  ", c)
        return

    for size, d in files:
        p = data_path(a.data_dir, size, d)
        t0 = time.perf_counter()
        made = ensure_data(p, size, d)
        print(f"data {'generated' if made else 'ok       '} {p} ({time.perf_counter() - t0:.0f}s)", flush=True)
    if a.prepare_only:
        return

    if not a.pybin or not (os.path.isfile(a.pybin) and os.access(a.pybin, os.X_OK)):
        sys.exit(f"--pybin {a.pybin!r} is not an executable file (activate the venv)")
    if shutil.which("mpirun") is None:
        sys.exit("mpirun not found on PATH")
    check_flags(a.pybin, a.k_flag, a.iter_flag)

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
          f"thread-cap={'off' if a.no_cap else a.cores}", flush=True)

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
        run_id = (f"A_{did}_s{c['size_mb']}_np{c['np']}_k{c['k']}_i{c['iters']}"
                  f"_f{c['features']}_r{rep}")
        log_path = f"results/logs/{run_id}.log"
        trace_path = f"results/traces/{run_id}.csv"
        dfile = data_path(a.data_dir, c["size_mb"], c["features"])
        cmd = (["mpirun", "--oversubscribe", "-np", str(c["np"])] + exports +
               [a.pybin, "-m", "mpj_spark.core.main_mpi", "--app", "kmeans",
                "--input", dfile, a.k_flag, str(c["k"]), a.iter_flag, str(c["iters"])])
        print(f"[{datetime.now():%H:%M:%S}] START {k_}/{len(todo)}  {run_id}", flush=True)
        wall, code, stats, sys_s = run_cmd(cmd, env, log_path, a.timeout, a.pss)
        cores_alloc, heap_alloc = rb.parse_alloc(log_path)
        if stats:
            write_trace(trace_path, stats, wall)
        scpu = rb.mean([s[0] for s in sys_s])
        smem = rb.mean([s[1] for s in sys_s])
        smem_peak = rb.mx([s[1] for s in sys_s])
        savail = min([s[2] for s in sys_s]) if sys_s else None
        rows = stats or {0: {"pid": "", "cpu": [], "rss": [], "pss": []}}
        for drv, rec in sorted(rows.items()):
            w.writerow({
                "design_id": did, "run_id": run_id,
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "host": socket.gethostname(), "app": "kmeans", "size_mb": c["size_mb"],
                "np": c["np"], "workers": c["np"] - 1, "k": c["k"], "iters": c["iters"],
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
                "data_file": os.path.relpath(dfile),
                "trace_file": trace_path if stats else ""})
        f.flush()
        elapsed.append(wall)
        eta = (sum(elapsed) / len(elapsed)) * (len(todo) - k_)
        print(f"[{datetime.now():%H:%M:%S}] DONE {k_}/{len(todo)}  {run_id}  wall={wall:.1f}s "
              f"exit={code}  ETA<={eta / 3600:.1f}h", flush=True)
        if code == 124:
            timed_out.add(did)
        consec = 0 if code in (0, 124) else consec + 1
        if consec >= MAX_CONSEC_FAILS:
            print(f"[!] {consec} consecutive failed runs (not timeouts) -- aborting. "
                  f"Check {log_path}", flush=True)
            f.close()
            sys.exit(3)
        time.sleep(rb.SETTLE_S)
    f.close()
    print(f"ALL DONE. rows in {a.out}")


if __name__ == "__main__":
    main()
