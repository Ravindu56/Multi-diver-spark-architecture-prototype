#!/usr/bin/env python3
"""
doe_rightsize.py -- Set D: right-sizing labels for the resource allocator (Objectives 2c/2d).

Sets A/B measured the demand of a workload at ONE generous allocation.  This set asks the
allocation question: for a fixed workload, how few cores per driver can it use before it
slows down?  A handful of representative workloads per app is run at several core counts
(Spark core budget SLOTS_OVERRIDE = numeric-library thread caps = cores), the 'knee' is the
smallest core count whose wall time is within --tol of the best, and each workload gets a
label: knee cores, the CPU it actually used there, and a memory limit sized from the
observed RSS (the heap is not the lever for total memory, so the limit is derived from RSS).

Phases:
  run     (default)  python scripts/doe_rightsize.py --pybin "$(which python)"
  labels  python scripts/doe_rightsize.py --labels results/doe_rightsize.csv --tol 0.05 0.10

Workloads (editable, WORKLOADS below): 6 per app spanning small/medium/large size and 1-3
drivers.  Combinations whose estimated total memory exceeds --mem-budget-gb are skipped
(more cores means more worker processes and more memory).  Order is randomised and every
(workload, cores) cell runs once before any second repetition.

Reuses the sampler/watchdog/data generators of doe_logreg.py and doe_kmeans.py: a run
whose drivers sit idle for --stall-s is killed (exit 125) and retried once; a timed-out
cell is not repeated.  The first line of each run log is the exact mpirun command.
"""
import argparse, csv, fcntl, math, os, random, shutil, socket, sys, time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import resource_benchmark as rb  # noqa: E402
import doe_kmeans as dk  # noqa: E402
import doe_logreg as dl  # noqa: E402

WORKLOADS = [
    dict(id="L1", app="logreg", size_mb=25, np=3, iters=30, features=20, k=0),
    dict(id="L2", app="logreg", size_mb=100, np=3, iters=30, features=20, k=0),
    dict(id="L3", app="logreg", size_mb=200, np=3, iters=30, features=20, k=0),
    dict(id="L4", app="logreg", size_mb=50, np=2, iters=30, features=20, k=0),
    dict(id="L5", app="logreg", size_mb=150, np=2, iters=30, features=20, k=0),
    dict(id="L6", app="logreg", size_mb=100, np=4, iters=10, features=40, k=0),
    dict(id="K1", app="kmeans", size_mb=25, np=3, iters=20, features=16, k=5),
    dict(id="K2", app="kmeans", size_mb=100, np=3, iters=20, features=16, k=5),
    dict(id="K3", app="kmeans", size_mb=200, np=3, iters=20, features=16, k=5),
    dict(id="K4", app="kmeans", size_mb=50, np=2, iters=20, features=16, k=5),
    dict(id="K5", app="kmeans", size_mb=100, np=2, iters=20, features=16, k=5),
    dict(id="K6", app="kmeans", size_mb=100, np=4, iters=20, features=16, k=5),
]
MAX_CONSEC_FAILS = 3

FIELDS = ["workload_id", "app", "run_id", "attempt", "timestamp", "host", "size_mb", "np",
          "workers", "iters", "features", "k", "cores", "heap_mb", "cap_threads", "rep",
          "driver_idx", "role", "pid", "wall_s", "exit_code", "n_samples",
          "cpu_avg_pct", "cpu_peak_pct", "cpu_p95_pct",
          "rss_avg_mb", "rss_peak_mb", "rss_p95_mb",
          "sys_cpu_avg_pct", "sys_mem_avg_mb", "sys_mem_peak_mb", "sys_mem_avail_min_mb",
          "cores_alloc", "heap_mb_alloc", "iters_logged", "data_file", "trace_file"]


def est_total_gb(wl, cores):
    workers = wl["np"] - 1
    if wl["app"] == "logreg":
        per = 2.7 + 0.024 * wl["size_mb"] / workers + 0.3 * (cores - 4)
    else:
        per = 1.6 + 0.05 * max(0, cores - 4)
    return workers * per + 1.0


def compute_labels(path, tols, ref_cores, out_path):
    import pandas as pd
    df = pd.read_csv(path)
    df = df[df.exit_code == 0]
    drv = df[df.role == "driver"]
    run = drv.groupby("run_id").agg(
        workload_id=("workload_id", "first"), app=("app", "first"), cores=("cores", "first"),
        rep=("rep", "first"), size_mb=("size_mb", "first"), np=("np", "first"),
        iters=("iters", "first"), features=("features", "first"), k=("k", "first"),
        wall=("wall_s", "first"), cpu_avg=("cpu_avg_pct", "mean"), cpu_p95=("cpu_p95_pct", "mean"),
        rss_p95=("rss_p95_mb", "max"), rss_peak=("rss_peak_mb", "max")).reset_index()
    cell = run.groupby(["workload_id", "cores"]).agg(
        app=("app", "first"), size_mb=("size_mb", "first"), np=("np", "first"),
        iters=("iters", "first"), features=("features", "first"), k=("k", "first"),
        n=("wall", "size"), wall=("wall", "mean"), wall_sd=("wall", "std"),
        cpu_avg=("cpu_avg", "mean"), cpu_p95=("cpu_p95", "mean"),
        rss_p95=("rss_p95", "mean"), rss_peak=("rss_peak", "max")).reset_index()
    print("\nwall time (s) by cores:")
    print(cell.pivot(index="workload_id", columns="cores", values="wall").round(1).to_string())
    rows = []
    for wid, g in cell.groupby("workload_id"):
        g = g.sort_values("cores")
        best = g.wall.min()
        for tol in tols:
            ok = g[g.wall <= best * (1.0 + tol)]
            r = ok.iloc[0]
            mem = max(1.25 * r.rss_p95, 1.10 * r.rss_peak)
            rows.append({
                "workload_id": wid, "app": r.app, "size_mb": r.size_mb, "np": r.np,
                "iters": r.iters, "features": r.features, "k": r.k, "tol": tol,
                "knee_cores": int(r.cores), "wall_best_s": round(best, 1),
                "wall_at_knee_s": round(r.wall, 1),
                "slowdown_pct": round((r.wall / best - 1.0) * 100.0, 1),
                "cpu_avg_cores_at_knee": round(r.cpu_avg / 100.0, 2),
                "cpu_p95_cores_at_knee": round(r.cpu_p95 / 100.0, 2),
                "rss_p95_mb_at_knee": round(r.rss_p95), "rss_peak_mb_at_knee": round(r.rss_peak),
                "rec_cpus": int(r.cores), "rec_mem_mb": int(math.ceil(mem / 64.0) * 64),
                "cores_freed_vs_ref": ref_cores - int(r.cores), "cells_tested": len(g),
                "reps_per_cell": int(g.n.min())})
    out = pd.DataFrame(rows)
    out.to_csv(out_path, index=False)
    print("\nlabels written to " + out_path)
    print(out[["workload_id", "app", "size_mb", "np", "tol", "knee_cores", "slowdown_pct",
               "cpu_avg_cores_at_knee", "rss_p95_mb_at_knee", "rec_cpus", "rec_mem_mb"]].to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=os.getcwd())
    ap.add_argument("--out", default="results/doe_rightsize.csv")
    ap.add_argument("--apps", nargs="+", default=["logreg", "kmeans"])
    ap.add_argument("--workloads", nargs="+", default=None, help="subset of workload ids, e.g. L1 K2")
    ap.add_argument("--cores-logreg", type=int, nargs="+", default=[1, 2, 3, 4, 6, 8], dest="cores_lr")
    ap.add_argument("--cores-kmeans", type=int, nargs="+", default=[1, 2, 3, 4], dest="cores_km")
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--seed", type=int, default=44)
    ap.add_argument("--heap-logreg", type=int, default=6144, dest="heap_lr")
    ap.add_argument("--heap-kmeans", type=int, default=3072, dest="heap_km")
    ap.add_argument("--mem-budget-gb", type=float, default=12.0, dest="budget")
    ap.add_argument("--timeout", type=int, default=3600)
    ap.add_argument("--stall-s", type=int, default=300, dest="stall_s")
    ap.add_argument("--lr-iter-flag", default="--logreg-iter", dest="lr_iter")
    ap.add_argument("--lr-features-flag", default="--logreg-features", dest="lr_feat")
    ap.add_argument("--km-k-flag", default="--kmeans-k", dest="km_k")
    ap.add_argument("--km-iter-flag", default="--kmeans-iter", dest="km_iter")
    ap.add_argument("--template", default="shared_storage/logreg_data.csv")
    ap.add_argument("--data-dir", default="shared_storage", dest="data_dir")
    ap.add_argument("--pybin", default=sys.executable)
    ap.add_argument("--design-only", action="store_true", dest="design_only")
    ap.add_argument("--prepare-only", action="store_true", dest="prepare_only")
    ap.add_argument("--labels", default=None, help="compute labels from this results CSV and exit")
    ap.add_argument("--tol", type=float, nargs="+", default=[0.05, 0.10])
    ap.add_argument("--ref-cores", type=int, default=4, dest="ref_cores")
    ap.add_argument("--labels-out", default="results/rightsize_labels.csv", dest="labels_out")
    a = ap.parse_args()

    if a.labels:
        compute_labels(a.labels, a.tol, a.ref_cores, a.labels_out)
        return

    wls = [w for w in WORKLOADS if w["app"] in a.apps and (a.workloads is None or w["id"] in a.workloads)]
    cells, skipped = [], []
    for w in wls:
        for c in (a.cores_lr if w["app"] == "logreg" else a.cores_km):
            (cells if est_total_gb(w, c) <= a.budget else skipped).append((w, c))
    print(f"design: {len(wls)} workloads, {len(cells)} (workload, cores) cells x {a.reps} reps = "
          f"{len(cells) * a.reps} runs; {len(skipped)} cells skipped by the {a.budget} GB memory guard", flush=True)
    for w in wls:
        cs = [c for (x, c) in cells if x["id"] == w["id"]]
        sk = [c for (x, c) in skipped if x["id"] == w["id"]]
        print(f"  {w['id']} {w['app']:6s} size={w['size_mb']:3d} np={w['np']} iters={w['iters']} "
              f"feat={w['features']} k={w['k']}  cores={cs}" + (f"  skipped={sk}" if sk else ""), flush=True)
    if a.design_only:
        return

    info = dl.template_info(a.template)
    for w in wls:
        if w["app"] == "logreg":
            p = dl.data_path(a.data_dir, w["size_mb"], w["features"])
            made = dl.ensure_data(p, w["size_mb"], w["features"], info)
        else:
            p = dk.data_path(a.data_dir, w["size_mb"], w["features"])
            made = dk.ensure_data(p, w["size_mb"], w["features"])
        w["data"] = p
        print(f"data {'generated' if made else 'ok       '} {p}", flush=True)
    if a.prepare_only:
        return

    if not a.pybin or not (os.path.isfile(a.pybin) and os.access(a.pybin, os.X_OK)):
        sys.exit(f"--pybin {a.pybin!r} is not an executable file (activate the venv)")
    if shutil.which("mpirun") is None:
        sys.exit("mpirun not found on PATH")
    need = ["--app", "--input"]
    if any(w["app"] == "logreg" for w in wls):
        need += [a.lr_iter] + ([] if a.lr_feat == "none" else [a.lr_feat])
    if any(w["app"] == "kmeans" for w in wls):
        need += [a.km_k, a.km_iter]
    dl.check_flags(a.pybin, need)

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    os.makedirs("results/logs", exist_ok=True)
    os.makedirs("results/traces", exist_ok=True)
    lock_f = open(a.out + ".lock", "w")
    try:
        fcntl.flock(lock_f, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        sys.exit(f"another run is already writing to {a.out}")

    done, timed_out = set(), set()
    if os.path.exists(a.out):
        with open(a.out, newline="") as fh:
            for row in csv.DictReader(fh):
                key = (row["workload_id"], row["cores"])
                if row.get("exit_code") == "0":
                    done.add(key + (row["rep"],))
                elif row.get("exit_code") == "124":
                    timed_out.add(key)
    new_file = not os.path.exists(a.out) or os.path.getsize(a.out) == 0
    f = open(a.out, "a", newline="")
    w_csv = csv.DictWriter(f, fieldnames=FIELDS)
    if new_file:
        w_csv.writeheader()

    rng = random.Random(a.seed)
    todo = []
    for rep in range(1, a.reps + 1):
        for (w, c) in rng.sample(cells, len(cells)):
            if (w["id"], str(c), str(rep)) not in done:
                todo.append((w, c, rep))
    print(f"[{datetime.now():%H:%M:%S}] todo={len(todo)} of {len(cells) * a.reps}  "
          f"heap logreg={a.heap_lr}MB kmeans={a.heap_km}MB  stall={a.stall_s}s", flush=True)

    elapsed, consec = [], 0
    for k_, (w, cores, rep) in enumerate(todo, 1):
        if (w["id"], str(cores)) in timed_out:
            continue
        env = os.environ.copy()
        env["SLOTS_OVERRIDE"] = str(cores)
        env["SPARK_DRIVER_MEMORY"] = f"{a.heap_lr if w['app'] == 'logreg' else a.heap_km}m"
        names = ["SLOTS_OVERRIDE", "SPARK_DRIVER_MEMORY", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]
        for v in names[2:]:
            env[v] = str(cores)
        exports = []
        for v in names:
            exports += ["-x", v]
        for attempt in (1, 2):
            run_id = f"D_{w['id']}_c{cores}_r{rep}" + ("" if attempt == 1 else "_retry")
            log_path = f"results/logs/{run_id}.log"
            trace_path = f"results/traces/{run_id}.csv"
            cmd = (["mpirun", "--oversubscribe", "-np", str(w["np"])] + exports +
                   [a.pybin, "-m", "mpj_spark.core.main_mpi", "--app", w["app"], "--input", w["data"]])
            if w["app"] == "logreg":
                cmd += [a.lr_iter, str(w["iters"])]
                if a.lr_feat != "none":
                    cmd += [a.lr_feat, str(w["features"])]
            else:
                cmd += [a.km_k, str(w["k"]), a.km_iter, str(w["iters"])]
            print(f"[{datetime.now():%H:%M:%S}] START {k_}/{len(todo)}  {run_id}", flush=True)
            wall, code, stats, sys_s = dl.run_cmd(cmd, env, log_path, a.timeout, False, a.stall_s)
            cores_alloc, heap_alloc = rb.parse_alloc(log_path)
            iters_logged = dl.parse_iters(log_path)
            if stats:
                dl.write_trace(trace_path, stats, wall)
            scpu = rb.mean([s[0] for s in sys_s])
            smem = rb.mean([s[1] for s in sys_s])
            smem_peak = rb.mx([s[1] for s in sys_s])
            savail = min([s[2] for s in sys_s]) if sys_s else None
            rows = stats or {0: {"pid": "", "cpu": [], "rss": [], "pss": []}}
            for drv, rec in sorted(rows.items()):
                w_csv.writerow({
                    "workload_id": w["id"], "app": w["app"], "run_id": run_id, "attempt": attempt,
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    "host": socket.gethostname(), "size_mb": w["size_mb"], "np": w["np"],
                    "workers": w["np"] - 1, "iters": w["iters"], "features": w["features"],
                    "k": w["k"], "cores": cores,
                    "heap_mb": a.heap_lr if w["app"] == "logreg" else a.heap_km,
                    "cap_threads": cores, "rep": rep, "driver_idx": drv,
                    "role": ("root" if drv == 0 else "driver") if rec["rss"] else "",
                    "pid": rec["pid"], "wall_s": f"{wall:.2f}", "exit_code": code,
                    "n_samples": len(rec["cpu"]),
                    "cpu_avg_pct": rb.fmt(rb.mean(rec["cpu"]), 1),
                    "cpu_peak_pct": rb.fmt(rb.mx(rec["cpu"]), 1),
                    "cpu_p95_pct": rb.fmt(rb.pctl(rec["cpu"], 95), 1),
                    "rss_avg_mb": rb.fmt(rb.mean(rec["rss"])),
                    "rss_peak_mb": rb.fmt(rb.mx(rec["rss"])),
                    "rss_p95_mb": rb.fmt(rb.pctl(rec["rss"], 95)),
                    "sys_cpu_avg_pct": rb.fmt(scpu, 1), "sys_mem_avg_mb": rb.fmt(smem),
                    "sys_mem_peak_mb": rb.fmt(smem_peak), "sys_mem_avail_min_mb": rb.fmt(savail),
                    "cores_alloc": cores_alloc, "heap_mb_alloc": heap_alloc,
                    "iters_logged": iters_logged, "data_file": os.path.relpath(w["data"]),
                    "trace_file": trace_path if stats else ""})
            f.flush()
            elapsed.append(wall)
            eta = (sum(elapsed) / len(elapsed)) * (len(todo) - k_)
            print(f"[{datetime.now():%H:%M:%S}] DONE {k_}/{len(todo)}  {run_id}  wall={wall:.1f}s "
                  f"exit={code}  ETA<={eta / 3600:.1f}h", flush=True)
            if code == 125 and attempt == 1:
                print("    retrying once after a stall", flush=True)
                time.sleep(rb.SETTLE_S)
                continue
            break
        if code == 124:
            timed_out.add((w["id"], str(cores)))
        consec = 0 if code in (0, 124, 125) else consec + 1
        if consec >= MAX_CONSEC_FAILS:
            print(f"[!] {consec} consecutive failed runs -- aborting. Check {log_path}", flush=True)
            f.close()
            sys.exit(3)
        time.sleep(rb.SETTLE_S)
    f.close()
    print(f"ALL DONE. rows in {a.out}\nnext: python scripts/doe_rightsize.py --labels {a.out}")


if __name__ == "__main__":
    main()
