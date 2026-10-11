#!/usr/bin/env python3
"""
doe_holdout.py -- Set E: validation runs for the demand predictor and the allocator labels.

Three sub-sets, all reusing the sampler / watchdog / data generators of Sets A, B and D:

  E1  holdout   Workloads at sizes that were NOT in the training sets (A/B used 25-200 MB at
                25/50/100/150/200; D used 25/50/100/150/200) -- interior sizes 75 and 125 MB and
                extension sizes beyond 200 MB -- run at the Set A/B reference allocation
                (--ref-cores cores, thread caps = cores) so they are directly comparable.
  E2  rep3      A third repetition of each Set D workload at its knee core count
                (rec_cpus from rightsize_labels.csv), to estimate the noise ceiling.
  E3  memresp   Memory-response curve for K2, L2, L4 at their knee cores: the run is capped
                at factor x rec_mem_mb per driver (factors descending).  A capped run that dies
                is a result (the under-allocation cliff), not an error: it is recorded, lower
                factors of that workload are skipped, and it never aborts the queue.

Memory cap modes (--mem-mode):
  cgroup  systemd-run --user --scope -p MemoryMax=<workers*factor*rec_mem + root> -p MemorySwapMax=0
          (a real cap on the whole mpirun tree; needs cgroup v2 + a user systemd session)
  heap    only sets SPARK_DRIVER_MEMORY.  Earlier floors showed heap is NOT the lever for total
          RSS, so this will show no penalty -- use it only if cgroup mode is unavailable.

Usage:
  python scripts/doe_holdout.py --design-only
  python scripts/doe_holdout.py --pybin "$(which python)" --prepare-only
  nohup python scripts/doe_holdout.py --pybin "$(which python)" > results/doe_holdout_log.txt 2>&1 &
  python scripts/doe_holdout.py --summarize results/doe_holdout.csv --d-csv results/doe_rightsize.csv
"""
import argparse, csv, fcntl, os, random, shutil, socket, subprocess, sys, time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import resource_benchmark as rb  # noqa: E402
import doe_kmeans as dk  # noqa: E402
import doe_logreg as dl  # noqa: E402
import doe_rightsize as dr  # noqa: E402

# (kind, app, size_mb, np, iters, features, k)
HOLDOUT = [
    ("interior", "logreg", 75, 2, 30, 20, 0), ("interior", "logreg", 75, 3, 30, 20, 0),
    ("interior", "logreg", 125, 2, 30, 20, 0), ("interior", "logreg", 125, 3, 30, 20, 0),
    ("extrap", "logreg", 300, 2, 30, 20, 0), ("extrap", "logreg", 250, 3, 30, 20, 0),
    ("interior", "kmeans", 75, 2, 20, 16, 5), ("interior", "kmeans", 75, 3, 20, 16, 5),
    ("interior", "kmeans", 125, 2, 20, 16, 5), ("interior", "kmeans", 125, 3, 20, 16, 5),
    ("extrap", "kmeans", 300, 3, 20, 16, 5), ("extrap", "kmeans", 400, 3, 20, 16, 5),
]
MEM_WORKLOADS = ["K2", "L2", "L4"]
MEM_FACTORS = [0.75, 0.5, 0.33]
MAX_CONSEC_FAILS = 3
FIELDS = dr.FIELDS + ["set", "kind", "mem_factor", "mem_limit_mb"]


def read_labels(path, tol):
    out = {}
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            if abs(float(r["tol"]) - tol) < 1e-9:
                out[r["workload_id"]] = {"cpus": int(r["rec_cpus"]), "mem": int(r["rec_mem_mb"])}
    return out


def build_items(a, labels):
    items = []
    wl_by_id = {w["id"]: w for w in dr.WORKLOADS}
    if "E1" in a.sets:
        for (kind, app, size, np_, it, feat, k) in HOLDOUT:
            wid = f"H{app[0].upper()}{size}_n{np_}"
            items.append(dict(set="E1", wid=wid, kind=kind, app=app, size_mb=size, np=np_, iters=it,
                              features=feat, k=k, cores=a.ref_cores, factor=None, limit=None,
                              reps=list(range(1, a.reps + 1)), heap_mem=None))
    if "E2" in a.sets or "E3" in a.sets:
        missing = [w for w in (a.rep3_workloads or [x["id"] for x in dr.WORKLOADS]) + MEM_WORKLOADS if w not in labels]
        if not labels or (("E2" in a.sets or "E3" in a.sets) and missing and not labels):
            sys.exit(f"no labels at tol={a.tol} in {a.labels_csv}")
    if "E2" in a.sets:
        for wid in (a.rep3_workloads or [x["id"] for x in dr.WORKLOADS]):
            w = wl_by_id[wid]
            items.append(dict(set="E2", wid=wid, kind="rep3", app=w["app"], size_mb=w["size_mb"], np=w["np"],
                              iters=w["iters"], features=w["features"], k=w["k"], cores=labels[wid]["cpus"],
                              factor=None, limit=None, reps=[3], heap_mem=None))
    if "E3" in a.sets:
        for wid in MEM_WORKLOADS:
            w = wl_by_id[wid]
            for f in sorted(a.factors, reverse=True):
                per = f * labels[wid]["mem"]
                items.append(dict(set="E3", wid=wid, kind="memresp", app=w["app"], size_mb=w["size_mb"],
                                  np=w["np"], iters=w["iters"], features=w["features"], k=w["k"],
                                  cores=labels[wid]["cpus"], factor=f,
                                  limit=int((w["np"] - 1) * per + a.root_mb),
                                  reps=list(range(1, a.reps + 1)), heap_mem=max(450, int(per))))
    return items


def fkey(f):
    return "" if f is None else f"{f:g}"


def cgroup_ok():
    try:
        r = subprocess.run(["systemd-run", "--user", "--scope", "--quiet", "-p", "MemoryMax=256M",
                            "-p", "MemorySwapMax=0", "true"], capture_output=True, timeout=20)
        return r.returncode == 0
    except Exception:
        return False


def run_table(df):
    drv = df[df.role == "driver"]
    m = drv.groupby("run_id").agg(cpu_avg=("cpu_avg_pct", "mean"), cpu_p95=("cpu_p95_pct", "mean"),
                                  rss_p95=("rss_p95_mb", "max"), rss_peak=("rss_peak_mb", "max"))
    b = df.groupby("run_id").agg(workload_id=("workload_id", "first"), app=("app", "first"),
                                 cores=("cores", "first"), rep=("rep", "first"), wall=("wall_s", "first"),
                                 exit=("exit_code", "first"), size_mb=("size_mb", "first"),
                                 np=("np", "first"), set=("set", "first"), mem_factor=("mem_factor", "first"),
                                 mem_limit=("mem_limit_mb", "first"))
    return b.join(m).reset_index()


def summarize(e_path, d_path):
    import pandas as pd
    pd.set_option("display.width", 200)
    e = pd.read_csv(e_path)
    e_runs = run_table(e)
    d_runs = None
    if d_path and os.path.exists(d_path):
        d = pd.read_csv(d_path)
        d["set"] = "D"; d["mem_factor"] = float("nan"); d["mem_limit_mb"] = float("nan")
        d_runs = run_table(d)
        d_runs = d_runs[d_runs["exit"] == 0]
    ok = e_runs[e_runs["exit"] == 0]

    h = ok[ok.set == "E1"]
    if len(h):
        print("\nE1 holdout (reference allocation) -- per workload means:")
        t = h.groupby(["app", "size_mb", "np"]).agg(runs=("wall", "size"), wall_s=("wall", "mean"),
            cpu_avg=("cpu_avg", "mean"), cpu_p95=("cpu_p95", "mean"), rss_p95=("rss_p95", "mean"),
            rss_peak=("rss_peak", "max")).round(1)
        t["cpu_avg"] = (t.cpu_avg / 100).round(2); t["cpu_p95"] = (t.cpu_p95 / 100).round(2)
        print(t.to_string())
        print("compare these against the model's predictions (train on Sets A/B only); report error vs size.")

    r3 = ok[ok.set == "E2"]
    if len(r3) and d_runs is not None:
        allr = pd.concat([d_runs, r3], ignore_index=True)
        cells = []
        for (wid, c), g in allr.groupby(["workload_id", "cores"]):
            if len(g) >= 3:
                cells.append({"workload_id": wid, "cores": c, "n": len(g),
                              "cv_wall": g.wall.std() / g.wall.mean(),
                              "cv_rss_p95": g.rss_p95.std() / g.rss_p95.mean(),
                              "cv_cpu_avg": g.cpu_avg.std() / g.cpu_avg.mean()})
        if cells:
            c = pd.DataFrame(cells)
            print("\nE2 noise (coefficient of variation over >=3 reps, one knee cell per workload):")
            print((c.set_index("workload_id") * 1).round(3).to_string())
            print("median CV  wall %.3f  rss_p95 %.3f  cpu_avg %.3f" % (c.cv_wall.median(), c.cv_rss_p95.median(), c.cv_cpu_avg.median()))
            print("noise-ceiling R2 ~ 1 - CV^2 / (spread of the target / mean)^2 : compare with the model's R2.")

    m = e_runs[e_runs.set == "E3"]
    if len(m):
        base = {}
        if d_runs is not None:
            for (wid, c), g in d_runs.groupby(["workload_id", "cores"]):
                base[(wid, c)] = (g.wall.mean(), g.rss_p95.mean())
        rows = []
        for (wid, f), g in m.groupby(["workload_id", "mem_factor"]):
            good = g[g["exit"] == 0]
            b = base.get((wid, g.cores.iloc[0]))
            w = good.wall.mean() if len(good) else float("nan")
            rows.append({"workload": wid, "factor": f, "limit_mb": int(g.mem_limit.iloc[0]), "runs": len(g),
                         "ok": len(good), "exit_codes": ",".join(sorted(set(str(x) for x in g["exit"]))),
                         "wall_s": round(w, 1), "slowdown_pct": round((w / b[0] - 1) * 100, 1) if b else None,
                         "rss_peak_mb": good.rss_peak.max() if len(good) else None})
        print("\nE3 memory response (factor x rec_mem_mb per driver; baseline = Set D knee cell):")
        print(pd.DataFrame(rows).sort_values(["workload", "factor"], ascending=[True, False]).to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/doe_holdout.csv")
    ap.add_argument("--sets", nargs="+", default=["E1", "E2", "E3"], choices=["E1", "E2", "E3"])
    ap.add_argument("--labels-csv", default="results/rightsize_labels.csv", dest="labels_csv")
    ap.add_argument("--tol", type=float, default=0.05)
    ap.add_argument("--rep3-workloads", nargs="+", default=None, dest="rep3_workloads")
    ap.add_argument("--factors", type=float, nargs="+", default=MEM_FACTORS)
    ap.add_argument("--mem-mode", choices=["cgroup", "heap"], default="cgroup", dest="mem_mode")
    ap.add_argument("--root-mb", type=int, default=512, dest="root_mb",
                    help="allowance for the root rank + launcher added to the cgroup cap")
    ap.add_argument("--ref-cores", type=int, default=4, dest="ref_cores")
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--seed", type=int, default=46)
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
    ap.add_argument("--summarize", default=None)
    ap.add_argument("--d-csv", default="results/doe_rightsize.csv", dest="d_csv")
    a = ap.parse_args()

    if a.summarize:
        summarize(a.summarize, a.d_csv)
        return

    labels = {}
    if "E2" in a.sets or "E3" in a.sets:
        if not os.path.exists(a.labels_csv):
            sys.exit(f"{a.labels_csv} not found (run: python scripts/doe_rightsize.py --labels results/doe_rightsize.csv)")
        labels = read_labels(a.labels_csv, a.tol)
    items = build_items(a, labels)

    ok_items, skipped = [], []
    for it in items:
        gb = dr.est_total_gb(it, it["cores"])
        (ok_items if (it["set"] != "E1" or gb <= a.budget) else skipped).append((it, gb))
    n_runs = sum(len(it["reps"]) for it, _ in ok_items)
    print(f"design: {len(ok_items)} cells, {n_runs} runs, {len(skipped)} cells skipped by the "
          f"{a.budget} GB memory guard (mem-mode {a.mem_mode})", flush=True)
    for it, gb in ok_items:
        extra = f" cap={it['limit']}MB(x{it['factor']:g})" if it["factor"] else ""
        print(f"  {it['set']} {it['wid']:12s} {it['kind']:8s} {it['app']:6s} size={it['size_mb']:3d} np={it['np']} "
              f"cores={it['cores']} reps={it['reps']} est={gb:.1f}GB{extra}", flush=True)
    for it, gb in skipped:
        print(f"  SKIPPED {it['wid']} est={gb:.1f}GB > {a.budget}GB (raise --mem-budget-gb only after checking `free -h`)", flush=True)
    if a.design_only:
        return
    items = [it for it, _ in ok_items]

    info = dl.template_info(a.template)
    for it in items:
        if it["app"] == "logreg":
            p = dl.data_path(a.data_dir, it["size_mb"], it["features"])
            made = dl.ensure_data(p, it["size_mb"], it["features"], info)
        else:
            p = dk.data_path(a.data_dir, it["size_mb"], it["features"])
            made = dk.ensure_data(p, it["size_mb"], it["features"])
        it["data"] = p
        print(f"data {'generated' if made else 'ok       '} {p}", flush=True)
    if a.prepare_only:
        return

    if not a.pybin or not (os.path.isfile(a.pybin) and os.access(a.pybin, os.X_OK)):
        sys.exit(f"--pybin {a.pybin!r} is not an executable file (activate the venv)")
    if shutil.which("mpirun") is None:
        sys.exit("mpirun not found on PATH")
    if any(it["set"] == "E3" for it in items) and a.mem_mode == "cgroup" and not cgroup_ok():
        sys.exit("cgroup memory cap unavailable (systemd-run --user --scope -p MemoryMax failed). "
                 "Fix the user session / cgroup v2, or use --mem-mode heap (heap is a weak lever), "
                 "or run with --sets E1 E2 now and E3 later.")
    need = ["--app", "--input"]
    if any(i["app"] == "logreg" for i in items):
        need += [a.lr_iter] + ([] if a.lr_feat == "none" else [a.lr_feat])
    if any(i["app"] == "kmeans" for i in items):
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
    dead_factor, failed_cells = {}, set()
    if os.path.exists(a.out):
        with open(a.out, newline="") as fh:
            for row in csv.DictReader(fh):
                fk = row.get("mem_factor", "")
                fk = "" if fk == "" else f"{float(fk):g}"
                key = (row["set"], row["workload_id"], row["cores"], fk)
                code = row.get("exit_code")
                if code == "0" or (row["set"] == "E3" and code not in ("124", "125", "")):
                    done.add(key + (row["rep"],))
                    if code != "0":
                        failed_cells.add(key)
                        dead_factor[row["workload_id"]] = max(dead_factor.get(row["workload_id"], 0.0), float(fk))
                elif code == "124":
                    timed_out.add(key)
    new_file = not os.path.exists(a.out) or os.path.getsize(a.out) == 0
    f = open(a.out, "a", newline="")
    w_csv = csv.DictWriter(f, fieldnames=FIELDS)
    if new_file:
        w_csv.writeheader()

    rng = random.Random(a.seed)
    todo = []
    e1 = [i for i in items if i["set"] == "E1"]
    for rep in range(1, a.reps + 1):
        for it in rng.sample(e1, len(e1)):
            todo.append((it, rep))
    e2 = [i for i in items if i["set"] == "E2"]
    for it in rng.sample(e2, len(e2)):
        todo.append((it, it["reps"][0]))
    e3 = [i for i in items if i["set"] == "E3"]
    order = rng.sample(MEM_WORKLOADS, len(MEM_WORKLOADS))
    for wid in order:
        for it in sorted([i for i in e3 if i["wid"] == wid], key=lambda x: -x["factor"]):
            for rep in it["reps"]:
                todo.append((it, rep))
    todo = [(it, rep) for it, rep in todo
            if (it["set"], it["wid"], str(it["cores"]), fkey(it["factor"]), str(rep)) not in done]
    print(f"[{datetime.now():%H:%M:%S}] todo={len(todo)} runs  stall={a.stall_s}s", flush=True)

    elapsed, consec = [], 0
    for k_, (it, rep) in enumerate(todo, 1):
        cell = (it["set"], it["wid"], str(it["cores"]), fkey(it["factor"]))
        if cell in timed_out or cell in failed_cells:
            continue
        if it["set"] == "E3" and it["wid"] in dead_factor and it["factor"] < dead_factor[it["wid"]]:
            print(f"    skip {it['wid']} x{it['factor']:g}: a higher factor already failed", flush=True)
            continue
        cores = it["cores"]
        heap = a.heap_lr if it["app"] == "logreg" else a.heap_km
        if it["set"] == "E3" and a.mem_mode == "heap":
            heap = it["heap_mem"]
        env = os.environ.copy()
        names = ["SLOTS_OVERRIDE", "SPARK_DRIVER_MEMORY", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]
        env["SLOTS_OVERRIDE"] = str(cores)
        env["SPARK_DRIVER_MEMORY"] = f"{heap}m"
        for v in names[2:]:
            env[v] = str(cores)
        exports = []
        for v in names:
            exports += ["-x", v]
        for attempt in (1, 2):
            run_id = (f"{it['set']}_{it['wid']}_c{cores}" + (f"_m{int(round(it['factor'] * 100)):03d}" if it["factor"] else "")
                      + f"_r{rep}" + ("" if attempt == 1 else "_retry"))
            log_path = f"results/logs/{run_id}.log"
            trace_path = f"results/traces/{run_id}.csv"
            cmd = ["mpirun", "--oversubscribe", "-np", str(it["np"])] + exports + \
                  [a.pybin, "-m", "mpj_spark.core.main_mpi", "--app", it["app"], "--input", it["data"]]
            if it["app"] == "logreg":
                cmd += [a.lr_iter, str(it["iters"])]
                if a.lr_feat != "none":
                    cmd += [a.lr_feat, str(it["features"])]
            else:
                cmd += [a.km_k, str(it["k"]), a.km_iter, str(it["iters"])]
            if it["set"] == "E3" and a.mem_mode == "cgroup":
                cmd = ["systemd-run", "--user", "--scope", "--quiet", "-p", f"MemoryMax={it['limit']}M",
                       "-p", "MemorySwapMax=0"] + cmd
            print(f"[{datetime.now():%H:%M:%S}] START {k_}/{len(todo)}  {run_id}", flush=True)
            wall, code, stats, sys_s = dl.run_cmd(cmd, env, log_path, a.timeout, False, a.stall_s)
            cores_alloc, heap_alloc = rb.parse_alloc(log_path)
            iters_logged = dl.parse_iters(log_path)
            if stats:
                dl.write_trace(trace_path, stats, wall)
            scpu = rb.mean([s[0] for s in sys_s]); smem = rb.mean([s[1] for s in sys_s])
            smem_peak = rb.mx([s[1] for s in sys_s])
            savail = min([s[2] for s in sys_s]) if sys_s else None
            rows = stats or {0: {"pid": "", "cpu": [], "rss": [], "pss": []}}
            for drv, rec in sorted(rows.items()):
                w_csv.writerow({
                    "workload_id": it["wid"], "app": it["app"], "run_id": run_id, "attempt": attempt,
                    "timestamp": datetime.now().isoformat(timespec="seconds"), "host": socket.gethostname(),
                    "size_mb": it["size_mb"], "np": it["np"], "workers": it["np"] - 1, "iters": it["iters"],
                    "features": it["features"], "k": it["k"], "cores": cores, "heap_mb": heap,
                    "cap_threads": cores, "rep": rep, "driver_idx": drv,
                    "role": (("root" if drv == 0 else "driver") if rec["rss"] else ""),
                    "pid": rec["pid"], "wall_s": f"{wall:.2f}", "exit_code": code, "n_samples": len(rec["cpu"]),
                    "cpu_avg_pct": rb.fmt(rb.mean(rec["cpu"]), 1), "cpu_peak_pct": rb.fmt(rb.mx(rec["cpu"]), 1),
                    "cpu_p95_pct": rb.fmt(rb.pctl(rec["cpu"], 95), 1),
                    "rss_avg_mb": rb.fmt(rb.mean(rec["rss"])), "rss_peak_mb": rb.fmt(rb.mx(rec["rss"])),
                    "rss_p95_mb": rb.fmt(rb.pctl(rec["rss"], 95)),
                    "sys_cpu_avg_pct": rb.fmt(scpu, 1), "sys_mem_avg_mb": rb.fmt(smem),
                    "sys_mem_peak_mb": rb.fmt(smem_peak), "sys_mem_avail_min_mb": rb.fmt(savail),
                    "cores_alloc": cores_alloc, "heap_mb_alloc": heap_alloc, "iters_logged": iters_logged,
                    "data_file": os.path.relpath(it["data"]), "trace_file": trace_path if stats else "",
                    "set": it["set"], "kind": it["kind"], "mem_factor": "" if it["factor"] is None else it["factor"],
                    "mem_limit_mb": "" if it["limit"] is None else it["limit"]})
            f.flush()
            elapsed.append(wall)
            eta = (sum(elapsed) / len(elapsed)) * (len(todo) - k_)
            print(f"[{datetime.now():%H:%M:%S}] DONE {k_}/{len(todo)}  {run_id}  wall={wall:.1f}s exit={code}  "
                  f"ETA<={eta / 3600:.1f}h", flush=True)
            if code == 125 and attempt == 1:
                print("    retrying once after a stall", flush=True)
                time.sleep(rb.SETTLE_S)
                continue
            break
        if code == 124:
            timed_out.add(cell)
        if it["set"] == "E3":
            if code not in (0, 124, 125):
                print(f"    {it['wid']} at x{it['factor']:g} of rec_mem died (exit {code}) -- recorded as the cliff; "
                      f"lower factors of {it['wid']} will be skipped", flush=True)
                failed_cells.add(cell)
                dead_factor[it["wid"]] = max(dead_factor.get(it["wid"], 0), it["factor"])
        else:
            consec = 0 if code in (0, 124, 125) else consec + 1
            if consec >= MAX_CONSEC_FAILS:
                print(f"[!] {consec} consecutive failed runs -- aborting. Check {log_path}", flush=True)
                f.close()
                sys.exit(3)
        time.sleep(rb.SETTLE_S)
    f.close()
    print(f"ALL DONE. rows in {a.out}\nnext: python scripts/doe_holdout.py --summarize {a.out} --d-csv results/doe_rightsize.csv")


if __name__ == "__main__":
    main()
