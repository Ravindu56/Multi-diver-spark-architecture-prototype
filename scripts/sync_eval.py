#!/usr/bin/env python3
"""scripts/sync_eval.py -- controlled evaluation of the LogReg sync modes.

Question answered: which synchronisation mode reaches a given model quality
fastest?  Equal *rounds* are not equal *work* (FedAvg stops early on loss,
gossip / async-PS / hybrid always run --logreg-iter rounds, async converges
slower per round), so modes are compared by TIME TO TARGET LOSS, not by
end-to-end wall time.

Modes (all warm-started, same objective: mean log-loss + 0.5*reg*||w||^2):
  ps_sync_fedavg_mpi, gossip, ps_async, hybrid_ps_allreduce
(``none`` and the queue FedAvg write a different metrics schema and are not
evaluated here.  K-Means is not wired for these modes yet.)

Per run it reads the workers' metrics CSVs (written by the mode itself) and
derives, per round:
  loss_after_round[t] = loss of the model that ENTERS round t+1
      - FedAvg: exact global loss (``global_loss`` column)
      - others: row-weighted mean of the workers' ``loss_start`` (each worker's
        own model on its own partition; for async/gossip an approximation)
  elapsed[t]          = cumulative mean per-round time over workers
  sync time split (gossip: gossip_time_s; hybrid: allreduce_time_s + ps_time_s):
      transfer ~ the smallest sync time among workers in that round (the
      straggler arrives last and does not wait); wait_w = sync_w - transfer.
      FedAvg / async CSVs carry no sync timing -> reported as NaN.

  python scripts/sync_eval.py plan
  python scripts/sync_eval.py run    --pybin "$(which python)" --reps 5 --workers 2
  python scripts/sync_eval.py run    --smoke      # 1 rep of each mode
  python scripts/sync_eval.py select --out-dir results/sync_eval
"""
import argparse, csv, glob, hashlib, itertools, os, random, shlex, signal, socket, subprocess, sys, time
from datetime import datetime

import numpy as np
import pandas as pd

MODES = ["ps_sync_fedavg_mpi", "gossip", "ps_async", "hybrid_ps_allreduce"]
METRIC_FILES = {
    "ps_sync_fedavg_mpi": ("worker_*_fedavg_mpi_metrics.csv", []),
    "gossip": ("worker_*_gossip_metrics.csv", ["gossip_time_s"]),
    "ps_async": ("worker_*_async_ps_metrics.csv", []),
    "hybrid_ps_allreduce": ("worker_*_hybrid_metrics.csv", ["allreduce_time_s", "ps_time_s"]),
}
RUN_FIELDS = ["run_id", "timestamp", "host", "commit", "dataset_sha256", "mode", "workers", "cores",
              "rounds_cfg", "rep", "seed", "exit_code", "wall_s", "rounds_logged", "loss_after_last",
              "mean_round_s", "transfer_s", "wait_s", "wait_share", "run_dir"]
ROUND_FIELDS = ["run_id", "mode", "workers", "round", "elapsed_s", "loss_after"]


# ───────────────────────── pure helpers (unit-tested) ─────────────────────────
def load_rounds(run_dir, mode):
    """Long DataFrame of the workers' per-round rows for one run (empty if none)."""
    pattern, sync_cols = METRIC_FILES[mode]
    frames = []
    for p in sorted(glob.glob(os.path.join(run_dir, pattern))):
        df = pd.read_csv(p)
        if df.empty:
            continue
        out = pd.DataFrame({
            "worker": df["worker_id"], "round": df["iteration"], "iter_s": df["iter_time_s"],
            "rows": df["row_count"],
            "loss": pd.to_numeric(df["global_loss"] if "global_loss" in df else df["loss_start"],
                                  errors="coerce"),
        })
        out["sync_s"] = df[sync_cols].sum(axis=1) if sync_cols else np.nan
        out["exact"] = "global_loss" in df
        frames.append(out)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _row_weighted(sub):
    """Row-weighted mean of the workers' losses in one round (NaN-safe)."""
    ok = sub["loss"].notna()
    if not ok.any():
        return float("nan")
    return float(np.average(sub.loc[ok, "loss"], weights=sub.loc[ok, "rows"]))


def summarise_run(df):
    """(rounds DataFrame, summary dict) for one run's long frame."""
    if df.empty:
        return pd.DataFrame(columns=["round", "elapsed_s", "loss_after"]), {}
    g = df.groupby("round")
    per = pd.DataFrame({"round_s": g["iter_s"].mean()})
    if bool(df["exact"].iloc[0]):
        per["loss_enter"] = g["loss"].first()
    else:
        per["loss_enter"] = pd.Series({r: _row_weighted(sub) for r, sub in df.groupby("round")})
    per["elapsed_s"] = per["round_s"].cumsum()
    rounds = per.index.to_numpy()
    after = pd.DataFrame({
        "round": rounds[:-1],
        "elapsed_s": per["elapsed_s"].to_numpy()[:-1],
        "loss_after": per["loss_enter"].to_numpy()[1:],
    })
    summary = {"rounds_logged": int(len(per)), "mean_round_s": float(per["round_s"].mean()),
               "loss_after_last": float(after["loss_after"].iloc[-1]) if len(after) else np.nan}
    if df["sync_s"].notna().any():
        transfer = df.groupby("round")["sync_s"].min()
        wait = (df["sync_s"] - df["round"].map(transfer)).groupby(df["round"]).mean()
        total = float(per["round_s"].sum())
        summary.update(transfer_s=float(transfer.sum()), wait_s=float(wait.sum()),
                       wait_share=float(wait.sum() / total) if total > 0 else np.nan)
    else:
        summary.update(transfer_s=np.nan, wait_s=np.nan, wait_share=np.nan)
    return after, summary


def time_to_target(after, target):
    """Elapsed seconds at the first round whose resulting model meets the target (NaN if never)."""
    hit = after[after["loss_after"] <= target]
    return float(hit["elapsed_s"].iloc[0]) if len(hit) else float("nan")


def pick_target(rounds, rel, abs_floor=1e-6):
    """Quality target per worker count: best median-over-reps best loss, plus a tolerance."""
    best = rounds.groupby(["workers", "mode", "run_id"])["loss_after"].min().reset_index()
    ref = best.groupby(["workers", "mode"])["loss_after"].median().groupby("workers").min()
    return {int(w): float(v + max(abs_floor, rel * abs(v))) for w, v in ref.items()}


def _median(series):
    """Median ignoring NaN; NaN (no warning) when nothing is left."""
    vals = series.dropna()
    return float(vals.median()) if len(vals) else float("nan")


def iqr(xs):
    s = sorted(xs)
    n = len(s)
    return s[int(0.25 * (n - 1))], s[int(round(0.75 * (n - 1)))]


def select_best(runs, rounds, rel=1e-3, min_reps=3):
    """Per worker count: mode table + the fastest mode that reaches the target."""
    ok = runs[runs["exit_code"] == 0]
    targets = pick_target(rounds[rounds["run_id"].isin(ok["run_id"])], rel)
    rows, verdicts = [], []
    for w, tgt in sorted(targets.items()):
        table = {}
        for mode in MODES:
            rr = ok[(ok["mode"] == mode) & (ok["workers"] == w)]
            if rr.empty:
                continue
            ttt = []
            for rid in rr["run_id"]:
                after = rounds[rounds["run_id"] == rid].sort_values("round")
                ttt.append(time_to_target(after, tgt))
            reached = [t for t in ttt if not np.isnan(t)]
            table[mode] = dict(n_ok=len(rr), n_reached=len(reached), ttt=reached,
                               wall=_median(rr["wall_s"]),
                               final=_median(rr["loss_after_last"]),
                               wait=_median(rr["wait_share"]))
            t = table[mode]
            rows.append(dict(workers=w, mode=mode, n_ok=t["n_ok"], n_reached=t["n_reached"],
                             median_ttt_s=float(np.median(reached)) if reached else np.nan,
                             ttt_iqr=("" if not reached else "%.1f-%.1f" % iqr(reached)),
                             median_wall_s=t["wall"], median_final_loss=t["final"],
                             median_wait_share=t["wait"], target_loss=tgt))
        cand = {m: t for m, t in table.items() if t["n_reached"] >= min_reps}
        if not cand:
            verdicts.append((w, tgt, "no mode reached the target in >= %d runs" % min_reps))
            continue
        ranked = sorted(cand.items(), key=lambda kv: float(np.median(kv[1]["ttt"])))
        win, ws = ranked[0]
        if len(ranked) == 1:
            sep = "(only candidate)"
        else:
            nxt = ranked[1]
            sep = ("SEPARABLE" if iqr(ws["ttt"])[1] < iqr(nxt[1]["ttt"])[0]
                   else "NOT separable from " + nxt[0])
        verdicts.append((w, tgt, "best=%s  median time-to-target %.1fs  %s" %
                         (win, float(np.median(ws["ttt"])), sep)))
    return pd.DataFrame(rows), verdicts


# ───────────────────────── orchestration ─────────────────────────
def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()[:16]


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except Exception:
        return ""


def grid(workers, reps, seed):
    cells = [(m, w, r) for m, w, r in itertools.product(MODES, workers, range(1, reps + 1))]
    random.Random(seed).shuffle(cells)  # randomised order: no mode owns a time window
    return cells


def run_id(mode, w, cores, rep):
    return f"logreg_{mode}_w{w}_c{cores}_r{rep}"


def build_cmd(mode, w, cores, input_path, pybin, run_dir, rounds, features, caps, mpirun="mpirun"):
    cmd = [mpirun, "--oversubscribe", "-np", str(w + 1)]
    for var in caps:
        cmd += ["-x", var]
    return cmd + [pybin, "-m", "mpj_spark.core.main_mpi", "--app", "logreg", "--sync-mode", mode,
                  "--input", input_path, "--cores", str(cores), "--results-dir", run_dir,
                  "--logreg-iter", str(rounds), "--logreg-features", str(features)]


def done_ids(path):
    s = set()
    if os.path.exists(path):
        with open(path, newline="") as f:
            for r in csv.DictReader(f):
                if r["exit_code"] == "0":
                    s.add(r["run_id"])
    return s


def launch(cmd, log_path, timeout, repo):
    t0 = time.perf_counter()
    with open(log_path, "w") as lf:
        lf.write("CMD: " + " ".join(shlex.quote(c) for c in cmd) + "\n")
        lf.flush()
        p = subprocess.Popen(cmd, cwd=repo, stdout=lf, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            code = p.wait(timeout=timeout or None)
        except subprocess.TimeoutExpired:
            os.killpg(p.pid, signal.SIGKILL)
            p.wait()
            code = 124
        except KeyboardInterrupt:
            os.killpg(p.pid, signal.SIGTERM)
            p.wait()
            raise
    return code, time.perf_counter() - t0


def cmd_plan(a):
    cells = grid(a.workers, a.reps, a.seed)
    print(f"{len(cells)} runs: modes={MODES} workers={a.workers} reps={a.reps}")
    for m, w, r in cells[:10]:
        print("  ", m, f"w={w}", f"rep={r}")


def cmd_run(a):
    if a.app != "logreg":
        sys.exit("only --app logreg is wired for these four modes (K-Means has Allreduce/FedAvg only)")
    if not os.path.isfile(a.input):
        sys.exit(f"missing input: {a.input}")
    caps = ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]
    for c in caps:
        os.environ[c] = str(a.cores)
    os.makedirs(a.out_dir, exist_ok=True)
    runs_csv, rounds_csv = os.path.join(a.out_dir, "runs.csv"), os.path.join(a.out_dir, "rounds.csv")
    sha, commit, repo = sha256(a.input), git_commit(), os.getcwd()
    cells = grid(a.workers, 1 if a.smoke else a.reps, a.seed)
    done = done_ids(runs_csv)
    new_runs, new_rounds = not os.path.exists(runs_csv), not os.path.exists(rounds_csv)
    with open(runs_csv, "a", newline="") as fr, open(rounds_csv, "a", newline="") as fo:
        wr, wo = csv.DictWriter(fr, fieldnames=RUN_FIELDS), csv.DictWriter(fo, fieldnames=ROUND_FIELDS)
        if new_runs:
            wr.writeheader()
        if new_rounds:
            wo.writeheader()
        for i, (mode, wk, rep) in enumerate(cells, 1):
            rid = run_id(mode, wk, a.cores, rep)
            if rid in done:
                continue
            rdir = os.path.abspath(os.path.join(a.out_dir, "runs", rid))
            os.makedirs(rdir, exist_ok=True)
            cmd = build_cmd(mode, wk, a.cores, a.input, a.pybin, rdir, a.rounds, a.features, caps, a.mpirun)
            print(f"[{datetime.now():%H:%M:%S}] {i}/{len(cells)} {rid}", flush=True)
            code, wall = launch(cmd, os.path.join(rdir, "run.log"), a.timeout, repo)
            after, s = summarise_run(load_rounds(rdir, mode))
            wr.writerow(dict(run_id=rid, timestamp=datetime.now().isoformat(timespec="seconds"),
                             host=socket.gethostname(), commit=commit, dataset_sha256=sha, mode=mode,
                             workers=wk, cores=a.cores, rounds_cfg=a.rounds, rep=rep, seed=a.seed,
                             exit_code=code, wall_s=f"{wall:.2f}", run_dir=rdir,
                             **{k: s.get(k, "") for k in ("rounds_logged", "loss_after_last", "mean_round_s",
                                                          "transfer_s", "wait_s", "wait_share")}))
            for _, row in after.iterrows():
                wo.writerow(dict(run_id=rid, mode=mode, workers=wk, round=int(row["round"]),
                                 elapsed_s=row["elapsed_s"], loss_after=row["loss_after"]))
            fr.flush()
            fo.flush()
            print(f"    exit={code} wall={wall:.1f}s rounds={s.get('rounds_logged', 0)} "
                  f"loss_after_last={s.get('loss_after_last', float('nan')):.6f}", flush=True)
            time.sleep(a.settle)


def cmd_select(a):
    runs = pd.read_csv(os.path.join(a.out_dir, "runs.csv"))
    rounds = pd.read_csv(os.path.join(a.out_dir, "rounds.csv"))
    table, verdicts = select_best(runs, rounds, rel=a.rel_tol, min_reps=a.min_reps)
    table.to_csv(os.path.join(a.out_dir, "mode_table.csv"), index=False)
    print(table.round(4).to_string(index=False))
    print()
    for w, tgt, msg in verdicts:
        print(f"[workers={w}] target loss {tgt:.6f}: {msg}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = p.add_subparsers(dest="cmd", required=True)
    s = sp.add_parser("plan")
    s.add_argument("--workers", nargs="+", type=int, default=[2])
    s.add_argument("--reps", type=int, default=5)
    s.add_argument("--seed", type=int, default=7)
    s.set_defaults(f=cmd_plan)
    s = sp.add_parser("run")
    s.add_argument("--app", default="logreg")
    s.add_argument("--workers", nargs="+", type=int, default=[2])
    s.add_argument("--reps", type=int, default=5)
    s.add_argument("--seed", type=int, default=7)
    s.add_argument("--pybin", default=sys.executable)
    s.add_argument("--mpirun", default="mpirun")
    s.add_argument("--input", default="shared_storage/logreg_100mb.csv")
    s.add_argument("--cores", type=int, default=2)
    s.add_argument("--rounds", type=int, default=10)
    s.add_argument("--features", type=int, default=10)
    s.add_argument("--timeout", type=int, default=1800)
    s.add_argument("--settle", type=int, default=10)
    s.add_argument("--out-dir", default="results/sync_eval")
    s.add_argument("--smoke", action="store_true", help="1 rep of every mode to check wiring")
    s.set_defaults(f=cmd_run)
    s = sp.add_parser("select")
    s.add_argument("--out-dir", default="results/sync_eval")
    s.add_argument("--rel-tol", type=float, default=1e-3, help="target = best loss * (1 + rel-tol)")
    s.add_argument("--min-reps", type=int, default=3)
    s.set_defaults(f=cmd_select)
    a = p.parse_args()
    a.f(a)


if __name__ == "__main__":
    main()
