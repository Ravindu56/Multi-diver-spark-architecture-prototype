#!/usr/bin/env python3
"""scripts/compare_baseline.py -- multi-driver sync modes vs the FAIR single-driver baseline.

The baseline (scripts/baseline_eval.py) is the same framework code path, algorithm and
kernels run with ONE driver and the same TOTAL cores as the W-driver run it is compared
with (W drivers x c cores -> 1 driver x W*c cores), same data and round budget.  Runs are
matched on (workers, cores per driver, rounds).

K-Means: per (workers, cores, rounds) and mode -> round_speedup (baseline mean round time
         over the mode's), wall_speedup (also counts JVM and partition start-up) and the
         final Total WCSS relative to the baseline's.
LogReg:  uses sync_eval's time-to-target: the target loss comes from the multi-driver runs
         of that worker count, then BOTH sides are timed to it ->
         speedup = baseline median time / mode median time.

Speedup > 1 means the multi-driver mode beats the single driver with equal resources.

  python scripts/compare_baseline.py --app kmeans
  python scripts/compare_baseline.py --app logreg --rel-tol 1e-3
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sync_eval as se  # noqa: E402


def default_dirs(app):
    if app == "kmeans":
        return "results/sync_eval_kmeans_v2", "results/baseline_kmeans_v2"
    return "results/sync_eval_logreg_v2", "results/baseline_logreg_v2"


def _ratio(num, den):
    ok = pd.notna(num) and pd.notna(den) and den
    return float(num / den) if ok else float("nan")


def _med_list(xs):
    return float(np.median(xs)) if xs else float("nan")


def _baseline_rows(base_runs, cores_total, rounds_cfg):
    """Successful single-driver runs with the matching total core budget and round budget."""
    return base_runs[
        (base_runs["exit_code"] == 0)
        & (base_runs["workers"] == 1)
        & (base_runs["cores"] == cores_total)
        & (base_runs["rounds_cfg"] == rounds_cfg)
    ]


def compare_kmeans(fw_runs, base_runs):
    fw = fw_runs[fw_runs["exit_code"] == 0]
    rows = []
    for (w, c, rc), sub in sorted(fw.groupby(["workers", "cores", "rounds_cfg"])):
        if w <= 1:
            continue
        b = _baseline_rows(base_runs, c * w, rc)
        if b.empty:
            continue
        b_round = se._median(b["mean_round_s"])
        b_wall = se._median(b["wall_s"])
        b_wcss = se._median(b["final_wcss"])
        for mode in se.MODES:
            rr = sub[sub["mode"] == mode]
            if rr.empty:
                continue
            r = se._median(rr["mean_round_s"])
            wl = se._median(rr["wall_s"])
            wc = se._median(rr["final_wcss"])
            rows.append(dict(
                workers=w, cores_per_driver=c, baseline_cores=c * w, rounds_cfg=rc, mode=mode,
                n_ok=len(rr), n_base=len(b),
                round_s=r, base_round_s=b_round, round_speedup=_ratio(b_round, r),
                wall_s=wl, base_wall_s=b_wall, wall_speedup=_ratio(b_wall, wl),
                final_wcss=wc, base_final_wcss=b_wcss,
                wcss_vs_base_pct=(100.0 * (wc / b_wcss - 1.0)
                                  if pd.notna(b_wcss) and b_wcss else float("nan")),
            ))
    return pd.DataFrame(rows)


def _ttts(run_ids, rounds, target):
    out = []
    for rid in run_ids:
        after = rounds[rounds["run_id"] == rid].sort_values("round")
        t = se.time_to_target(after, target)
        if not np.isnan(t):
            out.append(t)
    return out


def compare_logreg(fw_runs, fw_rounds, base_runs, base_rounds, rel=1e-3):
    ok = fw_runs[fw_runs["exit_code"] == 0]
    targets = se.pick_target(fw_rounds[fw_rounds["run_id"].isin(ok["run_id"])], rel)
    rows = []
    for (w, c, rc), sub in sorted(ok.groupby(["workers", "cores", "rounds_cfg"])):
        if w <= 1 or w not in targets:
            continue
        tgt = targets[w]
        b = _baseline_rows(base_runs, c * w, rc)
        if b.empty:
            continue
        b_ttt = _ttts(b["run_id"], base_rounds, tgt)
        b_med = _med_list(b_ttt)
        for mode in se.MODES:
            rr = sub[sub["mode"] == mode]
            if rr.empty:
                continue
            f_ttt = _ttts(rr["run_id"], fw_rounds, tgt)
            f_med = _med_list(f_ttt)
            rows.append(dict(
                workers=w, cores_per_driver=c, baseline_cores=c * w, rounds_cfg=rc, mode=mode,
                target_loss=tgt, n_ok=len(rr), n_reached=len(f_ttt), median_ttt_s=f_med,
                n_base=len(b), base_reached=len(b_ttt), base_median_ttt_s=b_med,
                speedup=_ratio(b_med, f_med),
            ))
    return pd.DataFrame(rows)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--app", choices=["kmeans", "logreg"], required=True)
    ap.add_argument("--framework-dir", default=None)
    ap.add_argument("--baseline-dir", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--rel-tol", type=float, default=1e-3, help="LogReg target tolerance")
    a = ap.parse_args(argv)

    fw_default, base_default = default_dirs(a.app)
    fw_dir = a.framework_dir or fw_default
    base_dir = a.baseline_dir or base_default
    fw_runs = pd.read_csv(os.path.join(fw_dir, "runs.csv"))
    base_runs = pd.read_csv(os.path.join(base_dir, "runs.csv"))
    if a.app == "kmeans":
        table = compare_kmeans(fw_runs, base_runs)
        note = ("round_speedup > 1: the multi-driver mode is faster per round than ONE driver with "
                "the same total cores. wall_speedup also counts JVM/partition start-up. Gossip's "
                "WCSS is a sum over per-worker models and async's is approximate.")
    else:
        fw_rounds = pd.read_csv(os.path.join(fw_dir, "rounds.csv"))
        base_rounds = pd.read_csv(os.path.join(base_dir, "rounds.csv"))
        table = compare_logreg(fw_runs, fw_rounds, base_runs, base_rounds, rel=a.rel_tol)
        note = ("speedup > 1: the mode reaches the target loss sooner than ONE driver with the same "
                "total cores. The target comes from the multi-driver runs; gossip/async losses are "
                "approximations.")
    if table.empty:
        print("no comparable rows: the baseline must be run with --compare-workers and --cores "
              "matching the multi-driver runs, and the same --rounds")
        return 1
    out = a.out or f"results/compare_{a.app}_v2.csv"
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    table.to_csv(out, index=False)
    print(table.round(4).to_string(index=False))
    print()
    print(note)
    return 0


if __name__ == "__main__":
    sys.exit(main())
