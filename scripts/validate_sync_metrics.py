#!/usr/bin/env python3
"""scripts/validate_sync_metrics.py -- gate for per-iteration sync-mode logs (logreg).

Checks per (run_id, worker): method labelled, rounds contiguous, weights actually
moving after round 1 (stale/refit-from-scratch detector), workers agree on the
global model, final model comparable across methods, and group balance.
Exit code 1 when any FAIL is found.

  python scripts/validate_sync_metrics.py results/iter_metrics_clean.csv --out results/sync_validation
"""
import argparse, os, sys
import numpy as np
import pandas as pd

MOVE_TOL = 1e-6      # |delta| below this after round 1 => "not moving"
NORM_REL_TOL = 0.05  # final-norm spread allowed across methods (same data/params)
MIN_RUNS = 5         # repetitions needed per (method, workers) cell

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv"); ap.add_argument("--out", default="results/sync_validation")
    ap.add_argument("--reference", default=None, help="method whose final norm is the reference")
    a = ap.parse_args()
    df = pd.read_csv(a.csv)
    need = {"run_id","worker_id","iteration","weight_norm","weight_delta","method","num_workers"}
    if need - set(df.columns):
        sys.exit(f"missing columns: {sorted(need - set(df.columns))}")
    os.makedirs(a.out, exist_ok=True)
    findings = []
    def add(level, check, scope, detail):
        findings.append(dict(level=level, check=check, scope=scope, detail=detail))

    tr = df.sort_values(["run_id","worker_id","iteration"]).groupby(["run_id","worker_id"])
    rows = []
    for (rid, wid), g in tr:
        d = g.iloc[1:]
        rows.append(dict(run_id=rid, worker_id=wid, method=g.method.iloc[0], nw=g.num_workers.iloc[0],
            rounds=len(g), contiguous=bool((g.iteration.values == np.arange(1, len(g)+1)).all()),
            moving=bool((d.weight_delta.abs() > MOVE_TOL).any()) if len(d) else False,
            final_norm=g.weight_norm.iloc[-1],
            late_time_ratio=(g.iter_time_s.iloc[-3:].median()/g.iter_time_s.iloc[0]) if "iter_time_s" in g and len(g)>3 else np.nan))
    t = pd.DataFrame(rows); t.to_csv(f"{a.out}/trace_summary.csv", index=False)

    unl = t[t.method.isin(["legacy_unspecified","", "nan"]) | t.method.isna()]
    if len(unl): add("WARN","method_label","legacy_unspecified",f"{unl.run_id.nunique()} runs have no method; exclude from method comparison")
    for m, g in t.groupby("method"):
        if m == "legacy_unspecified": continue
        frac = 1 - g.moving.mean()
        add("FAIL" if frac > 0.5 else "OK", "weights_moving_after_round1", m,
            f"{frac:.0%} of {len(g)} traces have zero weight change after round 1")
        if not g.contiguous.all():
            add("FAIL","contiguous_rounds",m,f"{(~g.contiguous).sum()} traces have gaps/duplicates")
    # workers agree on global model each round
    d = df.groupby(["run_id","iteration"]).weight_norm.agg(lambda s: s.max()-s.min())
    bad = d[d > 1e-4]
    mm = df.drop_duplicates("run_id").set_index("run_id").method
    sync_modes = {"ps_sync_fedavg_mpi","ps_sync_fedavg_queue","hybrid_ps_allreduce"}
    bad_sync = [(r,i) for (r,i) in bad.index if mm.get(r) in sync_modes]
    add("FAIL" if bad_sync else "OK","workers_agree","sync modes",f"{len(bad_sync)} (run,round) pairs disagree across workers")
    # final model comparable across methods
    fin = t[t.method != "legacy_unspecified"].groupby("method").final_norm.median()
    ref = fin[a.reference] if a.reference in fin.index else fin.median()
    for m, v in fin.items():
        rel = abs(v-ref)/ref
        add("FAIL" if rel > NORM_REL_TOL else "OK","final_model_comparable",m,f"median final ||w||={v:.4f} vs reference {ref:.4f} ({rel:.0%})")
    # within-method drift across time (same method, different final model)
    rr = t.drop_duplicates("run_id")
    for m, g in rr[rr.method != "legacy_unspecified"].groupby("method"):
        if g.final_norm.round(2).nunique() > 1:
            add("WARN","final_model_changed_over_time",m,f"final ||w|| takes values {sorted(g.final_norm.round(3).unique())}; check commit/solver per run")
    # balance
    cnt = rr[rr.method != "legacy_unspecified"].groupby(["method","nw"]).run_id.nunique()
    for (m,nw),n in cnt.items():
        if n < MIN_RUNS: add("WARN","min_repetitions",f"{m}/nw={nw}",f"{n} runs < {MIN_RUNS}")
    f = pd.DataFrame(findings); f.to_csv(f"{a.out}/findings.csv", index=False)
    print(f.to_string(index=False))
    n_fail = (f.level=="FAIL").sum()
    print(f"\n{n_fail} FAIL, {(f.level=='WARN').sum()} WARN")
    sys.exit(1 if n_fail else 0)

if __name__ == "__main__":
    main()
