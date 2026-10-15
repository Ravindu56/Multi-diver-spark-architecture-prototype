#!/usr/bin/env python
# =============================================================================
# scripts/kmeans_memory_test.py
# Phase 6 — step A2: can K-Means memory be predicted better?
#
# PROTOCOL (fixed before this test was run)
# -----------------------------------------
# Targets     mem_mb_p95_max, mem_mb_peak_max (largest driver per run)
# Baseline    ridge_wl — the deployed memory model (alpha from the model card)
# Candidates  C1 rf_wl         forest per workload (card selection)
#             C2 ridge_log_wl  per-workload Ridge on log(memory): power law
#             C3 ridge_int_wl  per-workload Ridge with pairwise interactions
#             (C2, C3: alpha chosen from models.RIDGE_ALPHAS by the same CV)
# Test        leave-one-configuration-out CV on TRAIN+VAL (frozen test split
#             untouched), scored on K-Means rows
# Rule        per target, a candidate replaces ridge_wl FOR K-MEANS ONLY if it
#             beats it on K-Means rows (paired group-bootstrap 95 % CI of the
#             MAE difference below 0); if several qualify, the lowest K-Means
#             MAE wins.  LogReg memory keeps ridge_wl (at the noise limit).
# Then        test split and E1 K-Means rows as confirmation only.
#
# AMENDMENT 1 (added after the first run, before re-running)
# -----------------------------------------------------------
# The confirmation showed a candidate (pairwise interactions) that wins in
# range but extrapolates badly beyond the measured data per driver — a gap
# in the rule, which looked at in-range jobs only.  Added criterion,
# evaluated on TRAIN+VAL only:
#   Extrapolation check  hold out K-Means rows with more than 50 MB per
#             driver, train on everything else (LogReg of every size stays),
#             predict the held-out K-Means rows.
#   Rule      a candidate qualifies only if it beats ridge_wl in range (as
#             above) AND is not clearly worse in the extrapolation check
#             (95 % CI lower bound of the MAE difference <= 0).
# The test split and E1 have been seen for this decision, so they are no
# longer a blind confirmation; they are reported for completeness.
#
# Output:  data/models/kmeans_memory/  (git-ignored) — report.json etc.
#
#   python scripts/kmeans_memory_test.py
# =============================================================================

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

import pandas as pd

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from mpj_spark.resource_prediction import models as m  # noqa: E402

TRAINING_SET = "capped"
TARGETS = ["mem_mb_p95_max", "mem_mb_peak_max"]
WORKLOAD = "kmeans"
BASELINE = "ridge_wl"
EXTRAP_SHARD_MB = 50.0  # amendment 1: K-Means rows above this are held out


def _parse_args(argv=None):
    p = argparse.ArgumentParser(description="K-Means memory model test (step A2).")
    p.add_argument(
        "--drivers", default=os.path.join(_PROJECT_ROOT, "data", "local", "local_drivers.csv")
    )
    p.add_argument(
        "--model-card",
        default=os.path.join(_PROJECT_ROOT, "data", "models", "local_capped", "model_card.json"),
    )
    p.add_argument(
        "--out-dir", default=os.path.join(_PROJECT_ROOT, "data", "models", "kmeans_memory")
    )
    p.add_argument("--rf-trees", type=int, default=200)
    p.add_argument("--n-boot", type=int, default=2000)
    return p.parse_args(argv)


def candidates(card: dict, target: str, rf_trees: int) -> dict:
    """Baseline + candidate factories; C2 / C3 over the alpha grid."""
    num = m.TRAINING_SETS[TRAINING_SET]["numeric"]
    sel = card["choices"][target]["selected"]
    alpha = float(sel["ridge_wl"].split("|")[1])
    _, depth, leaf = sel["rf_wl"].split("|")
    out = {
        BASELINE: lambda: m.make_ridge_wl(alpha, num),
        "rf_wl": lambda: m.make_rf_wl(int(depth), int(leaf), n_estimators=rf_trees),
    }
    for a in m.RIDGE_ALPHAS:
        out[f"ridge_log_wl|{a}"] = lambda a=a: m.make_ridge_log_wl(a, num)
        out[f"ridge_int_wl|{a}"] = lambda a=a: m.make_ridge_int_wl(a, num)
    return out


def _mae(oof: pd.DataFrame) -> pd.Series:
    return oof.assign(ae=(oof["predicted"] - oof["actual"]).abs()).groupby("model")["ae"].mean()


def extrapolation_check(d, facs, keys, target, n_boot) -> dict:
    """Amendment 1: hold out K-Means rows above EXTRAP_SHARD_MB per driver."""
    features = m.features_for(TRAINING_SET)
    shard = d["dataset_mb"] / d["num_workers"]
    held = (d["workload_type"] == WORKLOAD) & (shard > EXTRAP_SHARD_MB)
    train, ev = d[~held], d[held]
    frames = []
    for fam, key in keys.items():
        pred = facs[key]().fit(train[features], train[target]).predict(ev[features])
        frames.append(
            pd.DataFrame(
                {
                    "config_group": ev["config_group"].values,
                    "workload_type": WORKLOAD,
                    "model": fam,
                    "actual": ev[target].values,
                    "predicted": pred,
                }
            )
        )
    long = pd.concat(frames, ignore_index=True)
    return {
        "rows": len(ev),
        "groups": int(ev["config_group"].nunique()),
        "mae": _mae(long).to_dict(),
        "paired": {
            f"{fam}_vs_{BASELINE}": m.group_bootstrap(long, fam, baseline=BASELINE, n_boot=n_boot)
            for fam in keys
            if fam != BASELINE
        },
    }


def run_target(trval, card, target, args) -> dict:
    features = m.features_for(TRAINING_SET)
    d = trval[trval[target].notna()]
    oof = m.group_cv_predictions(
        d, target, candidates(card, target, args.rf_trees), features=features
    )
    km = oof[oof["workload_type"] == WORKLOAD]
    mae = _mae(km)
    # alpha of C2 / C3 chosen by the same K-Means CV MAE (stated in the protocol)
    chosen = {BASELINE: BASELINE, "rf_wl": "rf_wl"}
    for fam in ("ridge_log_wl", "ridge_int_wl"):
        keys = [k for k in mae.index if k.startswith(fam + "|")]
        chosen[fam] = mae[keys].idxmin()
    km = km[km["model"].isin(chosen.values())].replace({"model": {v: k for k, v in chosen.items()}})
    paired = {
        f"{fam}_vs_{BASELINE}": m.group_bootstrap(km, fam, baseline=BASELINE, n_boot=args.n_boot)
        for fam in chosen
        if fam != BASELINE
    }
    lr = oof[(oof["workload_type"] != WORKLOAD) & oof["model"].isin(chosen.values())]
    lr = lr.replace({"model": {v: k for k, v in chosen.items()}})
    extrap = extrapolation_check(
        d, candidates(card, target, args.rf_trees), chosen, target, args.n_boot
    )
    winners = [
        fam
        for fam in chosen
        if fam != BASELINE
        and paired[f"{fam}_vs_{BASELINE}"]["mae_diff_ci95"][1] < 0
        and extrap["paired"][f"{fam}_vs_{BASELINE}"]["mae_diff_ci95"][0] <= 0
    ]
    k_mae = _mae(km)
    decision = min(winners, key=lambda f: k_mae[f]) if winners else BASELINE
    return {
        "chosen_keys": chosen,
        "kmeans": {
            fam: m.regression_metrics(g["actual"], g["predicted"]) for fam, g in km.groupby("model")
        },
        "logreg_info": {
            fam: m.regression_metrics(g["actual"], g["predicted"]) for fam, g in lr.groupby("model")
        },
        "paired": paired,
        "extrapolation_check": extrap,
        "qualified": winners,
        "decision": decision,
    }


def confirm(trval, raw, card, target, keys, args) -> list[dict]:
    """Test split and E1 K-Means rows, models refit on train+val (confirmation only)."""
    features = m.features_for(TRAINING_SET)
    facs = candidates(card, target, args.rf_trees)
    d = trval[trval[target].notna()]
    rows = []
    sets = {
        "test": raw[raw["split"] == "test"],
        "E1 interior": raw[(raw["experiment"] == "E1") & (raw["holdout_kind"] == "interior")],
        "E1 extrap": raw[(raw["experiment"] == "E1") & (raw["holdout_kind"] == "extrap")],
    }
    fitted = {fam: facs[key]().fit(d[features], d[target]) for fam, key in keys.items()}
    for name, rows_df in sets.items():
        h = rows_df[(rows_df["status"] == "ok") & (rows_df["workload_type"] == WORKLOAD)]
        h = m.add_derived(h[h[target].notna()])
        if len(h) < 2:
            continue
        for fam, mdl in fitted.items():
            met = m.regression_metrics(h[target], mdl.predict(h[features]))
            rows.append({"target": target, "set": name, "model": fam} | met)
    return rows


def main(argv=None):
    args = _parse_args(argv)
    with open(args.model_card) as fh:
        card = json.load(fh)
    raw = pd.read_csv(args.drivers)
    df, rows_info = m.select_training_rows(raw, TRAINING_SET)
    trval = df[df["split"].isin(["train", "val"])]
    os.makedirs(args.out_dir, exist_ok=True)

    results, confirmation = {}, []
    for target in TARGETS:
        results[target] = run_target(trval, card, target, args)
        confirmation += confirm(trval, raw, card, target, results[target]["chosen_keys"], args)
    pd.DataFrame(confirmation).to_csv(os.path.join(args.out_dir, "confirmation.csv"), index=False)
    with open(args.drivers, "rb") as fh:
        sha = hashlib.sha256(fh.read()).hexdigest()
    report = {
        "question": "Can K-Means memory be predicted better than by ridge_wl?",
        "protocol": "see script header (fixed before this test; amendment 1 after the first run)",
        "data": {"drivers_sha256": sha, "rows": rows_info},
        "results": results,
        "decision": {t: r["decision"] for t, r in results.items()},
    }
    with open(os.path.join(args.out_dir, "report.json"), "w") as fh:
        json.dump(report, fh, indent=2, default=str)

    for target, r in results.items():
        print(f"── {target} — K-Means rows, leave-one-configuration-out CV ──")
        for fam, met in r["kmeans"].items():
            info = r["logreg_info"].get(fam, {})
            print(
                f"   {fam:13s} ({r['chosen_keys'][fam]:18s}) K-Means MAE {met['mae']:7.1f} MB  "
                f"R2 {met['r2']:.3f}   | LogReg MAE {info.get('mae', float('nan')):7.1f} (info)"
            )
        for name, p in r["paired"].items():
            lo, hi = p["mae_diff_ci95"]
            print(f"   {name:24s} {p['mae_diff_vs_baseline']:+7.1f} MB [{lo:+.1f}, {hi:+.1f}]")
        x = r["extrapolation_check"]
        print(f"   extrapolation check ({x['rows']} K-Means rows, {x['groups']} groups held out):")
        for name, p in x["paired"].items():
            lo, hi = p["mae_diff_ci95"]
            fam = name.split("_vs_")[0]
            print(
                f"     {name:24s} MAE {x['mae'][fam]:7.1f} vs {x['mae'][BASELINE]:7.1f} MB   "
                f"diff {p['mae_diff_vs_baseline']:+7.1f} [{lo:+.1f}, {hi:+.1f}]"
            )
        print(f"   → decision for K-Means: {r['decision']}  (qualified: {r['qualified']})")
    print("── confirmation only: K-Means rows of the test split and E1 ──")
    cf = pd.DataFrame(confirmation)
    print(
        cf.pivot_table(index=["target", "set"], columns="model", values="mae", sort=False)
        .round(1)
        .to_string()
    )
    print(f"[kmeans memory] outputs written to {args.out_dir}")
    return report


if __name__ == "__main__":
    main()
