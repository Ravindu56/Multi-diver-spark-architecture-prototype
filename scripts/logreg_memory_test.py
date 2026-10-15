#!/usr/bin/env python
# =============================================================================
# scripts/logreg_memory_test.py
# Phase 6 — step A2b: can LogReg memory be predicted better than by ridge_wl?
#
# WHY
# ---
# Step A2 (scripts/kmeans_memory_test.py) tested other memory models for
# K-Means only.  The A4 evaluation then showed, in the model card's CV
# metrics, that the per-workload power law ridge_log_wl also has a lower
# out-of-fold MAE on LogReg memory (p95 212 vs 273 MB, peak 204 vs 262 MB).
# That in-range CV result was SEEN before this protocol was written; the
# beyond-range checks below were not.
#
# PROTOCOL (fixed before this test was run) — step A2 with its amendment 1,
# applied to LogReg, plus a cores check
# -----------------------------------------------------------------------
# Targets     mem_mb_p95_max, mem_mb_peak_max (largest driver per run)
# Baseline    ridge_wl — the deployed LogReg memory model (card alpha)
# Candidates  C1 rf_wl         forest per workload (card selection)
#             C2 ridge_log_wl  per-workload Ridge on log(memory), at the
#                              model card's alpha (the one that would be
#                              deployed)
#             C3 ridge_int_wl  per-workload Ridge with pairwise interactions,
#                              alpha by the LogReg CV MAE as in A2 (not a
#                              deployable family: a win would be reported,
#                              not deployed, without a further step)
# In range    leave-one-configuration-out CV on TRAIN+VAL (frozen test split
#             untouched), scored on LogReg rows; paired group-bootstrap 95 %
#             CI of the MAE difference against ridge_wl
# Beyond range  on TRAIN+VAL only, each model predicts the held-out rows the
#             way the predictor does: the model at the largest training
#             value, grown by the TREND_POLICY trend (size: Scout-informed β;
#             cores: none) — calibration.stress_split / stress_predict:
#               size   hold out LogReg rows above 100 MB per driver
#               cores  hold out LogReg rows above 4 cores, scored on Set D
#                      rows (the core sweeps)
# Rule        per target, a candidate replaces ridge_wl FOR LOGREG only if it
#             beats it in range (CI upper bound < 0) AND is not clearly worse
#             in either beyond-range check (CI lower bound <= 0); if several
#             qualify, the lowest in-range LogReg MAE wins
# Then        test split and E1 LogReg rows, models refit on train+val and
#             predicted with the same trend: confirmation only (seen before,
#             not blind)
# If a deployable candidate wins, predictor.WORKLOAD_FAMILY_POLICY gets the
# LogReg entry and the predictor is retrained (its stress-test floors are
# recomputed for the new family).
#
# Output:  data/models/logreg_memory/  (git-ignored) — report.json,
#          confirmation.csv
#
#   python scripts/logreg_memory_test.py
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
for _p in (_PROJECT_ROOT, _SCRIPT_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import kmeans_memory_test as a2  # noqa: E402

from mpj_spark.resource_prediction import calibration as cal  # noqa: E402
from mpj_spark.resource_prediction import models as m  # noqa: E402
from mpj_spark.resource_prediction import predictor as pr  # noqa: E402

TRAINING_SET = "capped"
TARGETS = ["mem_mb_p95_max", "mem_mb_peak_max"]
WORKLOAD = "logreg"
BASELINE = "ridge_wl"
AXES = ("size", "cores")


def _parse_args(argv=None):
    p = argparse.ArgumentParser(description="LogReg memory model test (step A2b).")
    p.add_argument(
        "--drivers", default=os.path.join(_PROJECT_ROOT, "data", "local", "local_drivers.csv")
    )
    p.add_argument(
        "--model-card",
        default=os.path.join(_PROJECT_ROOT, "data", "models", "local_capped", "model_card.json"),
    )
    p.add_argument(
        "--scout-hosts", default=os.path.join(_PROJECT_ROOT, "data", "scout", "scout_hosts.csv")
    )
    p.add_argument(
        "--out-dir", default=os.path.join(_PROJECT_ROOT, "data", "models", "logreg_memory")
    )
    p.add_argument("--rf-trees", type=int, default=200)
    p.add_argument("--n-boot", type=int, default=2000)
    return p.parse_args(argv)


def chosen_keys(card: dict, target: str, lr_mae: pd.Series) -> dict[str, str]:
    """family → factory key: C2 at the card alpha, C3 by the LogReg CV MAE."""
    ints = [k for k in lr_mae.index if k.startswith("ridge_int_wl|")]
    return {
        BASELINE: BASELINE,
        "rf_wl": "rf_wl",
        "ridge_log_wl": card["choices"][target]["selected"]["ridge_log_wl"],
        "ridge_int_wl": lr_mae[ints].idxmin(),
    }


def _predict_like_predictor(facs, keys, train, ev, target, axis, scout) -> pd.DataFrame:
    """Long frame: every family's prediction of `ev` (model + policy trend)."""
    features = m.features_for(TRAINING_SET)
    frames = []
    for fam, key in keys.items():
        pred, _ = cal.stress_predict(
            facs[key](), train, ev, target, WORKLOAD, axis, pr.TREND_POLICY, scout, features
        )
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
    return pd.concat(frames, ignore_index=True)


def beyond_range(d, facs, keys, target, scout, n_boot) -> dict:
    out = {}
    for axis in AXES:
        train, ev = cal.stress_split(d, axis, WORKLOAD)
        long = _predict_like_predictor(facs, keys, train, ev, target, axis, scout)
        out[axis] = {
            "rows": len(ev),
            "groups": int(ev["config_group"].nunique()),
            "trend": pr.TREND_POLICY.get(axis, {}).get(WORKLOAD, {}).get(target),
            "mae": a2._mae(long).to_dict(),
            "paired": {
                f"{fam}_vs_{BASELINE}": m.group_bootstrap(
                    long, fam, baseline=BASELINE, n_boot=n_boot
                )
                for fam in keys
                if fam != BASELINE
            },
        }
    return out


def run_target(trval, card, target, scout, args) -> dict:
    features = m.features_for(TRAINING_SET)
    d = trval[trval[target].notna()]
    facs = a2.candidates(card, target, args.rf_trees)
    oof = m.group_cv_predictions(d, target, facs, features=features)
    lr = oof[oof["workload_type"] == WORKLOAD]
    keys = chosen_keys(card, target, a2._mae(lr))
    lr = lr[lr["model"].isin(keys.values())].replace({"model": {v: k for k, v in keys.items()}})
    paired = {
        f"{fam}_vs_{BASELINE}": m.group_bootstrap(lr, fam, baseline=BASELINE, n_boot=args.n_boot)
        for fam in keys
        if fam != BASELINE
    }
    extrap = beyond_range(d, facs, keys, target, scout, args.n_boot)
    winners = [
        fam
        for fam in keys
        if fam != BASELINE
        and paired[f"{fam}_vs_{BASELINE}"]["mae_diff_ci95"][1] < 0
        and all(
            x["paired"][f"{fam}_vs_{BASELINE}"]["mae_diff_ci95"][0] <= 0 for x in extrap.values()
        )
    ]
    lr_mae = a2._mae(lr)
    decision = min(winners, key=lambda f: lr_mae[f]) if winners else BASELINE
    return {
        "chosen_keys": keys,
        "logreg": {
            fam: m.regression_metrics(g["actual"], g["predicted"]) for fam, g in lr.groupby("model")
        },
        "paired": paired,
        "beyond_range": extrap,
        "qualified": winners,
        "decision": decision,
        "deployable": decision in pr.SOURCE_NAMES,
    }


def confirm(trval, raw, card, target, keys, scout, args) -> list[dict]:
    """Test split and E1 LogReg rows; refit on train+val, size trend applied."""
    facs = a2.candidates(card, target, args.rf_trees)
    d = trval[trval[target].notna()]
    sets = {
        "test": raw[raw["split"] == "test"],
        "E1 interior": raw[(raw["experiment"] == "E1") & (raw["holdout_kind"] == "interior")],
        "E1 extrap": raw[(raw["experiment"] == "E1") & (raw["holdout_kind"] == "extrap")],
    }
    rows = []
    for name, df in sets.items():
        h = df[(df["status"] == "ok") & (df["workload_type"] == WORKLOAD)]
        h = m.add_derived(h[h[target].notna()])
        if len(h) < 2:
            continue
        long = _predict_like_predictor(facs, keys, d, h, target, "size", scout)
        for fam, g in long.groupby("model"):
            met = m.regression_metrics(g["actual"], g["predicted"])
            rows.append({"target": target, "set": name, "model": fam} | met)
    return rows


def main(argv=None):
    args = _parse_args(argv)
    with open(args.model_card) as fh:
        card = json.load(fh)
    raw = pd.read_csv(args.drivers)
    df, rows_info = m.select_training_rows(raw, TRAINING_SET)
    trval = df[df["split"].isin(["train", "val"])]
    scout = pd.read_csv(args.scout_hosts) if os.path.isfile(args.scout_hosts) else None
    os.makedirs(args.out_dir, exist_ok=True)

    results, confirmation = {}, []
    for target in TARGETS:
        results[target] = run_target(trval, card, target, scout, args)
        keys = results[target]["chosen_keys"]
        confirmation += confirm(trval, raw, card, target, keys, scout, args)
    pd.DataFrame(confirmation).to_csv(os.path.join(args.out_dir, "confirmation.csv"), index=False)
    with open(args.drivers, "rb") as fh:
        sha = hashlib.sha256(fh.read()).hexdigest()
    report = {
        "question": "Can LogReg memory be predicted better than by ridge_wl?",
        "protocol": "see script header (fixed before this test)",
        "data": {"drivers_sha256": sha, "rows": rows_info},
        "results": results,
        "decision": {t: r["decision"] for t, r in results.items()},
    }
    with open(os.path.join(args.out_dir, "report.json"), "w") as fh:
        json.dump(report, fh, indent=2, default=str)

    for target, r in results.items():
        print(f"── {target} — LogReg rows, leave-one-configuration-out CV ──")
        for fam, met in r["logreg"].items():
            print(
                f"   {fam:13s} ({r['chosen_keys'][fam]:18s}) MAE {met['mae']:7.1f} MB  "
                f"R2 {met['r2']:.3f}"
            )
        for name, p in r["paired"].items():
            lo, hi = p["mae_diff_ci95"]
            print(f"   {name:24s} {p['mae_diff_vs_baseline']:+7.1f} MB [{lo:+.1f}, {hi:+.1f}]")
        for axis, x in r["beyond_range"].items():
            print(
                f"   beyond range, {axis} ({x['rows']} rows, {x['groups']} groups held out, "
                f"trend {x['trend']}):"
            )
            for name, p in x["paired"].items():
                lo, hi = p["mae_diff_ci95"]
                fam = name.split("_vs_")[0]
                print(
                    f"     {name:24s} MAE {x['mae'][fam]:7.1f} vs {x['mae'][BASELINE]:7.1f} MB"
                    f"   diff {p['mae_diff_vs_baseline']:+7.1f} [{lo:+.1f}, {hi:+.1f}]"
                )
        print(f"   → decision for LogReg: {r['decision']}  (qualified: {r['qualified']})")
    print("── confirmation only: LogReg rows of the test split and E1 ──")
    cf = pd.DataFrame(confirmation)
    print(
        cf.pivot_table(index=["target", "set"], columns="model", values="mae", sort=False)
        .round(1)
        .to_string()
    )
    print(f"[logreg memory] outputs written to {args.out_dir}")
    return report


if __name__ == "__main__":
    main()
