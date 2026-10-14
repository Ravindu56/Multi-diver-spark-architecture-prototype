#!/usr/bin/env python
# =============================================================================
# scripts/per_workload_rf_test.py
# Phase 6 — fix 1b: should the CPU random forest be fitted PER WORKLOAD?
#
# WHY  (found on E1, so E1 is not a blind check for this decision)
# ---
# A single forest answers a K-Means job with more data per driver than any
# K-Means run (K-Means measured up to 100 MB per driver) from LogReg runs of
# that size (LogReg measured up to 200 MB), which peak at far more cores.
#
# PROTOCOL (fixed before this test was run)
# -----------------------------------------
# Targets     cpu_cores_p95, cpu_cores_mean
# Candidates  rf     one forest for all workloads (deployed)
#             rf_wl  the same forest, fitted per workload inside one model
#             (hyper-parameters: the rf selection in the model card)
# Test T3     train+val only: hold out K-Means rows with more than 50 MB per
#             driver; train on everything else (LogReg of every size stays);
#             predict the held-out K-Means rows
# In range    leave-one-configuration-out CV on train+val (all rows)
# Rule        per target, adopt rf_wl only if
#               (a) it beats rf in T3 (paired group-bootstrap 95 % CI of the
#                   MAE difference below 0), and
#               (b) it is not clearly worse in range (CI lower bound <= 0)
# Then        E1 (interior and extrapolation) as confirmation.
#
# Output:  data/models/per_workload_rf/  (git-ignored) — report.json etc.
#
#   python scripts/per_workload_rf_test.py
# =============================================================================

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

import numpy as np
import pandas as pd

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from mpj_spark.resource_prediction import models as m  # noqa: E402

TRAINING_SET = "capped"
TARGETS = ["cpu_cores_p95", "cpu_cores_mean"]
T3_WORKLOAD, T3_SHARD_MB = "kmeans", 50.0
HOLDOUT_SET = "E1"


def _parse_args(argv=None):
    p = argparse.ArgumentParser(description="Per-workload random forest test (fix 1b).")
    p.add_argument(
        "--drivers", default=os.path.join(_PROJECT_ROOT, "data", "local", "local_drivers.csv")
    )
    p.add_argument(
        "--model-card",
        default=os.path.join(_PROJECT_ROOT, "data", "models", "local_capped", "model_card.json"),
    )
    p.add_argument(
        "--out-dir", default=os.path.join(_PROJECT_ROOT, "data", "models", "per_workload_rf")
    )
    p.add_argument("--rf-trees", type=int, default=200)
    p.add_argument("--n-boot", type=int, default=2000)
    return p.parse_args(argv)


def factories(card: dict, target: str, rf_trees: int) -> dict:
    _, depth, leaf = card["choices"][target]["selected"]["rf"].split("|")
    d, lf = int(depth), int(leaf)
    return {
        "rf": lambda: m.make_rf(d, lf, n_estimators=rf_trees),
        "rf_wl": lambda: m.make_rf_wl(d, lf, n_estimators=rf_trees),
    }


def _shard(df: pd.DataFrame) -> np.ndarray:
    return (df["dataset_mb"] / df["num_workers"]).to_numpy(float)


def t3(trval, facs, target, n_boot) -> dict:
    features = m.features_for(TRAINING_SET)
    d = trval[trval[target].notna()]
    held = (d["workload_type"] == T3_WORKLOAD).to_numpy() & (_shard(d) > T3_SHARD_MB)
    train, ev = d[~held], d[held]
    actual = ev[target].to_numpy(float)
    preds = {
        n: f().fit(train[features], train[target]).predict(ev[features]) for n, f in facs.items()
    }
    long = pd.concat(
        [
            pd.DataFrame(
                {
                    "config_group": ev["config_group"].values,
                    "workload_type": ev["workload_type"].values,
                    "model": n,
                    "actual": actual,
                    "predicted": p,
                }
            )
            for n, p in preds.items()
        ],
        ignore_index=True,
    )
    return {
        "rows": len(ev),
        "groups": int(ev["config_group"].nunique()),
        "metrics": {n: m.regression_metrics(actual, p) for n, p in preds.items()},
        "paired": m.group_bootstrap(long, "rf_wl", baseline="rf", n_boot=n_boot),
    }


def in_range(trval, facs, target, n_boot) -> dict:
    features = m.features_for(TRAINING_SET)
    oof = m.group_cv_predictions(trval, target, facs, features=features)
    return {
        "metrics": {
            n: m.regression_metrics(g["actual"], g["predicted"]) for n, g in oof.groupby("model")
        },
        "paired": m.group_bootstrap(oof, "rf_wl", baseline="rf", n_boot=n_boot),
    }


def confirm_e1(trval, holdout, facs, target) -> list[dict]:
    features = m.features_for(TRAINING_SET)
    d, h = trval[trval[target].notna()], holdout[holdout[target].notna()]
    rows = []
    for n, f in facs.items():
        pred = f().fit(d[features], d[target]).predict(h[features])
        for kind in ("interior", "extrap"):
            for wl in ("all", "kmeans", "logreg"):
                mask = (h["holdout_kind"] == kind).to_numpy()
                if wl != "all":
                    mask = mask & (h["workload_type"] == wl).to_numpy()
                if mask.sum() >= 2:
                    met = m.regression_metrics(h[target].to_numpy(float)[mask], pred[mask])
                    rows.append({"target": target, "model": n, "kind": kind, "workload": wl} | met)
    return rows


def main(argv=None):
    args = _parse_args(argv)
    with open(args.model_card) as fh:
        card = json.load(fh)
    raw = pd.read_csv(args.drivers)
    df, rows_info = m.select_training_rows(raw, TRAINING_SET)
    trval = df[df["split"].isin(["train", "val"])]  # frozen test split untouched
    holdout = raw[(raw["split"] == "holdout") & (raw["experiment"] == HOLDOUT_SET)]
    holdout = m.add_derived(holdout[holdout["status"] == "ok"])
    os.makedirs(args.out_dir, exist_ok=True)

    results, decision, confirm = {}, {}, []
    for target in TARGETS:
        facs = factories(card, target, args.rf_trees)
        r = {
            "T3": t3(trval, facs, target, args.n_boot),
            "in_range": in_range(trval, facs, target, args.n_boot),
        }
        results[target] = r
        beats_t3 = r["T3"]["paired"]["mae_diff_ci95"][1] < 0
        not_worse = r["in_range"]["paired"]["mae_diff_ci95"][0] <= 0
        decision[target] = {
            "adopt_rf_wl": bool(beats_t3 and not_worse),
            "beats_rf_in_T3": bool(beats_t3),
            "not_clearly_worse_in_range": bool(not_worse),
        }
        confirm += confirm_e1(trval, holdout, facs, target)

    pd.DataFrame(confirm).to_csv(os.path.join(args.out_dir, "e1_confirmation.csv"), index=False)
    with open(args.drivers, "rb") as fh:
        sha = hashlib.sha256(fh.read()).hexdigest()
    report = {
        "question": "Should the CPU random forest be fitted per workload?",
        "protocol": "see script header (fixed before this test; motivated by E1)",
        "data": {"drivers_sha256": sha, "rows": rows_info},
        "results": results,
        "decision": decision,
    }
    with open(os.path.join(args.out_dir, "report.json"), "w") as fh:
        json.dump(report, fh, indent=2, default=str)

    for target, r in results.items():
        t, c = r["T3"], r["in_range"]
        print(f"── {target} ──")
        print(
            f"   T3 ({t['rows']} K-Means rows, {t['groups']} groups): MAE rf "
            f"{t['metrics']['rf']['mae']:.3f}  rf_wl {t['metrics']['rf_wl']['mae']:.3f}  "
            f"diff {t['paired']['mae_diff_vs_baseline']:+.3f} "
            f"[{t['paired']['mae_diff_ci95'][0]:+.3f}, {t['paired']['mae_diff_ci95'][1]:+.3f}]"
        )
        print(
            f"   in-range CV: MAE rf {c['metrics']['rf']['mae']:.3f} (R2 {c['metrics']['rf']['r2']:.3f})  "
            f"rf_wl {c['metrics']['rf_wl']['mae']:.3f} (R2 {c['metrics']['rf_wl']['r2']:.3f})  "
            f"diff {c['paired']['mae_diff_vs_baseline']:+.3f} "
            f"[{c['paired']['mae_diff_ci95'][0]:+.3f}, {c['paired']['mae_diff_ci95'][1]:+.3f}]"
        )
        print(f"   decision: {decision[target]}")
    print("── confirmation only: E1 ──")
    cf = pd.DataFrame(confirm)
    print(
        cf.pivot_table(index=["target", "kind", "workload"], columns="model", values=["mae", "r2"])
        .round(3)
        .to_string()
    )
    print(f"[per-workload rf] outputs written to {args.out_dir}")
    return report


if __name__ == "__main__":
    main()
