#!/usr/bin/env python
# =============================================================================
# scripts/extrapolation_model_tests.py
# Phase 6 — fix 1: which model predicts CPU best for jobs with MORE DATA PER
# DRIVER than the laptop measured?
#
# PROTOCOL (fixed before any result was seen)
# -------------------------------------------
# Targets     cpu_cores_p95, cpu_cores_mean
# Candidates  rf (deployed), ridge_wl, ridge — hyper-parameters as selected by
#             CV in the model card (data/models/local_capped/model_card.json)
# Tests       train+val only (the frozen test split is never used):
#               T1  train dataset_mb <= 100, predict dataset_mb > 100
#               T2  train dataset_mb <= 50,  predict dataset_mb > 100
#             scored on rows whose data per driver exceeds the largest
#             data per driver of that workload in training
# Rule        per workload and target, a candidate replaces rf for
#             out-of-range requests only if it beats rf (paired group-
#             bootstrap 95 % CI of the MAE difference below 0) in EVERY test
#             that has out-of-range rows for that workload; if several
#             qualify, the lower MAE in T2 wins.  E1 extrapolation runs are
#             scored afterwards as confirmation only.
#
# Inputs:  data/local/local_drivers.csv, model card
# Output:  data/models/extrapolation/  (git-ignored)
#   extrap_metrics.csv, extrap_paired.csv, report.json (incl. "decision")
#
#   python scripts/extrapolation_model_tests.py
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
CANDIDATES = ["rf", "ridge_wl", "ridge"]
TESTS = {"T1": 100.0, "T2": 50.0}  # train on dataset_mb <= value, predict > 100
PREDICT_ABOVE_MB = 100.0
HOLDOUT_SET = "E1"


def _parse_args(argv=None):
    p = argparse.ArgumentParser(description="CPU model choice beyond the measured data size.")
    p.add_argument(
        "--drivers", default=os.path.join(_PROJECT_ROOT, "data", "local", "local_drivers.csv")
    )
    p.add_argument(
        "--model-card",
        default=os.path.join(_PROJECT_ROOT, "data", "models", "local_capped", "model_card.json"),
    )
    p.add_argument(
        "--out-dir", default=os.path.join(_PROJECT_ROOT, "data", "models", "extrapolation")
    )
    p.add_argument("--rf-trees", type=int, default=200)
    p.add_argument("--n-boot", type=int, default=2000)
    return p.parse_args(argv)


def factory_from_key(key: str, rf_trees: int):
    """Model factory from a CV selection key such as 'rf|8|2' or 'ridge_wl|10.0'."""
    num = m.TRAINING_SETS[TRAINING_SET]["numeric"]
    family, *params = key.split("|")
    if family == "rf":
        depth, leaf = int(params[0]), int(params[1])
        return lambda: m.make_rf(depth, leaf, n_estimators=rf_trees)
    if family == "ridge_wl":
        return lambda: m.make_ridge_wl(float(params[0]), num)
    if family == "ridge":
        return lambda: m.make_ridge(float(params[0]), num)
    raise ValueError(f"unsupported candidate {key!r}")


def _shard(df: pd.DataFrame) -> np.ndarray:
    return (df["dataset_mb"] / df["num_workers"]).to_numpy(float)


def run_test(df, factories, target, train_max, n_boot):
    """Per workload: metrics and paired comparisons on out-of-range rows."""
    features = m.features_for(TRAINING_SET)
    d = df[df[target].notna()]
    train = d[d["dataset_mb"] <= train_max]
    fitted = {name: f().fit(train[features], train[target]) for name, f in factories.items()}
    out = {}
    for wl, ev in d[d["dataset_mb"] > PREDICT_ABOVE_MB].groupby("workload_type"):
        s_max = float(_shard(train[train["workload_type"] == wl]).max())
        ev = ev[_shard(ev) > s_max]
        if ev.empty:
            continue
        actual = ev[target].to_numpy(float)
        preds = {name: mdl.predict(ev[features]) for name, mdl in fitted.items()}
        long = pd.concat(
            [
                pd.DataFrame(
                    {
                        "config_group": ev["config_group"].values,
                        "workload_type": wl,
                        "model": name,
                        "actual": actual,
                        "predicted": p,
                    }
                )
                for name, p in preds.items()
            ],
            ignore_index=True,
        )
        out[wl] = {
            "s_max": s_max,
            "rows": len(ev),
            "groups": int(ev["config_group"].nunique()),
            "metrics": {n: m.regression_metrics(actual, p) for n, p in preds.items()},
            "paired": {
                f"{n}_vs_rf": m.group_bootstrap(long, n, baseline="rf", n_boot=n_boot)
                for n in factories
                if n != "rf"
            },
        }
    return out


def decide(results: dict) -> dict:
    """Apply the protocol rule: {target: {workload: chosen candidate}}."""
    decision = {}
    for target, by_test in results.items():
        workloads = {wl for test in by_test.values() for wl in test}
        for wl in sorted(workloads):
            tests = {name: t[wl] for name, t in by_test.items() if wl in t}
            winners = [
                c
                for c in CANDIDATES
                if c != "rf"
                and all(t["paired"][f"{c}_vs_rf"]["mae_diff_ci95"][1] < 0 for t in tests.values())
            ]
            ref = tests.get("T2") or next(iter(tests.values()))
            choice = min(winners, key=lambda c: ref["metrics"][c]["mae"]) if winners else "rf"
            decision.setdefault(target, {})[wl] = {
                "model": choice,
                "tests": sorted(tests),
                "qualified": winners,
            }
    return decision


def main(argv=None):
    args = _parse_args(argv)
    with open(args.model_card) as fh:
        card = json.load(fh)
    raw = pd.read_csv(args.drivers)
    df, rows_info = m.select_training_rows(raw, TRAINING_SET)
    trval = df[df["split"].isin(["train", "val"])]  # frozen test split untouched
    os.makedirs(args.out_dir, exist_ok=True)

    results, metrics_rows, paired_rows, keys = {}, [], [], {}
    for target in TARGETS:
        keys[target] = {c: card["choices"][target]["selected"][c] for c in CANDIDATES}
        factories = {c: factory_from_key(k, args.rf_trees) for c, k in keys[target].items()}
        for test, train_max in TESTS.items():
            res = run_test(trval, factories, target, train_max, args.n_boot)
            results.setdefault(target, {})[test] = res
            for wl, r in res.items():
                for name, met in r["metrics"].items():
                    metrics_rows.append(
                        {"target": target, "test": test, "workload": wl, "model": name} | met
                    )
                for comp, p in r["paired"].items():
                    lo, hi = p["mae_diff_ci95"]
                    paired_rows.append(
                        {
                            "target": target,
                            "test": test,
                            "workload": wl,
                            "comparison": comp,
                            "mae_diff": p["mae_diff_vs_baseline"],
                            "ci_low": lo,
                            "ci_high": hi,
                            "groups": p["n_groups"],
                        }
                    )
    decision = decide(results)

    # confirmation only: E1 extrapolation, models refit on all train+val rows
    features = m.features_for(TRAINING_SET)
    e1 = raw[(raw["split"] == "holdout") & (raw["experiment"] == HOLDOUT_SET)]
    e1 = m.add_derived(e1[(e1["status"] == "ok") & (e1["holdout_kind"] == "extrap")])
    confirm = []
    for target in TARGETS:
        d = trval[trval[target].notna()]
        h = e1[e1[target].notna()]
        for c, k in keys[target].items():
            mdl = factory_from_key(k, args.rf_trees)().fit(d[features], d[target])
            pred = mdl.predict(h[features])
            for wl in ["all", *sorted(h["workload_type"].unique())]:
                mask = (
                    (h["workload_type"] == wl).to_numpy() if wl != "all" else np.ones(len(h), bool)
                )
                confirm.append(
                    {"target": target, "workload": wl, "model": c}
                    | m.regression_metrics(h[target].to_numpy(float)[mask], pred[mask])
                )

    pd.DataFrame(metrics_rows).to_csv(os.path.join(args.out_dir, "extrap_metrics.csv"), index=False)
    pd.DataFrame(paired_rows).to_csv(os.path.join(args.out_dir, "extrap_paired.csv"), index=False)
    pd.DataFrame(confirm).to_csv(os.path.join(args.out_dir, "e1_confirmation.csv"), index=False)
    with open(args.drivers, "rb") as fh:
        sha = hashlib.sha256(fh.read()).hexdigest()
    report = {
        "question": "Which model predicts CPU best beyond the measured data per driver?",
        "protocol": "see script header (fixed before results)",
        "candidates": keys,
        "data": {"drivers_sha256": sha, "rows": rows_info},
        "results": results,
        "decision": decision,
    }
    with open(os.path.join(args.out_dir, "report.json"), "w") as fh:
        json.dump(report, fh, indent=2, default=str)

    mt = pd.DataFrame(metrics_rows)
    print("── MAE / R² on rows beyond the training data per driver ──")
    print(
        mt.pivot_table(index=["target", "test", "workload"], columns="model", values=["mae", "r2"])
        .round(3)
        .to_string()
    )
    print("── paired MAE difference vs rf, 95 % CI (negative = candidate better) ──")
    print(pd.DataFrame(paired_rows).round(3).to_string(index=False))
    print("── decision (protocol rule) ──")
    print(json.dumps(decision, indent=1))
    print("── confirmation only: E1 extrapolation, models refit on train+val ──")
    cf = pd.DataFrame(confirm)
    print(
        cf.pivot_table(index=["target", "workload"], columns="model", values=["mae", "r2"])
        .round(3)
        .to_string()
    )
    print(f"[extrapolation] outputs written to {args.out_dir}")
    return report


if __name__ == "__main__":
    main()
