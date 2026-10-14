#!/usr/bin/env python
# =============================================================================
# scripts/transfer_trend_tests.py
# Phase 6 — which growth trend should the predictor use beyond the laptop's
# measured range, and does Scout make it better?  (transfer steps 2 and 3)
#
# PROTOCOL (fixed before any result was seen)
# -------------------------------------------
# Trends   demand ∝ x^β along size (data per driver) or cores, applied only
#          beyond the largest x in training (transfer.anchor_and_scale).
# Methods  local          base model's own prediction (no trend)
#          flat           anchored at x_max, no growth
#          scout_trend    β from Scout
#          local_trend    β from the local training rows
#          combined       Scout-informed β = (n·β_local + 3·β_scout)/(n + 3)
#          (a method whose β is unavailable is reported as not available)
# Base     random forest for CPU, per-workload Ridge for memory (CV choices,
#          fixed hyper-parameters); trained on TRAIN+VAL subsets only — the
#          frozen test split is never used.
# Tests    size S1 (scarce)  train dataset_mb <= 50,  predict dataset_mb > 100
#          size S2 (seen in option A)  train <= 100,  predict > 100
#          cores C1  K-Means: train cores <= 2, predict Set D rows with 3-4 cores
#                    LogReg:  train cores <= 4, predict Set D rows with 6-8 cores
#          (training keeps every other row; only the predicted range is held out)
# Scored   rows beyond x_max; paired group-bootstrap MAE differences, 95 % CI.
# Rules    per axis × workload × target:
#   source  "combined" if it beats local_trend (CI < 0) in the scarce test
#           (size S1, cores C1) and is not clearly worse (CI > 0) in size S2;
#           otherwise "local_trend"
#   use     the chosen trend is used only if it beats "local" (CI < 0) in the
#           main test (size S2, cores C1); otherwise the model's own prediction
#           is kept and the request is flagged out of range
#
# Inputs:  data/scout/scout_hosts.csv, data/local/local_drivers.csv
# Output:  data/models/transfer_trends/  (git-ignored)
#   trend_metrics.csv, trend_paired.csv, report.json (incl. "policy")
#
#   python scripts/transfer_trend_tests.py
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
from mpj_spark.resource_prediction import transfer as t  # noqa: E402

TRAINING_SET = "capped"
METHODS = ["local", "flat", "scout_trend", "local_trend", "combined"]
PAIRS = [
    ("local_trend", "local"),
    ("combined", "local"),
    ("scout_trend", "local"),
    ("combined", "local_trend"),
    ("scout_trend", "local_trend"),
]

# (test name, axis, workload, train filter, eval filter) — see PROTOCOL
TESTS = [
    ("size_S1", "size", None, lambda d: d["dataset_mb"] <= 50, lambda d: d["dataset_mb"] > 100),
    ("size_S2", "size", None, lambda d: d["dataset_mb"] <= 100, lambda d: d["dataset_mb"] > 100),
    (
        "cores_C1",
        "cores",
        "kmeans",
        lambda d: ~((d["workload_type"] == "kmeans") & (d["cores_per_driver"] > 2)),
        lambda d: (d["experiment"] == "D") & (d["cores_per_driver"] > 2),
    ),
    (
        "cores_C1",
        "cores",
        "logreg",
        lambda d: ~((d["workload_type"] == "logreg") & (d["cores_per_driver"] > 4)),
        lambda d: (d["experiment"] == "D") & (d["cores_per_driver"] > 4),
    ),
]
SCARCE_TEST = {"size": "size_S1", "cores": "cores_C1"}
MAIN_TEST = {"size": "size_S2", "cores": "cores_C1"}


def _parse_args(argv=None):
    p = argparse.ArgumentParser(description="Growth-trend tests for the predictor.")
    p.add_argument(
        "--scout-hosts", default=os.path.join(_PROJECT_ROOT, "data", "scout", "scout_hosts.csv")
    )
    p.add_argument(
        "--drivers", default=os.path.join(_PROJECT_ROOT, "data", "local", "local_drivers.csv")
    )
    p.add_argument(
        "--out-dir", default=os.path.join(_PROJECT_ROOT, "data", "models", "transfer_trends")
    )
    p.add_argument("--rf-trees", type=int, default=200)
    p.add_argument("--n-boot", type=int, default=2000)
    return p.parse_args(argv)


def base_factory(target: str, rf_trees: int):
    num = m.TRAINING_SETS[TRAINING_SET]["numeric"]
    if target.startswith("cpu"):
        return lambda: m.make_rf(5, 2, n_estimators=rf_trees)
    return lambda: m.make_ridge_wl(1.0, num)


def run_case(df, scout, test, axis, workload, train_f, eval_f, target, args):
    """One test × workload × target: predictions of every method on the eval rows."""
    features = m.features_for(TRAINING_SET)
    d = df[df[target].notna()]
    train = d[train_f(d)]
    evals = d[eval_f(d) & (d["workload_type"] == workload)]
    wl_train = train[train["workload_type"] == workload]
    if evals.empty or wl_train.empty:
        return None
    x_max = float(t.axis_value(wl_train, axis).max())
    beyond = t.axis_value(evals, axis) > x_max
    if not beyond.any():
        return None
    model = base_factory(target, args.rf_trees)().fit(train[features], train[target])
    local = t.local_elasticity(train, target, workload, axis)
    scout = t.scout_elasticity(scout_hosts=scout, target=target, workload=workload, axis=axis)
    comb = t.combine_elasticity(local, scout)
    betas = {
        "flat": 0.0,
        "scout_trend": scout["beta"] if scout else None,
        "local_trend": local["beta"] if local else None,
        "combined": comb["beta"] if comb else None,
    }
    ev = evals[beyond]
    preds = {"local": t.anchor_and_scale(model, ev, features, x_max, None, axis)}
    for name, beta in betas.items():
        preds[name] = (
            np.full(len(ev), np.nan)
            if beta is None
            else t.anchor_and_scale(model, ev, features, x_max, beta, axis)
        )
    return {
        "info": {
            "test": test,
            "axis": axis,
            "workload": workload,
            "target": target,
            "x_max": x_max,
            "train_rows": len(train),
            "rows_beyond": int(beyond.sum()),
            "groups_beyond": int(ev["config_group"].nunique()),
            "beta_local": local,
            "beta_scout": scout,
            "beta_combined": comb,
        },
        "rows": ev,
        "preds": preds,
    }


def score(case, target, n_boot):
    ev, preds = case["rows"], case["preds"]
    actual = ev[target].to_numpy(float)
    metrics = []
    for name in METHODS:
        ok = ~np.isnan(preds[name])
        row = {k: case["info"][k] for k in ("test", "axis", "workload", "target")}
        row["method"] = name
        row |= m.regression_metrics(actual[ok], preds[name][ok]) if ok.any() else {"n": 0}
        metrics.append(row)
    long = pd.concat(
        [
            pd.DataFrame(
                {
                    "config_group": ev["config_group"].values,
                    "workload_type": ev["workload_type"].values,
                    "model": name,
                    "actual": actual,
                    "predicted": preds[name],
                }
            )
            for name in METHODS
            if not np.isnan(preds[name]).any()
        ],
        ignore_index=True,
    )
    paired = {}
    available = set(long["model"])
    for a, b in PAIRS:
        if a in available and b in available:
            paired[f"{a}_vs_{b}"] = m.group_bootstrap(long, a, baseline=b, n_boot=n_boot)
    return metrics, paired


def _beats(paired: dict, a: str, b: str) -> bool:
    p = paired.get(f"{a}_vs_{b}")
    return bool(p) and p["mae_diff_ci95"][1] < 0


def _clearly_worse(paired: dict, a: str, b: str) -> bool:
    p = paired.get(f"{a}_vs_{b}")
    return bool(p) and p["mae_diff_ci95"][0] > 0


def decide(results: dict) -> dict:
    """Apply the protocol rules: {axis: {workload: {target: decision}}}."""
    policy = {}
    keys = {(r["axis"], r["workload"], r["target"]) for r in results.values()}
    for axis, wl, target in sorted(keys):
        scarce = results.get((SCARCE_TEST[axis], axis, wl, target), {}).get("paired", {})
        main = results.get((MAIN_TEST[axis], axis, wl, target), {}).get("paired", {})
        seen = (
            results.get(("size_S2", axis, wl, target), {}).get("paired", {})
            if axis == "size"
            else {}
        )
        combined_ok = _beats(scarce, "combined", "local_trend") and not _clearly_worse(
            seen, "combined", "local_trend"
        )
        source = "combined" if combined_ok else "local_trend"
        use = _beats(main, source, "local")
        policy.setdefault(axis, {}).setdefault(wl, {})[target] = {
            "source": source if use else None,
            "use_trend": use,
            "rule": "combined beats local_trend in the scarce test and is not clearly worse "
            "in size S2; trend used only if it beats local in the main test",
        }
    return policy


def _fmt_beta(b: dict | None, groups: bool = False) -> str:
    if not b:
        return "-"
    return f"{b['beta']:+.3f}" + (f" ({b['groups']} fam)" if groups else "")


def main(argv=None):
    args = _parse_args(argv)
    raw = pd.read_csv(args.drivers)
    df, rows_info = m.select_training_rows(raw, TRAINING_SET)
    df = df[df["split"].isin(["train", "val"])]  # frozen test split untouched
    scout = pd.read_csv(args.scout_hosts)
    os.makedirs(args.out_dir, exist_ok=True)

    results, metrics, paired_rows = {}, [], []
    for test, axis, workload, train_f, eval_f in TESTS:
        for wl in [workload] if workload else sorted(df["workload_type"].unique()):
            for target in m.TARGETS:
                case = run_case(df, scout, test, axis, wl, train_f, eval_f, target, args)
                if case is None:
                    continue
                rows, paired = score(case, target, args.n_boot)
                metrics += rows
                for name, p in paired.items():
                    paired_rows.append(
                        {**{k: case["info"][k] for k in ("test", "axis", "workload", "target")}}
                        | {
                            "comparison": name,
                            "mae_diff": p["mae_diff_vs_baseline"],
                            "ci_low": p["mae_diff_ci95"][0],
                            "ci_high": p["mae_diff_ci95"][1],
                            "groups": p["n_groups"],
                        }
                    )
                results[(test, axis, wl, target)] = case["info"] | {"paired": paired}
    policy = decide(results)
    pd.DataFrame(metrics).to_csv(os.path.join(args.out_dir, "trend_metrics.csv"), index=False)
    pd.DataFrame(paired_rows).to_csv(os.path.join(args.out_dir, "trend_paired.csv"), index=False)

    def _sha(path):
        with open(path, "rb") as fh:
            return hashlib.sha256(fh.read()).hexdigest()

    report = {
        "question": "Which growth trend should the predictor use beyond the measured range, "
        "and does Scout improve it?",
        "protocol": "see script header (fixed before results)",
        "scout_prior_groups": t.SCOUT_PRIOR_GROUPS,
        "data": {
            "drivers_sha256": _sha(args.drivers),
            "scout_hosts_sha256": _sha(args.scout_hosts),
            "rows": rows_info,
        },
        "cases": {"|".join(k): v for k, v in results.items()},
        "policy": policy,
    }
    with open(os.path.join(args.out_dir, "report.json"), "w") as fh:
        json.dump(report, fh, indent=2, default=str)

    mt = pd.DataFrame(metrics)
    print("── MAE on rows beyond the training range ──")
    print(
        mt.pivot_table(
            index=["test", "workload", "target"], columns="method", values="mae", sort=False
        )
        .round(3)
        .to_string()
    )
    print("── elasticities β (local / Scout / combined) ──")
    for (test, _axis, wl, target), info in results.items():
        bl, bs, bc = info["beta_local"], info["beta_scout"], info["beta_combined"]
        print(
            f"   {test:9s} {wl:7s} {target:16s} local "
            f"{_fmt_beta(bl, True)}  scout {_fmt_beta(bs)}  combined {_fmt_beta(bc)}  | "
            f"{info['rows_beyond']} rows, {info['groups_beyond']} groups beyond "
            f"x_max={info['x_max']:g}"
        )
    print("── paired MAE differences, 95 % CI (negative = first method better) ──")
    pr = pd.DataFrame(paired_rows)
    print(pr.round(3).to_string(index=False))
    print("── policy (protocol rules) ──")
    print(json.dumps(policy, indent=1))
    print(f"[trends] outputs written to {args.out_dir}")
    return report


if __name__ == "__main__":
    main()
