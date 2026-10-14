#!/usr/bin/env python
# =============================================================================
# scripts/transfer_size_trend.py
# Phase 6 — transfer option A: does Scout's DATA-SIZE TREND help predict jobs
# with more data per driver than measured locally?
#
# PROTOCOL (fixed before any result was seen)
# -------------------------------------------
# Scope      LogReg (Scout records input sizes only for LogReg among the local
#            workloads; K-Means is reported as not applicable).
# Source     β_scout = slope of log(target) on log(input size) within each
#            Scout VM type (single-node hosts, labels as in transfer.py).
# Base model the family chosen by CV for each target, fixed hyper-parameters:
#            random forest for CPU, per-workload Ridge for memory.
# Methods    for a request with shard s = dataset_mb / workers above the
#            largest local shard s_max (rows at or below s_max are unchanged):
#              local        the base model's own prediction
#              flat         base(at s_max)                     (no growth)
#              scout_trend  base(at s_max) × (s / s_max)^β_scout
#              local_trend  base(at s_max) × (s / s_max)^β_local (β from the
#                           local training rows — the fair comparison)
# Test 1     size stress test on TRAIN+VAL only (frozen test split untouched):
#            train on dataset_mb ≤ 100, predict dataset_mb > 100.
# Test 2     E1 extrapolation runs (models refit on train+val) — descriptive
#            only: there are very few LogReg rows beyond the local range.
# Decision   Scout "helps" for a target if scout_trend beats BOTH local and
#            local_trend on the rows beyond s_max in Test 1, with the 95 %
#            group-bootstrap interval of the MAE difference below 0.
#
# Inputs:  data/scout/scout_hosts.csv, data/local/local_drivers.csv
# Output:  data/models/transfer_size/  (git-ignored)
#   size_trend_metrics.csv, size_trend_predictions.csv, report.json
#
#   python scripts/transfer_size_trend.py
#
# CLI FLAGS
#   --scout-hosts PATH  --drivers PATH  --out-dir PATH
#   --workload NAME     workload with a Scout size trend (default logreg)
#   --small-max MB      Test 1 trains on dataset_mb <= this (default 100)
#   --rf-trees N        trees in the random forest (default 200)
#   --n-boot N          bootstrap resamples (default 2000)
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
HOLDOUT_SET = "E1"
METHODS = ["local", "flat", "scout_trend", "local_trend"]
COMPARISONS = [("scout_trend", "local"), ("scout_trend", "local_trend")]


def _parse_args(argv=None):
    p = argparse.ArgumentParser(description="Scout data-size trend transfer (option A).")
    p.add_argument(
        "--scout-hosts", default=os.path.join(_PROJECT_ROOT, "data", "scout", "scout_hosts.csv")
    )
    p.add_argument(
        "--drivers", default=os.path.join(_PROJECT_ROOT, "data", "local", "local_drivers.csv")
    )
    p.add_argument(
        "--out-dir", default=os.path.join(_PROJECT_ROOT, "data", "models", "transfer_size")
    )
    p.add_argument("--workload", default="logreg")
    p.add_argument("--small-max", type=float, default=100.0)
    p.add_argument("--rf-trees", type=int, default=200)
    p.add_argument("--n-boot", type=int, default=2000)
    return p.parse_args(argv)


def base_factory(target: str, rf_trees: int):
    """CV-chosen family per target with fixed hyper-parameters."""
    num = m.TRAINING_SETS[TRAINING_SET]["numeric"]
    if target.startswith("cpu"):
        return lambda: m.make_rf(5, 2, n_estimators=rf_trees)
    return lambda: m.make_ridge_wl(1.0, num)


def _shard(df: pd.DataFrame) -> np.ndarray:
    return (df["dataset_mb"] / df["num_workers"]).to_numpy(float)


def predict_methods(model, X, features, s_max, beta_scout, beta_local) -> dict:
    """Predictions of every method; a method whose β is unavailable is NaN, never "local"."""
    out = {"local": t.anchor_and_scale(model, X, features, s_max, None)}
    for name, beta in (("flat", 0.0), ("scout_trend", beta_scout), ("local_trend", beta_local)):
        if beta is None:
            out[name] = np.full(len(X), np.nan)
        else:
            out[name] = t.anchor_and_scale(model, X, features, s_max, beta)
    return out


def _rows(test, target, workload, scope, actual, preds, extra=None) -> list[dict]:
    out = []
    for name in METHODS:
        pred = preds[name]
        ok = ~np.isnan(pred)
        row = {"test": test, "target": target, "workload": workload, "rows": scope, "method": name}
        row |= m.regression_metrics(actual[ok], pred[ok]) if ok.any() else {"n": 0}
        out.append(row | (extra or {}))
    return out


def run_target(df, holdout, scout, target, args) -> tuple[list, list, dict]:
    wl = args.workload
    features = m.features_for(TRAINING_SET)
    factory = base_factory(target, args.rf_trees)
    scout_beta = t.scout_size_elasticity(scout, target, wl)
    beta_scout = scout_beta["beta"] if scout_beta else None
    metrics, preds_out = [], []
    info = {"scout": scout_beta}

    # ── Test 1: size stress test on train+val ─────────────────────────────
    d = df[df[target].notna()]
    train = d[d["dataset_mb"] <= args.small_max]
    evals = d[(d["dataset_mb"] > args.small_max) & (d["workload_type"] == wl)]
    model = factory().fit(train[features], train[target])
    s_max = float(_shard(train[train["workload_type"] == wl]).max())
    beta_local = t.local_size_elasticity(train, target, wl)
    preds = predict_methods(model, evals, features, s_max, beta_scout, beta_local)
    actual = evals[target].to_numpy(float)
    beyond = _shard(evals) > s_max
    metrics += _rows("size_stress", target, wl, "all", actual, preds)
    metrics += _rows(
        "size_stress",
        target,
        wl,
        "beyond_s_max",
        actual[beyond],
        {k: v[beyond] for k, v in preds.items()},
    )
    long = pd.concat(
        [
            pd.DataFrame(
                {
                    "config_group": evals["config_group"].values[beyond],
                    "workload_type": wl,
                    "model": name,
                    "actual": actual[beyond],
                    "predicted": preds[name][beyond],
                }
            )
            for name in METHODS
        ],
        ignore_index=True,
    )
    paired = {}
    if beyond.any() and beta_scout is not None:
        for a, b in COMPARISONS:
            if long.loc[long["model"] == b, "predicted"].notna().all():
                paired[f"{a}_vs_{b}"] = m.group_bootstrap(long, a, baseline=b, n_boot=args.n_boot)
    # decision rule: Scout must beat BOTH comparisons, CI entirely below 0
    helps = len(paired) == len(COMPARISONS) and all(
        p["mae_diff_ci95"][1] < 0 for p in paired.values()
    )
    info["size_stress"] = {
        "train_rows": len(train),
        "eval_rows": len(evals),
        "rows_beyond_s_max": int(beyond.sum()),
        "groups_beyond_s_max": int(evals.loc[beyond, "config_group"].nunique()),
        "s_max_shard_mb": s_max,
        "beta_local": beta_local,
        "paired": paired,
        "scout_helps": helps,
    }
    p1 = evals[["run_key", "driver_idx", "config_group", "dataset_mb", "num_workers"]].copy()
    p1["test"], p1["target"], p1["actual"] = "size_stress", target, actual
    for name in METHODS:
        p1[name] = preds[name]
    preds_out.append(p1)

    # ── Test 2: E1 extrapolation (descriptive) ────────────────────────────
    h = holdout[holdout[target].notna() & (holdout["workload_type"] == wl)]
    full = factory().fit(d[features], d[target])
    s_max_full = float(_shard(d[d["workload_type"] == wl]).max())
    beta_local_full = t.local_size_elasticity(d, target, wl)
    if len(h):
        hp = predict_methods(full, h, features, s_max_full, beta_scout, beta_local_full)
        h_actual = h[target].to_numpy(float)
        h_beyond = _shard(h) > s_max_full
        metrics += _rows(
            "E1",
            target,
            wl,
            "beyond_s_max",
            h_actual[h_beyond],
            {k: v[h_beyond] for k, v in hp.items()},
        )
        p2 = h[["run_key", "driver_idx", "config_group", "dataset_mb", "num_workers"]].copy()
        p2["test"], p2["target"], p2["actual"] = "E1", target, h_actual
        for name in METHODS:
            p2[name] = hp[name]
        preds_out.append(p2)
        info["E1"] = {
            "rows_beyond_s_max": int(h_beyond.sum()),
            "s_max_shard_mb": s_max_full,
            "beta_local": beta_local_full,
        }
    return metrics, preds_out, info


def main(argv=None):
    args = _parse_args(argv)
    raw = pd.read_csv(args.drivers)
    df, rows_info = m.select_training_rows(raw, TRAINING_SET)
    df = df[df["split"].isin(["train", "val"])]  # frozen test split untouched
    holdout = raw[(raw["split"] == "holdout") & (raw["experiment"] == HOLDOUT_SET)]
    holdout = holdout[(holdout["status"] == "ok") & holdout["thread_cap"].notna()]
    holdout = m.add_derived(holdout) if len(holdout) else holdout
    scout = pd.read_csv(args.scout_hosts)
    os.makedirs(args.out_dir, exist_ok=True)

    metrics, preds, info = [], [], {}
    for target in m.TARGETS:
        rows, p, i = run_target(df, holdout, scout, target, args)
        metrics += rows
        preds += p
        info[target] = i
    metrics = pd.DataFrame(metrics)
    metrics.to_csv(os.path.join(args.out_dir, "size_trend_metrics.csv"), index=False)
    pd.concat(preds).to_csv(os.path.join(args.out_dir, "size_trend_predictions.csv"), index=False)

    def _sha(path):
        with open(path, "rb") as fh:
            return hashlib.sha256(fh.read()).hexdigest()

    other = sorted(set(df["workload_type"]) - {args.workload})
    report = {
        "question": "Does Scout's data-size trend improve predictions beyond the local size range?",
        "protocol": "see script header (fixed before results); decision on Test 1, rows beyond s_max",
        "workload": args.workload,
        "not_applicable": {wl: "Scout records no input size for this workload" for wl in other}
        if args.workload == "logreg"
        else {},
        "small_max_mb": args.small_max,
        "data": {
            "drivers_sha256": _sha(args.drivers),
            "scout_hosts_sha256": _sha(args.scout_hosts),
            "rows": rows_info,
        },
        "targets": info,
    }
    with open(os.path.join(args.out_dir, "report.json"), "w") as fh:
        json.dump(report, fh, indent=2, default=str)

    print(f"── Scout size elasticity β ({args.workload}, within VM type) ──")
    for target, i in info.items():
        s = i["scout"]
        bl = i["size_stress"]["beta_local"]
        print(
            f"   {target:16s} β_scout {s['beta']:+.3f} ({s['vm_types']} VM types)   "
            f"β_local {bl if bl is None else round(bl, 3)}"
            if s
            else f"   {target:16s} no Scout trend"
        )
    show = metrics[["test", "target", "rows", "method", "n", "mae", "r2"]]
    print("── MAE by method ──")
    print(show.round(3).to_string(index=False))
    print(
        "── paired differences on rows beyond s_max (Test 1), 95 % CI; negative = Scout better ──"
    )
    for target, i in info.items():
        for name, p in i["size_stress"]["paired"].items():
            lo, hi = p["mae_diff_ci95"]
            print(
                f"   {target:16s} {name:28s} {p['mae_diff_vs_baseline']:+.3f} [{lo:+.3f}, {hi:+.3f}]"
            )
        print(f"   {target:16s} → Scout helps: {i['size_stress']['scout_helps']}")
    print(f"[transfer-size] outputs written to {args.out_dir}")
    return report


if __name__ == "__main__":
    main()
