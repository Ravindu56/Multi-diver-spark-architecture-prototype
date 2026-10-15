#!/usr/bin/env python
# =============================================================================
# scripts/upper_bound_tests.py
# Phase 6 — step A3: how should the upper bounds (safety limits) be set?
#
# PROTOCOL (fixed before this test was run)
# -----------------------------------------
# Models      the deployed family per target and workload (predictor
#             FAMILY_POLICY / WORKLOAD_FAMILY_POLICY / best_cv), CV-selected
#             hyper-parameters from the model card
# Residuals   leave-one-configuration-out out-of-fold predictions on TRAIN+VAL
# Methods     models.MARGIN_METHODS (quantile q = 0.9):
#               additive_rows    current method
#               additive_groups  group-weighted, conformal level correction
#               relative_groups  same on (actual − predicted) / predicted
# Calibration-CV  configuration groups split into 5 folds (hash of the group
#             name); margins set on 4 folds' residuals, coverage and mean
#             width measured on the 5th; repeated for every fold
# Rule        per target × workload, among methods with calibration-CV
#             coverage >= 0.88, the smallest mean width; if none reaches 0.88,
#             the highest coverage
# Beyond the measured range
#             relative group-weighted margins from train+val stress tests that
#             hold out larger jobs / more cores and predict them as the
#             predictor would (model + TREND_POLICY trend):
#               size   K-Means data per driver > 50 MB, LogReg > 100 MB
#               cores  K-Means cores > 2, LogReg cores > 4
#             applied to out-of-range requests as a floor:
#               upper = max(in-range upper, predicted × (1 + extrapolation margin))
# Confirmation  test split and E1 coverage (seen before for other decisions,
#             so not blind)
#
# AMENDMENT 1 (after the first run, before re-running)
# ----------------------------------------------------
# All Set A K-Means runs are at 4 cores, so holding out K-Means rows above 2
# cores also removes every Set A design from training; scoring those Set A
# rows measured errors from unseen job designs, not from more cores (+253 %
# memory margin).  The cores stress tests are therefore scored on Set D rows
# only (the core sweeps: same job, only cores differ), as in the cores-trend
# test of scripts/transfer_trend_tests.py.  Training is unchanged.
# The stress-test code is shared with the training script:
# mpj_spark/resource_prediction/calibration.py.
#
# Output:  data/models/upper_bounds/  (git-ignored) — report.json etc.
#
#   python scripts/upper_bound_tests.py
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

from mpj_spark.resource_prediction import calibration as cal  # noqa: E402
from mpj_spark.resource_prediction import models as m  # noqa: E402
from mpj_spark.resource_prediction import predictor as pr  # noqa: E402
from mpj_spark.resource_prediction import transfer as t  # noqa: E402

TRAINING_SET = "capped"
QUANTILE = 0.9
N_FOLDS = 5
MIN_COVERAGE = 0.88


def _parse_args(argv=None):
    p = argparse.ArgumentParser(description="Upper-bound calibration tests (step A3).")
    p.add_argument(
        "--drivers", default=os.path.join(_PROJECT_ROOT, "data", "local", "local_drivers.csv")
    )
    p.add_argument(
        "--model-dir", default=os.path.join(_PROJECT_ROOT, "data", "models", "local_capped")
    )
    p.add_argument(
        "--scout-hosts", default=os.path.join(_PROJECT_ROOT, "data", "scout", "scout_hosts.csv")
    )
    p.add_argument(
        "--out-dir", default=os.path.join(_PROJECT_ROOT, "data", "models", "upper_bounds")
    )
    p.add_argument("--rf-trees", type=int, default=200)
    return p.parse_args(argv)


def deployed_keys(card: dict) -> dict:
    """{target: {workload: CV selection key}} of the families the predictor uses."""
    out = {}
    for target in m.TARGETS:
        default = pr._pick_family(card, target, "best_cv")
        sel = card["choices"][target]["selected"]
        out[target] = {}
        for wl in card["valid_inputs"]:
            fam = pr.WORKLOAD_FAMILY_POLICY.get(target, {}).get(wl, default)
            out[target][wl] = sel[fam]
    return out


def fold_of(group: str) -> int:
    return int(hashlib.sha256(group.encode()).hexdigest()[:8], 16) % N_FOLDS


def oof_deployed(trval, target, keys, rf_trees) -> pd.DataFrame:
    """Out-of-fold predictions of the deployed family for each workload's rows."""
    num = m.TRAINING_SETS[TRAINING_SET]["numeric"]
    features = m.features_for(TRAINING_SET)
    d = trval[trval[target].notna()]
    facs = {k: m.factory_from_key(k, num, rf_trees) for k in set(keys.values())}
    oof = m.group_cv_predictions(d, target, facs, features=features)
    keep = oof["model"] == oof["workload_type"].map(keys)
    return oof[keep].reset_index(drop=True)


def calibration_cv(oof: pd.DataFrame) -> dict:
    """{workload: {method: coverage, width}} over N_FOLDS group folds."""
    out = {}
    folds = oof["config_group"].map(fold_of)
    for wl in sorted(oof["workload_type"].unique()):
        w = oof["workload_type"] == wl
        res = {}
        for method in m.MARGIN_METHODS:
            hit, width = [], []
            for f in range(N_FOLDS):
                cal, ev = oof[w & (folds != f)], oof[w & (folds == f)]
                if ev.empty:
                    continue
                margin = m.calibrate_margin(cal, QUANTILE, method)
                upper = m.apply_margin(ev["predicted"].to_numpy(float), margin, method)
                hit.append(ev["actual"].to_numpy(float) <= upper)
                width.append(upper - ev["predicted"].to_numpy(float))
            res[method] = {
                "coverage": float(np.concatenate(hit).mean()),
                "mean_width": float(np.concatenate(width).mean()),
            }
        out[wl] = res
    return out


def decide(cv: dict) -> dict:
    out = {}
    for wl, res in cv.items():
        ok = [mt for mt, r in res.items() if r["coverage"] >= MIN_COVERAGE]
        if ok:
            out[wl] = min(ok, key=lambda mt: res[mt]["mean_width"])
        else:
            out[wl] = max(res, key=lambda mt: res[mt]["coverage"])
    return out


def confirm(raw, card, target, margins_by_method, extrap, model_dir) -> list[dict]:
    """Coverage / width of the deployed predictor's points under each method."""
    pred = pr.ResourcePredictor.load(model_dir)
    trends = card["growth_trends"]
    sets = {
        "test": raw[raw["split"] == "test"],
        "E1 interior": raw[(raw["experiment"] == "E1") & (raw["holdout_kind"] == "interior")],
        "E1 extrap": raw[(raw["experiment"] == "E1") & (raw["holdout_kind"] == "extrap")],
    }
    rows = []
    for name, df in sets.items():
        df = df[(df["status"] == "ok") & df[target].notna()]
        if df.empty:
            continue
        reqs = [
            pr.PredictionRequest(
                r.workload_type, r.dataset_mb, int(r.num_workers), r.cores_per_driver,
                int(r.iterations), int(r.num_features), int(r.num_clusters),
            )
            for r in df.itertuples()
        ]  # fmt: skip
        point = np.array([getattr(r, target) for r in pred.predict_many(reqs)])
        actual = df[target].to_numpy(float)
        for method, margins in margins_by_method.items():
            for wl in ("all", *sorted(df["workload_type"].unique())):
                mask = (
                    (df["workload_type"] == wl).to_numpy()
                    if wl != "all"
                    else np.ones(len(df), bool)
                )
                ups = []
                for k in np.flatnonzero(mask):
                    w = df["workload_type"].iloc[k]
                    up = m.apply_margin(point[k], m.margin_for(margins, w), method)
                    for axis in ("size", "cores"):
                        x_max = trends[axis][w][target]["x_max"]
                        value = t.axis_value(df.iloc[[k]], axis)[0]
                        r_ext = extrap.get(axis, {}).get(w, {}).get("margin")
                        if value > x_max and r_ext is not None:
                            up = max(up, point[k] * (1 + r_ext))
                    ups.append(up)
                ups = np.array(ups)
                rows.append(
                    {
                        "target": target,
                        "set": name,
                        "workload": wl,
                        "method": method,
                        "n": int(mask.sum()),
                        "coverage": float((actual[mask] <= ups).mean()),
                        "mean_width": float((ups - point[mask]).mean()),
                    }
                )
    return rows


def main(argv=None):
    args = _parse_args(argv)
    with open(os.path.join(args.model_dir, "model_card.json")) as fh:
        card = json.load(fh)
    raw = pd.read_csv(args.drivers)
    df, rows_info = m.select_training_rows(raw, TRAINING_SET)
    trval = df[df["split"].isin(["train", "val"])]
    scout = pd.read_csv(args.scout_hosts) if os.path.isfile(args.scout_hosts) else None
    os.makedirs(args.out_dir, exist_ok=True)

    keys = deployed_keys(card)
    results, confirmation = {}, []
    for target in m.TARGETS:
        oof = oof_deployed(trval, target, keys[target], args.rf_trees)
        cv = calibration_cv(oof)
        choice = decide(cv)
        full = {method: {} for method in m.MARGIN_METHODS}
        for wl, g in oof.groupby("workload_type"):
            for method in m.MARGIN_METHODS:
                full[method][wl] = m.calibrate_margin(g, QUANTILE, method)
        for method in m.MARGIN_METHODS:
            full[method]["pooled"] = m.calibrate_margin(oof, QUANTILE, method)
        extrap = cal.stress_margins(
            trval, target, keys[target], pr.TREND_POLICY, scout, TRAINING_SET, args.rf_trees
        )
        results[target] = {
            "families": keys[target],
            "calibration_cv": cv,
            "decision": choice,
            "margins_full_oof": full,
            "extrapolation_margins": extrap,
        }
        confirmation += confirm(raw, card, target, full, extrap, args.model_dir)

    pd.DataFrame(confirmation).to_csv(os.path.join(args.out_dir, "confirmation.csv"), index=False)
    with open(args.drivers, "rb") as fh:
        sha = hashlib.sha256(fh.read()).hexdigest()
    report = {
        "question": "How should the upper bounds be calibrated?",
        "protocol": "see script header (fixed before this test; amendment 1 after the first run)",
        "quantile": QUANTILE,
        "data": {"drivers_sha256": sha, "rows": rows_info},
        "results": results,
        "decision": {tg: r["decision"] for tg, r in results.items()},
    }
    with open(os.path.join(args.out_dir, "report.json"), "w") as fh:
        json.dump(report, fh, indent=2, default=str)

    for target, r in results.items():
        print(f"── {target}  (families {r['families']}) ──")
        for wl, res in r["calibration_cv"].items():
            cells = "  ".join(
                f"{mt}: cover {v['coverage']:.3f} width {v['mean_width']:.3f}"
                for mt, v in res.items()
            )
            print(f"   {wl:7s} {cells}")
            print(f"           → chosen: {r['decision'][wl]}")
        for axis, by_wl in r["extrapolation_margins"].items():
            for wl, e in by_wl.items():
                print(
                    f"   beyond range ({axis}, {wl}): +{100 * e['margin']:.1f} %  "
                    f"({e['rows']} rows, {e['groups']} groups, trend {e['trend']})"
                )
    print("── confirmation (not blind): coverage / mean width by method ──")
    cf = pd.DataFrame(confirmation)
    print(
        cf[cf["workload"] != "all"]
        .pivot_table(
            index=["target", "set", "workload"], columns="method", values="coverage", sort=False
        )
        .round(3)
        .to_string()
    )
    print(f"[upper bounds] outputs written to {args.out_dir}")
    return report


if __name__ == "__main__":
    main()
