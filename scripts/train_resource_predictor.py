#!/usr/bin/env python
# =============================================================================
# scripts/train_resource_predictor.py
# Phase 6 — train and evaluate the per-driver resource predictor (2b)
#
# STAGE 1 — local-only baselines
# ------------------------------
# For each target (models.TARGETS: CPU p95 and mean per driver, memory p95 and
# peak of the largest driver per run) this script:
#   Models: lookup, ridge (shared slopes), ridge_wl (per-workload slopes), rf.
#   1. runs leave-one-configuration-out CV on TRAIN+VAL: every configuration
#      group is predicted by models trained on all other groups,
#   2. picks Ridge alpha and RF depth/leaf size by CV MAE,
#   3. reports CV errors with 95 % group-bootstrap intervals and paired
#      differences against the lookup baseline,
#   4. calibrates the empirical upper margin PER WORKLOAD from out-of-fold
#      residuals (never from TEST),
#   5. refits the chosen models on TRAIN+VAL and evaluates ONCE on TEST,
#   6. scores the same models on the E1 holdout runs, separately for
#      interior sizes and extrapolation (handoff validation protocol),
#   7. reports the noise ceiling: the R² that replicate runs of one exact
#      configuration (Set D + E2) reach — no model can be expected to beat it,
#   8. stores growth trends (elasticities along data size and cores, from the
#      local train+val rows, from Scout, and Scout-informed) in the model card;
#      the predictor applies the ones its TREND_POLICY selects beyond the
#      measured range (see scripts/transfer_trend_tests.py).
#
# The model card also records, per workload, the input values the models
# were trained on (valid_inputs) and the thread-cap rule of the training
# runs; mpj_spark/resource_prediction/predictor.py uses both to flag
# requests outside the measured domain.
#
# TRAINING SETS (never mixed; see models.TRAINING_SETS)
#   capped    thread-capped DOE runs, handoff inputs   (default)
#   uncapped  runs without a thread cap (none in the current data)
#
# Input:  data/local/local_drivers.csv  (scripts/build_local_dataset.py)
# Output: data/models/local_<training-set>/  (git-ignored)
#   metrics.csv          test metrics per target × model × workload
#   cv_metrics.csv       out-of-fold CV metrics per target × model × workload
#   predictions_test.csv per-row test predictions and upper bounds
#   holdout_metrics.csv  E1 holdout metrics per target × model × kind
#   predictions_holdout.csv  per-row E1 predictions
#   <target>__<model>.joblib   fitted models (refit on train+val)
#   model_card.json      data, features, valid inputs, margins, choices, versions
#
#   python scripts/train_resource_predictor.py
#
# CLI FLAGS
#   --drivers PATH         local_drivers.csv     (default: data/local/local_drivers.csv)
#   --training-set NAME    capped (default) | uncapped
#   --out-dir PATH         output directory      (default: data/models/local_<set>)
#   --unknown-alloc MODE   drop (default) — rows without recorded cores/heap
#                          (or, for capped, without the workload design)
#                          are excluded until those inputs are confirmed
#   --upper-quantile Q     residual quantile for the upper bound (default 0.9)
#   --scout-hosts PATH     Scout hosts table for the Scout trends (skipped if absent)
# =============================================================================

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys

import joblib
import numpy as np
import pandas as pd
import sklearn

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from mpj_spark.resource_prediction import models as m  # noqa: E402
from mpj_spark.resource_prediction import transfer as t  # noqa: E402
from mpj_spark.resource_prediction.predictor import CARD_SCHEMA_VERSION  # noqa: E402

# inputs with at most this many distinct training values also list them
_MAX_LISTED_VALUES = 12


def _parse_args(argv=None):
    p = argparse.ArgumentParser(description="Train the per-driver resource predictor.")
    p.add_argument(
        "--drivers", default=os.path.join(_PROJECT_ROOT, "data", "local", "local_drivers.csv")
    )
    p.add_argument("--training-set", choices=sorted(m.TRAINING_SETS), default="capped")
    p.add_argument("--out-dir", default=None)
    p.add_argument("--unknown-alloc", choices=["drop"], default="drop")
    p.add_argument("--upper-quantile", type=float, default=0.9)
    p.add_argument(
        "--scout-hosts", default=os.path.join(_PROJECT_ROOT, "data", "scout", "scout_hosts.csv")
    )
    return p.parse_args(argv)


# holdout set scored by the final models (interior / extrapolation kinds)
HOLDOUT_SCORED_SET = "E1"
# replicate sets for the noise ceiling (same configuration repeated)
NOISE_CEILING_SETS = ("D", "E2")


def load_training_rows(path: str, training_set: str) -> tuple[pd.DataFrame, dict]:
    return m.select_training_rows(pd.read_csv(path), training_set)


def load_holdout_rows(path: str, training_set: str) -> pd.DataFrame:
    """E1 holdout rows of the same environment (empty if the table has none)."""
    df = pd.read_csv(path)
    if "experiment" not in df or "split" not in df:
        return df.iloc[0:0]
    h = df[(df["split"] == "holdout") & (df["experiment"] == HOLDOUT_SCORED_SET)]
    h = h[h["status"] == "ok"]
    capped = h["thread_cap"].notna()
    h = h[capped if training_set == "capped" else ~capped]
    return m.add_derived(h) if len(h) else h


def noise_ceiling(path: str, target: str) -> dict | None:
    """
    R² reached by the mean of replicate runs of each exact configuration
    (Sets D + E2): 1 − within-configuration SS / total SS.
    """
    df = pd.read_csv(path)
    if "exact_config" not in df or "experiment" not in df:
        return None
    d = df[df["experiment"].isin(NOISE_CEILING_SETS) & (df["status"] == "ok")]
    d = d[d[target].notna()]
    d = d[d.groupby("exact_config")[target].transform("size") >= 2]
    if d.empty:
        return None
    y = d[target].to_numpy(float)
    within = d[target] - d.groupby("exact_config")[target].transform("mean")
    total = ((y - y.mean()) ** 2).sum()
    return {
        "r2": float(1 - (within**2).sum() / total) if total > 0 else None,
        "n_configs": int(d["exact_config"].nunique()),
        "n_rows": len(d),
        "within_config_sd": float(np.sqrt((within**2).mean())),
    }


def _fit_predict(model, train, test, target, features):
    model.fit(train[features], train[target])
    return model, model.predict(test[features])


def _cv_rows(oof: pd.DataFrame, target: str) -> list[dict]:
    """Pooled out-of-fold metrics per model, overall and per workload."""
    rows = []
    for name, g in oof.groupby("model"):
        for wl, gw in [("all", g)] + list(g.groupby("workload_type")):
            rows.append(
                {
                    "target": target,
                    "unit": m.TARGETS[target],
                    "model": name,
                    "workload": wl,
                    "n_groups": int(gw["config_group"].nunique()),
                    **m.regression_metrics(gw["actual"], gw["predicted"]),
                }
            )
    return rows


def _holdout_rows(h, target, name, pred, upper) -> list[dict]:
    """E1 metrics of one model, overall and per holdout kind."""
    rows = []
    kinds = [("all", h.index)] + [(k, g.index) for k, g in h.groupby("holdout_kind")]
    for kind, idx in kinds:
        mask = h.index.isin(idx)
        rows.append(
            {
                "target": target,
                "unit": m.TARGETS[target],
                "model": name,
                "kind": kind,
                **m.regression_metrics(h[target][mask], pred[mask], upper[mask]),
            }
        )
    return rows


def select_and_train(
    df: pd.DataFrame,
    target: str,
    q: float,
    training_set: str = "uncapped",
    holdout: pd.DataFrame | None = None,
) -> tuple[list[dict], list[dict], list, dict]:
    """Return (test metric rows, CV metric rows, test prediction frames, choices)."""
    features = m.features_for(training_set)
    num = m.TRAINING_SETS[training_set]["numeric"]
    d = df[df[target].notna()]
    trval = d[d["split"].isin(["train", "val"])]
    test = d[d["split"] == "test"]

    # ── selection by leave-one-configuration-out CV on TRAIN+VAL ──────────
    candidates = {"lookup": lambda: m.LookupModel(features)}
    candidates |= {f"ridge|{a}": (lambda a=a: m.make_ridge(a, num)) for a in m.RIDGE_ALPHAS}
    candidates |= {f"ridge_wl|{a}": (lambda a=a: m.make_ridge_wl(a, num)) for a in m.RIDGE_ALPHAS}
    candidates |= {
        f"rf|{p['max_depth']}|{p['min_samples_leaf']}": (lambda p=p: m.make_rf(**p))
        for p in m.RF_GRID
    }
    oof_all = m.group_cv_predictions(trval, target, candidates, features=features)
    cand_mae = (
        oof_all.assign(ae=(oof_all["predicted"] - oof_all["actual"]).abs())
        .groupby("model")["ae"]
        .mean()
    )
    best = {
        family: cand_mae[[c for c in cand_mae.index if c.split("|")[0] == family]].idxmin()
        for family in ("lookup", "ridge", "ridge_wl", "rf")
    }
    factories = {family: candidates[key] for family, key in best.items()}
    oof = oof_all[oof_all["model"].isin(best.values())].replace(
        {"model": {v: k for k, v in best.items()}}
    )
    cv_rows = _cv_rows(oof, target)
    uncertainty = {
        name: m.group_bootstrap(oof, name, baseline=None if name == "lookup" else "lookup")
        for name in factories
    }
    # does a per-workload slope improve on the shared-slope Ridge?
    uncertainty["ridge_wl_vs_ridge"] = m.group_bootstrap(oof, "ridge_wl", baseline="ridge")

    h = holdout[holdout[target].notna()] if holdout is not None else None
    rows, preds, fitted, margins_by_model = [], [], {}, {}
    holdout_rows, holdout_preds = [], []
    for name, factory in factories.items():
        # per-workload upper margins from out-of-fold residuals of the selection
        margins = m.upper_margins(oof[oof["model"] == name], q)
        margins_by_model[name] = margins
        # final model on TRAIN+VAL, evaluated once on TEST
        model, test_pred = _fit_predict(factory(), trval, test, target, features)
        upper = test_pred + np.array([m.margin_for(margins, w) for w in test["workload_type"]])
        fitted[name] = model
        for wl, idx in [("all", test.index)] + [
            (w, g.index) for w, g in test.groupby("workload_type")
        ]:
            mask = test.index.isin(idx)
            rows.append(
                {
                    "target": target,
                    "unit": m.TARGETS[target],
                    "model": name,
                    "workload": wl,
                    "upper_margin": np.nan if wl == "all" else m.margin_for(margins, wl),
                    **m.regression_metrics(test[target][mask], test_pred[mask], upper[mask]),
                }
            )
        p = test[["run_key", "driver_idx", *features]].copy()
        p["target"], p["model"] = target, name
        p["actual"], p["predicted"], p["upper"] = test[target].values, test_pred, upper
        if name == "lookup":
            p["lookup_level"] = model.level_used_
        preds.append(p)
        # the same final model on the E1 holdout (never used for fitting)
        if h is not None and len(h):
            h_pred = model.predict(h[features])
            h_upper = h_pred + np.array([m.margin_for(margins, w) for w in h["workload_type"]])
            holdout_rows += _holdout_rows(h, target, name, h_pred, h_upper)
            ph = h[["run_key", "driver_idx", "holdout_kind", *features]].copy()
            ph["target"], ph["model"] = target, name
            ph["actual"], ph["predicted"], ph["upper"] = h[target].values, h_pred, h_upper
            holdout_preds.append(ph)

    choices = {
        "selected": best,
        "cv_mae_by_candidate": cand_mae.round(4).to_dict(),
        "cv_uncertainty": uncertainty,
        "upper_margins": margins_by_model,
        "n_trainval_rows": len(trval),
        "n_trainval_groups": int(trval["config_group"].nunique()),
        "n_test_rows": len(test),
        "n_test_groups": int(test["config_group"].nunique()),
        "n_holdout_rows": 0 if h is None else len(h),
        "fitted": fitted,
        "holdout_rows": holdout_rows,
        "holdout_preds": holdout_preds,
    }
    return rows, cv_rows, preds, choices


def valid_inputs(df: pd.DataFrame, features: list[str]) -> dict:
    """Per workload: min / max (and the distinct values, if few) of each numeric input."""
    out = {}
    for wl, g in df.groupby("workload_type"):
        out[wl] = {}
        for col in features:
            if col in m.CATEGORICAL_FEATURES or col in m.DERIVED_FEATURES:
                continue
            vals = sorted(float(v) for v in g[col].dropna().unique())
            entry = {"min": vals[0], "max": vals[-1]}
            if len(vals) <= _MAX_LISTED_VALUES:
                entry["values"] = vals
            out[wl][col] = entry
    return out


def growth_trends(trval: pd.DataFrame, scout: pd.DataFrame | None) -> dict:
    """{axis: {workload: {target: x_max + local / Scout / Scout-informed elasticity}}}."""
    out = {}
    for axis in t.TREND_AXES:
        for wl, g in trval.groupby("workload_type"):
            x_max = float(t.axis_value(g, axis).max())
            for target in m.TARGETS:
                local = t.local_elasticity(trval, target, wl, axis)
                scout_b = t.scout_elasticity(scout, target, wl, axis) if scout is not None else None
                out.setdefault(axis, {}).setdefault(wl, {})[target] = {
                    "x_max": x_max,
                    "local": local,
                    "scout": scout_b,
                    "combined": t.combine_elasticity(local, scout_b),
                }
    return out


def thread_cap_policy(df: pd.DataFrame) -> str:
    """How the thread cap related to the cores in the training runs."""
    if "thread_cap" not in df or df["thread_cap"].isna().all():
        return "uncapped"
    if (df["thread_cap"] == df["cores_per_driver"]).all():
        return "equals_cores"
    return "mixed"


def main(argv=None):
    args = _parse_args(argv)
    if args.out_dir is None:
        args.out_dir = os.path.join(_PROJECT_ROOT, "data", "models", f"local_{args.training_set}")
    df, info = load_training_rows(args.drivers, args.training_set)
    holdout = load_holdout_rows(args.drivers, args.training_set)
    features = m.features_for(args.training_set)
    os.makedirs(args.out_dir, exist_ok=True)

    all_rows, all_cv, all_preds, card_targets = [], [], [], {}
    all_holdout, all_holdout_preds, ceilings = [], [], {}
    for target in m.TARGETS:
        rows, cv_rows, preds, choices = select_and_train(
            df, target, args.upper_quantile, args.training_set, holdout
        )
        all_rows += rows
        all_cv += cv_rows
        all_preds += preds
        all_holdout += choices.pop("holdout_rows")
        all_holdout_preds += choices.pop("holdout_preds")
        ceilings[target] = noise_ceiling(args.drivers, target)
        for name, model in choices.pop("fitted").items():
            joblib.dump(model, os.path.join(args.out_dir, f"{target}__{name}.joblib"))
        card_targets[target] = choices

    metrics = pd.DataFrame(all_rows)
    metrics.to_csv(os.path.join(args.out_dir, "metrics.csv"), index=False)
    cv_metrics = pd.DataFrame(all_cv)
    cv_metrics.to_csv(os.path.join(args.out_dir, "cv_metrics.csv"), index=False)
    pd.concat(all_preds).to_csv(os.path.join(args.out_dir, "predictions_test.csv"), index=False)
    holdout_metrics = pd.DataFrame(all_holdout)
    if all_holdout:
        holdout_metrics.to_csv(os.path.join(args.out_dir, "holdout_metrics.csv"), index=False)
        pd.concat(all_holdout_preds).to_csv(
            os.path.join(args.out_dir, "predictions_holdout.csv"), index=False
        )

    with open(args.drivers, "rb") as fh:
        data_sha = hashlib.sha256(fh.read()).hexdigest()
    trval = df[df["split"].isin(["train", "val"])]
    card = {
        "card_schema_version": CARD_SCHEMA_VERSION,
        "model_version": f"{args.training_set}-{data_sha[:12]}",
        "stage": "local_only_baselines",
        "training_set": args.training_set,
        "training_set_description": m.TRAINING_SETS[args.training_set]["description"],
        "protocol": "leave-one-configuration-out CV on train+val for selection, upper-margin calibration and uncertainty; single final evaluation on test",
        "data": {"drivers_csv": os.path.relpath(args.drivers, _PROJECT_ROOT), "sha256": data_sha},
        "rows": info,
        "features": features,
        "targets": m.TARGETS,
        "thread_caps_seen": sorted(df["thread_cap"].dropna().unique().tolist())
        if "thread_cap" in df
        else [],
        "thread_cap_policy": thread_cap_policy(df),
        "valid_inputs": valid_inputs(trval, features),
        "valid_inputs_note": "train+val rows the deployed models were fitted on",
        "valid_ranges": {c: [float(df[c].min()), float(df[c].max())] for c in features[1:]}
        | {"workload_type": sorted(df["workload_type"].unique())},
        "rows_by_split": df["split"].value_counts().to_dict(),
        "holdout": {
            "set": HOLDOUT_SCORED_SET,
            "rows_by_kind": holdout["holdout_kind"].value_counts().to_dict()
            if len(holdout)
            else {},
        },
        "noise_ceiling": ceilings,
        "growth_trends": growth_trends(
            trval,
            pd.read_csv(args.scout_hosts) if os.path.isfile(args.scout_hosts) else None,
        ),
        "growth_trends_note": "x_max = largest data per driver (size) or cores (cores) in "
        "train+val; beta: demand ∝ x^beta; applied by the predictor's TREND_POLICY only",
        "upper_quantile": args.upper_quantile,
        "choices": card_targets,
        "versions": {
            "python": platform.python_version(),
            "scikit-learn": sklearn.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
    }
    with open(os.path.join(args.out_dir, "model_card.json"), "w") as fh:
        json.dump(card, fh, indent=2, default=str)

    show = metrics[metrics["workload"] == "all"][
        [
            "target",
            "unit",
            "model",
            "n",
            "mae",
            "rmse",
            "mape_pct",
            "r2",
            "underprediction_rate",
            "upper_bound_coverage",
        ]
    ]
    print(f"[train] rows used: {info}")
    cv_show = cv_metrics[cv_metrics["workload"] == "all"][
        ["target", "unit", "model", "n_groups", "mae", "mape_pct", "underprediction_rate"]
    ]
    print("── leave-one-configuration-out CV (train+val) ──")
    print(cv_show.round(3).to_string(index=False))
    for target, ch in card_targets.items():
        for name, u in ch["cv_uncertainty"].items():
            diff = (
                f"  diff vs {u['baseline']} {u['mae_diff_vs_baseline']:+.3f} "
                f"[{u['mae_diff_ci95'][0]:+.3f}, {u['mae_diff_ci95'][1]:+.3f}]"
                if "baseline" in u
                else ""
            )
            print(
                f"   {target:15s} {name:17s} MAE {u['mae']:.3f} "
                f"[{u['mae_ci95'][0]:.3f}, {u['mae_ci95'][1]:.3f}]{diff}"
            )
    margin_rows = [
        {"target": t, "model": name, **margins}
        for t, ch in card_targets.items()
        for name, margins in ch["upper_margins"].items()
    ]
    print(f"── upper margins per workload (q = {args.upper_quantile}, out-of-fold) ──")
    print(pd.DataFrame(margin_rows).round(3).to_string(index=False))
    print("── final evaluation on TEST ──")
    print(show.round(3).to_string(index=False))
    if all_holdout:
        print(f"── {HOLDOUT_SCORED_SET} holdout (models refit on train+val) ──")
        cols = ["target", "model", "kind", "n", "mae", "r2", "upper_bound_coverage"]
        print(holdout_metrics[cols].round(3).to_string(index=False))
    print("── noise ceiling (replicates of Sets " + " + ".join(NOISE_CEILING_SETS) + ") ──")
    for target, c in ceilings.items():
        if c:
            print(f"   {target:16s} R2 {c['r2']:.3f}  ({c['n_configs']} configurations)")
    print(f"[train] outputs written to {args.out_dir}")
    return metrics


if __name__ == "__main__":
    main()
