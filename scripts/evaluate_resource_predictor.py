#!/usr/bin/env python
# =============================================================================
# scripts/evaluate_resource_predictor.py
# Phase 6 — step A4: final evaluation of the deployed resource predictor
#
# WHAT IS EVALUATED
# -----------------
# The one predictor the allocator calls: ResourcePredictor.load(model_dir)
# with its defaults (model "best_cv", trends "policy"), i.e. FAMILY_POLICY,
# WORKLOAD_FAMILY_POLICY, TREND_POLICY, MARGIN_POLICY and the stress-test
# floors, through predict_many() exactly as the allocator will call it.
#
# Scored sets (never used for fitting, tuning or margin calibration)
#   test         frozen test split of Sets A, B, D
#   E1 interior  holdout sizes 75 / 125 MB
#   E1 extrap    holdout sizes 300 / 400 MB (beyond the measured range)
# Both were looked at during steps A1-A3: confirmation, not blind.
#
# Metrics (architecture section 7), per target — overall, per workload, per
# number of workers, per cores per driver:
#   MAE, RMSE (cores, MB); mean signed error (+ = over-prediction); MAPE;
#   under-prediction rate and rate of > 10 % under-prediction; R² (overall and
#   per workload only, and only with >= 3 configuration groups — compare with
#   the noise ceiling of replicate runs);
#   upper-bound coverage, rate above the upper bound, mean upper-bound width;
#   share of requests flagged out of range
# Uncertainty  95 % group bootstrap of MAE (configuration groups resampled)
#   and paired MAE differences on the same rows (PAIRS)
# Ablation (same rows, same models)
#   deployed            growth trends of TREND_POLICY (Scout-informed / local)
#   local_trends_only   the fixed rule's literal outcome: local trends only
#   no_trends           the models' own extrapolation
#   lookup              profile lookup (nearest measured configuration)
#   plus Scout label transfer (scripts/transfer_experiment.py) and growth-trend
#   tests (scripts/transfer_trend_tests.py), collated from their reports
# Leakage audit  exact-configuration overlap of every scored set with
#   train+val; one split per configuration group; no post-run column among
#   the inputs
# Latency  model load, one request, a batch of 100 requests (on the machine
#   this runs on: re-run on the laptop for the thesis figure)
#
# Output:  data/models/evaluation/  (git-ignored)
#   metrics.csv, paired.csv, report.json, tables.md (the tables of
#   docs/p6-evaluation-report.md)
#
#   python scripts/evaluate_resource_predictor.py
# =============================================================================

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time

import numpy as np
import pandas as pd
import sklearn

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
for _p in (_PROJECT_ROOT, _SCRIPT_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import train_resource_predictor as trp  # noqa: E402

from mpj_spark.resource_prediction import models as m  # noqa: E402
from mpj_spark.resource_prediction import predictor as pr  # noqa: E402

N_BOOT = 2000
# variant → (model, trends) passed to ResourcePredictor.load
VARIANTS = {
    "deployed": ("best_cv", "policy"),
    "local_trends_only": ("best_cv", "local_only"),
    "no_trends": ("best_cv", "off"),
    "lookup": ("lookup", "off"),
}
PAIRS = [("deployed", "lookup"), ("deployed", "local_trends_only"), ("deployed", "no_trends")]
BY = ("workload_type", "num_workers", "cores_per_driver")
# measured during or after a run, or bookkeeping: never a pre-launch input
POST_RUN_COLUMNS = (
    "wall_s", "exit_code", "status", "n_samples", "iters_logged",
    "cpu_cores_mean", "cpu_cores_p95", "cpu_cores_peak",
    "mem_mb_mean", "mem_mb_p95", "mem_mb_peak", "mem_mb_p95_max", "mem_mb_peak_max",
    "host_cpu_avg_pct", "host_mem_avg_mb", "host_mem_peak_mb", "host_mem_avail_min_mb",
    "run_id", "run_key", "timestamp", "rep", "split", "config_group", "exact_config",
)  # fmt: skip


def _parse_args(argv=None):
    p = argparse.ArgumentParser(description="Final evaluation of the resource predictor (A4).")
    p.add_argument(
        "--drivers", default=os.path.join(_PROJECT_ROOT, "data", "local", "local_drivers.csv")
    )
    p.add_argument(
        "--model-dir", default=os.path.join(_PROJECT_ROOT, "data", "models", "local_capped")
    )
    p.add_argument(
        "--models-root",
        default=os.path.join(_PROJECT_ROOT, "data", "models"),
        help="folder holding the transfer_capped / transfer_trends reports",
    )
    p.add_argument("--out-dir", default=os.path.join(_PROJECT_ROOT, "data", "models", "evaluation"))
    p.add_argument("--latency-reps", type=int, default=200)
    return p.parse_args(argv)


def _sha(path: str) -> str:
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def _load_json(path: str) -> dict | None:
    if not os.path.isfile(path):
        return None
    with open(path) as fh:
        return json.load(fh)


# ── predictions ───────────────────────────────────────────────────────────


def scored_sets(raw: pd.DataFrame) -> dict[str, pd.DataFrame]:
    ok = raw[raw["status"] == "ok"]
    e1 = ok[ok["experiment"] == "E1"]
    return {
        "test": ok[ok["split"] == "test"],
        "E1 interior": e1[e1["holdout_kind"] == "interior"],
        "E1 extrap": e1[e1["holdout_kind"] == "extrap"],
    }


def to_requests(df: pd.DataFrame) -> list[pr.PredictionRequest]:
    return [
        pr.PredictionRequest(
            r.workload_type, float(r.dataset_mb), int(r.num_workers), float(r.cores_per_driver),
            int(r.iterations), int(r.num_features), int(r.num_clusters),
        )
        for r in df.itertuples()
    ]  # fmt: skip


def predict_long(pred: pr.ResourcePredictor, df: pd.DataFrame) -> pd.DataFrame:
    """One row per (driver row, target) with a measured value."""
    res = pred.predict_many(to_requests(df))
    frames = []
    for target in m.TARGETS:
        actual = df[target].to_numpy(float)
        keep = ~np.isnan(actual)
        frames.append(
            pd.DataFrame(
                {
                    "target": target,
                    "workload_type": df["workload_type"].to_numpy()[keep],
                    "num_workers": df["num_workers"].to_numpy()[keep],
                    "cores_per_driver": df["cores_per_driver"].to_numpy()[keep],
                    "config_group": df["config_group"].to_numpy()[keep],
                    "actual": actual[keep],
                    "predicted": np.array([getattr(r, target) for r in res], float)[keep],
                    "upper": np.array([getattr(r, f"{target}_upper") for r in res], float)[keep],
                    "status": np.array([r.status for r in res])[keep],
                    "source": np.array([r.source[target] for r in res])[keep],
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


# ── metrics ───────────────────────────────────────────────────────────────


MIN_GROUPS_R2 = 3


def metrics(g: pd.DataFrame, r2: bool = True) -> dict:
    """
    regression_metrics plus upper-bound width and out-of-range share.  R² only
    where it can mean something: not for narrow slices (r2=False) and not
    with fewer than MIN_GROUPS_R2 configuration groups.
    """
    out = m.regression_metrics(g["actual"], g["predicted"], g["upper"])
    n_groups = int(g["config_group"].nunique())
    if not r2 or n_groups < MIN_GROUPS_R2:
        out["r2"] = None
    width = g["upper"] - g["predicted"]
    pos = g["predicted"].where(g["predicted"] > 0)
    return out | {
        "n_groups": n_groups,
        "above_upper_rate": 1 - out["upper_bound_coverage"],
        "mean_upper_width": float(width.mean()),
        "mean_upper_width_pct": float((width / pos).mean() * 100),
        "flagged_out_of_range": float((g["status"] == "out_of_domain").mean()),
    }


def _key(v) -> str:
    return pr._fmt(v) if isinstance(v, (int, float, np.integer, np.floating)) else str(v)


def metric_table(long: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (variant, set_name, target), g in long.groupby(["variant", "set", "target"], sort=False):
        base = {"variant": variant, "set": set_name, "target": target, "unit": m.TARGETS[target]}
        rows.append(base | {"by": "all", "value": "all"} | metrics(g))
        for col in BY:
            for v, gv in g.groupby(col):
                rows.append(base | {"by": col, "value": _key(v)} | metrics(gv, r2=col == BY[0]))
    return pd.DataFrame(rows)


def paired_table(long: pd.DataFrame) -> pd.DataFrame:
    """Group-bootstrap MAE intervals and paired differences on the same rows."""
    rows = []
    oof = long.rename(columns={"variant": "model"})
    for (set_name, target), g in oof.groupby(["set", "target"], sort=False):
        for wl in ("all", *sorted(g["workload_type"].unique())):
            s = g if wl == "all" else g[g["workload_type"] == wl]
            for model, baseline in PAIRS:
                b = m.group_bootstrap(s, model, baseline=baseline, n_boot=N_BOOT)
                rows.append(
                    {
                        "set": set_name,
                        "target": target,
                        "workload": wl,
                        "model": model,
                        "baseline": baseline,
                        "n_groups": b["n_groups"],
                        "mae": b["mae"],
                        "mae_ci95_low": b["mae_ci95"][0],
                        "mae_ci95_high": b["mae_ci95"][1],
                        "mae_diff": b["mae_diff_vs_baseline"],
                        "diff_ci95_low": b["mae_diff_ci95"][0],
                        "diff_ci95_high": b["mae_diff_ci95"][1],
                    }
                )
    return pd.DataFrame(rows)


def noise_ceilings(path: str) -> dict:
    return {
        target: {wl: trp.noise_ceiling(path, target, None if wl == "all" else wl)
                 for wl in ("all", "kmeans", "logreg")}
        for target in m.TARGETS
    }  # fmt: skip


# ── audit, latency, transfer ──────────────────────────────────────────────


def leakage_audit(raw: pd.DataFrame, sets: dict, features: list[str]) -> dict:
    trval = raw[raw["split"].isin(["train", "val"])]
    seen_cfg, seen_grp = set(trval["exact_config"]), set(trval["config_group"])
    scored = {**sets, "E2 (noise ceiling only, not scored)": raw[raw["experiment"] == "E2"]}
    overlap = {
        name: {
            "exact_configs": int(df["exact_config"].nunique()),
            "exact_configs_in_train_val": len(set(df["exact_config"]) & seen_cfg),
            "groups_in_train_val": len(set(df["config_group"]) & seen_grp),
        }
        for name, df in scored.items()
    }
    main = raw[raw["split"].isin(["train", "val", "test"])]
    return {
        "overlap_with_train_val": overlap,
        "groups_in_more_than_one_split": int(
            (main.groupby("config_group")["split"].nunique() > 1).sum()
        ),
        "inputs": features,
        "post_run_columns_among_inputs": sorted(set(features) & set(POST_RUN_COLUMNS)),
        "heap_is_input": "heap_mb_per_driver" in features,
        "iterations": "planned count from the design; realised iters_logged is not an input",
        "scaling": "fitted inside each scikit-learn pipeline on training rows only",
    }


def latency(model_dir: str, reqs: list, reps: int) -> dict:
    t0 = time.perf_counter()
    p = pr.ResourcePredictor.load(model_dir)
    load_s = time.perf_counter() - t0
    for r in reqs[:5]:  # warm-up
        p.predict(r)
    single = []
    for i in range(reps):
        t0 = time.perf_counter()
        p.predict(reqs[i % len(reqs)])
        single.append(time.perf_counter() - t0)
    batch = (reqs * (100 // len(reqs) + 1))[:100]
    times = []
    for _ in range(10):
        t0 = time.perf_counter()
        p.predict_many(batch)
        times.append(time.perf_counter() - t0)
    return {
        "load_s": load_s,
        "single_request_ms_median": 1000 * float(np.median(single)),
        "single_request_ms_p95": 1000 * float(np.percentile(single, 95)),
        "batch_100_ms_median": 1000 * float(np.median(times)),
        "per_request_in_batch_ms": 10 * float(np.median(times)),
        "machine": {
            "platform": platform.platform(),
            "processor": platform.processor() or platform.machine(),
            "cpu_count": os.cpu_count(),
            "python": platform.python_version(),
        },
        "note": "forests predict with n_jobs=-1; one request includes input checks, "
        "feature row, every target's model, trends and upper bounds",
    }


def transfer_summary(models_root: str, drivers_sha: str) -> dict:
    """Collate the Scout transfer experiments (they are not re-run here)."""
    out = {}
    d = os.path.join(models_root, "transfer_capped")
    rep = _load_json(os.path.join(d, "report.json"))
    if rep:
        test = pd.read_csv(os.path.join(d, "test_metrics.csv"))
        curve = pd.read_csv(os.path.join(d, "learning_curve_summary.csv"))
        out["label_transfer"] = {
            "same_local_data": rep["local"]["sha256"] == drivers_sha,
            "scout": {
                k: rep["source"].get(k)
                for k in ("file", "sha256", "setting", "rows", "shared_features",
                          "not_transferred", "label_mapping")
            },
            "test": test[test["workload"] == "all"][
                ["target", "model", "n", "mae", "mean_signed_error", "r2"]
            ].to_dict("records"),
            "paired_cv": {tg: v["paired_cv"] for tg, v in rep["targets"].items()},
            "learning_curve": curve.to_dict("records"),
        }  # fmt: skip
    d = os.path.join(models_root, "transfer_trends")
    rep = _load_json(os.path.join(d, "report.json"))
    if rep:
        tm = pd.read_csv(os.path.join(d, "trend_metrics.csv"))
        out["trend_transfer"] = {
            "same_local_data": rep["data"]["drivers_sha256"] == drivers_sha,
            "mae": tm[["test", "workload", "target", "method", "n", "mae"]].to_dict("records"),
            "policy": rep["policy"],
        }
    return out


# ── markdown tables ───────────────────────────────────────────────────────


def _md(df: pd.DataFrame) -> str:
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for row in df.itertuples(index=False):
        lines.append("| " + " | ".join("" if v is None else str(v) for v in row) + " |")
    return "\n".join(lines)


def _num(v, unit: str) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "–"
    return f"{v:.0f}" if unit == "MB" else f"{v:.2f}"


def _pct(v) -> str:
    return "–" if v is None or np.isnan(v) else f"{100 * v:.0f} %"


def _r2(v) -> str:
    return "–" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.3f}"


def tables_md(report: dict, mt: pd.DataFrame, pt: pd.DataFrame) -> str:
    out = []
    dep = mt[mt["variant"] == "deployed"]
    nc = report["noise_ceiling"]

    out.append("## Headline\n")
    rows = []
    for r in dep[dep["by"] == "all"].itertuples():
        rows.append(
            {
                "set": r.set,
                "target": r.target,
                "typical error (MAE)": f"{_num(r.mae, r.unit)} {r.unit} ({r.mape_pct:.0f} %)",
                "R² (noise ceiling)": f"{_r2(r.r2)} ({_r2((nc[r.target]['all'] or {}).get('r2'))})",
                "real use within the upper bound": _pct(r.upper_bound_coverage),
            }
        )
    out.append(_md(pd.DataFrame(rows)))

    dp = report["deployed_predictor"]
    out.append("\n## Deployed models\n")
    rows = []
    for target, by_wl in dp["families"].items():
        for wl, fam in by_wl.items():
            trends = [
                f"{axis}: {src}"
                for axis, by in dp["trend_policy"].items()
                if (src := by.get(wl, {}).get(target))
            ]
            rows.append(
                {
                    "target": target,
                    "workload": wl,
                    "model": pr.SOURCE_NAMES[fam],
                    "growth trend beyond range": ", ".join(trends) or "model's own",
                    "upper-bound method": dp["margin_policy"]
                    .get(target, {})
                    .get(wl, dp["default_margin_method"]),
                }
            )
    out.append(_md(pd.DataFrame(rows)))

    out.append("\n## Valid input ranges (measured; outside = flagged out of range)\n")
    rows = []
    for wl, cols in dp["valid_inputs"].items():
        for col, dom in cols.items():
            values = ", ".join(pr._fmt(v) for v in dom.get("values", []))
            rows.append({"workload": wl, "input": col, "measured values": values})
    out.append(_md(pd.DataFrame(rows)))

    out.append("\n## Upper-bound floors beyond the measured range (stress tests)\n")
    rows = []
    for target, by_axis in dp["extrapolation_margins"].items():
        for axis, by_wl in by_axis.items():
            for wl, e in by_wl.items():
                rows.append(
                    {
                        "target": target,
                        "beyond the largest measured": "data per driver" if axis == "size"
                        else "cores",
                        "workload": wl,
                        "floor": f"prediction + {100 * e['margin']:.0f} %",
                        "held-out rows (groups)": f"{e['rows']} ({e['groups']})",
                    }
                )  # fmt: skip
    out.append(_md(pd.DataFrame(rows)))

    out.append("\n## Data used\n")
    out.append(_md(pd.DataFrame(report["data"]["by_experiment_split"])))

    out.append("\n## Deployed predictor — all rows\n")
    rows = []
    for r in dep[dep["by"] == "all"].itertuples():
        ci = pt[(pt.set == r.set) & (pt.target == r.target) & (pt.workload == "all")].iloc[0]
        u = r.unit
        rows.append(
            {
                "set": r.set,
                "target": r.target,
                "n (groups)": f"{r.n} ({r.n_groups})",
                "MAE [95 % CI]": f"{_num(r.mae, u)} [{_num(ci.mae_ci95_low, u)}–"
                f"{_num(ci.mae_ci95_high, u)}] {u}",
                "RMSE": _num(r.rmse, u),
                "bias": _num(r.mean_signed_error, u),
                "R²": _r2(r.r2),
                "noise ceiling R²": _r2((nc[r.target]["all"] or {}).get("r2")),
                "under-predicted": _pct(r.underprediction_rate),
                "above upper bound": _pct(r.above_upper_rate),
                "upper bound width": f"+{_num(r.mean_upper_width, u)} {u} "
                f"(+{r.mean_upper_width_pct:.0f} %)",
                "flagged out of range": _pct(r.flagged_out_of_range),
            }
        )
    out.append(_md(pd.DataFrame(rows)))

    out.append("\n## Deployed predictor — per workload\n")
    rows = []
    for r in dep[dep["by"] == "workload_type"].itertuples():
        ci = pt[(pt.set == r.set) & (pt.target == r.target) & (pt.workload == r.value)].iloc[0]
        u = r.unit
        rows.append(
            {
                "set": r.set,
                "target": r.target,
                "workload": r.value,
                "n (groups)": f"{r.n} ({r.n_groups})",
                "MAE [95 % CI]": f"{_num(r.mae, u)} [{_num(ci.mae_ci95_low, u)}–"
                f"{_num(ci.mae_ci95_high, u)}] {u}",
                "bias": _num(r.mean_signed_error, u),
                "R²": _r2(r.r2),
                "noise ceiling R²": _r2((nc[r.target][r.value] or {}).get("r2")),
                "replicate SD": _num((nc[r.target][r.value] or {}).get("within_config_sd"), u),
                "under-predicted": _pct(r.underprediction_rate),
                "above upper bound": _pct(r.above_upper_rate),
            }
        )
    out.append(_md(pd.DataFrame(rows)))

    for col, title in (("num_workers", "number of workers"), ("cores_per_driver", "cores")):
        out.append(f"\n## Deployed predictor — per {title} (MAE / above upper bound)\n")
        s = dep[dep["by"] == col]
        rows = []
        for (set_name, value), g in s.groupby(["set", "value"], sort=False):
            row = {"set": set_name, title: value}
            for r in g.itertuples():
                row[r.target] = f"{_num(r.mae, r.unit)} / {_pct(r.above_upper_rate)} (n {r.n})"
            rows.append(row)
        out.append(_md(pd.DataFrame(rows)))

    out.append("\n## Ablation on the same rows — MAE (paired difference vs deployed)\n")
    rows = []
    allrows = mt[mt["by"] == "all"]
    for (set_name, target), g in allrows.groupby(["set", "target"], sort=False):
        u = m.TARGETS[target]
        row = {"set": set_name, "target": target}
        for r in g.itertuples():
            cell = _num(r.mae, u)
            if r.variant != "deployed":
                p = pt[(pt.set == set_name) & (pt.target == target) & (pt.workload == "all")
                       & (pt.baseline == r.variant)]  # fmt: skip
                if len(p):
                    p = p.iloc[0]
                    cell += (
                        f" (deployed {p.mae_diff:+.{0 if u == 'MB' else 2}f} "
                        f"[{p.diff_ci95_low:+.{0 if u == 'MB' else 2}f}, "
                        f"{p.diff_ci95_high:+.{0 if u == 'MB' else 2}f}])"
                    )
            row[r.variant] = cell
        rows.append(row)
    out.append(_md(pd.DataFrame(rows)))

    tr = report.get("transfer", {})
    lt = tr.get("label_transfer")
    if lt:
        out.append("\n## Scout label transfer — frozen test split, MAE (R²)\n")
        t = pd.DataFrame(lt["test"])
        rows = []
        for target, g in t.groupby("target", sort=False):
            u = m.TARGETS[target]
            rows.append(
                {"target": target}
                | {r.model: f"{_num(r.mae, u)} ({_r2(r.r2)})" for r in g.itertuples()}
            )
        out.append(_md(pd.DataFrame(rows)))
        out.append("\n## Scout label transfer — learning curve, median MAE\n")
        c = pd.DataFrame(lt["learning_curve"])
        piv = c.pivot_table(
            index=["target", "k_groups"], columns="model", values="mae_median", sort=False
        ).reset_index()
        for col in piv.columns[2:]:
            piv[col] = [
                _num(v, m.TARGETS[tg]) for v, tg in zip(piv[col], piv["target"], strict=True)
            ]
        out.append(_md(piv))
    tt = tr.get("trend_transfer")
    if tt:
        out.append("\n## Growth-trend transfer tests — MAE of the held-out larger jobs\n")
        t = pd.DataFrame(tt["mae"])
        piv = t.pivot_table(
            index=["test", "workload", "target"], columns="method", values="mae", sort=False
        ).reset_index()
        for col in piv.columns[3:]:
            piv[col] = [
                _num(v, m.TARGETS[tg]) for v, tg in zip(piv[col], piv["target"], strict=True)
            ]
        out.append(_md(piv))

    out.append("\n## Leakage audit\n")
    la = report["leakage_audit"]
    out.append(_md(pd.DataFrame([{"set": k} | v for k, v in la["overlap_with_train_val"].items()])))
    out.append(
        f"\nGroups in more than one split: {la['groups_in_more_than_one_split']}. "
        f"Post-run columns among the inputs: {la['post_run_columns_among_inputs'] or 'none'}. "
        f"Heap is an input: {la['heap_is_input']}."
    )

    lat = report["latency"]
    out.append("\n## Latency\n")
    out.append(
        _md(
            pd.DataFrame(
                [
                    {
                        "load (s)": f"{lat['load_s']:.2f}",
                        "one request, median (ms)": f"{lat['single_request_ms_median']:.1f}",
                        "one request, p95 (ms)": f"{lat['single_request_ms_p95']:.1f}",
                        "batch of 100 (ms)": f"{lat['batch_100_ms_median']:.0f}",
                        "per request in a batch (ms)": f"{lat['per_request_in_batch_ms']:.2f}",
                        "machine": f"{lat['machine']['processor']}, "
                        f"{lat['machine']['cpu_count']} CPUs",
                    }
                ]
            )
        )
    )
    return "\n".join(out) + "\n"


# ── main ──────────────────────────────────────────────────────────────────


def data_summary(raw: pd.DataFrame) -> list[dict]:
    ok = raw[raw["status"] == "ok"]
    g = ok.groupby(["experiment", "split"], dropna=False)
    return [
        {
            "experiment": exp,
            "split": split,
            "driver rows": len(d),
            "runs": int(d["run_key"].nunique()),
            "configuration groups": int(d["config_group"].nunique()),
        }
        for (exp, split), d in g
    ]


def main(argv=None):
    args = _parse_args(argv)
    raw = pd.read_csv(args.drivers)
    drivers_sha = _sha(args.drivers)
    sets = scored_sets(raw)
    os.makedirs(args.out_dir, exist_ok=True)

    frames = []
    for variant, (model, trends) in VARIANTS.items():
        pred = pr.ResourcePredictor.load(args.model_dir, model, trends)
        for set_name, df in sets.items():
            frames.append(predict_long(pred, df).assign(variant=variant, set=set_name))
    long = pd.concat(frames, ignore_index=True)
    mt = metric_table(long)
    pt = paired_table(long)

    deployed = pr.ResourcePredictor.load(args.model_dir)
    card = deployed.card
    families = {
        target: {wl: deployed.family_for(target, wl) for wl in card["valid_inputs"]}
        for target in m.TARGETS
    }
    report = {
        "question": "How good is the deployed resource predictor (objective 2b)?",
        "protocol": "see script header",
        "data": {
            "drivers_csv": os.path.relpath(args.drivers, _PROJECT_ROOT),
            "sha256": drivers_sha,
            "card_trained_on_same_data": card["data"]["sha256"] == drivers_sha,
            "by_experiment_split": data_summary(raw),
            "failed_runs": int((raw["status"] != "ok").sum()),
            "sampling_interval_s_median": float((raw["wall_s"] / raw["n_samples"]).median()),
            "host_mem_available_min_mb": float(raw["host_mem_avail_min_mb"].min()),
        },
        "deployed_predictor": {
            "model_version": card["model_version"],
            "card_schema_version": card["card_schema_version"],
            "families": families,
            "trend_policy": pr.TREND_POLICY,
            "margin_policy": pr.MARGIN_POLICY,
            "default_margin_method": pr.DEFAULT_MARGIN_METHOD,
            "upper_quantile": card["upper_quantile"],
            "valid_inputs": card["valid_inputs"],
            "extrapolation_margins": card["extrapolation_margins"],
            "versions": card["versions"],
        },
        "noise_ceiling": noise_ceilings(args.drivers),
        "leakage_audit": leakage_audit(raw, sets, card["features"]),
        "latency": latency(args.model_dir, to_requests(sets["test"]), args.latency_reps),
        "transfer": transfer_summary(args.models_root, drivers_sha),
        "versions": {
            "python": platform.python_version(),
            "scikit-learn": sklearn.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
    }

    mt.to_csv(os.path.join(args.out_dir, "metrics.csv"), index=False)
    pt.to_csv(os.path.join(args.out_dir, "paired.csv"), index=False)
    with open(os.path.join(args.out_dir, "report.json"), "w") as fh:
        json.dump(report, fh, indent=2, default=str)
    md = tables_md(report, mt, pt)
    with open(os.path.join(args.out_dir, "tables.md"), "w") as fh:
        fh.write(md)
    print(md)
    print(f"[evaluate] outputs written to {args.out_dir}")
    return report


if __name__ == "__main__":
    main()
