# =============================================================================
# mpj_spark/resource_prediction/predictor.py
# Phase 6 — pre-launch resource prediction for one job (Objective 2b → 2c)
#
# PURPOSE
# -------
# The single entry point for asking the trained models
# "how much CPU and memory will EACH driver of this job use?"
#
#   predictor = ResourcePredictor.load("data/models/local_capped")
#   result = predictor.predict(PredictionRequest(
#       workload_type="logreg", dataset_mb=100, num_workers=2,
#       cores_per_driver=4, iterations=30, num_features=20))
#
# A PredictionResult holds, for the job's drivers:
#   cpu_cores_p95, cpu_cores_mean        CPU of each driver (cores)
#   mem_mb_p95_max, mem_mb_peak_max      memory of the LARGEST driver (MB) —
#                                        every driver gets the same limit
#   *_upper                              prediction + empirical margin
# The derived inputs of the handoff (data per driver and its products) are
# computed from the request; the heap is only needed by models trained with it.
# and where the numbers come from (source per target, training set, model
# version), plus whether the request lies outside the measured domain
# (out_of_domain + reasons) or is not covered at all (status "unsupported").
#
# RULES
# -----
#   * pre-launch inputs only; `iterations` is the PLANNED iteration count
#   * missing / invalid mandatory inputs are rejected (ValueError)
#   * inputs outside the values measured for that workload are flagged,
#     never silently trusted; values between measured ones become notes;
#     so is data per driver (dataset_mb / num_workers) beyond the largest
#     measured, even when each input alone is in range
#   * no numbers are invented for workloads without training data — they
#     come back "unsupported" so the allocator uses its static fallback
#   * this module predicts DEMAND; turning demand into Docker limits, with a
#     safety factor and host-total checks, is the allocator's job
#
# Beyond the measured range, TREND_POLICY may replace the model's own
# extrapolation by a growth trend: the request is predicted at the largest
# measured value and grown by (value / x_max)^beta (beta from the model card;
# "local" = local data, "combined" = Scout-informed).  Such requests stay
# flagged out_of_domain and the note says which trend was applied.
#
# Upper margins: per workload, the q-quantile of out-of-fold residuals from
# leave-one-configuration-out CV on train+val (never the test split); see
# scripts/train_resource_predictor.py.
# =============================================================================

from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass, field

import joblib
import numpy as np
import pandas as pd
import sklearn

from mpj_spark.resource_prediction import models as m

# model card layout this module understands (written by the training script)
CARD_SCHEMA_VERSION = 4

# model family in the model card → provenance name reported to the allocator
SOURCE_NAMES: dict[str, str] = {
    "lookup": "profile_lookup",
    "ridge": "local_ridge",
    "ridge_wl": "local_ridge_wl",
    "rf": "local_rf",
    "rf_wl": "local_rf_wl",
    "ridge_log_wl": "local_ridge_log_wl",
}
MODEL_CHOICES = ["best_cv", *SOURCE_NAMES]

# "best_cv" = the family with the lowest out-of-fold MAE among BEST_CV_FAMILIES,
# except where FAMILY_POLICY fixes the family by a pre-registered test:
#   CPU → rf_wl (forest per workload): scripts/per_workload_rf_test.py showed a
#   single forest answers K-Means jobs beyond the K-Means range from LogReg runs
#   (T3: CPU p95 MAE 6.35 → 0.47 cores) while in-range accuracy is not worse.
BEST_CV_FAMILIES = ("lookup", "ridge", "ridge_wl", "rf")
FAMILY_POLICY: dict[str, str] = {"cpu_cores_p95": "rf_wl", "cpu_cores_mean": "rf_wl"}
# Per-workload exceptions in "best_cv" mode (target → workload → family):
#   K-Means peak memory → ridge_log_wl (per-workload power law):
#   scripts/kmeans_memory_test.py — beats ridge_wl in range (−4.6 MB, CI below
#   0) and in the extrapolation check (−14.6 MB); K-Means memory p95 and all
#   LogReg memory keep ridge_wl (no candidate qualified there).
# Upper-bound method per target and workload (models.MARGIN_METHODS); default
# "additive_rows".  Decided by scripts/upper_bound_tests.py (calibration-CV on
# train+val: the narrowest method with coverage >= 0.88): LogReg CPU uses a
# relative margin (same coverage, narrower), everything else the original one.
MARGIN_POLICY: dict[str, dict[str, str]] = {
    "cpu_cores_p95": {"logreg": "relative_groups"},
    "cpu_cores_mean": {"logreg": "relative_groups"},
}
DEFAULT_MARGIN_METHOD = "additive_rows"

WORKLOAD_FAMILY_POLICY: dict[str, dict[str, str]] = {
    "mem_mb_peak_max": {"kmeans": "ridge_log_wl"},
}

# Growth trends used BEYOND the measured range: axis → workload → target →
# beta source ("combined" = Scout-informed, "local" = local data only).
# Evidence: scripts/transfer_trend_tests.py (data/models/transfer_trends/report.json)
#   size   LogReg memory → "combined" (Scout-informed).  DESIGN CHOICE made
#          after the fixed-rule tests, whose literal outcome was "local": the
#          rule could not compare the two in the scarce test because local data
#          had no size trend there — where Scout's trend alone cut the memory
#          error by 106 MB (11 groups).  With full local data the Scout-informed
#          rate equals the local one (0.362 vs 0.362; +5 MB in test S2), so
#          predictions are unchanged today and Scout's trend becomes the
#          automatic fallback when local data cannot show a trend.
#   cores  K-Means, all targets → "local"; Scout's VM-size trend does not
#          transfer (bigger AWS VMs also have more memory) and was clearly worse.
# Everything else beyond the range keeps the model's own prediction.
TREND_POLICY: dict[str, dict[str, dict[str, str]]] = {
    "size": {"logreg": {"mem_mb_p95_max": "combined", "mem_mb_peak_max": "combined"}},
    "cores": {"kmeans": {target: "local" for target in m.TARGETS}},
}
TREND_CHOICES: dict[str, dict] = {
    "policy": TREND_POLICY,
    # the fixed rule's literal outcome: local trends only (for comparison)
    "local_only": {
        axis: {wl: dict.fromkeys(targets, "local") for wl, targets in by_wl.items()}
        for axis, by_wl in TREND_POLICY.items()
    },
    "off": {},
}
TREND_SOURCE_LABELS = {"combined": "Scout-informed", "local": "local"}


@dataclass
class PredictionRequest:
    """One job, described with pre-launch inputs only (values per driver)."""

    workload_type: str
    dataset_mb: float
    num_workers: int  # number of Spark drivers (MPI ranks >= 1)
    cores_per_driver: float
    iterations: int | None = None  # planned, not realised after early stopping
    num_features: int | None = None
    num_clusters: int = 0  # K-Means k; ignored (0) for other workloads
    heap_mb_per_driver: float | None = None  # only for models trained with heap
    thread_cap: int | None = None  # BLAS/OpenMP threads; None = same as cores

    def feature_row(self) -> dict:
        row = asdict(self)
        row.pop("thread_cap")
        if row["workload_type"] != "kmeans":
            row["num_clusters"] = 0
        shard = (
            row["dataset_mb"] / row["num_workers"]
            if row["dataset_mb"] is not None and row["num_workers"]
            else None
        )
        row["shard_mb"] = shard
        for name, col in (("shard_x_iters", "iterations"), ("shard_x_features", "num_features")):
            row[name] = shard * row[col] if shard is not None and row[col] is not None else None
        return row


@dataclass
class PredictionResult:
    """Predicted demand: CPU of each driver, memory of the largest driver; with provenance."""

    status: str  # "ok" | "out_of_domain" | "unsupported"
    workload_type: str
    num_workers: int
    cpu_cores_p95: float | None = None
    mem_mb_p95_max: float | None = None
    cpu_cores_mean: float | None = None
    mem_mb_peak_max: float | None = None
    cpu_cores_p95_upper: float | None = None
    mem_mb_p95_max_upper: float | None = None
    cpu_cores_mean_upper: float | None = None
    mem_mb_peak_max_upper: float | None = None
    upper_quantile: float | None = None
    source: dict = field(default_factory=dict)  # target → provenance name
    out_of_domain: bool = False
    reasons: list = field(default_factory=list)  # why out of domain / unsupported
    notes: list = field(default_factory=list)  # in-domain remarks
    training_set: str | None = None
    model_version: str | None = None
    schema_version: int = CARD_SCHEMA_VERSION

    def to_dict(self) -> dict:
        return asdict(self)


def _fmt(v: float) -> str:
    return str(int(v)) if float(v).is_integer() else f"{v:.2f}"


def _pick_family(card: dict, target: str, model: str) -> str:
    """'best_cv' = FAMILY_POLICY, else the lowest out-of-fold MAE among BEST_CV_FAMILIES."""
    if model != "best_cv":
        if model not in SOURCE_NAMES:
            raise ValueError(f"unknown model {model!r}; choose from {MODEL_CHOICES}")
        return model
    cv = card["choices"][target]["cv_uncertainty"]
    if target in FAMILY_POLICY and FAMILY_POLICY[target] in cv:
        return FAMILY_POLICY[target]
    return min(BEST_CV_FAMILIES, key=lambda fam: cv[fam]["mae"])


class ResourcePredictor:
    """Trained per-driver CPU / memory models plus the checks around them."""

    def __init__(
        self,
        card: dict,
        models: dict,
        families: dict[str, str],
        trends: str = "policy",
        overrides: dict[str, dict[str, str]] | None = None,
    ):
        self.card = card
        self.models = models  # target → {family → fitted model}
        self.families = families  # target → default family
        self.overrides = overrides or {}  # target → workload → family
        self.features = card["features"]
        self.valid = card["valid_inputs"]
        if trends not in TREND_CHOICES:
            raise ValueError(f"unknown trends {trends!r}; choose from {list(TREND_CHOICES)}")
        self.trend_policy = TREND_CHOICES[trends]

    @classmethod
    def load(
        cls, model_dir: str, model: str = "best_cv", trends: str = "policy"
    ) -> ResourcePredictor:
        """Load the model card and one fitted model per target from model_dir."""
        with open(os.path.join(model_dir, "model_card.json")) as fh:
            card = json.load(fh)
        if card.get("card_schema_version") != CARD_SCHEMA_VERSION:
            raise ValueError(
                f"model card in {model_dir!r} has layout "
                f"{card.get('card_schema_version')!r}, expected {CARD_SCHEMA_VERSION}; "
                "re-run scripts/train_resource_predictor.py"
            )
        families = {t: _pick_family(card, t, model) for t in m.TARGETS}
        overrides = {}
        if model == "best_cv":
            for t, by_wl in WORKLOAD_FAMILY_POLICY.items():
                cv = card["choices"][t]["cv_uncertainty"]
                overrides[t] = {wl: fam for wl, fam in by_wl.items() if fam in cv}
        models = {}
        for t, fam in families.items():
            for f in {fam, *overrides.get(t, {}).values()}:
                path = os.path.join(model_dir, f"{t}__{f}.joblib")
                models.setdefault(t, {})[f] = joblib.load(path)
        return cls(card, models, families, trends, overrides)

    def family_for(self, target: str, workload: str) -> str:
        """Model family that answers `target` for `workload`."""
        return self.overrides.get(target, {}).get(workload, self.families[target])

    # ── checks ────────────────────────────────────────────────────────────

    def _empty_result(self, req: PredictionRequest, status: str) -> PredictionResult:
        return PredictionResult(
            status=status,
            workload_type=req.workload_type,
            num_workers=req.num_workers,
            upper_quantile=self.card["upper_quantile"],
            source={t: SOURCE_NAMES[self.family_for(t, req.workload_type)] for t in self.families},
            training_set=self.card["training_set"],
            model_version=self.card["model_version"],
        )

    def _validate(self, req: PredictionRequest) -> dict:
        """Feature row of a supported workload; ValueError on missing / invalid input."""
        row = req.feature_row()
        problems = []
        for col in self.features:
            if col in m.CATEGORICAL_FEATURES or col in m.DERIVED_FEATURES:
                continue  # derived inputs follow from the checked ones
            v = row.get(col)
            if v is None or not math.isfinite(float(v)):
                problems.append(f"{col} is missing")
            elif col == "num_clusters":
                if req.workload_type == "kmeans" and v < 1:
                    problems.append("num_clusters (k) must be >= 1 for kmeans")
            elif v <= 0:
                problems.append(f"{col} must be > 0 (got {v})")
        if problems:
            raise ValueError("invalid pre-launch input: " + "; ".join(problems))
        return {col: row[col] for col in self.features}

    def _domain(self, req: PredictionRequest, row: dict, res: PredictionResult) -> None:
        """Flag inputs outside the measured domain of this workload."""
        wl = req.workload_type
        for col, dom in self.valid[wl].items():
            v = float(row[col])
            if v < dom["min"] or v > dom["max"]:
                res.reasons.append(
                    f"{col}={_fmt(v)} outside the measured range "
                    f"{_fmt(dom['min'])}-{_fmt(dom['max'])} for {wl}"
                )
            elif "values" in dom and v not in dom["values"]:
                measured = ", ".join(_fmt(x) for x in dom["values"])
                res.notes.append(
                    f"{col}={_fmt(v)} was not measured for {wl} (measured: {measured}); "
                    "the model interpolates"
                )
        # each input can be in range while their combination is not: 125 MB on
        # 1 worker = 125 MB per driver, beyond the largest measured (card x_max)
        by_target = self.card.get("growth_trends", {}).get("size", {}).get(wl, {})
        shard_max = max((e["x_max"] for e in by_target.values() if e), default=None)
        shard = row["dataset_mb"] / row["num_workers"]
        if shard_max is not None and shard > shard_max:
            res.reasons.append(
                f"data per driver {_fmt(shard)} MB (dataset_mb / num_workers) beyond the "
                f"largest measured {_fmt(shard_max)} MB for {wl}"
            )
        cap = req.thread_cap if req.thread_cap is not None else req.cores_per_driver
        policy = self.card.get("thread_cap_policy")
        if policy == "equals_cores" and cap != req.cores_per_driver:
            res.reasons.append(
                f"thread_cap={_fmt(cap)} differs from cores_per_driver="
                f"{_fmt(req.cores_per_driver)}; all training runs used thread cap = cores"
            )
        elif policy == "uncapped" and req.thread_cap is not None:
            res.reasons.append("thread cap requested, but the models were trained on uncapped runs")
        trained = self.card.get("versions", {}).get("scikit-learn")
        if trained and trained != sklearn.__version__:
            res.notes.append(
                f"models saved with scikit-learn {trained}, running {sklearn.__version__}"
            )
        res.out_of_domain = bool(res.reasons)
        res.status = "out_of_domain" if res.out_of_domain else "ok"

    # ── prediction ────────────────────────────────────────────────────────

    def predict(self, request: PredictionRequest) -> PredictionResult:
        return self.predict_many([request])[0]

    def predict_many(self, requests: list[PredictionRequest]) -> list[PredictionResult]:
        """One result per request; supported requests are predicted in one batch."""
        results, rows, idx = [], [], []
        for i, req in enumerate(requests):
            if req.workload_type not in self.valid:
                res = self._empty_result(req, "unsupported")
                res.source = {t: "none" for t in self.families}
                res.reasons.append(
                    f"no training data for workload {req.workload_type!r} in the "
                    f"{self.card['training_set']!r} training set; use the allocator's "
                    "static fallback profile"
                )
                results.append(res)
                continue
            row = self._validate(req)
            res = self._empty_result(req, "ok")
            self._domain(req, row, res)
            results.append(res)
            rows.append(row)
            idx.append(i)
        if not rows:
            return results
        X = pd.DataFrame(rows, columns=self.features)
        for target, by_family in self.models.items():
            fams = np.array([self.family_for(target, results[i].workload_type) for i in idx])
            values = np.empty(len(idx))
            for f in set(fams):  # each family predicts only the rows routed to it
                values[fams == f] = by_family[f].predict(X[fams == f])
            for k, (i, row) in enumerate(zip(idx, rows, strict=True)):
                res, fam = results[i], fams[k]
                model, value = by_family[fam], values[k]
                method = MARGIN_POLICY.get(target, {}).get(res.workload_type, DEFAULT_MARGIN_METHOD)
                margins = self.card["choices"][target]["calibrated_margins"][fam][method]
                margin = m.margin_for(margins, res.workload_type)
                value, upper = self._apply_trends(target, model, row, value, margin, method, res)
                upper = self._extrapolation_floor(target, row, value, upper, res)
                if value < 0:  # linear models can dip below zero; demand cannot
                    res.notes.append(f"{target} prediction {value:.3f} clipped to 0")
                    value, upper = 0.0, max(upper, 0.0)
                setattr(res, target, float(value))
                setattr(res, f"{target}_upper", float(upper))
        return results

    def _extrapolation_floor(self, target, row, value, upper, res) -> float:
        """Beyond the measured range: upper >= value × (1 + stress-test margin)."""
        wl = res.workload_type
        for axis in ("size", "cores"):
            entry = self.card.get("growth_trends", {}).get(axis, {}).get(wl, {}).get(target)
            extra = self.card.get("extrapolation_margins", {}).get(target, {}).get(axis, {}).get(wl)
            if not entry or not extra:
                continue
            x = (
                row["dataset_mb"] / row["num_workers"]
                if axis == "size"
                else row["cores_per_driver"]
            )
            if x > entry["x_max"]:
                floor = value * (1 + extra["margin"])
                if floor > upper:
                    upper = floor
                    res.notes.append(
                        f"{target}: upper bound widened to +{100 * extra['margin']:.0f} % "
                        f"beyond the measured {axis} (stress test)"
                    )
        return upper

    def _apply_trends(self, target, model, row, value, margin, method, res) -> tuple[float, float]:
        """Prediction and upper bound, grown by TREND_POLICY trends beyond the measured range."""
        wl = res.workload_type
        anchor, factor, applied = dict(row), 1.0, []
        for axis, by_wl in self.trend_policy.items():
            source = by_wl.get(wl, {}).get(target)
            entry = self.card.get("growth_trends", {}).get(axis, {}).get(wl, {}).get(target)
            if not source or not entry:
                continue
            x = (
                row["dataset_mb"] / row["num_workers"]
                if axis == "size"
                else row["cores_per_driver"]
            )
            x_max = entry["x_max"]
            if x <= x_max:
                continue
            beta = (entry.get(source) or {}).get("beta")
            if beta is None:
                res.notes.append(f"{target}: no {source} {axis} trend available; model kept")
                continue
            if axis == "size":
                anchor["dataset_mb"] = x_max * row["num_workers"]
            else:
                anchor["cores_per_driver"] = x_max
            factor *= (x / x_max) ** beta
            label = TREND_SOURCE_LABELS.get(source, source)
            applied.append(f"{axis}^{beta:.2f} {label} trend from {x_max:g} to {x:g}")
        if not applied:
            return float(value), float(m.apply_margin(value, margin, method))
        anchor_df = m.add_derived(pd.DataFrame([anchor]))[self.features]
        base = float(model.predict(anchor_df)[0])
        res.source[target] = f"{res.source[target]}+trend"
        res.notes.append(
            f"{target}: grown ×{factor:.2f} beyond the measured range ({'; '.join(applied)})"
        )
        return base * factor, m.apply_margin(base, margin, method) * factor
