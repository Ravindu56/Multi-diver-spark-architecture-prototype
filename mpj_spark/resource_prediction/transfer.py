# =============================================================================
# mpj_spark/resource_prediction/transfer.py
# Phase 6 — Scout → local transfer experiment (Objective 2b)
#
# QUESTION
# --------
# Does knowledge from the Scout dataset (AWS EC2, whole-VM sar traces) improve
# per-driver CPU / memory prediction on the local testbed, compared with
# models trained on local data only — in particular when local data is scarce?
#
# SOURCE MODEL (deliberately simple and readable)
# ----------------------------------------------
# One straight line per workload, fitted on Scout single-node hosts:
#     source_target ≈ a_w + b_w · log2(cores)
# Only `workload_type` and `cores` are shared between the domains:
#   * Scout vcpus_per_host  ↔ local cores_per_driver
#   * data size is NOT used: local data per driver (about 8–200 MB) lies far
#     below every Scout input (≥ 769 MiB) — that would be pure extrapolation
#   * the workload design (iterations, features, k) has no Scout counterpart;
#     only the local correction sees it
# Label mapping (scope differs — documented, not hidden):
#   cpu_cores_p95   ← host_cpu_busy_p95_cores                 (whole VM)
#   cpu_cores_mean  ← host_cpu_busy_mean_pct / 100 × vcpus    (whole VM)
#   mem_mb_p95_max  ← host_mem_app_peak_mib   (whole VM; Scout has no p95)
#   mem_mb_peak_max ← host_mem_app_peak_mib   (whole VM, no cache)
#
# TRANSFER METHODS (all see the source prediction as column `source_pred`)
# -----------------------------------------------------------------------
#   scout_only      the source line applied unchanged (zero-shot)
#   scout_offset    source line + per-workload mean local residual, shrunk
#                   towards 0 with few local rows: n / (n + k)
#   scout_residual  source line + Ridge (per-workload slopes) fitted on the
#                   local residual (y − source_pred), using all local inputs
# Local-only comparators come from models.py (lookup, ridge_wl, rf).
# =============================================================================

from __future__ import annotations

import numpy as np
import pandas as pd

from mpj_spark.resource_prediction import models as m

# local target → (Scout column or derivation, unit note)
SOURCE_LABELS: dict[str, str] = {
    "cpu_cores_p95": "host_cpu_busy_p95_cores",
    "mem_mb_p95_max": "host_mem_app_peak_mib (Scout records no p95)",
    "cpu_cores_mean": "host_cpu_busy_mean_pct / 100 * vcpus_per_host",
    "mem_mb_peak_max": "host_mem_app_peak_mib",
}
SOURCE_SETTING = "single_node"  # one Spark application per VM ≈ one driver


def scout_source_frame(scout_hosts: pd.DataFrame) -> pd.DataFrame:
    """Scout host rows mapped onto the local column names used for transfer."""
    s = scout_hosts[scout_hosts["setting"] == SOURCE_SETTING].copy()
    return pd.DataFrame(
        {
            "workload_type": s["workload_type"],
            "cores_per_driver": s["vcpus_per_host"].astype(float),
            "cpu_cores_p95": s["host_cpu_busy_p95_cores"],
            "mem_mb_p95_max": s["host_mem_app_peak_mib"],
            "cpu_cores_mean": s["host_cpu_busy_mean_pct"] / 100.0 * s["vcpus_per_host"],
            "mem_mb_peak_max": s["host_mem_app_peak_mib"],
        }
    )


# =============================================================================
# Source model
# =============================================================================


class SourceLines:
    """Per-workload least-squares line  y = a + b·log2(cores)  fitted on Scout."""

    def fit(self, source: pd.DataFrame, target: str) -> SourceLines:
        self.target = target
        self.lines_ = {}
        self.core_range_ = {}
        for wl, g in source.dropna(subset=[target]).groupby("workload_type"):
            x = np.log2(g["cores_per_driver"].to_numpy(float))
            y = g[target].to_numpy(float)
            if len(np.unique(x)) >= 2:
                b, a = np.polyfit(x, y, 1)
            else:
                a, b = float(np.mean(y)), 0.0
            self.lines_[wl] = (float(a), float(b))
            self.core_range_[wl] = (
                float(g["cores_per_driver"].min()),
                float(g["cores_per_driver"].max()),
            )
        all_a = np.mean([a for a, _ in self.lines_.values()])
        all_b = np.mean([b for _, b in self.lines_.values()])
        self.fallback_ = (float(all_a), float(all_b))
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        out = np.empty(len(df))
        for i, (wl, cores) in enumerate(
            zip(df["workload_type"], df["cores_per_driver"], strict=True)
        ):
            a, b = self.lines_.get(wl, self.fallback_)
            out[i] = a + b * np.log2(float(cores))
        return out

    def extrapolating(self, df: pd.DataFrame) -> np.ndarray:
        """True where local cores lie outside the Scout range of that workload."""
        flags = []
        for wl, cores in zip(df["workload_type"], df["cores_per_driver"], strict=True):
            lo, hi = self.core_range_.get(wl, (np.inf, -np.inf))
            flags.append(not (lo <= cores <= hi))
        return np.array(flags)

    def describe(self) -> dict:
        return {
            wl: {"intercept": a, "slope_per_log2_core": b, "scout_core_range": self.core_range_[wl]}
            for wl, (a, b) in self.lines_.items()
        }


# =============================================================================
# Transfer estimators — X must contain FEATURES + "source_pred"
# =============================================================================


class ScoutOnly:
    """Zero-shot: the source prediction, unchanged."""

    def fit(self, X: pd.DataFrame, y) -> ScoutOnly:
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return X["source_pred"].to_numpy(float)


class ScoutOffset:
    """Source prediction + per-workload mean local residual, shrunk by n/(n+k)."""

    def __init__(self, shrink_k: float = 2.0):
        self.shrink_k = shrink_k

    def fit(self, X: pd.DataFrame, y) -> ScoutOffset:
        resid = pd.Series(np.asarray(y, float) - X["source_pred"].to_numpy(float), index=X.index)
        stats = resid.groupby(X["workload_type"]).agg(["mean", "count"])
        self.offsets_ = {
            wl: r["mean"] * r["count"] / (r["count"] + self.shrink_k) for wl, r in stats.iterrows()
        }
        n = len(resid)
        self.global_offset_ = float(resid.mean() * n / (n + self.shrink_k)) if n else 0.0
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        off = X["workload_type"].map(self.offsets_).fillna(self.global_offset_).to_numpy(float)
        return X["source_pred"].to_numpy(float) + off


class ScoutResidual:
    """Source prediction + per-workload Ridge on the local residual (all local inputs)."""

    def __init__(self, alpha: float = 1.0, numeric: list[str] = m.NUMERIC_FEATURES):
        self.alpha = alpha
        self.features = [*m.CATEGORICAL_FEATURES, *numeric]
        self.numeric = numeric

    def fit(self, X: pd.DataFrame, y) -> ScoutResidual:
        resid = np.asarray(y, float) - X["source_pred"].to_numpy(float)
        self.model_ = m.make_ridge_wl(self.alpha, self.numeric).fit(X[self.features], resid)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return X["source_pred"].to_numpy(float) + self.model_.predict(X[self.features])


def method_factories(rf_trees: int = 200, training_set: str = "uncapped") -> dict:
    """All compared methods; every factory returns an object with fit/predict on X."""
    num = m.TRAINING_SETS[training_set]["numeric"]
    feats = m.features_for(training_set)
    return {
        "local_lookup": lambda: m.LookupModel(feats),
        "local_ridge_wl": lambda: m.make_ridge_wl(1.0, num),
        "local_rf": lambda: m.make_rf(5, 2, n_estimators=rf_trees),
        "scout_only": ScoutOnly,
        "scout_offset": ScoutOffset,
        "scout_residual": lambda: ScoutResidual(1.0, num),
    }


def _fit_predict(
    factory,
    train: pd.DataFrame,
    test: pd.DataFrame,
    target: str,
    features: list[str] = m.FEATURES,
) -> np.ndarray:
    cols = [*features, "source_pred"]
    model = factory()
    # local-only pipelines must not see the source prediction
    if isinstance(model, ScoutOnly | ScoutOffset | ScoutResidual):
        return model.fit(train[cols], train[target]).predict(test[cols])
    return model.fit(train[features], train[target]).predict(test[features])


# =============================================================================
# Evaluation protocols
# =============================================================================


def group_cv(
    df: pd.DataFrame, target: str, factories: dict, features: list[str] = m.FEATURES
) -> pd.DataFrame:
    """Leave-one-configuration-out predictions for every method."""
    frames = []
    for group in df["config_group"].unique():
        held, rest = df[df["config_group"] == group], df[df["config_group"] != group]
        for name, factory in factories.items():
            frames.append(
                pd.DataFrame(
                    {
                        "config_group": group,
                        "workload_type": held["workload_type"].values,
                        "model": name,
                        "actual": held[target].values,
                        "predicted": _fit_predict(factory, rest, held, target, features),
                    }
                )
            )
    return pd.concat(frames, ignore_index=True)


def sample_groups(groups: pd.DataFrame, k: int, rng: np.random.Generator) -> list[str]:
    """k configuration groups, at least one per workload when k allows."""
    by_wl = {wl: list(g["config_group"]) for wl, g in groups.groupby("workload_type")}
    chosen = []
    for wl in sorted(by_wl):
        if len(chosen) < k:
            chosen.append(by_wl[wl][rng.integers(len(by_wl[wl]))])
    rest = [g for g in groups["config_group"] if g not in chosen]
    extra = rng.choice(rest, size=max(0, min(k, len(groups)) - len(chosen)), replace=False)
    return chosen + list(extra)


def learning_curve(
    df: pd.DataFrame,
    target: str,
    factories: dict,
    ks: list[int],
    reps: int = 20,
    seed: int = 0,
    features: list[str] = m.FEATURES,
) -> pd.DataFrame:
    """
    For each k, repeatedly train on k random configuration groups (the
    'adaptation set', identical for every method) and evaluate on all other
    groups.  Uses train+val only — the frozen test split is never touched.
    """
    rng = np.random.default_rng(seed)
    groups = df[["config_group", "workload_type"]].drop_duplicates("config_group")
    rows = []
    for k in ks:
        if k >= len(groups):
            continue
        for rep in range(reps):
            picked = sample_groups(groups, k, rng)
            train = df[df["config_group"].isin(picked)]
            evals = df[~df["config_group"].isin(picked)]
            for name, factory in factories.items():
                pred = _fit_predict(factory, train, evals, target, features)
                err = pred - evals[target].to_numpy(float)
                rows.append(
                    {
                        "target": target,
                        "k_groups": k,
                        "rep": rep,
                        "model": name,
                        "n_train_rows": len(train),
                        "n_eval_rows": len(evals),
                        "mae": float(np.mean(np.abs(err))),
                        "mean_signed_error": float(np.mean(err)),
                    }
                )
    return pd.DataFrame(rows)


# =============================================================================
# Option A — transfer of the DATA-SIZE TREND (extrapolation beyond local sizes)
# =============================================================================
#
# Scout jobs are larger (769–1537 MiB) than any local job, so the one thing
# Scout may know that local data does not is how demand GROWS with data size.
# The trend is transferred as an elasticity β (demand ∝ size^β), estimated
# within each Scout VM type (fixed effects), and applied only to requests with
# more data per driver than measured locally:
#
#   prediction = base(anchor at the largest local shard) × (shard / s_max)^β
#
# Scout records input sizes for LogReg only (K-Means rows have no size), so
# the trend exists for LogReg only.  The fair comparison is the same
# anchor-and-scale with β estimated from LOCAL data.

# local target → Scout column used for the size trend (same mapping as above)
SIZE_TREND_SOURCE: dict[str, str] = {
    "cpu_cores_p95": "host_cpu_busy_p95_cores",
    "mem_mb_p95_max": "host_mem_app_peak_mib",
    "cpu_cores_mean": "cpu_cores_mean",  # derived from busy % × vCPUs
    "mem_mb_peak_max": "host_mem_app_peak_mib",
}


def _within_slope(x: np.ndarray, y: np.ndarray, groups: np.ndarray) -> tuple[float, int]:
    """Least-squares slope of y on x after removing each group's mean (fixed effects)."""
    df = pd.DataFrame({"x": x, "y": y, "g": groups})
    df = df[df.groupby("g")["x"].transform("nunique") >= 2]
    if df.empty:
        return float("nan"), 0
    dx = df["x"] - df.groupby("g")["x"].transform("mean")
    dy = df["y"] - df.groupby("g")["y"].transform("mean")
    denom = float((dx**2).sum())
    return (float((dx * dy).sum()) / denom if denom > 0 else float("nan")), int(df["g"].nunique())


def scout_size_elasticity(scout_hosts: pd.DataFrame, target: str, workload: str) -> dict | None:
    """β of `target` with respect to input size on Scout single-node hosts, or None."""
    s = scout_hosts[
        (scout_hosts["setting"] == SOURCE_SETTING)
        & (scout_hosts["workload_type"] == workload)
        & scout_hosts["input_mib"].notna()
    ].copy()
    if s.empty:
        return None
    s["cpu_cores_mean"] = s["host_cpu_busy_mean_pct"] / 100.0 * s["vcpus_per_host"]
    y = s[SIZE_TREND_SOURCE[target]].to_numpy(float)
    ok = y > 0
    beta, n_groups = _within_slope(
        np.log(s["input_mib"].to_numpy(float)[ok]), np.log(y[ok]), s["vm_type"].to_numpy()[ok]
    )
    if not np.isfinite(beta):
        return None
    return {"beta": beta, "vm_types": n_groups, "rows": int(ok.sum())}


def local_size_elasticity(df: pd.DataFrame, target: str, workload: str) -> float | None:
    """β of `target` with respect to data per driver, within configurations that differ only in size."""
    d = df[(df["workload_type"] == workload) & df[target].notna() & (df[target] > 0)]
    if d.empty:
        return None
    key_cols = [c for c in m.FEATURES + m.DESIGN_FEATURES if c in d and c != "dataset_mb"]
    key = d[key_cols].astype(str).agg("|".join, axis=1).to_numpy()
    shard = (d["dataset_mb"] / d["num_workers"]).to_numpy(float)
    beta, _ = _within_slope(np.log(shard), np.log(d[target].to_numpy(float)), key)
    return beta if np.isfinite(beta) else None


def anchor_and_scale(
    model, X: pd.DataFrame, features: list[str], s_max: float, beta: float | None
) -> np.ndarray:
    """
    Model prediction, except for rows with more data per driver than s_max:
    those are predicted at s_max and scaled by (shard / s_max)^beta
    (beta None = keep the model's own prediction, beta 0 = flat).
    """
    pred = model.predict(X[features])
    shard = (X["dataset_mb"] / X["num_workers"]).to_numpy(float)
    beyond = shard > s_max
    if beta is None or not beyond.any():
        return pred
    anchor = X.loc[beyond].copy()
    anchor["dataset_mb"] = s_max * anchor["num_workers"]
    anchor = m.add_derived(anchor)
    out = pred.copy()
    out[beyond] = model.predict(anchor[features]) * (shard[beyond] / s_max) ** beta
    return out
