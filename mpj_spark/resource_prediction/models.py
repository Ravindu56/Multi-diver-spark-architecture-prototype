# =============================================================================
# mpj_spark/resource_prediction/models.py
# Phase 6 — per-driver CPU / memory demand models (P6-02 CPU, P6-03 memory)
#
# PURPOSE
# -------
# Small, transparent regressors that predict resource demand from PRE-RUN
# inputs only (what is known before the job is launched).  Inputs of the
# "capped" training set (predictor handoff):
#
#   workload_type, dataset_mb, num_workers, cores_per_driver,
#   iterations, num_features, num_clusters (K-Means k, 0 otherwise)
#   + derived: shard_mb = dataset_mb / num_workers, shard_mb x iterations,
#              shard_mb x num_features
#
# Targets (TARGETS): per-driver CPU p95 and mean in cores; memory as the
# largest driver of the run (RSS p95 and peak, MB), because every driver of a
# job gets the same memory limit.
#
# TRAINING SETS (thread-capped and uncapped runs are never mixed):
#   capped    runs with a BLAS/OpenMP thread cap and a recorded workload
#             design (Sets A, B, D) — the environment the allocator deploys
#   uncapped  runs without a thread cap: none in the current data; kept so
#             such runs are never mixed into the capped models
#
# Models (one per target; CPU and memory are trained separately):
#   lookup  median of the most specific matching configuration seen in
#           training, falling back to coarser matches  (baseline)
#   ridge     regularised linear regression on scaled / one-hot features
#   ridge_wl  the same with a separate slope per workload (interactions)
#   rf      small random forest (non-linear comparator)
#   rf_wl   the same forest fitted separately per workload, inside ONE model
#           object, so a workload never borrows another workload's data when
#           a request lies outside its own measured range
#
# Preprocessing lives inside each scikit-learn Pipeline, so scaling and
# encoding are fitted on training rows only (no leakage).
#
# An empirical UPPER bound per model is calibrated on validation residuals:
#   upper = prediction + q-quantile of (actual − prediction) on validation
# The allocator (later step) adds its own safety margin on top.
# =============================================================================

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin, clone
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    OneHotEncoder,
    PolynomialFeatures,
    StandardScaler,
)

# =============================================================================
# Feature contract
# =============================================================================

CATEGORICAL_FEATURES = ["workload_type"]
NUMERIC_FEATURES = ["dataset_mb", "num_workers", "cores_per_driver", "heap_mb_per_driver"]
FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES
DESIGN_FEATURES = ["iterations", "num_features", "num_clusters"]
# engineered inputs of the predictor handoff, computed from pre-launch inputs
DERIVED_FEATURES = ["shard_mb", "shard_x_iters", "shard_x_features"]

# numeric inputs used in log2 form by the linear models (all strictly positive)
LOG_FEATURES = (
    "dataset_mb",
    "heap_mb_per_driver",
    "iterations",
    "num_features",
    *DERIVED_FEATURES,
)

TRAINING_SETS: dict[str, dict] = {
    "capped": {
        # no heap: it is fixed per workload in the runs and not an allocator input
        "numeric": ["dataset_mb", "num_workers", "cores_per_driver"]
        + DESIGN_FEATURES
        + DERIVED_FEATURES,
        "description": "thread-capped DOE runs (Sets A, B, D) with the handoff inputs",
    },
    "uncapped": {
        "numeric": NUMERIC_FEATURES,
        "description": "runs without a thread cap (none in the current data)",
    },
}


def add_derived(df: pd.DataFrame) -> pd.DataFrame:
    """Copy of df with the derived inputs (data per driver and its products)."""
    out = df.copy()
    shard = out["dataset_mb"] / out["num_workers"]
    out["shard_mb"] = shard
    out["shard_x_iters"] = shard * out["iterations"] if "iterations" in out else np.nan
    out["shard_x_features"] = shard * out["num_features"] if "num_features" in out else np.nan
    return out


def features_for(training_set: str) -> list[str]:
    return CATEGORICAL_FEATURES + TRAINING_SETS[training_set]["numeric"]


def select_training_rows(df: pd.DataFrame, training_set: str) -> tuple[pd.DataFrame, dict]:
    """
    Rows of one environment that may train a model: main set, finished OK,
    known allocation and — for the capped set — a recorded design.
    """
    info = {"training_set": training_set, "rows_in_file": len(df)}
    capped = df["thread_cap"].notna() if "thread_cap" in df else pd.Series(False, df.index)
    env = capped if training_set == "capped" else ~capped
    d = df[df["in_main_set"] & (df["status"] == "ok") & env]
    info["rows_env_main_ok"] = len(d)
    required = ["alloc_known"] + (["design_known"] if training_set == "capped" else [])
    unknown = ~d[required].all(axis=1)
    info["rows_dropped_unknown_inputs"] = int(unknown.sum())
    info["workloads_dropped_entirely"] = sorted(
        set(d.loc[unknown, "workload_type"]) - set(d.loc[~unknown, "workload_type"])
    )
    d = add_derived(d[~unknown])
    check_features(d, features_for(training_set))
    return d, info


# Fallback keys of the lookup baseline after the full feature key, most →
# least specific (levels whose columns are not model inputs are skipped).
LOOKUP_LEVELS: list[list[str]] = [
    ["workload_type", "dataset_mb", "num_workers", "cores_per_driver"],
    ["workload_type", "dataset_mb", "num_workers"],
    ["workload_type"],
    [],
]

# target → unit.  The first two are the targets named in the predictor handoff.
TARGETS: dict[str, str] = {
    "cpu_cores_p95": "cores",  # per driver
    "mem_mb_p95_max": "MB",  # largest driver of the run
    "cpu_cores_mean": "cores",  # per driver
    "mem_mb_peak_max": "MB",  # largest driver of the run
}

MODEL_FAMILIES = ("lookup", "ridge", "ridge_wl", "rf", "rf_wl", "ridge_log_wl")

RIDGE_ALPHAS = [0.1, 1.0, 10.0, 100.0]
RF_GRID = [{"max_depth": d, "min_samples_leaf": leaf} for d in (3, 5, 8) for leaf in (2, 5)]


def check_features(df: pd.DataFrame, features: list[str] = FEATURES) -> None:
    """Reject rows that miss a mandatory pre-run input."""
    missing = [c for c in features if c not in df]
    if missing:
        raise ValueError(f"missing feature columns: {missing}")
    bad = df[features].isna().any(axis=1)
    if bad.any():
        raise ValueError(f"{int(bad.sum())} rows have missing pre-run inputs")


# =============================================================================
# Models
# =============================================================================


def _log2(x):
    return np.log2(np.asarray(x, dtype=float))


def make_ridge(alpha: float = 1.0, numeric: list[str] = NUMERIC_FEATURES) -> Pipeline:
    """Ridge on one-hot workload + standardised log2(size, heap, ...) and counts."""
    pre = ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
            (
                "log",
                Pipeline([("log2", FunctionTransformer(_log2)), ("scale", StandardScaler())]),
                [c for c in numeric if c in LOG_FEATURES],
            ),
            ("lin", StandardScaler(), [c for c in numeric if c not in LOG_FEATURES]),
        ]
    )
    return Pipeline([("pre", pre), ("model", Ridge(alpha=alpha))])


def make_rf(
    max_depth: int = 5, min_samples_leaf: int = 2, seed: int = 0, n_estimators: int = 200
) -> Pipeline:
    """Small random forest; trees need no scaling, only the one-hot workload."""
    pre = ColumnTransformer(
        [("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES)],
        remainder="passthrough",
    )
    rf = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=seed,
        n_jobs=-1,
    )
    return Pipeline([("pre", pre), ("model", rf)])


class WorkloadInteractions(BaseEstimator, TransformerMixin):
    """
    Features for a per-workload linear model:
        [ one-hot workload | z(numeric) | one-hot workload × z(numeric) ]
    where z() standardises the LOG_FEATURES in log2 form and the counts with
    the TRAINING mean/std.  The interaction block gives every workload its own
    slope for each input, so Ridge is no longer forced to share one slope
    across K-Means and LogReg.  An unseen workload gets an all-zero one-hot
    (shared slopes only).
    """

    def __init__(self, numeric: list[str] = NUMERIC_FEATURES):
        self.numeric = numeric

    def _numeric(self, X: pd.DataFrame) -> np.ndarray:
        cols = [
            np.log2(X[c].to_numpy(float)) if c in LOG_FEATURES else X[c].to_numpy(float)
            for c in self.numeric
        ]
        return np.column_stack(cols)

    def fit(self, X: pd.DataFrame, y=None):
        self.workloads_ = sorted(X["workload_type"].unique())
        num = self._numeric(X)
        self.mean_ = num.mean(axis=0)
        std = num.std(axis=0)
        self.std_ = np.where(std > 0, std, 1.0)  # constant columns (e.g. workers)
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        z = (self._numeric(X) - self.mean_) / self.std_
        onehot = np.column_stack(
            [(X["workload_type"] == w).to_numpy(float) for w in self.workloads_]
        )
        inter = np.column_stack([onehot[:, [i]] * z for i in range(onehot.shape[1])])
        return np.hstack([onehot, z, inter])


class PerWorkloadModel(BaseEstimator, RegressorMixin):
    """
    One copy of `base` per workload, each fitted on that workload's rows only;
    predict() routes every row to the copy of its workload.  Without it, a
    single forest answers e.g. a K-Means job with more data per driver than
    any K-Means run from LogReg runs of that size.  A workload not seen in
    fit() is refused (the predictor reports such requests as unsupported).
    """

    def __init__(self, base=None):
        self.base = base

    def fit(self, X: pd.DataFrame, y) -> PerWorkloadModel:
        y = np.asarray(y, dtype=float)
        workloads = X["workload_type"].to_numpy()
        self.models_ = {
            wl: clone(self.base).fit(X[workloads == wl], y[workloads == wl])
            for wl in np.unique(workloads)
        }
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        workloads = X["workload_type"].to_numpy()
        unseen = sorted(set(workloads) - set(self.models_))
        if unseen:
            raise ValueError(f"no per-workload model for {unseen}")
        out = np.empty(len(X))
        for wl in np.unique(workloads):
            mask = workloads == wl
            out[mask] = self.models_[wl].predict(X[mask])
        return out


def make_rf_wl(
    max_depth: int = 5, min_samples_leaf: int = 2, seed: int = 0, n_estimators: int = 200
) -> PerWorkloadModel:
    """make_rf fitted per workload (see PerWorkloadModel)."""
    return PerWorkloadModel(make_rf(max_depth, min_samples_leaf, seed, n_estimators))


def make_ridge_log_wl(
    alpha: float = 1.0, numeric: list[str] = NUMERIC_FEATURES
) -> TransformedTargetRegressor:
    """
    make_ridge_wl on log(target): a per-workload POWER-LAW model
    (target ∝ Π x^β on the log inputs), the form of the growth laws seen
    in the data.  Predictions are exp(·) of the log prediction.
    """
    return TransformedTargetRegressor(
        regressor=make_ridge_wl(alpha, numeric), func=np.log, inverse_func=np.exp
    )


def make_ridge_int_wl(
    alpha: float = 1.0, numeric: list[str] = NUMERIC_FEATURES
) -> PerWorkloadModel:
    """
    Ridge with all pairwise products of the standardised inputs (e.g. size ×
    cores), fitted per workload (see PerWorkloadModel).
    """
    pre = ColumnTransformer(
        [
            (
                "log",
                FunctionTransformer(_log2),
                [c for c in numeric if c in LOG_FEATURES],
            ),
            ("lin", "passthrough", [c for c in numeric if c not in LOG_FEATURES]),
        ]
    )
    base = Pipeline(
        [
            ("pre", pre),
            ("scale", StandardScaler()),
            ("pairs", PolynomialFeatures(degree=2, include_bias=False)),
            ("model", Ridge(alpha=alpha)),
        ]
    )
    return PerWorkloadModel(base)


def make_ridge_wl(alpha: float = 1.0, numeric: list[str] = NUMERIC_FEATURES) -> Pipeline:
    """Ridge with per-workload slopes (see WorkloadInteractions)."""
    return Pipeline([("pre", WorkloadInteractions(numeric)), ("model", Ridge(alpha=alpha))])


class LookupModel:
    """
    Median of the training rows that share the most specific configuration
    key available: the full feature key first, then the coarser keys of
    LOOKUP_LEVELS.  `level_used_` records which level answered each
    prediction, so fallbacks are visible in the report.
    """

    def __init__(self, features: list[str] = FEATURES):
        self.features = features
        self.levels = [features] + [
            lvl for lvl in LOOKUP_LEVELS if set(lvl) <= set(features) and lvl != features
        ]

    def fit(self, X: pd.DataFrame, y) -> LookupModel:
        y = pd.Series(np.asarray(y, dtype=float), index=X.index)
        self.tables_ = []
        for keys in self.levels:
            if keys:
                self.tables_.append(y.groupby([X[k] for k in keys]).median().to_dict())
            else:
                self.tables_.append({(): float(y.median())})
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        preds, levels = [], []
        for _, row in X.iterrows():
            for level, (keys, table) in enumerate(zip(self.levels, self.tables_, strict=True)):
                key = (
                    tuple(row[k] for k in keys) if len(keys) > 1 else (row[keys[0]] if keys else ())
                )
                if key in table:
                    preds.append(table[key])
                    levels.append(level)
                    break
        self.level_used_ = np.array(levels)
        return np.array(preds, dtype=float)


# =============================================================================
# Metrics
# =============================================================================


def regression_metrics(y_true, y_pred, upper=None) -> dict:
    """Errors in physical units; R² only meaningful with enough spread."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    err = y_pred - y_true
    out = {
        "n": int(len(y_true)),
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "mean_signed_error": float(np.mean(err)),
        "mape_pct": float(np.mean(np.abs(err) / np.maximum(np.abs(y_true), 1e-9)) * 100),
        "underprediction_rate": float(np.mean(err < 0)),
        "underprediction_gt10pct_rate": float(np.mean(err < -0.10 * np.abs(y_true))),
    }
    var = np.var(y_true)
    out["r2"] = float(1 - np.mean(err**2) / var) if var > 0 else None
    if upper is not None:
        out["upper_bound_coverage"] = float(np.mean(np.asarray(upper) >= y_true))
    return out


def upper_margin(y_true, y_pred, quantile: float = 0.9) -> float:
    """Additive margin so that `pred + margin` covers `quantile` of residuals."""
    resid = np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)
    return float(max(np.quantile(resid, quantile), 0.0))


# a workload gets its own margin only with at least this many out-of-fold rows
MIN_ROWS_WORKLOAD_MARGIN = 20


def upper_margins(oof: pd.DataFrame, quantile: float = 0.9) -> dict[str, float]:
    """
    Upper margins per workload from out-of-fold rows (columns workload_type,
    actual, predicted), plus "pooled" over all rows.  Workloads differ in how
    noisy they are, so one shared margin is too wide for one and too narrow
    for the other; a workload with too few rows uses the pooled margin.
    """
    out = {"pooled": upper_margin(oof["actual"], oof["predicted"], quantile)}
    for wl, g in oof.groupby("workload_type"):
        if len(g) >= MIN_ROWS_WORKLOAD_MARGIN:
            out[wl] = upper_margin(g["actual"], g["predicted"], quantile)
    return out


def margin_for(margins: dict[str, float], workload: str) -> float:
    return margins.get(workload, margins["pooled"])


# =============================================================================
# Leave-one-configuration-out cross-validation (train+val only; test untouched)
# =============================================================================

# Fixed hyper-parameters for CV so the CV estimate is not tuned on its own folds.
CV_FACTORIES = {
    "lookup": LookupModel,
    "ridge": lambda: make_ridge(1.0),
    "ridge_wl": lambda: make_ridge_wl(1.0),
    "rf": lambda: make_rf(5, 2),
}


def cv_factories(training_set: str = "uncapped") -> dict:
    """CV_FACTORIES for the inputs of one training set."""
    num = TRAINING_SETS[training_set]["numeric"]
    return {
        "lookup": lambda: LookupModel(features_for(training_set)),
        "ridge": lambda: make_ridge(1.0, num),
        "ridge_wl": lambda: make_ridge_wl(1.0, num),
        "rf": lambda: make_rf(5, 2),
    }


def group_cv_predictions(
    df: pd.DataFrame,
    target: str,
    factories: dict | None = None,
    group_col: str = "config_group",
    features: list[str] = FEATURES,
) -> pd.DataFrame:
    """
    Out-of-fold predictions: each configuration group is predicted by a model
    trained on all OTHER groups, so repeats of the held-out configuration are
    never seen in training.  Returns one row per (input row, model).
    """
    factories = CV_FACTORIES if factories is None else factories
    d = df[df[target].notna()]
    frames = []
    for group in d[group_col].unique():
        held = d[d[group_col] == group]
        rest = d[d[group_col] != group]
        for name, factory in factories.items():
            model = factory().fit(rest[features], rest[target])
            frames.append(
                pd.DataFrame(
                    {
                        "row": held.index,
                        group_col: group,
                        "workload_type": held["workload_type"].values,
                        "model": name,
                        "actual": held[target].values,
                        "predicted": model.predict(held[features]),
                    }
                )
            )
    return pd.concat(frames, ignore_index=True)


def group_bootstrap(
    oof: pd.DataFrame,
    model: str,
    baseline: str | None = None,
    n_boot: int = 2000,
    seed: int = 0,
    group_col: str = "config_group",
) -> dict:
    """
    95 % bootstrap interval of MAE by resampling configuration GROUPS (the
    independent unit).  With `baseline`, also the paired MAE difference
    model − baseline on the same resampled groups (negative = model better).
    """
    rng = np.random.default_rng(seed)

    def per_group_abs(name):
        s = oof[oof["model"] == name]
        return (
            s.assign(ae=(s["predicted"] - s["actual"]).abs())
            .groupby(group_col)["ae"]
            .agg(["sum", "count"])
        )

    a = per_group_abs(model)
    groups = a.index.to_numpy()
    b = per_group_abs(baseline).reindex(groups) if baseline else None
    maes, diffs = [], []
    for _ in range(n_boot):
        pick = rng.choice(groups, size=len(groups), replace=True)
        sa = a.loc[pick]
        mae = sa["sum"].sum() / sa["count"].sum()
        maes.append(mae)
        if b is not None:
            sb = b.loc[pick]
            diffs.append(mae - sb["sum"].sum() / sb["count"].sum())
    out = {
        "mae": float(a["sum"].sum() / a["count"].sum()),
        "mae_ci95": [float(np.percentile(maes, 2.5)), float(np.percentile(maes, 97.5))],
        "n_groups": int(len(groups)),
    }
    if b is not None:
        point = out["mae"] - float(b["sum"].sum() / b["count"].sum())
        out |= {
            "baseline": baseline,
            "mae_diff_vs_baseline": point,
            "mae_diff_ci95": [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))],
        }
    return out
