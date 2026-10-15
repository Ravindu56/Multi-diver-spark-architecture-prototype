# =============================================================================
# mpj_spark/resource_prediction/calibration.py
# Phase 6 — upper bounds beyond the measured range (step A3)
#
# PURPOSE
# -------
# In-range upper bounds come from out-of-fold residuals (models.calibrated_
# margins).  For requests BEYOND the measured data per driver or cores they are
# too narrow, so this module derives extra, relative margins from TRAIN+VAL
# stress tests that hold out larger jobs / more cores and predict them the way
# the predictor does (deployed model + TREND_POLICY growth trend):
#
#   size   K-Means data per driver > 50 MB, LogReg > 100 MB
#   cores  K-Means cores > 2, LogReg cores > 4 — scored on Set D sweep rows
#          only (amendment 1 of scripts/upper_bound_tests.py: every Set A
#          K-Means run is at 4 cores, so holding out "> 2 cores" also removes
#          all Set A designs from training, and their errors would measure
#          the missing designs, not the extra cores)
#
# The predictor applies them as a floor for out-of-range requests:
#   upper = max(in-range upper, predicted × (1 + extrapolation margin))
# =============================================================================

from __future__ import annotations

import numpy as np
import pandas as pd

from mpj_spark.resource_prediction import models as m
from mpj_spark.resource_prediction import transfer as t

QUANTILE = 0.9
# axis → workload → rows above this value are held out
STRESS: dict[str, dict[str, float]] = {
    "size": {"kmeans": 50.0, "logreg": 100.0},
    "cores": {"kmeans": 2.0, "logreg": 4.0},
}
# axis → data sets whose held-out rows are scored (None = all)
STRESS_SCORED_SETS: dict[str, tuple[str, ...] | None] = {"size": None, "cores": ("D",)}


def stress_margins(
    trval: pd.DataFrame,
    target: str,
    keys: dict[str, str],
    trend_policy: dict,
    scout: pd.DataFrame | None = None,
    training_set: str = "capped",
    n_estimators: int | None = None,
) -> dict:
    """
    {axis: {workload: {margin, rows, groups, trend}}}: relative group-weighted
    q-quantile of (actual − predicted) / predicted on the held-out rows.
    `keys` = {workload: CV selection key of the deployed family}.
    """
    num = m.TRAINING_SETS[training_set]["numeric"]
    features = m.features_for(training_set)
    d = trval[trval[target].notna()]
    out = {}
    for axis, by_wl in STRESS.items():
        for wl, limit in by_wl.items():
            if wl not in keys:
                continue
            held = (d["workload_type"] == wl).to_numpy() & (t.axis_value(d, axis) > limit)
            scored_sets = STRESS_SCORED_SETS[axis]
            scored = held.copy()
            if scored_sets is not None and "experiment" in d:
                scored &= d["experiment"].isin(scored_sets).to_numpy()
            train, ev = d[~held], d[scored]
            if ev.empty or train[train["workload_type"] == wl].empty:
                continue
            model = m.factory_from_key(keys[wl], num, n_estimators)()
            model.fit(train[features], train[target])
            source = trend_policy.get(axis, {}).get(wl, {}).get(target)
            beta = None
            if source:
                local = t.local_elasticity(train, target, wl, axis)
                sc = t.scout_elasticity(scout, target, wl, axis) if scout is not None else None
                info = t.combine_elasticity(local, sc) if source == "combined" else local
                beta = info["beta"] if info else None
            x_max = float(t.axis_value(train[train["workload_type"] == wl], axis).max())
            pred = t.anchor_and_scale(model, ev, features, x_max, beta, axis)
            rows = pd.DataFrame(
                {
                    "actual": ev[target].to_numpy(float),
                    "predicted": np.asarray(pred, dtype=float),
                    "config_group": ev["config_group"].values,
                }
            )
            out.setdefault(axis, {})[wl] = {
                "margin": m.calibrate_margin(rows, QUANTILE, "relative_groups"),
                "rows": len(ev),
                "groups": int(ev["config_group"].nunique()),
                "trend": source,
            }
    return out
