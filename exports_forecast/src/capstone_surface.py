"""Capstone: OOS RMSE vs. feature count vs. regularization strength.

Controlled experiment, not a comparison across the differently-sized
datasets from Layers 1/2/simplification (those differ in sample size AND
feature count at once, which would confound the surface). Instead: fix the
dataset to Layer 2's (96 rows, same 36-month test window throughout), add
features one at a time in the order SHAP ranked them for the Layer 2 GBM
(most important first), and grid Ridge's alpha at each cumulative feature
count. That isolates exactly the two things this project has been asking
about - does feature count help, does regularization strength help - with
everything else held constant.

This is a diagnostic/descriptive surface, not a hyperparameter search: the
whole grid is reported, not just the best cell, and no "optimal" point from
it is claimed as a deployable model. Picking the single best (count, alpha)
cell post-hoc and calling it the new headline result would be exactly the
test-set-tuning mistake caught and fixed in run_ensemble.py.
"""

import numpy as np
import pandas as pd

from layer2_model import build_dataset
from layer1_model import make_ridge_forecaster
from walk_forward import walk_forward_x, rmse

N_TEST = 36
ALPHA_GRID = [0.1, 1, 3, 10, 30, 100, 300, 1000]

# SHAP importance order from the Layer 2 GBM (most to least important) -
# see 04_layer2_sentiment.ipynb for the source ranking.
FEATURE_ORDER = [
    "target_lag1", "oil_wti", "samsung", "sox", "usd_krw",
    "samsung_sentiment", "skhynix", "bdry", "target_lag12", "trends",
]


def build_surface():
    y, X = build_dataset()
    assert set(FEATURE_ORDER) == set(X.columns)

    rows = []
    for count in range(1, len(FEATURE_ORDER) + 1):
        cols = FEATURE_ORDER[:count]
        X_subset = X[cols]
        for alpha in ALPHA_GRID:
            forecaster = make_ridge_forecaster(alpha=alpha)
            results = walk_forward_x(y, X_subset, forecaster, n_test=N_TEST)
            rows.append({"n_features": count, "alpha": alpha, "rmse": rmse(results)})
            print(f"  n_features={count:>2}  alpha={alpha:>6}  RMSE={rmse(results):,.0f}")

    return pd.DataFrame(rows)


if __name__ == "__main__":
    surface = build_surface()
    surface.to_csv("../data/capstone_surface.csv", index=False)
    print(f"\nwrote {len(surface)} rows to ../data/capstone_surface.csv")

    best = surface.loc[surface["rmse"].idxmin()]
    print(f"\nlowest RMSE in the grid (descriptive only, NOT a chosen model): "
          f"n_features={best.n_features:.0f}, alpha={best.alpha}, RMSE={best.rmse:,.0f}")
