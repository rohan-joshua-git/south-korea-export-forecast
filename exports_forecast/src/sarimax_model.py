"""SARIMAX: Layer 0's SARIMA structure, plus genuinely leading/direct
exogenous regressors, fit jointly in one state-space model rather than
routed through a separate GBM/Ridge on top of lag features the way
Layer 1/2 did.

Two exogenous variables, deliberately minimal (Occam's razor, and the
capstone notebook's own finding that more features need more
regularization - a two-variable SARIMAX doesn't need a regularization
argument the way a 9-feature Ridge did):

  wsts_asia_pacific  - actual semiconductor billings revenue, not a
                        market-price proxy (see wsts_billings.py)
  busan_export_tons  - satellite-tracked physical export tonnage through
                        Busan, IMF PortWatch (see portwatch.py)

Both are real, freely-sourced, no-key-required data - unlike everything
else added in this pass, neither needed a workaround. busan_export_tons'
2019-01 start is the binding constraint on the merged dataset (91 months
before target-month alignment), which is comfortably enough for the
assignment's 36-month walk-forward minimum, unlike the Customs flash
feature's 50-row dataset (see run_layer1_customs.py) - this is the first
alt-data attempt in the project that can actually be tested at the
required rigor rather than only exploratorily.
"""

import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

from data import load_exports
from features import wsts_asia_pacific, busan_export_tons

warnings.filterwarnings("ignore")


def build_dataset():
    y = load_exports()
    X = pd.concat([wsts_asia_pacific(), busan_export_tons()], axis=1)
    df = pd.concat([y.rename("y"), X], axis=1).dropna()
    return df["y"], df.drop(columns="y")


def select_sarimax_order(y, X=None, seasonal_period=12):
    """AIC grid search, same idea as baseline.py's select_sarima_order,
    but deliberately a smaller grid (p,q,P,Q capped at 1, not 2) - this
    dataset is 88 rows against Layer 0's 436, and an earlier attempt at
    baseline.py's full grid picked (2,1,2)(1,1,1,12) here, which produced
    numerically unstable, non-converged forecasts (RMSE in the
    quadrillions) at the walk-forward origins with the least training
    data. Seasonal differencing alone consumes 12 observations; a 6-ish
    parameter ARMA structure on top of that isn't safely identifiable
    with only ~40-50 effective rows. Capping the grid is itself an
    Occam's-razor-consistent response to that failure, not an arbitrary
    restriction - use the same capped grid for both the SARIMA and
    SARIMAX comparison so the two are chosen by an identical procedure.

    X=None runs a plain SARIMA order search (used for the same-window,
    no-exog comparison in run_sarimax.py) through the same code path,
    rather than a second copy of this function."""
    candidates = [
        (p, 1, q, P, 1, Q)
        for p in (0, 1)
        for q in (0, 1)
        for P in (0, 1)
        for Q in (0, 1)
    ]

    best_order, best_aic = None, np.inf
    for p, d, q, P, D, Q in candidates:
        try:
            model = SARIMAX(
                y, exog=X,
                order=(p, d, q),
                seasonal_order=(P, D, Q, seasonal_period),
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            fit = model.fit(disp=False)
            if fit.aic < best_aic:
                best_aic = fit.aic
                best_order = ((p, d, q), (P, D, Q, seasonal_period))
        except Exception:
            continue

    return best_order


def make_sarimax_forecaster(order, seasonal_order):
    def fit_predict(y_train, X_train, x_pred):
        model = SARIMAX(
            y_train, exog=X_train,
            order=order, seasonal_order=seasonal_order,
            enforce_stationarity=False, enforce_invertibility=False,
        )
        fit = model.fit(disp=False)
        return fit.forecast(1, exog=x_pred).values[0]

    return fit_predict


if __name__ == "__main__":
    y, X = build_dataset()
    print(f"{len(y)} usable rows, {X.shape[1]} exog features, "
          f"{y.index.min().date()} to {y.index.max().date()}")
