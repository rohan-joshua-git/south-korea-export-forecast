"""Layer 0: the models everything else has to beat.

Two dumb benchmarks (naive, seasonal naive) plus SARIMA. The dumb
benchmarks matter as much as SARIMA here - if SARIMA can't beat a
seasonal random walk, that's already an interesting result.
"""

import warnings
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.filterwarnings("ignore")


def naive_forecast(train, horizon):
    return np.repeat(train.iloc[-1], horizon)


def seasonal_naive_forecast(train, horizon, season=12):
    last_season = train.iloc[-season:]
    reps = int(np.ceil(horizon / season))
    return np.tile(last_season.values, reps)[:horizon]


def select_sarima_order(series, seasonal_period=12):
    """Small grid search on AIC, run once on the full series up to the
    start of the test window. Refitting this search at every walk-forward
    origin would be needlessly slow and the order rarely moves much."""
    candidates = [
        (p, 1, q, P, 1, Q)
        for p in (0, 1, 2)
        for q in (0, 1, 2)
        for P in (0, 1)
        for Q in (0, 1)
    ]

    best_order, best_aic = None, np.inf
    for p, d, q, P, D, Q in candidates:
        try:
            model = SARIMAX(
                series,
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


def make_sarima_forecaster(order, seasonal_order):
    def fit_predict(train, horizon):
        model = SARIMAX(
            train,
            order=order,
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        fit = model.fit(disp=False)
        return fit.forecast(horizon).values

    return fit_predict
