"""Rolling-origin walk-forward validation.

Standard expanding-window backtest: train on everything up to time t,
forecast h steps ahead, slide t forward by one period, refit, repeat.
This is what "at least 36 months out-of-sample" means in practice -
36 separate origins, each refit on only the data that would have been
available at that point in time.
"""

import pandas as pd


def walk_forward(series, fit_predict, n_test, horizon=1, min_train=60):
    """
    series: pd.Series indexed by date, target to forecast
    fit_predict: fn(train_series, horizon) -> array of length `horizon` forecasts
    n_test: number of origins to evaluate (>=36 for the assignment)
    horizon: steps ahead per forecast (1 = one-month-ahead)
    min_train: minimum observations before the first origin

    Returns a DataFrame with one row per origin: date, actual, forecast.
    """
    n = len(series)
    first_origin = n - n_test - horizon + 1
    if first_origin < min_train:
        raise ValueError(
            f"Not enough data: need {min_train + n_test + horizon - 1} obs, have {n}"
        )

    rows = []
    for origin in range(first_origin, n - horizon + 1):
        train = series.iloc[:origin]
        actual = series.iloc[origin + horizon - 1]
        target_date = series.index[origin + horizon - 1]

        forecast = fit_predict(train, horizon)[-1]

        rows.append({"date": target_date, "actual": actual, "forecast": forecast})

    return pd.DataFrame(rows).set_index("date")


def walk_forward_x(y, X, fit_predict, n_test, min_train=36):
    """
    Same rolling-origin idea as walk_forward(), but for models that take
    exogenous features. One-step-ahead only (this project never needs
    multi-step here, and it keeps fit_predict's signature simple).

    y: pd.Series, target
    X: pd.DataFrame, features aligned to y's index (same dates)
    fit_predict: fn(y_train, X_train, x_pred_row) -> scalar forecast
    """
    n = len(y)
    first_origin = n - n_test
    if first_origin < min_train:
        raise ValueError(f"Not enough data: need {min_train + n_test} obs, have {n}")

    rows = []
    for origin in range(first_origin, n):
        y_train = y.iloc[:origin]
        X_train = X.iloc[:origin]
        x_pred = X.iloc[[origin]]
        actual = y.iloc[origin]
        target_date = y.index[origin]

        forecast = fit_predict(y_train, X_train, x_pred)

        rows.append({"date": target_date, "actual": actual, "forecast": forecast})

    return pd.DataFrame(rows).set_index("date")


def rmse(results):
    err = results["actual"] - results["forecast"]
    return (err ** 2).mean() ** 0.5


def mape(results):
    return ((results["actual"] - results["forecast"]).abs() / results["actual"]).mean() * 100
