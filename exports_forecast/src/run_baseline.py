"""Runs the Layer 0 walk-forward backtest and prints the comparison table.

This is the anchor result every later layer (structured alt-data, DART
sentiment, satellite) has to beat to earn its place in the model.
"""

import time
from data import load_exports
from walk_forward import walk_forward, rmse, mape
from baseline import (
    naive_forecast,
    seasonal_naive_forecast,
    select_sarima_order,
    make_sarima_forecaster,
)

N_TEST = 36
HORIZON = 1

if __name__ == "__main__":
    series = load_exports()

    order_selection_cutoff = len(series) - N_TEST
    order, seasonal_order = select_sarima_order(series.iloc[:order_selection_cutoff])
    print(f"SARIMA order selected on pre-test data: {order} x {seasonal_order}")

    sarima_forecast = make_sarima_forecaster(order, seasonal_order)

    models = {
        "naive": naive_forecast,
        "seasonal_naive": seasonal_naive_forecast,
        "sarima": sarima_forecast,
    }

    results = {}
    for name, fn in models.items():
        t0 = time.time()
        results[name] = walk_forward(series, fn, n_test=N_TEST, horizon=HORIZON)
        print(f"{name:>15}  RMSE={rmse(results[name]):>14,.0f}  MAPE={mape(results[name]):>6.2f}%  ({time.time()-t0:.1f}s)")

    for name, df in results.items():
        df.to_csv(f"../data/backtest_{name}.csv")
