"""Runs the SARIMAX walk-forward backtest, benchmarked against SARIMA and
naive recomputed on the SAME window - not against Layer 0's official
436-row numbers, which cover a much longer and different period. Unlike
run_layer1_customs.py's 14-origin exploratory check, this one has enough
history (88 rows, PortWatch's 2019-01 start is the binding constraint) to
run the assignment's actual 36-month walk-forward minimum.
"""

import pandas as pd

from baseline import naive_forecast, make_sarima_forecaster
from sarimax_model import build_dataset, select_sarimax_order, make_sarimax_forecaster
from walk_forward import walk_forward, walk_forward_x, rmse, mape
from diebold_mariano import diebold_mariano

N_TEST = 36

if __name__ == "__main__":
    y, X = build_dataset()
    print(f"dataset: {len(y)} rows, {X.shape[1]} exog features, "
          f"{y.index.min().date()} to {y.index.max().date()}\n")

    order_cutoff = len(y) - N_TEST
    y_dev, X_dev = y.iloc[:order_cutoff], X.iloc[:order_cutoff]

    sarimax_order = select_sarimax_order(y_dev, X_dev)
    print(f"SARIMAX order selected on pre-test data: {sarimax_order}")
    sarima_order = select_sarimax_order(y_dev)  # same capped grid, X=None - see module docstring
    print(f"SARIMA (same window, no exog) order selected on pre-test data: {sarima_order}\n")

    sarimax_fn = make_sarimax_forecaster(*sarimax_order)
    sarima_fn = make_sarima_forecaster(*sarima_order)

    # Match min_train to order_cutoff: the order was selected using exactly
    # that many rows, so no walk-forward origin should train on fewer rows
    # than informed the order choice. An earlier attempt at min_train=40
    # (less than order_cutoff=52) produced explosive, non-converged
    # forecasts at the first several origins - a numerically unstable fit,
    # not a real result - confirming this matters, not just a formality.
    MIN_TRAIN = order_cutoff
    sarimax_results = walk_forward_x(y, X, sarimax_fn, n_test=N_TEST, min_train=MIN_TRAIN)
    sarima_results = walk_forward(y, sarima_fn, n_test=N_TEST, min_train=MIN_TRAIN)
    naive_results = walk_forward(y, naive_forecast, n_test=N_TEST, min_train=MIN_TRAIN)

    print(f"{'naive (same window)':>28}  RMSE={rmse(naive_results):>14,.0f}  MAPE={mape(naive_results):>6.2f}%")
    print(f"{'SARIMA (same window)':>28}  RMSE={rmse(sarima_results):>14,.0f}  MAPE={mape(sarima_results):>6.2f}%")
    print(f"{'SARIMAX (wsts+busan)':>28}  RMSE={rmse(sarimax_results):>14,.0f}  MAPE={mape(sarimax_results):>6.2f}%")

    print(f"\nFor reference only (NOT the same window/sample - see module docstring):")
    print(f"{'Layer 0 official SARIMA':>28}  MAPE=2.92%  (436-row series, 1990-2026)")

    sarimax_results.to_csv("../data/backtest_sarimax.csv")
    sarima_results.to_csv("../data/backtest_sarima_same_window.csv")
    naive_results.to_csv("../data/backtest_naive_same_window.csv")

    print(f"\nDiebold-Mariano, SARIMA (same window) vs. SARIMAX (H0: equal accuracy):")
    dm_stat, p_value = diebold_mariano(
        sarima_results["actual"].values, sarima_results["forecast"].values, sarimax_results["forecast"].values
    )
    print(f"  DM statistic (HLN-corrected): {dm_stat:.3f}")
    print(f"  p-value: {p_value:.4f}")
    print(f"  {'Significant at 5%' if p_value < 0.05 else 'NOT significant at 5%'} (n={len(sarima_results)})")
