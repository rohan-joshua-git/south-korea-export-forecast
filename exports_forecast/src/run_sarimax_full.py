"""The decisive test: SARIMA + WSTS Asia Pacific billings, at Layer 0's
actual training length (436 rows, 1990-2026), not the shorter window
run_sarimax.py was forced into by Busan PortWatch's 2019 start. WSTS goes
back to 1986, before the target series even starts, so this dataset needs
no truncation at all - the only previous constraint (sample length) is
gone here. If genuine industry billings data doesn't help SARIMA at full
sample length, that's a real answer about the feature, not a sample-size
excuse.

Uses baseline.py's original (uncapped) order grid, since the numerical
instability that forced a smaller grid in sarimax_model.py was specifically
a short-sample (88-row) problem - not expected to recur at 436 rows, and
worth confirming rather than assuming.
"""

import pandas as pd

from data import load_exports
from features import wsts_asia_pacific
from baseline import naive_forecast, select_sarima_order, make_sarima_forecaster
from sarimax_model import make_sarimax_forecaster
from walk_forward import walk_forward, walk_forward_x, rmse, mape
from diebold_mariano import diebold_mariano

N_TEST = 36

if __name__ == "__main__":
    y = load_exports()
    X = wsts_asia_pacific().rename("wsts_asia_pacific").to_frame()
    df = pd.concat([y.rename("y"), X], axis=1).dropna()
    y, X = df["y"], df.drop(columns="y")
    print(f"dataset: {len(y)} rows, {X.shape[1]} exog feature(s), "
          f"{y.index.min().date()} to {y.index.max().date()}\n")

    order_cutoff = len(y) - N_TEST
    order, seasonal_order = select_sarima_order(y.iloc[:order_cutoff])
    print(f"SARIMAX order selected on pre-test data (uncapped grid): {order} x {seasonal_order}")

    sarima_fn = make_sarima_forecaster(order, seasonal_order)
    sarimax_fn = make_sarimax_forecaster(order, seasonal_order)

    naive_results = walk_forward(y, naive_forecast, n_test=N_TEST)
    sarima_results = walk_forward(y, sarima_fn, n_test=N_TEST)
    sarimax_results = walk_forward_x(y, X, sarimax_fn, n_test=N_TEST, min_train=order_cutoff - N_TEST)

    print(f"\n{'naive':>28}  RMSE={rmse(naive_results):>14,.0f}  MAPE={mape(naive_results):>6.2f}%")
    print(f"{'SARIMA (Layer 0, no exog)':>28}  RMSE={rmse(sarima_results):>14,.0f}  MAPE={mape(sarima_results):>6.2f}%")
    print(f"{'SARIMAX (+ WSTS Asia Pac.)':>28}  RMSE={rmse(sarimax_results):>14,.0f}  MAPE={mape(sarimax_results):>6.2f}%")

    sarimax_results.to_csv("../data/backtest_sarimax_wsts_full.csv")

    print(f"\nDiebold-Mariano, SARIMA vs. SARIMAX+WSTS (H0: equal accuracy):")
    dm_stat, p_value = diebold_mariano(
        sarima_results["actual"].values, sarima_results["forecast"].values, sarimax_results["forecast"].values
    )
    print(f"  DM statistic (HLN-corrected): {dm_stat:.3f}")
    print(f"  p-value: {p_value:.4f}")
    print(f"  {'Significant at 5%' if p_value < 0.05 else 'NOT significant at 5%'} (n={len(sarima_results)})")
