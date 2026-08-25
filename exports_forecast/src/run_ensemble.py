"""SARIMA + Ridge(Layer 2) forecast combination.

Simple average-of-forecasts (Bates & Granger, 1969) - one of the most
robustly-replicated findings in the forecasting literature is that
combining two mediocre, differently-wrong models often beats either one
alone, and often beats "smarter" single models too. Worth trying precisely
because it's simple: two already-validated models, no new data, no new
model family.

Important methodological note: the blend weight MUST be chosen without
looking at the test window, exactly like SARIMA's order and Ridge's alpha
were. An earlier exploratory pass here grid-searched the weight directly
against the 36-month test set and found a "better" number (MAPE ~2.6%) -
that number is test-set-tuned and not a valid out-of-sample result. It's
reported below only as a ceiling estimate, not the headline result.
"""

import pandas as pd

from data import load_exports
from baseline import select_sarima_order, make_sarima_forecaster
from layer2_model import build_dataset as build_layer2_dataset
from layer1_model import make_ridge_forecaster
from walk_forward import walk_forward, walk_forward_x, rmse, mape
from diebold_mariano import diebold_mariano

N_TEST = 36
VAL_SIZE = 24
WEIGHT_GRID = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]


def blended_rmse(sarima_df, ridge_df, weight):
    actual = sarima_df["actual"]
    forecast = weight * sarima_df["forecast"] + (1 - weight) * ridge_df.reindex(sarima_df.index)["forecast"]
    df = actual.to_frame("actual").assign(forecast=forecast)
    return rmse(df), mape(df)


if __name__ == "__main__":
    series = load_exports()
    y2, X2 = build_layer2_dataset()

    order, seasonal_order = select_sarima_order(series.iloc[: len(series) - N_TEST])
    sarima_fn = make_sarima_forecaster(order, seasonal_order)
    ridge_fn = make_ridge_forecaster(alpha=10.0)

    # --- weight selection on the validation window (pre-test, not test) ---
    val_cutoff = len(y2) - N_TEST
    y2_dev, X2_dev = y2.iloc[:val_cutoff], X2.iloc[:val_cutoff]
    series_dev = series[series.index <= y2_dev.index.max()]

    sarima_val = walk_forward(series_dev, sarima_fn, n_test=VAL_SIZE)
    ridge_val = walk_forward_x(y2_dev, X2_dev, ridge_fn, n_test=VAL_SIZE)

    best_w, best_val_rmse = None, float("inf")
    for w in WEIGHT_GRID:
        r, _ = blended_rmse(sarima_val, ridge_val, w)
        if r < best_val_rmse:
            best_w, best_val_rmse = w, r
    print(f"weight selected on validation window (pre-test): sarima_weight={best_w}")

    # --- apply that weight, untouched, to the actual test window ---
    sarima_test = walk_forward(series, sarima_fn, n_test=N_TEST)
    ridge_test = walk_forward_x(y2, X2, ridge_fn, n_test=N_TEST)

    r, m = blended_rmse(sarima_test, ridge_test, best_w)
    print(f"\nHONEST result (weight chosen out-of-sample of the test window):")
    print(f"{'blend':>20}  RMSE={r:>14,.0f}  MAPE={m:>6.2f}%")

    blend_forecast = best_w * sarima_test["forecast"] + (1 - best_w) * ridge_test.reindex(sarima_test.index)["forecast"]
    blend_df = sarima_test["actual"].to_frame("actual").assign(forecast=blend_forecast)
    blend_df.to_csv("../data/backtest_blend.csv")

    seasonal_naive = pd.read_csv("../data/backtest_seasonal_naive.csv", index_col=0, parse_dates=True)
    print(f"\nReference:")
    print(f"{'naive':>20}  RMSE={2_847_251_652:>14,.0f}  MAPE={3.14:>6.2f}%")
    print(f"{'seasonal_naive':>20}  RMSE={rmse(seasonal_naive):>14,.0f}  MAPE={mape(seasonal_naive):>6.2f}%")
    print(f"{'sarima':>20}  RMSE={rmse(sarima_test):>14,.0f}  MAPE={mape(sarima_test):>6.2f}%")
    print(f"{'ridge_layer2':>20}  RMSE={rmse(ridge_test):>14,.0f}  MAPE={mape(ridge_test):>6.2f}%")

    print(f"\nFor context only - NOT a valid result (weight tuned directly on test set):")
    for w in WEIGHT_GRID:
        r, m = blended_rmse(sarima_test, ridge_test, w)
        print(f"  weight={w}  RMSE={r:>14,.0f}  MAPE={m:>6.2f}%")

    print(f"\nDiebold-Mariano test, SARIMA vs. blend (H0: equal accuracy):")
    dm_stat, p_value = diebold_mariano(
        sarima_test["actual"].values, sarima_test["forecast"].values, blend_df["forecast"].values
    )
    print(f"  DM statistic (HLN-corrected): {dm_stat:.3f}")
    print(f"  p-value: {p_value:.4f}")
    print(f"  {'Significant at 5%' if p_value < 0.05 else 'NOT significant at 5%'} (n={len(sarima_test)})")
