"""Runs the Layer 1 walk-forward backtest and compares against Layer 0."""

from walk_forward import walk_forward_x, rmse, mape
from layer1_model import build_dataset, make_gbm_forecaster, make_ridge_forecaster

N_TEST = 36

if __name__ == "__main__":
    y, X = build_dataset()
    print(f"dataset: {len(y)} rows, {X.shape[1]} features, {y.index.min().date()} to {y.index.max().date()}")

    models = {
        "gbm_layer1": make_gbm_forecaster(),
        "ridge_layer1": make_ridge_forecaster(alpha=10.0),
    }

    for name, fn in models.items():
        results = walk_forward_x(y, X, fn, n_test=N_TEST)
        print(f"{name:>15}  RMSE={rmse(results):>14,.0f}  MAPE={mape(results):>6.2f}%")
        results.to_csv(f"../data/backtest_{name}.csv")
