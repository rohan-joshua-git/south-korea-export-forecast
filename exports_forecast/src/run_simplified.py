"""Runs Ridge on the simplified feature sets and compares to Layer 0."""

from simplified_model import build_dataset
from layer1_model import make_ridge_forecaster
from walk_forward import walk_forward_x, rmse, mape

N_TEST = 36
ALPHA_GRID = [1.0, 3.0, 10.0, 30.0, 100.0]


def select_alpha(y, X, n_test, val_size=24):
    """Picks alpha using a validation window immediately before the test
    window - the test window itself is never touched by this choice."""
    y_dev = y.iloc[: len(y) - n_test]
    X_dev = X.iloc[: len(y) - n_test]

    best_alpha, best_rmse = None, float("inf")
    for alpha in ALPHA_GRID:
        forecaster = make_ridge_forecaster(alpha=alpha)
        val_results = walk_forward_x(y_dev, X_dev, forecaster, n_test=val_size)
        val_rmse = rmse(val_results)
        if val_rmse < best_rmse:
            best_alpha, best_rmse = alpha, val_rmse

    return best_alpha


if __name__ == "__main__":
    for use_sentiment in (False, True):
        y, X = build_dataset(use_sentiment)
        label = "ridge_with_sentiment" if use_sentiment else "ridge_long_history"

        alpha = select_alpha(y, X, N_TEST)
        forecaster = make_ridge_forecaster(alpha=alpha)
        results = walk_forward_x(y, X, forecaster, n_test=N_TEST)

        print(f"{label:>22}  alpha={alpha:>6}  RMSE={rmse(results):>14,.0f}  MAPE={mape(results):>6.2f}%  ({len(y)} rows, {X.shape[1]} features)")
        results.to_csv(f"../data/backtest_{label}.csv")
