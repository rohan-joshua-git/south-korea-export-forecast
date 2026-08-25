"""Exploratory check: does adding the Customs flash-export feature change
anything, now that it's actually sourced (see customs_flash.py)?

NOT a Layer result on the same footing as Layers 0-2. The case requires at
least 36 months of out-of-sample walk-forward; customs_flash_yoy's real
parseable coverage (2021-05 onward - see customs_flash.py and features.py
docstrings for why) leaves only 50 usable rows after alignment with the
rest of Layer 1's features and the lag terms. That caps the test window at
14 origins even with min_train shaved down, well under the assignment's
minimum. Reported honestly as a directional, exploratory check only - the
same discipline the capstone notebook applies to its own best-cell finding,
not a new headline number.

Also reports a same-window Ridge/GBM WITHOUT the customs feature, so the
comparison isn't just "worse numbers than the real Layer 1" (a smaller,
more recent window is a different, generally easier problem on its own -
see the mismatch already flagged between Layer 0's 436-row window and
Layer 1's 98-row one) but an apples-to-apples check of whether the customs
feature helps within the one window where it's available at all.
"""

from walk_forward import walk_forward_x, rmse, mape
from layer1_model import build_dataset, make_gbm_forecaster, make_ridge_forecaster

N_TEST = 14   # NOT the assignment's 36-month minimum - see module docstring
MIN_TRAIN = 30

if __name__ == "__main__":
    y_with, X_with = build_dataset(include_customs_flash=True)
    y_without, X_without = build_dataset(include_customs_flash=False)
    # align the "without" dataset to the exact same dates as the "with" one,
    # so the comparison isn't confounded by different windows
    common_index = y_with.index
    y_without, X_without = y_without.loc[common_index], X_without.loc[common_index]

    print(f"WITH customs_flash:    {len(y_with)} rows, {X_with.shape[1]} features, "
          f"{y_with.index.min().date()} to {y_with.index.max().date()}")
    print(f"WITHOUT (same window): {len(y_without)} rows, {X_without.shape[1]} features")
    print(f"n_test={N_TEST} (assignment minimum is 36 - this is exploratory only, see module docstring)\n")

    variants = {
        "gbm_with_customs": (X_with, make_gbm_forecaster()),
        "ridge_with_customs": (X_with, make_ridge_forecaster(alpha=10.0)),
        "gbm_without_customs_same_window": (X_without, make_gbm_forecaster()),
        "ridge_without_customs_same_window": (X_without, make_ridge_forecaster(alpha=10.0)),
    }

    for name, (X, fn) in variants.items():
        results = walk_forward_x(y_with, X, fn, n_test=N_TEST, min_train=MIN_TRAIN)
        print(f"{name:>34}  RMSE={rmse(results):>14,.0f}  MAPE={mape(results):>6.2f}%")
        results.to_csv(f"../data/backtest_{name}_EXPLORATORY.csv")
