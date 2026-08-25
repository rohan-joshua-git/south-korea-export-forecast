"""Diebold-Mariano test, with the Harvey-Leybourne-Newbold small-sample
correction (matters here - n=36 is not large enough to trust the
asymptotic normal approximation DM originally proposed).

Tests whether two forecasts' loss differential has non-zero mean. Squared-
error loss is used throughout this project (RMSE is the primary metric),
so that's what's compared here rather than absolute error.
"""

import numpy as np
from scipy import stats


def diebold_mariano(actual, forecast1, forecast2, h=1):
    """H0: forecast1 and forecast2 have equal predictive accuracy.
    Positive DM stat means forecast1 has larger loss (forecast2 better)."""
    e1 = actual - forecast1
    e2 = actual - forecast2
    d = e1**2 - e2**2

    n = len(d)
    d_bar = d.mean()

    # Newey-West-style long-run variance, lags = h-1 (0 for one-step)
    gamma0 = np.var(d, ddof=0)
    var_d = gamma0
    for lag in range(1, h):
        gamma_lag = np.cov(d[lag:], d[:-lag])[0, 1]
        var_d += 2 * gamma_lag
    var_d /= n

    dm_stat = d_bar / np.sqrt(var_d)

    # Harvey-Leybourne-Newbold (1997) small-sample correction
    hln_factor = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    dm_hln = dm_stat * hln_factor

    p_value = 2 * (1 - stats.t.cdf(np.abs(dm_hln), df=n - 1))

    return dm_hln, p_value


if __name__ == "__main__":
    import pandas as pd

    sarima = pd.read_csv("../data/backtest_sarima.csv", index_col=0, parse_dates=True)
    blend = pd.read_csv("../data/backtest_blend.csv", index_col=0, parse_dates=True)

    sarima, blend = sarima.align(blend, join="inner")
    stat, p = diebold_mariano(sarima["actual"].values, sarima["forecast"].values, blend["forecast"].values)

    print(f"n={len(sarima)}")
    print(f"DM statistic (HLN-corrected): {stat:.3f}")
    print(f"p-value: {p:.4f}")
    print(f"{'Significant at 5%' if p < 0.05 else 'NOT significant at 5%'}")
