"""Simpler Layer 1/2: hand-picked features instead of throwing everything in.

Dropped based on what Layers 1-2 actually showed: BDRY (SHAP: low
importance, and the thing capping the whole dataset at ~98 rows),
SK Hynix (redundant with Samsung + SOX as memory-cycle proxies, low SHAP),
Google Trends (SHAP importance of exactly zero), target_lag12 (low SHAP,
SARIMA's seasonal term already covers this better than a raw lag can).

Two variants, because keeping sentiment re-introduces the same short-history
problem BDRY caused:
  - long history (2000+, no sentiment): tests whether more data alone
    closes the gap to Layer 0
  - short history (2018+, with sentiment): tests whether the one feature
    that helped in Layer 2 is worth the row count it costs
"""

import pandas as pd

from data import load_exports
from features import usd_krw, oil, sox, samsung
from earnings_sentiment import build_sentiment_series


def build_dataset(use_sentiment):
    y = load_exports()
    X = pd.concat([usd_krw(), oil(), sox(), samsung()], axis=1)
    X["target_lag1"] = y.shift(1)
    if use_sentiment:
        X["samsung_sentiment"] = build_sentiment_series()

    df = pd.concat([y.rename("y"), X], axis=1).dropna()
    return df["y"], df.drop(columns="y")


if __name__ == "__main__":
    for use_sentiment in (False, True):
        y, X = build_dataset(use_sentiment)
        label = "with sentiment" if use_sentiment else "no sentiment, long history"
        print(f"{label:>28}: {len(y)} rows, {X.shape[1]} features, {y.index.min().date()} to {y.index.max().date()}")
