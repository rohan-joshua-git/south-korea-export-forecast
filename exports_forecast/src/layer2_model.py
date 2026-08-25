"""Layer 2: Layer 1's feature set plus Samsung earnings-call sentiment."""

import pandas as pd

from data import load_exports
from features import build_feature_matrix
from earnings_sentiment import build_sentiment_series


def build_dataset():
    y = load_exports()
    X = build_feature_matrix()
    X = X.copy()
    X["target_lag1"] = y.shift(1)
    X["target_lag12"] = y.shift(12)
    X["samsung_sentiment"] = build_sentiment_series()

    df = pd.concat([y.rename("y"), X], axis=1).dropna()
    return df["y"], df.drop(columns="y")


if __name__ == "__main__":
    y, X = build_dataset()
    print(f"{len(y)} usable rows, {X.shape[1]} features, {y.index.min().date()} to {y.index.max().date()}")
    print(list(X.columns))
