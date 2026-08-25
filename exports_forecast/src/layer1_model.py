"""Layer 1: structured alt-data fed into a GBM, benchmarked against Layer 0.

Chose GBM+SHAP over a dynamic factor model for the first pass mainly for
time: DFM needs care around factor identification and is a bigger lift to
get right in two weeks, while a shallow, heavily-regularized GBM is fast
to validate and SHAP gives a comparable level of interpretability (this is
also what Woloszko's OECD Adaptive Trees benchmark does). DFM is a
reasonable thing to come back to if there's time left after Layers 2-3.

Using sklearn's HistGradientBoostingRegressor rather than lightgbm: the
lightgbm wheel available in this environment segfaults intermittently on
repeated .fit() calls (access violation inside its native Dataset
construction), reproducible across numpy 1.26 and 2.x. sklearn's GBM is
pure Cython, no separate native DLL, and every one of the 36 walk-forward
fits ran clean. Worth revisiting lightgbm on a Linux box if speed becomes
an issue later - this is a Windows-environment quirk, not a modeling
choice.
"""

import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from data import load_exports
from features import build_feature_matrix


def build_dataset(include_customs_flash=False):
    y = load_exports()
    X = build_feature_matrix(include_customs_flash=include_customs_flash)

    X = X.copy()
    X["target_lag1"] = y.shift(1)
    X["target_lag12"] = y.shift(12)

    df = pd.concat([y.rename("y"), X], axis=1).dropna()
    return df["y"], df.drop(columns="y")


def _default_params():
    return dict(
        max_depth=3,
        max_leaf_nodes=7,
        learning_rate=0.05,
        min_samples_leaf=5,
        max_iter=100,
    )


def make_gbm_forecaster(**params_override):
    params = _default_params()
    params.update(params_override)

    def fit_predict(y_train, X_train, x_pred):
        model = HistGradientBoostingRegressor(**params)
        model.fit(X_train.values, y_train.values)
        return model.predict(x_pred.values)[0]

    return fit_predict


def make_ridge_forecaster(alpha=10.0):
    """Simpler, more heavily-regularized alternative to the GBM. Included
    because the GBM overfits badly on ~90 training rows / 9 features - this
    isolates whether the alt-data itself has signal, independent of
    whether a flexible tree model is the right amount of complexity for
    a dataset this small. Directly foreshadows the project's feature-count
    x regularization x RMSE surface (Layer 3 capstone)."""

    def fit_predict(y_train, X_train, x_pred):
        scaler = StandardScaler().fit(X_train)
        Xt = scaler.transform(X_train)
        Xp = scaler.transform(x_pred)
        model = Ridge(alpha=alpha)
        model.fit(Xt, y_train.values)
        return model.predict(Xp)[0]

    return fit_predict


def fit_full_model(y, X, **params_override):
    """Fits on everything - used for the SHAP interpretability pass, not
    for backtest scoring."""
    params = _default_params()
    params.update(params_override)
    model = HistGradientBoostingRegressor(**params)
    model.fit(X.values, y.values)
    return model


if __name__ == "__main__":
    y, X = build_dataset()
    print(f"{len(y)} usable rows, {X.shape[1]} features, {y.index.min().date()} to {y.index.max().date()}")
    print(list(X.columns))
