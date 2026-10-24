"""LEAR: LASSO-Estimated AutoRegressive model (Lago, Marcjasz, De Schutter & Weron, 2021).

The standard open benchmark in day-ahead price forecasting, re-implemented here
following the epftoolbox recipe (epftoolbox itself pins an old TensorFlow):
  * one LASSO per delivery hour (24 models), alpha chosen by LARS + AIC;
  * inputs: all 24 prices of D-1, D-2, D-3, D-7, exogenous forecasts for D and D-1,
    and day-of-week dummies;
  * median/MAD normalisation followed by an asinh variance-stabilising transform;
  * an ensemble over several calibration windows (here 1 and 3 years).
"""
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso, LassoLarsIC

LAGS = [1, 2, 3, 7]


class _AsinhScaler:
    def fit(self, x: np.ndarray):
        self.med = np.median(x, axis=0)
        mad = np.median(np.abs(x - self.med), axis=0) / 0.6745
        self.mad = np.where(mad < 1e-6, 1.0, mad)
        return self

    def transform(self, x):
        return np.arcsinh((x - self.med) / self.mad)

    def inverse(self, z):
        return np.sinh(z) * self.mad + self.med


def build_design(price: pd.DataFrame, exog: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (X, Y): one row per delivery date."""
    blocks = {}
    for lag in LAGS:
        sh = price.shift(lag)
        sh.columns = [f"p_d{lag}_h{h}" for h in range(24)]
        blocks[f"p{lag}"] = sh
    for name, w in exog.items():
        w = w.reindex(price.index)
        for lag in [0, 1]:
            sh = w.shift(lag)
            sh.columns = [f"{name}_d{lag}_h{h}" for h in range(24)]
            blocks[f"{name}{lag}"] = sh
    X = pd.concat(blocks.values(), axis=1)
    dow = pd.get_dummies(price.index.dayofweek, prefix="dow").astype(float)
    dow.index = price.index
    X = pd.concat([X, dow], axis=1)
    return X, price


def _fit_predict_window(X: np.ndarray, Y: np.ndarray, x_new: np.ndarray) -> np.ndarray:
    price_cols = [i for i in range(X.shape[1]) if i < 24 * len(LAGS)]
    # Scale prices (target and lags) with one scaler fitted on the target, exog per column.
    ys = _AsinhScaler().fit(Y.reshape(-1, 1))
    xs = _AsinhScaler().fit(X)
    Xt = xs.transform(X)
    xt_new = xs.transform(x_new)
    Xt[:, price_cols] = ys.transform(X[:, price_cols].reshape(-1, 1)).reshape(len(X), -1)
    xt_new[:, price_cols] = ys.transform(x_new[:, price_cols].reshape(-1, 1)).reshape(len(x_new), -1)
    Yt = ys.transform(Y.reshape(-1, 1)).reshape(Y.shape)
    out = np.zeros((len(x_new), 24))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for h in range(24):
            alpha = LassoLarsIC(criterion="aic", max_iter=2500).fit(Xt, Yt[:, h]).alpha_
            model = Lasso(alpha=max(alpha, 1e-6), max_iter=5000).fit(Xt, Yt[:, h])
            out[:, h] = model.predict(xt_new)
    return ys.inverse(out.reshape(-1, 1)).reshape(out.shape)


def walk_forward(price: pd.DataFrame, exog: dict, test_dates: pd.DatetimeIndex,
                 windows=(364, 1092), recal_every: int = 7) -> pd.DataFrame:
    """Forecast each test date using only data before it; refit every `recal_every` days."""
    X, Y = build_design(price, exog)
    valid = X.notna().all(axis=1)
    preds = []
    test_dates = [d for d in test_dates if d in X.index and valid.get(d, False)]
    for i in range(0, len(test_dates), recal_every):
        block = test_dates[i:i + recal_every]
        cutoff = block[0]
        x_new = X.loc[block].to_numpy(float)
        ens = []
        for w in windows:
            hist = X.index[(X.index < cutoff) & valid & Y.notna().all(axis=1)][-w:]
            ens.append(_fit_predict_window(X.loc[hist].to_numpy(float), Y.loc[hist].to_numpy(float), x_new))
        preds.append(pd.DataFrame(np.mean(ens, axis=0), index=block, columns=range(24)))
    out = pd.concat(preds)
    out.index.name = "date"
    return out
