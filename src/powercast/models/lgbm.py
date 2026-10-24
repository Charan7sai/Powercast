"""LightGBM point and quantile models, trained walk-forward with an expanding window.

One global model over all three zones (zone as a categorical feature, hour as a
feature): the zones are coupled, so pooling gives each zone's model more data.
"""
import json

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterSampler

from powercast.config import REPORTS, TRAIN_END, VAL_END, VAL_START
from powercast.transform.features import feature_columns

QUANTILES = [0.1, 0.5, 0.9]
BASE_PARAMS = dict(n_estimators=800, learning_rate=0.03, num_leaves=63, min_child_samples=50,
                   subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=1.0, verbose=-1)


def _xy(df: pd.DataFrame):
    X = df[feature_columns(df)].copy()
    X["zone"] = df["zone"].astype("category")
    return X, df["price"]


def make_model(params: dict, quantile: float | None = None) -> lgb.LGBMRegressor:
    p = dict(params)
    if quantile is None:
        p["objective"] = "regression"
    else:
        p.update(objective="quantile", alpha=quantile)
    return lgb.LGBMRegressor(**p)


def tune(gold: pd.DataFrame, n_iter: int = 12, seed: int = 0) -> dict:
    """Random search scored on the validation year only (train <= 2023, score 2024)."""
    train = gold[(gold["date"] <= TRAIN_END) & gold["price"].notna()]
    val = gold[(gold["date"] >= VAL_START) & (gold["date"] <= VAL_END)]
    Xtr, ytr = _xy(train)
    Xva, yva = _xy(val)
    space = dict(num_leaves=[31, 63, 127, 255], learning_rate=[0.02, 0.03, 0.05],
                 min_child_samples=[20, 50, 100, 200], colsample_bytree=[0.5, 0.7, 0.9],
                 reg_lambda=[0.0, 1.0, 5.0], n_estimators=[400, 800, 1200])
    results = []
    for cand in ParameterSampler(space, n_iter=n_iter, random_state=seed):
        params = {**BASE_PARAMS, **cand}
        m = make_model(params).fit(Xtr, ytr)
        mae = float(np.mean(np.abs(m.predict(Xva) - yva)))
        results.append({"params": cand, "val_mae": mae})
        print(f"  tune {cand} -> val MAE {mae:.2f}", flush=True)
    best = min(results, key=lambda r: r["val_mae"])
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "tuning.json").write_text(json.dumps({"best": best, "all": results}, indent=2))
    return {**BASE_PARAMS, **best["params"]}


def walk_forward(gold: pd.DataFrame, start: str, end: str | None = None, params: dict | None = None,
                 retrain_days: int = 28, train_start: str | None = None) -> pd.DataFrame:
    """Predict every date in [start, end] with models trained only on earlier dates.

    Retrains every `retrain_days` days. Returns rows with zone/date/hour/price plus
    columns `lgbm`, `q10`, `q50`, `q90`.
    """
    params = params or BASE_PARAMS
    data = gold if train_start is None else gold[gold["date"] >= train_start]
    dates = data.loc[data["date"] >= start, "date"].drop_duplicates().sort_values()
    if end is not None:
        dates = dates[dates <= end]
    out = []
    for i in range(0, len(dates), retrain_days):
        block = dates.iloc[i:i + retrain_days]
        # Prices for D-1 are known when forecasting D, so train on everything before the block.
        train = data[(data["date"] < block.iloc[0]) & data["price"].notna()]
        test = data[data["date"].isin(block)].copy()
        Xtr, ytr = _xy(train)
        Xte, _ = _xy(test)
        test["lgbm"] = make_model(params).fit(Xtr, ytr).predict(Xte)
        for q in QUANTILES:
            test[f"q{int(q * 100)}"] = make_model(params, q).fit(Xtr, ytr).predict(Xte)
        print(f"  lgbm block from {block.iloc[0].date()} ({len(train)} train rows)", flush=True)
        out.append(test[["zone", "date", "hour", "price", "lgbm", "q10", "q50", "q90"]])
    res = pd.concat(out, ignore_index=True)
    # Quantile models are fitted separately and can cross; sort to restore monotonicity.
    res[["q10", "q50", "q90"]] = np.sort(res[["q10", "q50", "q90"]].to_numpy(), axis=1)
    return res


def fit_final(gold: pd.DataFrame, params: dict) -> dict:
    """Models trained on all known prices, used for the live next-day forecast."""
    train = gold[gold["price"].notna()]
    X, y = _xy(train)
    models = {"lgbm": make_model(params).fit(X, y)}
    for q in QUANTILES:
        models[f"q{int(q * 100)}"] = make_model(params, q).fit(X, y)
    return models


def predict(models: dict, rows: pd.DataFrame) -> pd.DataFrame:
    X, _ = _xy(rows)
    out = rows[["zone", "date", "hour"]].copy()
    for name, m in models.items():
        out[name] = m.predict(X)
    out[["q10", "q50", "q90"]] = np.sort(out[["q10", "q50", "q90"]].to_numpy(), axis=1)
    return out
