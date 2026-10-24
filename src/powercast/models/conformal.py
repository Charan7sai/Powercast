"""Conformalized quantile regression (Romano et al., 2019), applied online.

For delivery date D, the correction is computed from the conformity scores of the
previous `window` days of out-of-sample forecasts (all of them settled before D),
so the calibration never peeks at the future. Target coverage: 80% (P10-P90).
"""
import numpy as np
import pandas as pd


def conformalize(pred: pd.DataFrame, window: int = 90, coverage: float = 0.8,
                 lo: str = "q10", hi: str = "q90") -> pd.DataFrame:
    out = pred.copy()
    out["score"] = np.maximum(out[lo] - out["price"], out["price"] - out[hi])
    out["cqr_lo"] = np.nan
    out["cqr_hi"] = np.nan
    for zone, g in out.groupby("zone"):
        daily_scores = g.groupby("date")["score"].apply(np.asarray)
        dates = daily_scores.index
        for i, d in enumerate(dates):
            if i < 30:
                continue  # not enough history to calibrate yet
            hist = np.concatenate(daily_scores.iloc[max(0, i - window):i].to_numpy())
            hist = hist[~np.isnan(hist)]
            n = len(hist)
            level = min(1.0, np.ceil((n + 1) * coverage) / n)
            q = np.quantile(hist, level, method="higher")
            idx = g.index[g["date"] == d]
            out.loc[idx, "cqr_lo"] = out.loc[idx, lo] - q
            out.loc[idx, "cqr_hi"] = out.loc[idx, hi] + q
    return out.drop(columns="score")


def latest_correction(pred: pd.DataFrame, zone: str, window: int = 90, coverage: float = 0.8) -> float:
    """Correction for the next (unsettled) day, from the most recent settled forecasts."""
    g = pred[(pred["zone"] == zone) & pred["price"].notna()]
    recent = g[g["date"] > g["date"].max() - pd.Timedelta(days=window)]
    s = np.maximum(recent["q10"] - recent["price"], recent["price"] - recent["q90"]).dropna().to_numpy()
    n = len(s)
    return float(np.quantile(s, min(1.0, np.ceil((n + 1) * coverage) / n), method="higher"))
