import numpy as np
import pandas as pd


def mae(y, f):
    return float(np.nanmean(np.abs(np.asarray(y) - np.asarray(f))))


def rmse(y, f):
    return float(np.sqrt(np.nanmean((np.asarray(y) - np.asarray(f)) ** 2)))


def rmae(y, f, naive):
    """MAE relative to the naive benchmark (< 1 means better than naive)."""
    return mae(y, f) / mae(y, naive)


def pinball(y, q, alpha):
    d = np.asarray(y) - np.asarray(q)
    return float(np.nanmean(np.maximum(alpha * d, (alpha - 1) * d)))


def coverage(y, lo, hi):
    y = np.asarray(y)
    return float(np.nanmean((y >= np.asarray(lo)) & (y <= np.asarray(hi))))


def width(lo, hi):
    return float(np.nanmean(np.asarray(hi) - np.asarray(lo)))


def point_table(df: pd.DataFrame, models: list[str], naive: str = "naive") -> pd.DataFrame:
    rows = []
    for m in models:
        rows.append({"model": m, "MAE": mae(df["price"], df[m]), "RMSE": rmse(df["price"], df[m]),
                     "rMAE": rmae(df["price"], df[m], df[naive])})
    return pd.DataFrame(rows).set_index("model")


def interval_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    pin = np.mean([pinball(df["price"], df[f"q{a}"], a / 100) for a in (10, 50, 90)])
    rows.append({"interval": "quantile LGBM P10-P90", "coverage": coverage(df["price"], df["q10"], df["q90"]),
                 "width": width(df["q10"], df["q90"]), "avg pinball": pin})
    if "cqr_lo" in df:
        d = df.dropna(subset=["cqr_lo"])
        rows.append({"interval": "conformalized (CQR)", "coverage": coverage(d["price"], d["cqr_lo"], d["cqr_hi"]),
                     "width": width(d["cqr_lo"], d["cqr_hi"]), "avg pinball": np.nan})
    return pd.DataFrame(rows).set_index("interval")
