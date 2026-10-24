"""Multivariate Diebold-Mariano test as used in epftoolbox (Lago et al., 2021).

Each delivery day is one observation: the loss differential is the difference in the
day's L1 error norm across all 24 hours, which avoids the strong intraday correlation.
One-sided: H1 is that forecast B is more accurate than forecast A.
"""
import numpy as np
import pandas as pd
from scipy import stats


def dm_test(y: pd.DataFrame, f_a: pd.DataFrame, f_b: pd.DataFrame) -> tuple[float, float]:
    """y, f_a, f_b: day x 24 frames aligned on index. Returns (DM statistic, p-value)."""
    e_a = (y - f_a).abs().sum(axis=1)
    e_b = (y - f_b).abs().sum(axis=1)
    d = (e_a - e_b).dropna().to_numpy()
    n = len(d)
    stat = d.mean() / (d.std(ddof=1) / np.sqrt(n))
    return float(stat), float(1 - stats.norm.cdf(stat))


def to_wide(df: pd.DataFrame, col: str) -> pd.DataFrame:
    return df.pivot_table(index="date", columns="hour", values=col)
