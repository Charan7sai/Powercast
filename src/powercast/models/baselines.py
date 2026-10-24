"""Naive benchmarks. Every model is reported relative to these."""
import numpy as np
import pandas as pd


def naive_d1(df: pd.DataFrame) -> np.ndarray:
    """Same hour yesterday."""
    return df["p_d1"].to_numpy()


def naive_d7(df: pd.DataFrame) -> np.ndarray:
    """Same hour last week."""
    return df["p_d7"].to_numpy()


def naive_epf(df: pd.DataFrame) -> np.ndarray:
    """The standard EPF naive model (Nowotarski & Weron): same hour last week for
    Monday, Saturday and Sunday, same hour yesterday for Tuesday to Friday.
    This is the reference for rMAE."""
    weekly = df["dow"].isin([0, 5, 6]).to_numpy()
    return np.where(weekly, df["p_d7"].to_numpy(), df["p_d1"].to_numpy())
