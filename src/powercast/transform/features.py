"""Gold feature table: one row per zone x delivery date x hour, using only information
available at the day-ahead gate closure (D-1 12:00 market time).

Leakage rules enforced here:
  * price features are lagged by at least one full day (D-1 prices are known on D-1);
  * weather and fundamentals for day D are *forecasts* issued before D;
  * nothing from day D's realised prices, load or weather is used.
"""
import holidays
import numpy as np
import pandas as pd

from powercast.config import GOLD, HOLIDAY_COUNTRY, MARKET_TZ, SILVER, ZONES
from powercast.transform.silver import to_day_hour

PRICE_LAGS = [1, 2, 3, 7]


def load_wide() -> dict:
    """{zone: {"price": wide, "<var>": wide, ...}} in day x hour layout."""
    out = {}
    for z in ZONES:
        d = {"price": to_day_hour(pd.read_parquet(SILVER / f"prices_{z}.parquet")["price"])}
        w = pd.read_parquet(SILVER / f"weather_{z}.parquet")
        for col in w.columns.drop("gap"):
            d[col] = to_day_hour(w[col])
        fpath = SILVER / f"fundamentals_{z}.parquet"
        if fpath.exists():
            f = pd.read_parquet(fpath)
            for col in f.columns.drop("gap"):
                d[col] = to_day_hour(f[col])
        out[z] = d
    return out


def _stack(wide: pd.DataFrame, name: str) -> pd.Series:
    s = wide.stack()
    s.index.names = ["date", "hour"]
    return s.rename(name)


def dst_transition_dates(years) -> set:
    idx = pd.date_range(f"{min(years)}-01-01", f"{max(years) + 1}-01-01", freq="1h", tz=MARKET_TZ)
    hours_per_day = pd.Series(1, index=idx).groupby(idx.date).sum()
    return set(pd.to_datetime(hours_per_day[hours_per_day != 24].index))


def build_zone_features(zone: str, wide: dict, extra_days: int = 1) -> pd.DataFrame:
    price = wide[zone]["price"]
    # Extend the calendar so the next delivery day (target unknown) gets a feature row.
    dates = pd.date_range(price.index.min(), price.index.max() + pd.Timedelta(days=extra_days), freq="D")
    price = price.reindex(dates)

    cols = [_stack(price, "price")]
    for lag in PRICE_LAGS:
        cols.append(_stack(price.shift(lag), f"p_d{lag}"))
    d1 = price.shift(1)
    for stat in ["min", "max", "mean", "std"]:
        cols.append(_stack(pd.DataFrame({h: getattr(d1, stat)(axis=1) for h in range(24)}), f"p_d1_{stat}"))
    cols.append(_stack(pd.DataFrame({h: d1[23] for h in range(24)}), "p_d1_last"))
    cols.append(_stack(pd.DataFrame({h: price.shift(7).mean(axis=1) for h in range(24)}), "p_d7_mean"))

    # Cross-zone columns use fixed names for every zone (own zone included), so the
    # pooled model sees the same feature meaning on every row.
    for other in ZONES:
        op = wide[other]["price"].reindex(dates).shift(1)
        tag = other.replace("-", "")
        cols.append(_stack(op, f"p_d1_{tag}"))
        cols.append(_stack(op.sub(d1), f"spread_d1_{tag}"))

    own_exog = [k for k in wide[zone] if k != "price"]
    for var in own_exog:
        w = wide[zone][var].reindex(dates)
        cols.append(_stack(w, var))
        cols.append(_stack(pd.DataFrame({h: w.mean(axis=1) for h in range(24)}), f"{var}_daymean"))
    # Cross-zone renewables drivers: neighbours' wind and sun move this zone's price too.
    for other in ZONES:
        tag = other.replace("-", "")
        for var in ["wind_speed_100m", "shortwave_radiation"]:
            cols.append(_stack(wide[other][var].reindex(dates), f"{var}_{tag}"))
        if "residual_load_fc" in wide[other]:
            cols.append(_stack(wide[other]["residual_load_fc"].reindex(dates), f"residual_load_fc_{tag}"))

    df = pd.concat(cols, axis=1).reset_index()
    df.insert(0, "zone", zone)

    hol = holidays.country_holidays(HOLIDAY_COUNTRY[zone], years=range(dates.year.min(), dates.year.max() + 1))
    dst = dst_transition_dates(range(dates.year.min(), dates.year.max() + 1))
    df["dow"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month
    df["doy"] = df["date"].dt.dayofyear
    df["is_weekend"] = (df["dow"] >= 5).astype(int)
    df["is_holiday"] = df["date"].dt.date.map(lambda d: d in hol).astype(int)
    df["is_dst_day"] = df["date"].isin(dst).astype(int)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    # Need at least a week of history for the lag features.
    return df[df["p_d7"].notna()].reset_index(drop=True)


def build_gold() -> pd.DataFrame:
    GOLD.mkdir(parents=True, exist_ok=True)
    wide = load_wide()
    gold = pd.concat([build_zone_features(z, wide) for z in ZONES], ignore_index=True)
    gold.to_parquet(GOLD / "features.parquet")
    return gold


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in ("zone", "date", "price")]
