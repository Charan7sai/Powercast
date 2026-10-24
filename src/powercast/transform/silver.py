"""Bronze -> silver: UTC hourly grid, dedupe, resample 15-min data, flag and fill short gaps.

Then convert to a market-time "day x 24 hours" layout, the standard convention in
electricity price forecasting: on the 23-hour spring DST day the missing hour is
filled with the previous hour, on the 25-hour autumn day the repeated hour is averaged.
"""
import pandas as pd

from powercast.config import MARKET_TZ, RAW, SILVER


def to_hourly(df: pd.DataFrame, max_gap_hours: int = 3) -> pd.DataFrame:
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df = df.resample("1h").mean()  # averages 15-minute products into the hourly price
    grid = pd.date_range(df.index.min(), df.index.max(), freq="1h", tz="UTC")
    df = df.reindex(grid)
    df["gap"] = df.isna().any(axis=1)
    filled = df.drop(columns="gap").interpolate(limit=max_gap_hours, limit_area="inside")
    filled["gap"] = df["gap"]
    filled.index.name = "ts_utc"
    return filled


def to_day_hour(series: pd.Series) -> pd.DataFrame:
    """UTC hourly series -> frame indexed by market date with columns 0..23."""
    local = series.tz_convert(MARKET_TZ)
    frame = pd.DataFrame({"v": local.values, "date": local.index.date, "hour": local.index.hour})
    wide = frame.groupby(["date", "hour"])["v"].mean().unstack("hour")
    wide = wide.reindex(columns=range(24)).ffill(axis=1).bfill(axis=1)
    wide.index = pd.to_datetime(wide.index)
    wide.index.name = "date"
    return wide


def build_silver(zone: str) -> dict[str, pd.DataFrame]:
    SILVER.mkdir(parents=True, exist_ok=True)
    out = {}
    for kind in ["prices", "weather", "fundamentals"]:
        path = RAW / f"{kind}_{zone}.parquet"
        if not path.exists():
            continue
        hourly = to_hourly(pd.read_parquet(path))
        hourly.to_parquet(SILVER / f"{kind}_{zone}.parquet")
        out[kind] = hourly
    return out
