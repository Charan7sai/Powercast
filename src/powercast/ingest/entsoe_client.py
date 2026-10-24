"""Optional ENTSO-E client for day-ahead load and wind/solar *forecasts*.

These are the strongest fundamentals (residual load = load - wind - solar), but the
ENTSO-E API needs a personal token. Register at transparency.entsoe.eu, then email
transparency@entsoe.eu with subject "Restful API access". Once you have it:

    pip install entsoe-py
    set ENTSOE_API_KEY=<token>
    python -m powercast.pipeline ingest

The feature builder picks these columns up automatically when the files exist.
"""
import os

import pandas as pd

from powercast.config import RAW

ENTSOE_CODES = {"NL": "NL", "DE-LU": "DE_LU", "FR": "FR"}


def available() -> bool:
    return bool(os.environ.get("ENTSOE_API_KEY"))


def update_fundamentals(zone: str, start: str, end: str) -> pd.DataFrame | None:
    if not available():
        return None
    from entsoe import EntsoePandasClient

    client = EntsoePandasClient(api_key=os.environ["ENTSOE_API_KEY"])
    code = ENTSOE_CODES[zone]
    s, e = pd.Timestamp(start, tz="UTC"), pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=2)
    load = client.query_load_forecast(code, start=s, end=e)
    load = load.iloc[:, 0] if isinstance(load, pd.DataFrame) else load
    ws = client.query_wind_and_solar_forecast(code, start=s, end=e, psr_type=None)
    df = pd.DataFrame({"load_fc": load})
    df["solar_fc"] = ws.filter(like="Solar").sum(axis=1)
    df["wind_fc"] = ws.filter(like="Wind").sum(axis=1)
    df = df.tz_convert("UTC").resample("1h").mean()
    df["residual_load_fc"] = df["load_fc"] - df["wind_fc"] - df["solar_fc"]
    df.index.name = "ts_utc"
    df.to_parquet(RAW / f"fundamentals_{zone}.parquet")
    return df
