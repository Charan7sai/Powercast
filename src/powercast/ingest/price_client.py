"""Day-ahead prices from the Energy-Charts API (Fraunhofer ISE, CC BY 4.0, no key needed).

Energy-Charts republishes the EPEX/ENTSO-E day-ahead auction results per bidding zone.
Recent data arrives in 15-minute resolution; the silver layer resamples to hourly.
"""
import time

import pandas as pd
import requests

from powercast.config import RAW

URL = "https://api.energy-charts.info/price"


def fetch_prices(zone: str, start: str, end: str) -> pd.DataFrame:
    """Return a UTC-indexed frame with one `price` column (EUR/MWh) for [start, end]."""
    frames = []
    # One request per calendar year keeps us well under the API's rate limit.
    starts = pd.date_range(pd.Timestamp(start).to_period("Y").start_time, end, freq="YS")
    for chunk_start in starts:
        chunk_start = max(chunk_start, pd.Timestamp(start))
        chunk_end = min(chunk_start + pd.offsets.YearEnd(0), pd.Timestamp(end))
        params = {"bzn": zone, "start": chunk_start.strftime("%Y-%m-%d"),
                  "end": chunk_end.strftime("%Y-%m-%d")}
        r, err = None, None
        for attempt in range(6):
            try:
                r = requests.get(URL, params=params, timeout=120)
                if r.status_code == 200:
                    break
                err = f"HTTP {r.status_code}: {r.text[:300]}"
            except requests.RequestException as e:
                err = repr(e)
            print(f"  energy-charts {zone} attempt {attempt + 1} failed ({err})", flush=True)
            time.sleep(10 * 2 ** attempt)
        else:
            raise RuntimeError(f"Energy-Charts request failed after retries: {err}")
        time.sleep(2)
        js = r.json()
        if not js.get("unix_seconds"):
            continue
        frames.append(pd.DataFrame(
            {"price": js["price"]},
            index=pd.to_datetime(js["unix_seconds"], unit="s", utc=True),
        ))
    df = pd.concat(frames).sort_index()
    df.index.name = "ts_utc"
    return df[~df.index.duplicated(keep="last")]


def update_prices(zone: str, start: str, end: str) -> pd.DataFrame:
    path = RAW / f"prices_{zone}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        old = pd.read_parquet(path)
        # Refetch the last week to pick up late corrections.
        start = max(pd.Timestamp(start), old.index.max().tz_convert(None) - pd.Timedelta(days=7)).strftime("%Y-%m-%d")
        new = fetch_prices(zone, start, end)
        df = pd.concat([old[old.index < new.index.min()], new])
    else:
        df = fetch_prices(zone, start, end)
    df.to_parquet(path)
    return df
