"""Archived weather *forecasts* from the Open-Meteo Historical Forecast API (free, no key).

Using archived forecasts rather than observed weather keeps the features honest: at
D-1 12:00 you only have a forecast of tomorrow's wind, sun and temperature.
For the live forecast the regular forecast API is used with the same variables.
"""
import time

import pandas as pd
import requests

from powercast.config import RAW, WEATHER_POINTS, WEATHER_VARS

HIST_URL = "https://historical-forecast-api.open-meteo.com/v1/forecast"
LIVE_URL = "https://api.open-meteo.com/v1/forecast"


def _get(url: str, params: dict) -> dict:
    err = None
    for attempt in range(5):
        try:
            r = requests.get(url, params=params, timeout=120)
            if r.status_code == 200:
                return r.json()
            err = f"HTTP {r.status_code}: {r.text[:300]}"
        except requests.RequestException as e:
            err = repr(e)
        print(f"  open-meteo attempt {attempt + 1} failed ({err})", flush=True)
        time.sleep(5 * 2 ** attempt)
    raise RuntimeError(f"Open-Meteo request failed after retries: {err}")


def _to_frame(js: dict) -> pd.DataFrame:
    h = js["hourly"]
    df = pd.DataFrame({v: h[v] for v in WEATHER_VARS}, index=pd.to_datetime(h["time"], utc=True))
    df.index.name = "ts_utc"
    return df


def fetch_point(lat: float, lon: float, start: str, end: str) -> pd.DataFrame:
    frames = []
    # Anchor on the start's year so an incremental start mid-year still yields a chunk.
    for y0 in pd.date_range(pd.Timestamp(start).to_period("Y").start_time, end, freq="YS"):
        y0 = max(y0, pd.Timestamp(start))
        y1 = min(y0 + pd.offsets.YearEnd(0), pd.Timestamp(end))
        js = _get(HIST_URL, {"latitude": lat, "longitude": lon, "hourly": ",".join(WEATHER_VARS),
                             "start_date": y0.strftime("%Y-%m-%d"), "end_date": y1.strftime("%Y-%m-%d"),
                             "timezone": "GMT"})
        frames.append(_to_frame(js))
    return pd.concat(frames)


def fetch_live(lat: float, lon: float, days: int = 3) -> pd.DataFrame:
    js = _get(LIVE_URL, {"latitude": lat, "longitude": lon, "hourly": ",".join(WEATHER_VARS),
                         "past_days": 2, "forecast_days": days, "timezone": "GMT"})
    return _to_frame(js)


def zone_weather(points_frames: dict) -> pd.DataFrame:
    """Average the points of one zone into zone-level weather columns."""
    return pd.concat(points_frames.values()).groupby(level=0).mean()


def update_weather(zone: str, start: str, end: str) -> pd.DataFrame:
    path = RAW / f"weather_{zone}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        old = pd.read_parquet(path)
        start = (old.index.max().tz_convert(None) - pd.Timedelta(days=3)).strftime("%Y-%m-%d")
    else:
        old = None
    pts = {name: fetch_point(lat, lon, start, end) for name, lat, lon in WEATHER_POINTS[zone]}
    new = zone_weather(pts)
    df = new if old is None else pd.concat([old[old.index < new.index.min()], new])
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df.to_parquet(path)
    return df
