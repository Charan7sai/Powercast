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
    for attempt in range(5):
        r = requests.get(url, params=params, timeout=120)
        if r.status_code == 200:
            return r.json()
        time.sleep(5 * 2 ** attempt)
    r.raise_for_status()


def _to_frame(js: dict) -> pd.DataFrame:
    h = js["hourly"]
    df = pd.DataFrame({v: h[v] for v in WEATHER_VARS}, index=pd.to_datetime(h["time"], utc=True))
    df.index.name = "ts_utc"
    return df


def fetch_point(lat: float, lon: float, start: str, end: str) -> pd.DataFrame:
    frames = []
    for y0 in pd.date_range(start, end, freq="YS"):
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
