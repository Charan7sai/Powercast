"""FastAPI service: `uvicorn powercast.api.main:app` then GET /forecast?zone=NL"""
from datetime import date as Date
from typing import Literal

import pandas as pd
from fastapi import FastAPI, HTTPException

from powercast.config import REPORTS

app = FastAPI(title="powercast", description="Day-ahead power price forecasts with P10/P50/P90 bands")


def _load(name: str) -> pd.DataFrame:
    path = REPORTS / name
    if not path.exists():
        raise HTTPException(503, f"{name} not built yet; run the pipeline first")
    return pd.read_parquet(path)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/forecast")
def forecast(zone: Literal["NL", "DE-LU", "FR"] = "NL", date: Date | None = None):
    """Hourly P10/P50/P90 (and conformal 80% band) for one zone and delivery date.

    Without `date` returns the latest live forecast; with a past date returns the
    walk-forward backtest forecast for that day plus the realised price.
    """
    if date is None:
        df = _load("forecast_latest.parquet")
    else:
        df = _load("predictions.parquet")
        df = df[df["date"] == pd.Timestamp(date)]
    df = df[df["zone"] == zone].sort_values("hour")
    if df.empty:
        raise HTTPException(404, "no forecast for that zone/date")
    cols = [c for c in ["hour", "q10", "q50", "q90", "cqr_lo", "cqr_hi", "price"] if c in df]
    return {"zone": zone, "delivery_date": str(df["date"].iloc[0].date()), "unit": "EUR/MWh",
            "hours": df[cols].round(2).astype(object).where(df[cols].notna(), None).to_dict("records")}
