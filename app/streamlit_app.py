"""Streamlit dashboard: `streamlit run app/streamlit_app.py`"""
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from powercast.config import REPORTS, TEST_START, ZONES  # noqa: E402

st.set_page_config(page_title="powercast", layout="wide")
st.title("powercast: day-ahead power price forecasts")


@st.cache_data
def load(name):
    p = REPORTS / name
    if not p.exists():
        return None
    return pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p)


zone = st.sidebar.selectbox("Bidding zone", ZONES)
fc, pred, bt = load("forecast_latest.parquet"), load("predictions.parquet"), load("trading_daily.parquet")


def fan(df, title, actual=True):
    fig = go.Figure()
    x = df["ts"] if "ts" in df else df["hour"]
    hi, lo = ("cqr_hi", "cqr_lo") if df.get("cqr_hi") is not None and df["cqr_hi"].notna().all() else ("q90", "q10")
    fig.add_trace(go.Scatter(x=x, y=df[hi], line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=x, y=df[lo], fill="tonexty", line=dict(width=0), name="80% band",
                             fillcolor="rgba(70,130,180,0.25)"))
    fig.add_trace(go.Scatter(x=x, y=df["q50"], name="P50 forecast", line=dict(color="steelblue")))
    if actual and "price" in df and df["price"].notna().any():
        fig.add_trace(go.Scatter(x=x, y=df["price"], name="actual", line=dict(color="black", dash="dot")))
    fig.update_layout(title=title, yaxis_title="EUR/MWh", height=380, margin=dict(t=40, b=20))
    return fig


tab1, tab2, tab3, tab4 = st.tabs(["Next-day forecast", "Backtest", "Battery trading", "Drift"])

with tab1:
    if fc is None:
        st.info("Run `python -m powercast.pipeline forecast` to produce the live forecast.")
    else:
        f = fc[fc["zone"] == zone].sort_values("hour")
        st.plotly_chart(fan(f, f"{zone}, delivery {f['date'].iloc[0].date()} (issued {f['issued_at'].iloc[0]:%Y-%m-%d %H:%M} UTC)",
                            actual=False), use_container_width=True)
        st.dataframe(f[["hour", "q10", "q50", "q90", "cqr_lo", "cqr_hi"]].round(1), hide_index=True)

with tab2:
    tables = REPORTS / "tables"
    if (tables / "point.csv").exists():
        st.subheader("Test-period accuracy (walk-forward, " + TEST_START + " onward)")
        st.dataframe(pd.read_csv(tables / "point.csv").round(3), hide_index=True)
        st.dataframe(pd.read_csv(tables / "interval.csv").round(3), hide_index=True)
        st.dataframe(pd.read_csv(tables / "dm_test.csv").round(4), hide_index=True)
        st.subheader("Where the model wins and loses")
        st.dataframe(pd.read_csv(tables / "regimes.csv").query("zone == @zone").round(2), hide_index=True)
    if pred is not None:
        p = pred[(pred["zone"] == zone) & (pred["date"] >= TEST_START)]
        dates = sorted(p["date"].dt.date.unique())
        start = st.select_slider("Week starting", options=dates, value=dates[-8])
        wk = p[(p["date"].dt.date >= start) & (p["date"].dt.date < start + pd.Timedelta(days=7))].copy()
        wk["ts"] = wk["date"] + pd.to_timedelta(wk["hour"], unit="h")
        st.plotly_chart(fan(wk.sort_values("ts"), f"{zone}: one week of backtest forecasts"), use_container_width=True)

with tab3:
    if bt is None:
        st.info("Run the `trade` stage.")
    else:
        b = bt[bt["zone"] == zone].sort_values("date")
        fig = go.Figure()
        for s in ["perfect", "risk_aware", "p50", "naive"]:
            fig.add_trace(go.Scatter(x=b["date"], y=b[s].cumsum(), name=s))
        fig.update_layout(title=f"{zone}: cumulative profit, 1 MW / 2 MWh battery", yaxis_title="EUR", height=420)
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(pd.read_csv(REPORTS / "tables" / "trading.csv").query("zone == @zone").round(1), hide_index=True)

with tab4:
    if pred is not None:
        p = pred[(pred["zone"] == zone) & pred["price"].notna()].copy()
        p["abs_err"] = (p["price"] - p["lgbm"]).abs()
        daily = p.groupby("date")["abs_err"].mean()
        roll = daily.rolling(30).mean()
        test_mae = daily[daily.index >= TEST_START].mean()
        fig = go.Figure([go.Scatter(x=roll.index, y=roll, name="rolling 30-day MAE")])
        fig.add_hline(y=test_mae, line_dash="dash", annotation_text=f"test-period MAE {test_mae:.1f}")
        fig.update_layout(title=f"{zone}: forecast drift", yaxis_title="EUR/MWh", height=380)
        st.plotly_chart(fig, use_container_width=True)
        if roll.iloc[-1] > 1.5 * test_mae:
            st.warning("Rolling MAE is more than 1.5x the test-period MAE: consider retraining.")
