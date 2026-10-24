# %% [markdown]
# # 05 - Result figures for the README
# Fan chart of one test week, cumulative battery profit, and interval coverage.

# %%
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from powercast.config import REPORTS, TEST_START

FIG = REPORTS / "figures"
pred = pd.read_parquet(REPORTS / "predictions.parquet")
bt = pd.read_parquet(REPORTS / "trading_daily.parquet")

# %% One week of DE-LU forecasts with conformal 80% band
wk = pred[(pred["zone"] == "DE-LU") & (pred["date"] >= "2026-05-04") & (pred["date"] < "2026-05-11")].copy()
wk["ts"] = wk["date"] + pd.to_timedelta(wk["hour"], unit="h")
wk = wk.sort_values("ts")
fig, ax = plt.subplots(figsize=(11, 3.8))
ax.fill_between(wk["ts"], wk["cqr_lo"], wk["cqr_hi"], alpha=0.25, label="80% conformal band")
ax.plot(wk["ts"], wk["q50"], label="P50 forecast")
ax.plot(wk["ts"], wk["price"], "k:", label="actual")
ax.plot(wk["ts"], wk["lear"], lw=0.8, alpha=0.7, label="LEAR")
ax.set(ylabel="EUR/MWh", title="DE-LU, one test week (forecast issued D-1 before 12:00)")
ax.legend(ncol=4, fontsize=8)
fig.tight_layout()
fig.savefig(FIG / "fan_week.png", dpi=130)

# %% Cumulative profit, all zones pooled
daily = bt.groupby("date")[["perfect", "risk_aware", "p50", "naive"]].sum().cumsum() / 1000
fig, ax = plt.subplots(figsize=(9, 3.8))
for c in daily:
    ax.plot(daily.index, daily[c], label=c)
ax.set(ylabel="kEUR", title="Cumulative profit, three 1 MW / 2 MWh batteries (NL, DE-LU, FR)")
ax.legend()
fig.tight_layout()
fig.savefig(FIG / "cumulative_profit.png", dpi=130)

# %% Rolling coverage: raw quantiles vs conformal
t = pred[pred["date"] >= TEST_START].dropna(subset=["cqr_lo"]).copy()
t["raw"] = t["price"].between(t["q10"], t["q90"])
t["cqr"] = t["price"].between(t["cqr_lo"], t["cqr_hi"])
cov = t.groupby("date")[["raw", "cqr"]].mean().rolling(30).mean()
fig, ax = plt.subplots(figsize=(9, 3.4))
ax.plot(cov.index, cov["raw"], label="quantile LightGBM P10-P90")
ax.plot(cov.index, cov["cqr"], label="conformalized")
ax.axhline(0.8, color="k", ls="--", lw=0.8)
ax.set(ylabel="30-day coverage", ylim=(0.3, 1), title="Interval coverage on the test period (target 80%)")
ax.legend()
fig.tight_layout()
fig.savefig(FIG / "coverage.png", dpi=130)
