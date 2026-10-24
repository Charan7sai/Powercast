# %% [markdown]
# # 01 - Exploratory data analysis
# Seasonality, price spikes, negative prices and the weather-price relationship.
# Runs as a script (`python notebooks/01_eda.py`) or cell by cell in VS Code / Jupyter.

# %%
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from powercast.config import REPORTS, SILVER, ZONES
from powercast.transform.silver import to_day_hour

FIG = REPORTS / "figures"
FIG.mkdir(parents=True, exist_ok=True)
prices = {z: pd.read_parquet(SILVER / f"prices_{z}.parquet")["price"] for z in ZONES}
weather = {z: pd.read_parquet(SILVER / f"weather_{z}.parquet") for z in ZONES}

# %% Daily mean price history: the 2022 crisis regime stands out
fig, ax = plt.subplots(figsize=(11, 4))
for z in ZONES:
    prices[z].resample("D").mean().rolling(7).mean().plot(ax=ax, label=z, lw=1)
ax.set(ylabel="EUR/MWh (7-day mean)", title="Day-ahead prices, 2019 to today")
ax.legend()
fig.tight_layout()
fig.savefig(FIG / "eda_price_history.png", dpi=130)

# %% Intraday profile by year: the solar "duck curve" deepens every year
fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), sharey=True)
for ax, z in zip(axes, ZONES):
    w = to_day_hour(prices[z])
    for year, g in w.groupby(w.index.year):
        if year >= 2019:
            ax.plot(g.mean().to_numpy(), label=str(year), lw=1.2)
    ax.set(title=z, xlabel="hour (market time)")
axes[0].set_ylabel("mean EUR/MWh")
axes[-1].legend(fontsize=7)
fig.tight_layout()
fig.savefig(FIG / "eda_intraday_profile.png", dpi=130)

# %% Share of negative-price hours by month x hour (DE-LU)
w = to_day_hour(prices["DE-LU"])
neg = (w < 0).groupby(w.index.month).mean() * 100
fig, ax = plt.subplots(figsize=(9, 3.6))
im = ax.imshow(neg.to_numpy(), aspect="auto", cmap="Blues")
ax.set(xlabel="hour", ylabel="month", title="DE-LU: % of hours with negative prices")
ax.set_yticks(range(12), neg.index)
fig.colorbar(im, ax=ax)
fig.tight_layout()
fig.savefig(FIG / "eda_negative_heatmap.png", dpi=130)

# %% Wind forecast vs price (2024+): more wind, lower price
fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
for ax, z in zip(axes, ZONES):
    df = pd.concat([prices[z], weather[z]["wind_speed_100m"]], axis=1).dropna()
    df = df[df.index >= "2024-01-01"].sample(5000, random_state=0)
    ax.scatter(df["wind_speed_100m"], df["price"], s=2, alpha=0.3)
    ax.set(title=z, xlabel="forecast wind speed at 100 m (km/h)", ylim=(-150, 400))
axes[0].set_ylabel("EUR/MWh")
fig.tight_layout()
fig.savefig(FIG / "eda_wind_vs_price.png", dpi=130)

# %% Spike statistics
for z in ZONES:
    p = prices[z][prices[z].index >= "2023-01-01"]
    print(f"{z}: top-5% threshold {p.quantile(0.95):.0f} EUR/MWh, negative hours {100 * (p < 0).mean():.1f}%")
