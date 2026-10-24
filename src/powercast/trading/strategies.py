"""Backtest of four battery strategies on the test period.

  perfect    - optimise on actual prices (upper bound, not achievable)
  naive      - optimise on the naive forecast
  p50        - optimise on the LightGBM median forecast
  risk_aware - optimise as if buying at P90 and selling at P10: the battery only
               trades when even a pessimistic reading of the forecast pays off
All schedules are settled at actual prices.
"""
import pandas as pd

from powercast.trading.battery_lp import optimise_day, settle


def backtest(pred: pd.DataFrame, **battery) -> pd.DataFrame:
    rows = []
    for (zone, date), g in pred.sort_values("hour").groupby(["zone", "date"]):
        if len(g) != 24 or g["price"].isna().any():
            continue
        actual = g["price"].to_numpy()
        plans = {
            "perfect": optimise_day(actual, **battery),
            "naive": optimise_day(g["naive"].to_numpy(), **battery),
            "p50": optimise_day(g["q50"].to_numpy(), **battery),
            "risk_aware": optimise_day(g["q90"].to_numpy(), g["q10"].to_numpy(), **battery),
        }
        row = {"zone": zone, "date": date}
        for name, (c, d) in plans.items():
            row[name] = settle(c, d, actual)
            row[f"{name}_mwh"] = float(d.sum())
        rows.append(row)
    return pd.DataFrame(rows)


STRATEGIES = ["perfect", "naive", "p50", "risk_aware"]


def summary(bt: pd.DataFrame) -> pd.DataFrame:
    out = []
    for zone, g in list(bt.groupby("zone")) + [("ALL", bt)]:
        for s in STRATEGIES:
            out.append({"zone": zone, "strategy": s, "profit_eur": g[s].sum(),
                        "capture_pct": 100 * g[s].sum() / g["perfect"].sum(),
                        "worst_day_eur": g[s].min(), "days_traded": int((g[f"{s}_mwh"] > 1e-6).sum()),
                        "days": len(g)})
    return pd.DataFrame(out)
