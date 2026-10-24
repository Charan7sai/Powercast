"""End-to-end pipeline. Run a stage with:  python -m powercast.pipeline <stage> [<stage> ...]

Stages: ingest, features, tune, backtest, evaluate, trade, forecast, stress, all, daily
"""
import json
import sys

import numpy as np
import pandas as pd

from powercast import config as C
from powercast.evaluate import metrics as M
from powercast.evaluate.dm_test import dm_test, to_wide
from powercast.models import baselines, conformal, lear, lgbm
from powercast.trading import strategies
from powercast.transform import features, silver

PRED = C.REPORTS / "predictions.parquet"
PARAMS = C.REPORTS / "best_params.json"


def _today() -> pd.Timestamp:
    return pd.Timestamp.now(tz=C.MARKET_TZ).normalize().tz_localize(None)


def stage_ingest():
    from powercast.ingest import entsoe_client
    from powercast.ingest.price_client import update_prices
    from powercast.ingest.weather_client import update_weather

    end = (_today() + pd.Timedelta(days=2)).strftime("%Y-%m-%d")
    for z in C.ZONES:
        p = update_prices(z, C.HISTORY_START, end)
        w = update_weather(z, C.HISTORY_START, end)
        f = entsoe_client.update_fundamentals(z, C.HISTORY_START, end)
        print(f"{z}: prices to {p.index.max()}, weather to {w.index.max()}"
              + ("" if f is None else f", fundamentals to {f.index.max()}"))


def stage_features():
    for z in C.ZONES:
        silver.build_silver(z)
    gold = features.build_gold()
    print(f"gold: {gold.shape}, {gold['date'].min().date()} .. {gold['date'].max().date()}")


def _gold() -> pd.DataFrame:
    return pd.read_parquet(C.GOLD / "features.parquet")


def _params() -> dict:
    return json.loads(PARAMS.read_text()) if PARAMS.exists() else lgbm.BASE_PARAMS


def stage_tune():
    best = lgbm.tune(_gold())
    C.REPORTS.mkdir(exist_ok=True)
    PARAMS.write_text(json.dumps(best, indent=2))
    print("best params:", best)


def stage_backtest():
    gold = _gold()
    pred = lgbm.walk_forward(gold, start=C.VAL_START, params=_params())
    pred = pred.merge(gold[["zone", "date", "hour", "p_d1", "p_d7", "dow", "is_holiday"]],
                      on=["zone", "date", "hour"])
    pred["naive"] = baselines.naive_epf(pred)
    pred["naive_d1"] = baselines.naive_d1(pred)
    pred["naive_d7"] = baselines.naive_d7(pred)
    pred = conformal.conformalize(pred)

    # LEAR on the test period only (it is the benchmark to beat, not used for calibration).
    wide = features.load_wide()
    test_dates = pd.DatetimeIndex(sorted(pred.loc[(pred["date"] >= C.TEST_START) & pred["price"].notna(), "date"].unique()))
    lear_rows = []
    for z in C.ZONES:
        exog = {k: wide[z][k] for k in ["temperature_2m", "wind_speed_100m", "shortwave_radiation", "residual_load_fc"]
                if k in wide[z]}
        lp = lear.walk_forward(wide[z]["price"], exog, test_dates)
        s = lp.stack().rename("lear").reset_index()
        s.columns = ["date", "hour", "lear"]
        s["zone"] = z
        lear_rows.append(s)
        print(f"  LEAR {z} done ({len(lp)} days)", flush=True)
    pred = pred.merge(pd.concat(lear_rows), on=["zone", "date", "hour"], how="left")
    pred.to_parquet(PRED)
    print(f"saved {PRED} ({len(pred)} rows)")


def _regimes(test: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for z, g in test.groupby("zone"):
        spike = g["price"] >= g["price"].quantile(0.95)
        masks = {"all hours": slice(None), "spike hours (top 5%)": spike, "negative prices": g["price"] < 0,
                 "weekends": g["dow"] >= 5, "holidays": g["is_holiday"] == 1}
        for name, m in masks.items():
            sub = g[m]
            if len(sub) == 0:
                continue
            rows.append({"zone": z, "regime": name, "hours": len(sub), "MAE naive": M.mae(sub["price"], sub["naive"]),
                         "MAE LEAR": M.mae(sub["price"], sub["lear"]), "MAE LGBM": M.mae(sub["price"], sub["lgbm"])})
    return pd.DataFrame(rows)


def stage_evaluate():
    pred = pd.read_parquet(PRED)
    test = pred[(pred["date"] >= C.TEST_START) & pred["price"].notna()].dropna(subset=["lear"])
    models = ["naive_d1", "naive_d7", "naive", "lear", "lgbm", "q50"]
    point = pd.concat({z: M.point_table(g, models) for z, g in test.groupby("zone")}
                      | {"ALL": M.point_table(test, models)}, names=["zone"])
    interval = pd.concat({z: M.interval_table(g) for z, g in test.groupby("zone")}
                         | {"ALL": M.interval_table(test)}, names=["zone"])
    dm = []
    for z, g in test.groupby("zone"):
        y = to_wide(g, "price")
        stat, p = dm_test(y, to_wide(g, "lear"), to_wide(g, "lgbm"))
        dm.append({"zone": z, "comparison": "LGBM better than LEAR", "DM stat": stat, "p-value": p})
    dm = pd.DataFrame(dm)
    regimes = _regimes(test)

    (C.REPORTS / "tables").mkdir(parents=True, exist_ok=True)
    point.to_csv(C.REPORTS / "tables" / "point.csv")
    interval.to_csv(C.REPORTS / "tables" / "interval.csv")
    dm.to_csv(C.REPORTS / "tables" / "dm_test.csv", index=False)
    regimes.to_csv(C.REPORTS / "tables" / "regimes.csv", index=False)
    with pd.option_context("display.float_format", "{:.3f}".format, "display.width", 160):
        print(point, "\n", interval, "\n", dm, "\n", regimes)
    _shap(test)


def _shap(test: pd.DataFrame):
    """SHAP summary for the final point model, on a sample of test rows."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import shap

    gold = _gold()
    train = gold[(gold["date"] < C.TEST_START) & gold["price"].notna()]
    X, y = lgbm._xy(train)
    model = lgbm.make_model(_params()).fit(X, y)
    sample = gold[gold["date"] >= C.TEST_START].dropna(subset=["price"]).sample(3000, random_state=0)
    Xs, _ = lgbm._xy(sample)
    sv = shap.TreeExplainer(model).shap_values(Xs)
    shap.summary_plot(sv, Xs, show=False, max_display=15)
    (C.REPORTS / "figures").mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(C.REPORTS / "figures" / "shap_summary.png", dpi=130)
    plt.close()
    imp = pd.Series(np.abs(sv).mean(axis=0), index=Xs.columns).sort_values(ascending=False)
    imp.to_csv(C.REPORTS / "tables" / "shap_importance.csv", header=["mean_abs_shap"])


def stage_trade():
    pred = pd.read_parquet(PRED)
    test = pred[(pred["date"] >= C.TEST_START) & pred["price"].notna()]
    bt = strategies.backtest(test)
    bt.to_parquet(C.REPORTS / "trading_daily.parquet")
    summ = strategies.summary(bt)
    summ.to_csv(C.REPORTS / "tables" / "trading.csv", index=False)
    print(summ.round(1).to_string(index=False))

    # Sensitivity: efficiency and battery duration, on the P50 strategy (all zones pooled).
    sens = []
    for eff in [0.80, 0.85, 0.90, 0.95]:
        for e in [1.0, 2.0, 4.0]:
            b = strategies.backtest(test, round_trip_eff=eff, energy_mwh=e)
            sens.append({"round_trip_eff": eff, "energy_mwh": e, "p50_profit": b["p50"].sum(),
                         "perfect_profit": b["perfect"].sum(), "capture_pct": 100 * b["p50"].sum() / b["perfect"].sum()})
    pd.DataFrame(sens).to_csv(C.REPORTS / "tables" / "trading_sensitivity.csv", index=False)
    print(pd.DataFrame(sens).round(1).to_string(index=False))


def stage_forecast():
    """Forecast the next delivery day that has no published price yet."""
    from powercast.ingest.weather_client import fetch_live, zone_weather

    # Refresh weather with the live forecast for the next days.
    for z in C.ZONES:
        path = C.RAW / f"weather_{z}.parquet"
        old = pd.read_parquet(path)
        live = zone_weather({n: fetch_live(la, lo) for n, la, lo in C.WEATHER_POINTS[z]})
        merged = pd.concat([old[old.index < live.index.min()], live])
        merged.to_parquet(path)
    stage_features()
    gold = _gold()
    target = gold.loc[gold["price"].isna(), "date"].min()
    rows = gold[gold["date"] == target]
    models = lgbm.fit_final(gold, _params())
    fc = lgbm.predict(models, rows)
    pred = pd.read_parquet(PRED)
    for z in C.ZONES:
        q = conformal.latest_correction(pred, z)
        m = fc["zone"] == z
        fc.loc[m, "cqr_lo"] = fc.loc[m, "q10"] - q
        fc.loc[m, "cqr_hi"] = fc.loc[m, "q90"] + q
    fc["issued_at"] = pd.Timestamp.now(tz="UTC")
    fc.to_parquet(C.REPORTS / "forecast_latest.parquet")
    print(f"forecast for {target.date()}:\n", fc.pivot_table(index="hour", columns="zone", values="q50").round(1))


def stage_stress():
    """2022 energy-crisis stress test: models trained on 2019-2021 only, walk-forward through 2022."""
    gold = _gold()
    pred = lgbm.walk_forward(gold, start="2022-01-01", end="2022-12-31", params=_params())
    pred = pred.merge(gold[["zone", "date", "hour", "p_d1", "p_d7", "dow"]], on=["zone", "date", "hour"])
    pred["naive"] = baselines.naive_epf(pred)
    rows = []
    for (z, month), g in pred.groupby(["zone", pred["date"].dt.to_period("M")]):
        rows.append({"zone": z, "month": str(month), "mean_price": g["price"].mean(),
                     "MAE naive": M.mae(g["price"], g["naive"]), "MAE LGBM": M.mae(g["price"], g["lgbm"]),
                     "rMAE": M.rmae(g["price"], g["lgbm"], g["naive"]),
                     "coverage P10-P90": M.coverage(g["price"], g["q10"], g["q90"])})
    out = pd.DataFrame(rows)
    out.to_csv(C.REPORTS / "tables" / "stress_2022.csv", index=False)
    print(out.round(2).to_string(index=False))


STAGES = {"ingest": stage_ingest, "features": stage_features, "tune": stage_tune, "backtest": stage_backtest,
          "evaluate": stage_evaluate, "trade": stage_trade, "forecast": stage_forecast, "stress": stage_stress}
GROUPS = {"all": ["ingest", "features", "tune", "backtest", "evaluate", "trade", "stress", "forecast"],
          "daily": ["ingest", "features", "forecast"]}

if __name__ == "__main__":
    C.REPORTS.mkdir(exist_ok=True)
    for arg in sys.argv[1:] or ["all"]:
        for name in GROUPS.get(arg, [arg]):
            print(f"=== {name} ===", flush=True)
            STAGES[name]()
