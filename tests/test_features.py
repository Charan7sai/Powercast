import numpy as np
import pandas as pd

from powercast.config import ZONES
from powercast.transform.features import build_zone_features
from powercast.transform.silver import to_day_hour


def _synthetic_wide(days=30, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-03-01", periods=days, freq="D")
    wide = {}
    for z in ZONES:
        d = {"price": pd.DataFrame(rng.normal(80, 20, (days, 24)), index=dates, columns=range(24))}
        for v in ["temperature_2m", "wind_speed_100m", "shortwave_radiation", "cloud_cover"]:
            d[v] = pd.DataFrame(rng.normal(0, 1, (days + 1, 24)),
                                index=pd.date_range(dates[0], periods=days + 1), columns=range(24))
        wide[z] = d
    return wide


def test_no_same_day_price_leakage():
    """Changing day D's prices must not change any feature of day D."""
    wide = _synthetic_wide()
    base = build_zone_features("NL", wide)
    target = base["date"].iloc[len(base) // 2]
    for z in ZONES:
        wide[z]["price"].loc[target] += 1000.0
    bumped = build_zone_features("NL", wide)
    cols = [c for c in base.columns if c not in ("zone", "date", "price")]
    a = base.loc[base["date"] == target, cols].reset_index(drop=True)
    b = bumped.loc[bumped["date"] == target, cols].reset_index(drop=True)
    pd.testing.assert_frame_equal(a, b)
    # ...but the next day's lag features must see it.
    nxt = target + pd.Timedelta(days=1)
    assert (bumped.loc[bumped["date"] == nxt, "p_d1"].to_numpy()
            - base.loc[base["date"] == nxt, "p_d1"].to_numpy()).min() > 999


def test_lags_point_to_right_day():
    wide = _synthetic_wide()
    f = build_zone_features("NL", wide)
    row = f.iloc[100]
    d, h = row["date"], row["hour"]
    p = wide["NL"]["price"]
    assert row["p_d1"] == p.loc[d - pd.Timedelta(days=1), h]
    assert row["p_d7"] == p.loc[d - pd.Timedelta(days=7), h]


def test_next_day_row_exists_without_target():
    f = build_zone_features("NL", _synthetic_wide())
    last = f[f["date"] == f["date"].max()]
    assert len(last) == 24 and last["price"].isna().all() and last["p_d1"].notna().all()


def test_dst_days_have_24_hours():
    # 2024-03-31 has 23 local hours, 2024-10-27 has 25.
    idx = pd.date_range("2024-03-30", "2024-11-01", freq="1h", tz="UTC")
    s = pd.Series(np.arange(len(idx), dtype=float), index=idx)
    w = to_day_hour(s)
    assert w.shape[1] == 24 and not w.isna().any().any()
    assert w.loc["2024-03-31", 2] == w.loc["2024-03-31", 1]  # missing hour filled from previous
