import numpy as np

from powercast.trading.battery_lp import optimise_day, settle


def _soc(c, d, eff=0.9):
    eta = np.sqrt(eff)
    return np.cumsum(eta * c - d / eta)


def test_buys_low_sells_high():
    prices = np.full(24, 50.0)
    prices[3] = 10.0
    prices[4] = 10.0
    prices[18] = 200.0
    prices[19] = 200.0
    c, d = optimise_day(prices)
    assert c[3] > 0.9 and c[4] > 0.9
    assert d[18] > 0.5 and d[19] > 0.5
    assert settle(c, d, prices) > 0


def test_respects_physics():
    rng = np.random.default_rng(1)
    for _ in range(20):
        p = rng.normal(80, 40, 24)
        c, d = optimise_day(p)
        soc = _soc(c, d)
        assert soc.min() >= -1e-6 and soc.max() <= 2.0 + 1e-6
        assert c.max() <= 1 + 1e-9 and d.max() <= 1 + 1e-9
        assert (np.sqrt(0.9) * c).sum() <= 2.0 + 1e-6  # one full cycle per day


def test_flat_prices_no_trade():
    c, d = optimise_day(np.full(24, 60.0))
    assert settle(c, d, np.full(24, 60.0)) == 0.0 and d.sum() < 1e-6


def test_risk_aware_skips_uncertain_day():
    p50 = np.full(24, 50.0)
    p50[18] = 70.0  # small spread...
    lo, hi = p50 - 30, p50 + 30  # ...dwarfed by uncertainty
    c, d = optimise_day(hi, lo)
    assert d.sum() < 1e-6
