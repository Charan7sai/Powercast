"""Daily battery arbitrage as a linear program.

Decision per hour h: charge c_h and discharge d_h (MW, = MWh for one hour).
    max   sum_h  sell_h * d_h - buy_h * c_h
    s.t.  s_h = sum_{k<=h} (eta_c * c_k - d_k / eta_d),   0 <= s_h <= E   (start empty)
          sum_h eta_c * c_h <= cycles * E                                (cycle limit)
          0 <= c_h, d_h <= P
Solved with HiGHS via scipy (in-process, fast enough for thousands of days).
"""
import numpy as np
from scipy.optimize import linprog

from powercast.config import BATTERY


def optimise_day(buy: np.ndarray, sell: np.ndarray | None = None, power_mw=None, energy_mwh=None,
                 round_trip_eff=None, max_cycles_per_day=None) -> tuple[np.ndarray, np.ndarray]:
    """Return (charge, discharge) schedules for 24 hours given buy/sell price vectors."""
    P = power_mw or BATTERY["power_mw"]
    E = energy_mwh or BATTERY["energy_mwh"]
    eff = round_trip_eff or BATTERY["round_trip_eff"]
    cycles = max_cycles_per_day or BATTERY["max_cycles_per_day"]
    sell = buy if sell is None else sell
    n = len(buy)
    eta = np.sqrt(eff)
    # x = [c_0..c_{n-1}, d_0..d_{n-1}]; linprog minimises, so negate profit.
    cost = np.concatenate([np.asarray(buy, float), -np.asarray(sell, float)])
    L = np.tril(np.ones((n, n)))
    soc = np.hstack([eta * L, -L / eta])          # state of charge after each hour
    A = np.vstack([soc, -soc, np.concatenate([np.full(n, eta), np.zeros(n)])[None, :]])
    b = np.concatenate([np.full(n, E), np.zeros(n), [cycles * E]])
    res = linprog(cost, A_ub=A, b_ub=b, bounds=[(0, P)] * (2 * n), method="highs")
    if not res.success:
        return np.zeros(n), np.zeros(n)
    x = res.x
    return x[:n], x[n:]


def settle(charge: np.ndarray, discharge: np.ndarray, actual: np.ndarray) -> float:
    """Profit in EUR when the schedule is settled at actual day-ahead prices."""
    return float(np.dot(actual, discharge - charge))
