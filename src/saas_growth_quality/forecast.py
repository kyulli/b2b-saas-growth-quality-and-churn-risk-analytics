"""List-MRR scenario forecast from the SQL bridge, with a rolling-origin backtest.

Each month: closing = opening x (1 + expansion - contraction - churn rate) + new MRR.
Rates are shares of opening MRR; new MRR is in dollars. Base uses the trailing 12 months.
The fan resamples whole historical months (keeping the four components jointly). Its out-of-sample coverage is
measured in diligence.rolling_coverage; it reflects month-to-month noise, not a regime change.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SCENARIOS = {  # multipliers on (new, expansion, contraction, churn)
    "Run-off": (0.0, 1.0, 1.0, 1.0),
    "Downside": (0.8, 0.8, 1.3, 1.3),
    "Base": (1.0, 1.0, 1.0, 1.0),
    "Upside": (1.15, 1.15, 0.85, 0.85),
}


def bridge_rates(bridge: pd.DataFrame) -> pd.DataFrame:
    b = bridge.copy()
    r = pd.DataFrame({"month": pd.to_datetime(b["month"]), "new": b["new_list"],
                      "expansion": b["expansion_list"] / b["opening_list"],
                      "contraction": -b["contraction_list"] / b["opening_list"],
                      "churn": -b["churn_list"] / b["opening_list"], "closing": b["closing_list"]})
    return r.set_index("month")


def project(opening: float, new: np.ndarray, exp: np.ndarray, con: np.ndarray, churn: np.ndarray) -> np.ndarray:
    out, m = [], opening
    for n, e, c, k in zip(new, exp, con, churn):
        m = m * (1 + e - c - k) + n
        out.append(m)
    return np.array(out)


def scenario_forecast(rates: pd.DataFrame, horizon: int = 12, lookback: int = 12) -> pd.DataFrame:
    hist = rates.iloc[-lookback:]
    base = hist[["new", "expansion", "contraction", "churn"]].mean()
    idx = pd.date_range(rates.index[-1] + pd.DateOffset(months=1), periods=horizon, freq="MS")
    out = {}
    for name, (mn, me, mc, mk) in SCENARIOS.items():
        out[name] = project(rates["closing"].iloc[-1], np.full(horizon, base["new"] * mn), np.full(horizon, base["expansion"] * me),
                            np.full(horizon, base["contraction"] * mc), np.full(horizon, base["churn"] * mk))
    return pd.DataFrame(out, index=idx)


def fan_paths(pre: pd.DataFrame, horizon: int, pool: int = 24, lookback: int = 12, n: int = 2000, seed: int = 42) -> np.ndarray:
    """One method used everywhere (forward fan, underwriting band, coverage test): resample whole months of rates from the
    last `pool` months, with new-logo MRR rescaled to its trailing `lookback`-month level so the band is centred on the
    same assumptions as the base case."""
    rng = np.random.default_rng(seed)
    hist = pre.iloc[-pool:][["new", "expansion", "contraction", "churn"]].to_numpy().copy()
    hist[:, 0] = hist[:, 0] / hist[:, 0].mean() * pre["new"].iloc[-lookback:].mean()
    opening = pre["closing"].iloc[-1]
    return np.array([project(opening, *hist[rng.integers(0, len(hist), horizon)].T) for _ in range(n)])


def bootstrap_fan(rates: pd.DataFrame, horizon: int = 12, pool: int = 24, n: int = 4000, seed: int = 42) -> pd.DataFrame:
    q = np.quantile(fan_paths(rates, horizon, pool, n=n, seed=seed), [0.05, 0.25, 0.5, 0.75, 0.95], axis=0)
    idx = pd.date_range(rates.index[-1] + pd.DateOffset(months=1), periods=horizon, freq="MS")
    return pd.DataFrame(q.T, index=idx, columns=["p05", "p25", "p50", "p75", "p95"])


def rolling_origin_backtest(rates: pd.DataFrame, horizon: int = 12, lookback: int = 12) -> pd.DataFrame:
    """Refit the base case at every origin with enough history and a fully observed horizon."""
    rows = []
    for i in range(lookback, len(rates) - horizon + 1):
        fc = scenario_forecast(rates.iloc[:i], horizon, lookback)["Base"].to_numpy()
        actual = rates["closing"].iloc[i:i + horizon].to_numpy()
        for h in range(horizon):
            rows.append({"origin": rates.index[i - 1], "horizon": h + 1, "forecast": fc[h], "actual": actual[h],
                         "pct_error": fc[h] / actual[h] - 1})
    return pd.DataFrame(rows)
