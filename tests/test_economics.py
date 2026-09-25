import numpy as np
import pandas as pd

from saas_growth_quality.forecast import project, rolling_origin_backtest
from saas_growth_quality.intervention import break_even, pilot_size


def test_projection_identity() -> None:
    out = project(100.0, np.array([10.0]), np.array([0.05]), np.array([0.02]), np.array([0.03]))
    assert np.isclose(out[0], 100 * 1.0 + 10)


def test_perfect_targeting_beats_random() -> None:
    rng = np.random.default_rng(0)
    s = pd.DataFrame({"customer_id": np.tile(np.arange(500), 4), "month": np.repeat(range(4), 500), "churn_next_3m": (rng.random(2000) < 0.1).astype(int), "mrr_amount": 1000.0})
    s["perfect"] = s["churn_next_3m"] + rng.random(2000) * 0.1
    assert break_even(s, "perfect")["break_even_save_rate"] < break_even(s, "x", seed=1)["break_even_save_rate"]


def test_pilot_size_shrinks_with_effect() -> None:
    assert pilot_size(0.14, 0.3) < pilot_size(0.14, 0.2) < pilot_size(0.14, 0.1)
    assert 900 < pilot_size(0.14, 0.3) < 1000


def test_backtest_horizons() -> None:
    idx = pd.date_range("2023-01-01", periods=30, freq="MS")
    r = pd.DataFrame({"new": 10.0, "expansion": 0.02, "contraction": 0.01, "churn": 0.01, "closing": 100 * 1.0 ** np.arange(30) + 10 * np.arange(30)}, index=idx)
    bt = rolling_origin_backtest(r)
    assert bt["horizon"].max() == 12 and bt["origin"].nunique() == 30 - 12 - 12 + 1


def test_underwriting_uses_only_pre_close_months() -> None:
    from saas_growth_quality.diligence import underwrite
    months = pd.date_range("2022-02-01", periods=40, freq="MS")
    b = pd.DataFrame({"month": months.strftime("%Y-%m-%d"), "opening_list": 100.0, "new_list": 5.0, "expansion_list": 1.0,
                      "contraction_list": -0.5, "churn_list": -1.0})
    b["closing_list"] = b[["opening_list", "new_list", "expansion_list", "contraction_list", "churn_list"]].sum(axis=1)
    u1 = underwrite(b, close="2024-12-01", end="2025-05-01", n=200)
    b.loc[pd.to_datetime(b["month"]) > "2024-12-01", ["new_list", "churn_list"]] = [50.0, -20.0]   # change the future only
    u2 = underwrite(b, close="2024-12-01", end="2025-05-01", n=200)
    assert np.allclose(u1["base"], u2["base"]) and np.allclose(u1["p95"], u2["p95"])
