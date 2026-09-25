import numpy as np
import pandas as pd

from saas_growth_quality.revenue_quality import concentration, quarterly_arr_bridge, revenue_to_cash


def test_invoices_equal_revenue_and_receivables_roll_forward(panel) -> None:
    subs = pd.read_csv(panel / "clean" / "subscriptions_clean.csv")
    pays = pd.read_csv(panel / "clean" / "payments_clean.csv")
    rc = revenue_to_cash(subs, pays)
    assert np.allclose(rc["invoiced"], rc["revenue"], atol=0.05)
    opening = rc["ending_ar"].shift(1).fillna(0)
    rolled = opening + rc["invoiced"] - rc["failed_uncollected"] - rc["cash_collected"]
    assert np.allclose(rolled, rc["ending_ar"], atol=0.05)


def test_quarterly_bridge_rolls_forward(panel) -> None:
    q = quarterly_arr_bridge(pd.read_csv(panel / "marts" / "mrr_bridge.csv"))
    assert np.allclose(q[["opening", "new", "expansion", "contraction", "churn"]].sum(axis=1), q["closing"], atol=0.05)
    assert np.allclose(q["opening"].iloc[1:].to_numpy(), q["closing"].iloc[:-1].to_numpy(), atol=0.05)


def test_concentration_bounds() -> None:
    c = concentration(pd.Series([50.0, 30.0, 20.0]))
    assert np.isclose(c["top1"], 0.5) and np.isclose(c["hhi"], (0.25 + 0.09 + 0.04) * 10_000)
    even = concentration(pd.Series(np.ones(100)))
    assert np.isclose(even["hhi"], 100)
