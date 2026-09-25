import numpy as np
import pandas as pd

from saas_growth_quality.growth import bridge_check, retention_12m


def test_mrr_bridge_reconciles(panel) -> None:
    assert bridge_check(pd.read_csv(panel / "marts" / "mrr_bridge.csv")) < 0.01


def test_sql_churn_matches_pandas(panel) -> None:
    b = pd.read_csv(panel / "marts" / "mrr_bridge.csv")
    s = pd.read_csv(panel / "clean" / "subscriptions_clean.csv")
    w = s.assign(v=np.where(s["status"] == "active", s["list_mrr"], 0.0)).pivot(index="customer_id", columns="month", values="v")
    prev, cur = w.shift(1, axis=1).iloc[:, 1:].fillna(0), w.iloc[:, 1:].fillna(0)
    assert np.allclose(-((prev > 0) & (cur == 0)).mul(prev).sum().values, b["churn_list"].values, atol=0.01)


def test_retention_bounds(panel) -> None:
    am = pd.read_csv(panel / "marts" / "account_month.csv")
    r = retention_12m(am, "2023-06-01", min_accounts=1).iloc[0]
    assert 0 < r["grr"] <= 1 and r["nrr"] >= r["grr"]
