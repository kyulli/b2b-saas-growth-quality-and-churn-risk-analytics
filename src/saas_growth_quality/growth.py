"""Revenue-quality metrics computed from the SQL marts (the marts are the source of truth)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

GROUPS = ["segment", "industry", "contract_term_months", "pricing_model", "channel"]


def load_marts(marts_dir: Path) -> dict[str, pd.DataFrame]:
    return {n: pd.read_csv(marts_dir / f"{n}.csv") for n in ["account_month", "mrr_bridge", "discount_dependence", "cohort_retention"]}


def bridge_check(bridge: pd.DataFrame) -> float:
    """Largest absolute gap in opening + new + expansion + contraction + churn = closing. Should be ~0."""
    rhs = bridge[["opening_list", "new_list", "expansion_list", "contraction_list", "churn_list"]].sum(axis=1)
    return float((rhs - bridge["closing_list"]).abs().max())


def retention_12m(am: pd.DataFrame, end: str, by: str | None = None, basis: str = "list_mrr", min_accounts: int = 30) -> pd.DataFrame:
    """Point-to-point 12-month GRR/NRR on the accounts active at end-12 months.
    GRR = sum(min(end, start)) / sum(start); NRR = sum(end) / sum(start). New logos are excluded by construction."""
    end = pd.Timestamp(end)
    start = (end - pd.DateOffset(months=12)).strftime("%Y-%m-%d")
    act = am[am["status"] == "active"]
    s = act[act["month"] == start].set_index("customer_id")
    e = act[act["month"] == end.strftime("%Y-%m-%d")].set_index("customer_id")[basis].reindex(s.index).fillna(0)
    d = pd.DataFrame({"start": s[basis], "end": e, "group": s[by] if by else "Portfolio"})
    out = d.groupby("group").apply(lambda x: pd.Series({
        "accounts": len(x), "start_mrr": x["start"].sum(),
        "grr": np.minimum(x["end"], x["start"]).sum() / x["start"].sum(),
        "nrr": x["end"].sum() / x["start"].sum(),
        "logo_churn": (x["end"] == 0).mean()}), include_groups=False)
    return out[out["accounts"] >= min_accounts].assign(period_end=end.date())


def discount_dependence(dd: pd.DataFrame, by: str | None = None, since: str | None = None) -> pd.DataFrame:
    """Share of gross new and expansion list MRR written at a discount above 15%."""
    d = dd if since is None else dd[dd["month"] >= since]
    keys = ([by] if by else []) + ["motion"]
    g = d.groupby(keys + ["deep_discount"])["gross_list_mrr"].sum().unstack("deep_discount", fill_value=0)
    return (g.get(1, 0) / g.sum(axis=1)).rename("deep_discount_share").unstack("motion") if by else (g.get(1, 0) / g.sum(axis=1)).rename("deep_discount_share")


def discounted_mrr_share(am: pd.DataFrame) -> pd.DataFrame:
    """Monthly: discount dollars / list MRR, and billed vs list MRR levels."""
    a = am[am["status"] == "active"]
    m = a.groupby("month").agg(list_mrr=("list_mrr", "sum"), billed_mrr=("billed_mrr", "sum"))
    m["discounted_mrr_share"] = 1 - m["billed_mrr"] / m["list_mrr"]
    m["list_yoy"] = m["list_mrr"].pct_change(12)
    m["billed_yoy"] = m["billed_mrr"].pct_change(12)
    return m
