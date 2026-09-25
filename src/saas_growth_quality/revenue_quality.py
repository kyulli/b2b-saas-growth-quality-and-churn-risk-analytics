"""Quality-of-revenue views a financial analyst would build in diligence.

ARR = 12 x month-end MRR (billed unless stated). Recognized revenue = billed MRR summed over the months of
the period (monthly billing, ratable recognition). Cash is assigned to the day an invoice is collected:
invoice date (1st of the month) + one month of payment terms + days late; failed invoices are never collected.
The panel has no terms field, so one-month terms (net-30 equivalent) are an assumption; measuring terms in
calendar months keeps every quarter holding exactly three due dates. Every account is billed monthly,
so deferred revenue is zero by construction; in real diligence annual upfront billing adds a deferred-revenue
line between ARR and cash.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

AS_OF = "2026-07-01"
PAYMENT_TERMS_MONTHS = 1


def _q(month: pd.Series) -> pd.Series:
    return pd.to_datetime(month).dt.to_period("Q").astype(str)


def customer_arr(am: pd.DataFrame, month: str = AS_OF) -> pd.DataFrame:
    a = am[(am["month"] == month) & (am["status"] == "active")].copy()
    a["arr"] = a["billed_mrr"] * 12
    a["list_arr"] = a["list_mrr"] * 12
    cols = ["customer_id", "segment", "industry", "channel", "pricing_model", "contract_term_months", "discount_pct",
            "months_to_renewal", "arr", "list_arr"]
    return a[cols].sort_values("arr", ascending=False).reset_index(drop=True)


def concentration(arr: pd.Series) -> dict[str, float]:
    s = arr.sort_values(ascending=False)
    share = s / s.sum()
    return {"customers": len(s), "top1": share.iloc[0], "top10": share.iloc[:10].sum(), "top20": share.iloc[:20].sum(),
            "top50": share.iloc[:50].sum(), "top10pct_of_customers": share.iloc[: int(np.ceil(0.1 * len(s)))].sum(),
            "hhi": float((share ** 2).sum() * 10_000)}


def concentration_trend(am: pd.DataFrame) -> pd.DataFrame:
    a = am[am["status"] == "active"]
    rows = []
    for m, g in a.groupby("month"):
        if pd.Timestamp(m).month in (3, 6, 9, 12) or m == a["month"].max():
            rows.append({"month": m, **concentration(g["billed_mrr"])})
    return pd.DataFrame(rows)


def quarterly_arr_bridge(bridge: pd.DataFrame) -> pd.DataFrame:
    """Monthly list-MRR bridge rolled up to quarters and annualized (x12). Only complete quarters."""
    b = bridge.copy()
    b["quarter"] = _q(b["month"])
    full = b.groupby("quarter")["month"].transform("size") == 3
    b = b[full]
    q = b.groupby("quarter").agg(opening=("opening_list", "first"), new=("new_list", "sum"), expansion=("expansion_list", "sum"),
                                 contraction=("contraction_list", "sum"), churn=("churn_list", "sum"), closing=("closing_list", "last"))
    return q * 12


def revenue_to_cash(subs: pd.DataFrame, pays: pd.DataFrame) -> pd.DataFrame:
    """Quarterly recognized revenue, invoicing, collections, receivables and failed collections."""
    s = subs[subs["status"] == "active"].assign(quarter=lambda d: _q(d["month"]))
    rev = s.groupby("quarter").agg(revenue=("mrr_amount", "sum"), ending_arr=("mrr_amount", lambda x: np.nan))
    end_month = s.groupby("quarter")["month"].max()
    rev["ending_arr"] = [s.loc[s["month"] == m, "mrr_amount"].sum() * 12 for m in end_month]
    p = pays.copy()
    p["invoice_date"] = pd.to_datetime(p["month"])
    p["collected_date"] = p["invoice_date"] + pd.DateOffset(months=PAYMENT_TERMS_MONTHS) + pd.to_timedelta(p["days_late"], unit="D")
    p["inv_q"], p["col_q"] = _q(p["invoice_date"]), _q(p["collected_date"])
    ok = p["payment_status"] != "failed"
    out = rev.join(p.groupby("inv_q")["invoice_amount"].sum().rename("invoiced"))
    out = out.join(p[ok].groupby("col_q")["invoice_amount"].sum().rename("cash_collected"))
    out = out.join(p[~ok].groupby("inv_q")["invoice_amount"].sum().rename("failed_uncollected"))
    ends = pd.PeriodIndex(out.index, freq="Q").end_time.normalize()
    out["ending_ar"] = [p.loc[ok & (p["invoice_date"] <= e) & (p["collected_date"] > e), "invoice_amount"].sum() for e in ends]
    full = s.groupby("quarter")["month"].nunique() == 3
    return out[full.reindex(out.index).fillna(False)].fillna(0)


def renewal_churn_rates(contracts: pd.DataFrame, customers: pd.DataFrame, since: str = "2025-08-01") -> pd.DataFrame:
    """Share of committed contracts reaching term in the last 12 months that did not renew, by segment."""
    c = contracts.merge(customers[["customer_id", "segment"]], on="customer_id")
    end = pd.to_datetime(c["end_date"])
    c = c[(c["term_months"] >= 12) & c["outcome"].isin(["renewed", "churned_at_renewal"]) & (end >= since) & (end < "2026-07-01")]
    return c.groupby("segment").agg(contracts=("contract_id", "size"),
                                    non_renewal_rate=("outcome", lambda o: (o == "churned_at_renewal").mean()))


def renewal_schedule(arr_table: pd.DataFrame, as_of: str = AS_OF) -> pd.DataFrame:
    """ARR by segment and the quarter its committed contract comes up for renewal; monthly contracts separate."""
    a = arr_table.copy()
    committed = a["contract_term_months"] > 1
    a["renewal_month"] = pd.Timestamp(as_of) + pd.to_timedelta(0, "D")
    a.loc[committed, "renewal_month"] = [pd.Timestamp(as_of) + pd.DateOffset(months=int(k)) for k in a.loc[committed, "months_to_renewal"]]
    a["bucket"] = np.where(committed, pd.to_datetime(a["renewal_month"]).dt.to_period("Q").astype(str), "Monthly (any month)")
    horizon = pd.Timestamp(as_of) + pd.DateOffset(months=12)
    a.loc[committed & (pd.to_datetime(a["renewal_month"]) > horizon), "bucket"] = "Beyond 12 months"
    return a.pivot_table(index="bucket", columns="segment", values="arr", aggfunc="sum", fill_value=0)


def build(root: Path) -> dict[str, pd.DataFrame | dict]:
    clean, marts = root / "data" / "clean", root / "outputs" / "marts"
    am = pd.read_csv(marts / "account_month.csv")
    arr = customer_arr(am)
    return {
        "customer_arr": arr,
        "concentration": concentration(arr["arr"]),
        "concentration_trend": concentration_trend(am),
        "arr_bridge_q": quarterly_arr_bridge(pd.read_csv(marts / "mrr_bridge.csv")),
        "revenue_to_cash": revenue_to_cash(pd.read_csv(clean / "subscriptions_clean.csv"), pd.read_csv(clean / "payments_clean.csv")),
        "renewal_rates": renewal_churn_rates(pd.read_csv(clean / "contracts_clean.csv"), pd.read_csv(clean / "customers_clean.csv")),
        "renewal_schedule": renewal_schedule(arr),
    }
