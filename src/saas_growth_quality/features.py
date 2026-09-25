"""Customer-month features built only from the trailing three months (t-2..t).

Label: the account is active at t and first appears as churned in t+1..t+3.
Rows whose label window runs past the panel end are dropped (label not yet mature).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

CORE_FEATURES = ["usage_ratio_3m", "discount_pct_3m", "days_late_3m", "ticket_count_3m", "list_mrr_growth_3m"]
EXTENDED_FEATURES = CORE_FEATURES + [
    "usage_trend_3m", "ticket_trend_3m", "sev1_tickets_3m", "modules_adopted_3m", "failed_payments_3m",
    "nps_latest", "nps_missing", "renewal_in_3m", "monthly_contract", "promo_active", "tenure_months",
]
HORIZON = 3


def build_feature_table(clean_dir: Path, output_path: Path) -> pd.DataFrame:
    read = lambda n: pd.read_csv(clean_dir / f"{n}_clean.csv", parse_dates=["month"] if n != "customers" else None)
    cust, subs, usage, pay = read("customers"), read("subscriptions"), read("usage"), read("payments")

    f = (subs.merge(cust, on="customer_id")
             .merge(usage, on=["customer_id", "month"], how="left")
             .merge(pay[["customer_id", "month", "days_late", "payment_status"]], on=["customer_id", "month"], how="left")
             .sort_values(["customer_id", "month"]).reset_index(drop=True))
    f["failed_payment"] = (f["payment_status"] == "failed").astype(int)
    g = f.groupby("customer_id", sort=False)
    roll = lambda col: g[col].transform(lambda s: s.rolling(3, min_periods=3).mean())

    f["usage_ratio_3m"] = roll("usage_ratio")
    f["discount_pct_3m"] = roll("discount_pct")
    f["days_late_3m"] = roll("days_late")
    f["ticket_count_3m"] = roll("ticket_count")
    # List-price growth, so a discount change is not counted twice (discount is its own feature).
    f["list_mrr_growth_3m"] = f["list_mrr"] / g["list_mrr"].shift(2) - 1
    f["usage_trend_3m"] = f["usage_ratio"] - g["usage_ratio"].shift(2)
    f["ticket_trend_3m"] = f["ticket_count"] - g["ticket_count"].shift(2)
    f["sev1_tickets_3m"] = g["sev1_tickets"].transform(lambda s: s.rolling(3, min_periods=3).sum())
    f["modules_adopted_3m"] = roll("modules_adopted")
    f["failed_payments_3m"] = g["failed_payment"].transform(lambda s: s.rolling(3, min_periods=3).sum())
    f["nps_latest"] = g["nps_score"].ffill()
    f["nps_missing"] = f["nps_latest"].isna().astype(int)
    f["nps_latest"] = f["nps_latest"].fillna(7.0)                      # neutral prior; the flag carries missingness
    f["renewal_in_3m"] = ((f["months_to_renewal"] <= HORIZON) & (f["contract_term_months"] > 1)).astype(int)
    f["monthly_contract"] = (f["contract_term_months"] == 1).astype(int)   # every month is a renewal decision
    f["promo_active"] = f["promo_discount"].astype(bool).astype(int)
    f["tenure_months"] = ((f["month"] - pd.to_datetime(f["signup_date"])).dt.days / 30.44).round()

    churn_month = f.loc[f["status"] == "churned"].groupby("customer_id")["month"].min()
    f["churn_month"] = f["customer_id"].map(churn_month)
    horizon_end = f["month"] + pd.DateOffset(months=HORIZON)
    f["churn_next_3m"] = (f["churn_month"].notna() & (f["churn_month"] > f["month"]) & (f["churn_month"] <= horizon_end)).astype(int)

    mature = horizon_end <= f["month"].max()
    out = f.loc[(f["status"] == "active") & mature].dropna(subset=CORE_FEATURES).drop(columns=["churn_month"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)
    return out
