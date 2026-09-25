"""Data-quality ledger and cleaning.

Policy: repair a value only when an accounting identity reconstructs it exactly
(billed = list x (1 - discount)); otherwise quarantine the record. Every rule writes
row-level issues before any change, so detection can be scored against the defects
the generator planted.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

AS_OF_DATE = pd.Timestamp("2026-07-31")
TABLES = ["customers", "subscriptions", "usage", "payments", "events", "contracts"]
CHILD_TABLES = TABLES[1:]
RULES = {
    "duplicate_customer_id": "keep_first",
    "missing_industry": "label_unknown",
    "future_signup": "quarantine_customer",
    "negative_mrr": "repair_sign_from_identity",
    "duplicate_subscription_month": "drop_exact_duplicate",
    "discount_out_of_range": "repair_from_identity",
    "orphan_foreign_key": "quarantine_row",
    "contract_date_inversion": "swap_dates",
}


def read_tables(folder: Path, suffix: str = "") -> dict[str, pd.DataFrame]:
    return {name: pd.read_csv(folder / f"{name}{suffix}.csv") for name in TABLES}


def _issue(table: str, rule: str, ids) -> pd.DataFrame:
    return pd.DataFrame({"table": table, "rule": rule, "record_id": list(ids), "action": RULES[rule]})


def run_quality_checks(raw_dir: Path, quality_dir: Path) -> pd.DataFrame:
    """Write the row-level issue ledger and a rule summary (detected vs injected when known)."""
    quality_dir.mkdir(parents=True, exist_ok=True)
    t = read_tables(raw_dir)
    cust, subs, con = t["customers"], t["subscriptions"], t["contracts"]
    sub_key = subs["customer_id"] + ":" + subs["month"].astype(str)

    parts = [
        _issue("customers", "duplicate_customer_id", cust.loc[cust.duplicated("customer_id"), "customer_id"]),
        _issue("customers", "missing_industry", cust.loc[cust["industry"].isna(), "customer_id"]),
        _issue("customers", "future_signup", cust.loc[pd.to_datetime(cust["signup_date"]) > AS_OF_DATE, "customer_id"]),
        _issue("subscriptions", "negative_mrr", sub_key[subs["mrr_amount"] < 0]),
        _issue("subscriptions", "duplicate_subscription_month", sub_key[subs.duplicated(["customer_id", "month"])]),
        _issue("subscriptions", "discount_out_of_range", sub_key[~subs["discount_pct"].between(0, 0.9)]),
        _issue("contracts", "contract_date_inversion", con.loc[pd.to_datetime(con["end_date"]) < pd.to_datetime(con["start_date"]), "contract_id"]),
    ]
    valid = set(cust["customer_id"])
    for name in CHILD_TABLES:
        orphan = t[name].loc[~t[name]["customer_id"].isin(valid), "customer_id"].unique()
        parts.append(_issue(name, "orphan_foreign_key", orphan))
    issues = pd.concat(parts, ignore_index=True)
    issues.to_csv(quality_dir / "issues.csv", index=False)

    summary = issues.groupby("rule").size().reindex(list(RULES), fill_value=0).rename("detected").to_frame()
    summary["action"] = pd.Series(RULES)
    injected_path = raw_dir / "injected_defects.json"
    if injected_path.exists():
        summary["injected"] = pd.Series(json.loads(injected_path.read_text()))
    summary.rename_axis("rule").reset_index().to_csv(quality_dir / "rule_summary.csv", index=False)
    return issues


def clean_raw_data(raw_dir: Path, clean_dir: Path, quality_dir: Path) -> dict[str, int]:
    clean_dir.mkdir(parents=True, exist_ok=True)
    raw = read_tables(raw_dir)
    t = {k: v.copy() for k, v in raw.items()}

    cust = t["customers"].drop_duplicates("customer_id", keep="first")
    cust = cust.loc[pd.to_datetime(cust["signup_date"]) <= AS_OF_DATE].copy()
    cust["industry"] = cust["industry"].fillna("Unknown")
    cust["signup_date"] = pd.to_datetime(cust["signup_date"]).dt.date.astype(str)
    valid = set(cust["customer_id"])

    subs = t["subscriptions"].drop_duplicates(["customer_id", "month"], keep="first").copy()
    expected = subs["list_mrr"] * (1 - subs["discount_pct"].clip(0, 0.9))
    neg = subs["mrr_amount"] < 0
    sign_ok = neg & np.isclose(-subs["mrr_amount"], expected, atol=0.02)
    subs.loc[sign_ok, "mrr_amount"] = -subs.loc[sign_ok, "mrr_amount"]
    bad = ~subs["discount_pct"].between(0, 0.9) & (subs["list_mrr"] > 0)
    subs.loc[bad, "discount_pct"] = (1 - subs.loc[bad, "mrr_amount"] / subs.loc[bad, "list_mrr"]).round(4)
    unresolved = (subs["mrr_amount"] < 0) | ~subs["discount_pct"].between(0, 0.9)
    subs = subs.loc[~unresolved]

    con = t["contracts"].copy()
    inv = pd.to_datetime(con["end_date"]) < pd.to_datetime(con["start_date"])
    con.loc[inv, ["start_date", "end_date"]] = con.loc[inv, ["end_date", "start_date"]].to_numpy()

    cleaned = {"customers": cust, "subscriptions": subs, "usage": t["usage"], "payments": t["payments"], "events": t["events"], "contracts": con}
    for name in CHILD_TABLES:
        cleaned[name] = cleaned[name].loc[cleaned[name]["customer_id"].isin(valid)]
    for name, frame in cleaned.items():
        frame.to_csv(clean_dir / f"{name}_clean.csv", index=False)

    # Post-clean assertions: the identity must hold on every billed row after repair.
    s = cleaned["subscriptions"]
    identity_gap = (s["mrr_amount"] - s["list_mrr"] * (1 - s["discount_pct"])).abs()
    manifest = {
        "as_of_date": AS_OF_DATE.date().isoformat(),
        "raw_rows": {k: len(v) for k, v in raw.items()},
        "clean_rows": {k: len(v) for k, v in cleaned.items()},
        "rows_repaired": {"negative_mrr": int(sign_ok.sum()), "discount_out_of_range": int(bad.sum()), "contract_date_inversion": int(inv.sum())},
        "rows_quarantined_unresolved": int(unresolved.sum()),
        "post_clean_identity_violations": int((identity_gap > 0.05).sum()),
        "post_clean_duplicate_keys": int(s.duplicated(["customer_id", "month"]).sum()),
        "issues_recorded": int(len(pd.read_csv(quality_dir / "issues.csv"))),
    }
    (quality_dir / "cleaning_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest["clean_rows"]
