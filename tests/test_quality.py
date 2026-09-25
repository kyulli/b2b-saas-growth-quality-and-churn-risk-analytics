import json

import pandas as pd


def test_every_planted_defect_is_detected_exactly(panel) -> None:
    s = pd.read_csv(panel / "quality" / "rule_summary.csv")
    assert len(s) == 8
    assert (s["detected"] == s["injected"]).all(), s


def test_cleaning_restores_identity_and_unique_keys(panel) -> None:
    m = json.loads((panel / "quality" / "cleaning_manifest.json").read_text())
    assert m["post_clean_identity_violations"] == 0
    assert m["post_clean_duplicate_keys"] == 0
    assert m["rows_quarantined_unresolved"] == 0
    subs = pd.read_csv(panel / "clean" / "subscriptions_clean.csv")
    assert (subs["mrr_amount"] >= 0).all() and subs["discount_pct"].between(0, 0.9).all()
    con = pd.read_csv(panel / "clean" / "contracts_clean.csv")
    assert (pd.to_datetime(con["end_date"]) >= pd.to_datetime(con["start_date"])).all()


def test_no_orphans_after_cleaning(panel) -> None:
    ids = set(pd.read_csv(panel / "clean" / "customers_clean.csv")["customer_id"])
    for t in ["subscriptions", "usage", "payments", "events", "contracts"]:
        assert pd.read_csv(panel / "clean" / f"{t}_clean.csv")["customer_id"].isin(ids).all(), t
