import shutil

import numpy as np
import pandas as pd

from saas_growth_quality.features import CORE_FEATURES, EXTENDED_FEATURES, build_feature_table


def test_labels_are_mature_and_binary(panel) -> None:
    f = pd.read_csv(panel / "features.csv", parse_dates=["month"])
    subs = pd.read_csv(panel / "clean" / "subscriptions_clean.csv", parse_dates=["month"])
    assert (f["month"] + pd.DateOffset(months=3) <= subs["month"].max()).all()
    assert set(f["churn_next_3m"].unique()) <= {0, 1}
    assert f[EXTENDED_FEATURES].notna().all().all()


def test_label_matches_first_churn_month(panel) -> None:
    f = pd.read_csv(panel / "features.csv", parse_dates=["month"])
    subs = pd.read_csv(panel / "clean" / "subscriptions_clean.csv", parse_dates=["month"])
    first = subs[subs["status"] == "churned"].groupby("customer_id")["month"].min()
    cm = f["customer_id"].map(first)
    expected = (cm > f["month"]) & (cm <= f["month"] + pd.DateOffset(months=3))
    assert (expected.astype(int) == f["churn_next_3m"]).all()


def test_features_do_not_look_ahead(panel, tmp_path) -> None:
    """Scrambling everything after month t must leave features at t unchanged."""
    clean = tmp_path / "clean"
    shutil.copytree(panel / "clean", clean)
    cutoff = "2022-10-01"
    for t in ["usage", "payments", "subscriptions"]:
        df = pd.read_csv(clean / f"{t}_clean.csv")
        late = df["month"] > cutoff
        for col in df.select_dtypes("number").columns:
            df.loc[late, col] = df.loc[late, col].sample(frac=1, random_state=0).to_numpy()
        df.to_csv(clean / f"{t}_clean.csv", index=False)
    a = pd.read_csv(panel / "features.csv", parse_dates=["month"])
    b = build_feature_table(clean, tmp_path / "f.csv")
    key = ["customer_id", "month"]
    m = a[a["month"] <= pd.Timestamp(cutoff)].merge(b, on=key, suffixes=("", "_b"))
    cols = [c for c in CORE_FEATURES + ["usage_trend_3m", "sev1_tickets_3m", "failed_payments_3m"]]
    for c in cols:
        assert np.allclose(m[c], m[f"{c}_b"], equal_nan=True), c
