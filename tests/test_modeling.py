import numpy as np
import pandas as pd

from saas_growth_quality.modeling import CLOSE, RULE_WEIGHTS, forward_folds, rule_renewal, rule_score, split


def _frame() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    months = pd.date_range("2024-01-01", periods=24, freq="MS")
    n = 50
    f = pd.DataFrame({"customer_id": np.tile([f"C{i}" for i in range(n)], len(months)), "month": np.repeat(months, n)})
    for c in ["usage_ratio_3m", "discount_pct_3m", "days_late_3m", "ticket_count_3m", "list_mrr_growth_3m"]:
        f[c] = rng.random(len(f))
    f["churn_next_3m"] = (rng.random(len(f)) < 0.2).astype(int)
    f["contract_term_months"], f["months_to_renewal"] = 12, rng.integers(1, 13, len(f))
    return f


def test_split_is_pre_close_vs_post_close() -> None:
    train, test, info = split(_frame())
    assert train["month"].max() + pd.DateOffset(months=3) <= CLOSE
    assert test["month"].min() > CLOSE
    assert train["month"].max() + pd.DateOffset(months=3) < test["month"].min()


def test_forward_folds_never_train_on_immature_labels() -> None:
    train, _, _ = split(_frame())
    for fit, val in forward_folds(train):
        assert train.loc[fit, "month"].max() + pd.DateOffset(months=3) < train.loc[val, "month"].min()


def test_rule_is_bounded_monotone_and_weights_sum_to_one() -> None:
    f = _frame()
    s = rule_score(f)
    assert s.between(0, 1).all() and abs(sum(RULE_WEIGHTS.values()) - 1) < 1e-9
    worse = f.assign(usage_ratio_3m=f["usage_ratio_3m"] * 0.5)
    assert (rule_score(worse) >= s).all()
    assert (rule_renewal(f) <= s + 1e-12).all()
