from pathlib import Path

import numpy as np
import pandas as pd

from saas_growth_quality.uncertainty import build_cluster_bootstrap_intervals


def test_paired_bootstrap_detects_a_better_score(tmp_path: Path) -> None:
    rng = np.random.default_rng(1)
    n, months = 300, 3
    y = np.repeat((rng.random(n) < 0.2).astype(int), months)
    signal = y + rng.normal(0, 1, n * months)
    df = pd.DataFrame({"customer_id": np.repeat([f"C{i}" for i in range(n)], months), "churn_next_3m": y,
                       "rule_score_score": rng.random(n * months), "logistic_score": 1 / (1 + np.exp(-signal))})
    df.to_csv(tmp_path / "s.csv", index=False)
    build_cluster_bootstrap_intervals(tmp_path / "s.csv", tmp_path / "u", n_bootstrap=100)
    p = pd.read_csv(tmp_path / "u" / "cluster_bootstrap_paired_difference.csv")
    auc = p[(p["model"] == "logistic") & (p["metric"] == "roc_auc")].iloc[0]
    assert auc["ci_lower_95"] > 0
