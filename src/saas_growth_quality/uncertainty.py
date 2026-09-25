"""Customer-cluster bootstrap for holdout metrics.

Each customer contributes up to six correlated holdout months, so rows are not independent.
Resampling customers keeps that dependence. Differences vs the rule score are bootstrapped
on the same draw (paired), which is the test that matters for a promotion decision.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

MODEL_NAMES = ["rule_score", "rule_calibrated", "rule_renewal", "logistic", "logistic_interact", "gbt"]
BASELINES = ["rule_score", "rule_renewal", "logistic"]


def _metrics(y: np.ndarray, s: np.ndarray, probability: bool) -> dict[str, float]:
    top = np.argsort(-s)[: max(1, int(np.ceil(0.10 * len(s))))]
    return {"roc_auc": roc_auc_score(y, s), "average_precision": average_precision_score(y, s),
            "brier_score": brier_score_loss(y, s) if probability else np.nan, "top10_churn_rate": y[top].mean()}


def build_cluster_bootstrap_intervals(scored_path: Path, output_dir: Path, n_bootstrap: int = 300, seed: int = 42) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    scored = pd.read_csv(scored_path)
    names = [n for n in MODEL_NAMES + ["oracle"] if f"{n}_score" in scored]
    codes, uniques = pd.factorize(scored["customer_id"])
    rows_of = pd.Series(np.arange(len(scored))).groupby(codes).apply(np.array).to_numpy()
    y = scored["churn_next_3m"].to_numpy()
    S = {n: scored[f"{n}_score"].to_numpy() for n in names}

    rng = np.random.default_rng(seed)
    observed = {n: _metrics(y, S[n], n != "rule_score") for n in names}
    draws = {n: [] for n in names}
    for _ in range(n_bootstrap):
        idx = np.concatenate(rows_of[rng.integers(0, len(uniques), len(uniques))])
        for n in names:
            draws[n].append(_metrics(y[idx], S[n][idx], n != "rule_score"))
    D = {n: pd.DataFrame(draws[n]) for n in names}

    marg = pd.DataFrame([{"model": n, "metric": m, "estimate": observed[n][m],
                          "ci_lower_95": D[n][m].quantile(0.025), "ci_upper_95": D[n][m].quantile(0.975)}
                         for n in names for m in observed[n]])
    marg.to_csv(output_dir / "cluster_bootstrap_model_metrics.csv", index=False)

    paired = []
    for base0 in BASELINES:
        for n in names:
            for m in observed[n]:
                base = "rule_calibrated" if (m == "brier_score" and base0 == "rule_score") else base0
                if n in (base0, "rule_score", "rule_calibrated") or (base0 == "logistic" and n not in ("gbt", "logistic_interact")) or base not in D or np.isnan(observed[n][m]):
                    continue
                diff = D[n][m] - D[base][m]
                paired.append({"model": n, "metric": m, "baseline": base, "difference": observed[n][m] - observed[base][m],
                               "ci_lower_95": diff.quantile(0.025), "ci_upper_95": diff.quantile(0.975),
                               "share_of_draws_better": float((diff < 0).mean() if m == "brier_score" else (diff > 0).mean())})
    pd.DataFrame(paired).to_csv(output_dir / "cluster_bootstrap_paired_difference.csv", index=False)
    return marg
