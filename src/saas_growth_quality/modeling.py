"""Churn-risk scoring on a pre-close / post-close split.

The book is assumed acquired at CLOSE (Dec 2024). Everything a buyer could know at close is training data:
feature months whose 3-month label matured by CLOSE. Feature months after CLOSE are the holdout, so every
reported holdout number is genuinely out of time.

Models
  rule_score         hand-set weights on the five core signals (0-1 ranking, not a probability)
  rule_calibrated    the same ranking mapped to probabilities on training months (for Brier only)
  rule_renewal       rule_score scaled by contract exposure (specified in the previous project round,
                     frozen before this holdout was generated)
  logistic           regularised logistic regression on 16 features; L1, L2 and Elastic Net compared on
                     forward folds and on coefficient stability (see penalty_diagnostics)
  logistic_interact  the chosen logistic plus explicit renewal-exposure interaction terms; tests whether a
                     tree model's gain is only interactions
  gbt                gradient boosted trees (histogram-based) on the same 16 features
  oracle             the simulator's true 3-month hazard; a ceiling, not a model
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, brier_score_loss, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from saas_growth_quality.features import CORE_FEATURES, EXTENDED_FEATURES, HORIZON

CLOSE = pd.Timestamp("2024-12-01")
RULE_WEIGHTS = {"usage": 0.35, "discount": 0.25, "late": 0.20, "tickets": 0.15, "growth": 0.05}
EARLY_EXIT = 0.25
TOP_SHARE = 0.10
INTERACT_WITH = ["usage_ratio_3m", "usage_trend_3m", "ticket_trend_3m", "ticket_count_3m"]
MODELS = ["rule_score", "rule_calibrated", "rule_renewal", "logistic", "logistic_interact", "gbt"]


# ---------------------------------------------------------------- rules
def rule_components(f: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "usage": (1 - f["usage_ratio_3m"]).clip(0, 1),
        "discount": (f["discount_pct_3m"] / 0.40).clip(0, 1),
        "late": (f["days_late_3m"] / 30).clip(0, 1),
        "tickets": (f["ticket_count_3m"] / 6).clip(0, 1),
        "growth": ((-f["list_mrr_growth_3m"]).clip(0, 0.30) / 0.30),
    }, index=f.index)


def rule_score(f: pd.DataFrame, weights: dict[str, float] = RULE_WEIGHTS) -> pd.Series:
    w = pd.Series(weights) / sum(weights.values())
    return (rule_components(f)[w.index] * w).sum(axis=1)


def exposed(f: pd.DataFrame) -> pd.Series:
    """Account can leave within the label window: monthly contract, or committed contract renewing within 3 months."""
    return (f["contract_term_months"] == 1) | ((f["months_to_renewal"] <= HORIZON) & (f["contract_term_months"] > 1))


def rule_renewal(f: pd.DataFrame, early_exit: float = EARLY_EXIT) -> pd.Series:
    return rule_score(f) * np.where(exposed(f), 1.0, early_exit)


def with_interactions(f: pd.DataFrame) -> pd.DataFrame:
    x = f[EXTENDED_FEATURES].copy()
    e = exposed(f).astype(float)
    x["exposed"] = e
    for c in INTERACT_WITH:
        x[f"exposed_x_{c}"] = e * f[c]
    return x


# ---------------------------------------------------------------- split and folds
def split(features: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    f = features.copy()
    f["month"] = pd.to_datetime(f["month"])
    last_mature_at_close = CLOSE - pd.DateOffset(months=HORIZON)
    train = f[f["month"] <= last_mature_at_close].reset_index(drop=True)
    test = f[f["month"] > CLOSE].reset_index(drop=True)
    info = {"close": str(CLOSE.date()),
            "train_months": [str(train["month"].min().date()), str(train["month"].max().date())],
            "holdout_months": [str(test["month"].min().date()), str(test["month"].max().date())],
            "excluded_months": "feature months after the last mature month and up to close (labels straddle close)",
            "train_rows": int(len(train)), "holdout_rows": int(len(test)),
            "train_positives": int(train["churn_next_3m"].sum()), "holdout_positives": int(test["churn_next_3m"].sum()),
            "train_churn_rate": round(float(train["churn_next_3m"].mean()), 4),
            "holdout_churn_rate": round(float(test["churn_next_3m"].mean()), 4)}
    return train, test, info


def forward_folds(train: pd.DataFrame, n_folds: int = 4) -> list[tuple[np.ndarray, np.ndarray]]:
    months = sorted(train["month"].unique())
    folds = []
    for m in months[-n_folds:]:
        fit = np.flatnonzero(train["month"] < pd.Timestamp(m) - pd.DateOffset(months=HORIZON))
        val = np.flatnonzero(train["month"] == m)
        folds.append((fit, val))
    return folds


# ---------------------------------------------------------------- estimators
def _logit(penalty: str = "l2", c: float = 1.0, l1_ratio: float | None = None) -> Pipeline:
    kw = {"l1": dict(solver="liblinear"), "l2": dict(solver="lbfgs"),
          "elasticnet": dict(solver="saga", l1_ratio=l1_ratio, tol=1e-3)}[penalty]
    return Pipeline([("scale", StandardScaler()), ("model", LogisticRegression(penalty=penalty, C=c, max_iter=5000, **kw))])


LOGIT_GRID = [{"model__penalty": ["l1"], "model__solver": ["liblinear"], "model__C": [0.01, 0.03, 0.1, 1.0]},
              {"model__penalty": ["l2"], "model__solver": ["lbfgs"], "model__C": [0.01, 0.03, 0.1, 1.0]},
              {"model__penalty": ["elasticnet"], "model__solver": ["saga"], "model__l1_ratio": [0.5],
               "model__C": [0.01, 0.03, 0.1, 1.0]}]
GBT_GRID = {"learning_rate": [0.05, 0.1], "max_leaf_nodes": [8, 16], "min_samples_leaf": [100, 400], "max_iter": [200]}


def evaluate(y: pd.Series, score: pd.Series, months: pd.Series, probability: bool = True) -> dict[str, float]:
    flag = score.groupby(months).transform(lambda s: s.rank(ascending=False, method="first") <= np.ceil(TOP_SHARE * len(s)))
    monthly = pd.DataFrame({"y": y, "s": score, "m": months}).groupby("m").apply(
        lambda d: roc_auc_score(d["y"], d["s"]) if d["y"].nunique() == 2 else np.nan, include_groups=False)
    base = y.mean()
    return {"roc_auc": roc_auc_score(y, score), "average_precision": average_precision_score(y, score),
            "brier_score": brier_score_loss(y, score) if probability else np.nan,
            "brier_skill": 1 - brier_score_loss(y, score) / (base * (1 - base)) if probability else np.nan,
            "mean_predicted": score.mean() if probability else np.nan, "actual_rate": base,
            "monthly_auc_mean": monthly.mean(), "monthly_auc_std": monthly.std(),
            "top10_churn_rate": y[flag].mean(), "top10_lift": y[flag].mean() / base,
            "precision_top10": precision_score(y, flag), "recall_top10": recall_score(y, flag),
            "f1_top10": f1_score(y, flag), "accuracy_top10": accuracy_score(y, flag)}


# ---------------------------------------------------------------- penalty diagnostics
def penalty_diagnostics(train: pd.DataFrame, output_dir: Path, c: float, n_boot: int = 40, seed: int = 7) -> pd.DataFrame:
    """Why L1, L2 or Elastic Net: collinearity (correlation, VIF, condition number), events per variable, and
    how stable each penalty's coefficients are when customers are resampled."""
    X = train[EXTENDED_FEATURES]
    corr = X.corr()
    corr.to_csv(output_dir / "feature_correlation.csv")
    Z = StandardScaler().fit_transform(X)
    inv = np.linalg.pinv(np.corrcoef(Z, rowvar=False))
    vif = pd.Series(np.diag(inv), index=EXTENDED_FEATURES, name="vif")
    cond = float(np.linalg.cond(Z))
    epv = train["churn_next_3m"].sum() / len(EXTENDED_FEATURES)

    codes, uniq = pd.factorize(train["customer_id"])
    rows_of = pd.Series(np.arange(len(train))).groupby(codes).apply(np.array).to_numpy()
    rng = np.random.default_rng(seed)
    coefs = {p: [] for p in ["l1", "l2", "elasticnet"]}
    for _ in range(n_boot):
        idx = np.concatenate(rows_of[rng.integers(0, len(uniq), len(uniq))])
        for p in coefs:
            m = _logit(p, c, 0.5 if p == "elasticnet" else None).fit(X.iloc[idx], train["churn_next_3m"].iloc[idx])
            coefs[p].append(m.named_steps["model"].coef_[0])
    rows = []
    for p, arr in coefs.items():
        a = np.array(arr)
        for j, feat in enumerate(EXTENDED_FEATURES):
            col = a[:, j]
            nz = np.abs(col) > 1e-6
            rows.append({"penalty": p, "feature": feat, "mean_coef": col.mean(), "sd_coef": col.std(),
                         "selected_share": nz.mean(),
                         "sign_consistency": max((col > 1e-6).mean(), (col < -1e-6).mean()) / max(nz.mean(), 1e-9) if nz.any() else np.nan})
    stab = pd.DataFrame(rows)
    stab.to_csv(output_dir / "penalty_coefficient_stability.csv", index=False)
    summary = {"condition_number": cond, "events_per_variable": float(epv),
               "max_abs_correlation_pair": max(((a, b, abs(corr.loc[a, b])) for i, a in enumerate(EXTENDED_FEATURES)
                                                for b in EXTENDED_FEATURES[i + 1:]), key=lambda t: t[2]),
               "vif": vif.round(2).to_dict(),
               "unstable_selection_features_l1": stab[(stab.penalty == "l1") & stab.selected_share.between(0.1, 0.9)]["feature"].tolist()}
    (output_dir / "penalty_diagnostics.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    return stab


def interaction_cells(test: pd.DataFrame) -> pd.DataFrame:
    """Churn and predicted risk in the eight cells formed by usage falling, tickets rising and renewal exposure."""
    t = test.assign(usage_falling=test["usage_trend_3m"] < -0.05, tickets_rising=test["ticket_trend_3m"] > 0, exposed=exposed(test))
    g = t.groupby(["exposed", "usage_falling", "tickets_rising"]).agg(
        rows=("churn_next_3m", "size"), actual=("churn_next_3m", "mean"),
        logistic=("logistic_score", "mean"), logistic_interact=("logistic_interact_score", "mean"), gbt=("gbt_score", "mean"))
    return g.reset_index()


# ---------------------------------------------------------------- main
def train_and_compare(features: pd.DataFrame, output_dir: Path, oracle: pd.DataFrame | None = None) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    train, test, info = split(features)
    folds = forward_folds(train)
    y = train["churn_next_3m"]

    lg = GridSearchCV(_logit(), LOGIT_GRID, cv=folds, scoring="roc_auc", n_jobs=-1).fit(train[EXTENDED_FEATURES], y)
    pd.DataFrame(lg.cv_results_)[["params", "mean_test_score", "std_test_score", "rank_test_score"]].to_csv(output_dir / "gridsearch_logistic.csv", index=False)
    best = {k.replace("model__", ""): v for k, v in lg.best_params_.items()}
    # Among settings within one CV standard error of the best, prefer the most stable penalty (L2 < EN < L1).
    cv = pd.DataFrame(lg.cv_results_)
    top = cv["mean_test_score"].max()
    se = cv.loc[cv["rank_test_score"] == 1, "std_test_score"].iloc[0] / np.sqrt(len(folds))
    near = cv[cv["mean_test_score"] >= top - se].copy()
    near["order"] = near["param_model__penalty"].map({"l2": 0, "elasticnet": 1, "l1": 2})
    pick = near.sort_values(["order", "param_model__C"]).iloc[0]
    chosen = {"penalty": pick["param_model__penalty"], "C": float(pick["param_model__C"]),
              "l1_ratio": 0.5 if pick["param_model__penalty"] == "elasticnet" else None,
              "cv_auc": float(pick["mean_test_score"]), "best_cv_auc": float(top), "one_se": float(se), "cv_best_params": best}
    logistic = _logit(chosen["penalty"], chosen["C"], chosen["l1_ratio"]).fit(train[EXTENDED_FEATURES], y)
    stab = penalty_diagnostics(train, output_dir, chosen["C"])

    li = _logit(chosen["penalty"], chosen["C"], chosen["l1_ratio"]).fit(with_interactions(train), y)
    gb = GridSearchCV(HistGradientBoostingClassifier(random_state=42), GBT_GRID, cv=folds, scoring="roc_auc", n_jobs=-1).fit(train[EXTENDED_FEATURES], y)
    pd.DataFrame(gb.cv_results_)[["params", "mean_test_score", "std_test_score", "rank_test_score"]].to_csv(output_dir / "gridsearch_gbt.csv", index=False)

    platt = LogisticRegression().fit(rule_score(train).to_frame(), y)
    platt_r = LogisticRegression().fit(rule_renewal(train).to_frame(), y)
    test["rule_score_score"] = rule_score(test)
    test["rule_calibrated_score"] = platt.predict_proba(test["rule_score_score"].to_frame())[:, 1]
    test["rule_renewal_score"] = platt_r.predict_proba(rule_renewal(test).to_frame())[:, 1]
    test["logistic_score"] = logistic.predict_proba(test[EXTENDED_FEATURES])[:, 1]
    test["logistic_interact_score"] = li.predict_proba(with_interactions(test))[:, 1]
    test["gbt_score"] = gb.best_estimator_.predict_proba(test[EXTENDED_FEATURES])[:, 1]
    if oracle is not None:
        o = oracle.assign(month=pd.to_datetime(oracle["month"]))
        test = test.merge(o, on=["customer_id", "month"], how="left")
        test["oracle_score"] = test["oracle_churn_3m"]

    names = MODELS + (["oracle"] if oracle is not None else [])
    table = pd.DataFrame({n: evaluate(test["churn_next_3m"], test[f"{n}_score"], test["month"], n != "rule_score") for n in names}).T
    table.rename_axis("model").reset_index().to_csv(output_dir / "model_comparison.csv", index=False)

    coef = pd.DataFrame({"feature": EXTENDED_FEATURES, "standardized_coef": logistic.named_steps["model"].coef_[0]})
    coef.to_csv(output_dir / "logistic_coefficients.csv", index=False)
    lic = pd.DataFrame({"feature": list(with_interactions(train.head(2)).columns), "standardized_coef": li.named_steps["model"].coef_[0]})
    lic.to_csv(output_dir / "logistic_interact_coefficients.csv", index=False)
    sample = test.sample(min(12000, len(test)), random_state=1)
    pi = permutation_importance(gb.best_estimator_, sample[EXTENDED_FEATURES], sample["churn_next_3m"], scoring="roc_auc", n_repeats=5, random_state=1)
    pd.DataFrame({"feature": EXTENDED_FEATURES, "auc_drop_mean": pi.importances_mean, "auc_drop_sd": pi.importances_std}) \
        .sort_values("auc_drop_mean", ascending=False).to_csv(output_dir / "gbt_permutation_importance.csv", index=False)
    interaction_cells(test).to_csv(output_dir / "interaction_cells.csv", index=False)
    rows = [{"component": "base", "shift": 0.0, "roc_auc": roc_auc_score(test["churn_next_3m"], rule_renewal(test))}]
    for k in RULE_WEIGHTS:
        for d in (-0.2, 0.2):
            w = dict(RULE_WEIGHTS, **{k: RULE_WEIGHTS[k] * (1 + d)})
            rows.append({"component": k, "shift": d, "roc_auc": roc_auc_score(test["churn_next_3m"], rule_score(test, w) * np.where(exposed(test), 1, EARLY_EXIT))})
    pd.DataFrame(rows).to_csv(output_dir / "rule_weight_sensitivity.csv", index=False)
    pd.DataFrame([{"early_exit": e, "roc_auc": roc_auc_score(test["churn_next_3m"], rule_renewal(test, e))} for e in (0.1, 0.25, 0.5, 1.0)]) \
        .to_csv(output_dir / "rule_renewal_sensitivity.csv", index=False)

    keep = ["customer_id", "month", "segment", "industry", "contract_term_months", "pricing_model", "channel", "list_mrr",
            "mrr_amount", "months_to_renewal", "churn_next_3m"] + EXTENDED_FEATURES + [f"{n}_score" for n in names]
    test[keep].to_csv(output_dir / "holdout_scored_accounts.csv", index=False)
    summary = {**info, "forward_folds": len(folds), "logistic_choice": chosen,
               "gbt_params": gb.best_params_, "gbt_cv_auc": float(gb.best_score_), "metrics": table.round(4).to_dict("index")}
    (output_dir / "model_comparison.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    return summary
