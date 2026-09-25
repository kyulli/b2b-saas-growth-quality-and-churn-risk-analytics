from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from saas_growth_quality.features import build_feature_table
from saas_growth_quality.generate import generate_raw_data
from saas_growth_quality.modeling import train_and_compare
from saas_growth_quality.quality import clean_raw_data, run_quality_checks
from saas_growth_quality.sql_pipeline import build_sql_marts
from saas_growth_quality.uncertainty import build_cluster_bootstrap_intervals


def run(root: Path, seed: int = 42, stages: str = "all") -> None:
    data, out = root / "data", root / "outputs"
    if stages in ("all", "data"):
        generate_raw_data(data / "raw", seed=seed)
        run_quality_checks(data / "raw", out / "quality")
        clean_raw_data(data / "raw", data / "clean", out / "quality")
        build_sql_marts(data / "clean", root / "sql", out / "marts")
        build_feature_table(data / "clean", out / "features" / "customer_month_features.csv")
    if stages in ("all", "model"):
        features = pd.read_csv(out / "features" / "customer_month_features.csv")
        oracle = pd.read_csv(data / "oracle" / "oracle_churn_3m.csv")
        train_and_compare(features, out / "model", oracle)
        build_cluster_bootstrap_intervals(out / "model" / "holdout_scored_accounts.csv", out / "uncertainty")
    if stages in ("all", "report"):
        from saas_growth_quality.reporting import build_reports
        build_reports(root)


def main() -> None:
    p = argparse.ArgumentParser(description="Run the SaaS growth-quality and churn-risk pipeline.")
    p.add_argument("--project-root", type=Path, default=Path.cwd())
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--stages", choices=["all", "data", "model", "report"], default="all")
    a = p.parse_args()
    run(a.project_root.resolve(), a.seed, a.stages)


if __name__ == "__main__":
    main()
