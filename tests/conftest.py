from pathlib import Path

import pytest

from saas_growth_quality.features import build_feature_table
from saas_growth_quality.generate import generate_raw_data
from saas_growth_quality.quality import clean_raw_data, run_quality_checks
from saas_growth_quality.sql_pipeline import build_sql_marts

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def panel(tmp_path_factory) -> Path:
    """A small end-to-end run: 400 initial accounts, 18 months, 1% defects."""
    d = tmp_path_factory.mktemp("panel")
    info = generate_raw_data(d / "raw", n_customers=400, n_months=18, monthly_arrivals=20, seed=11, defect_rate=0.01)
    run_quality_checks(d / "raw", d / "quality")
    clean_raw_data(d / "raw", d / "clean", d / "quality")
    build_sql_marts(d / "clean", ROOT / "sql", d / "marts")
    build_feature_table(d / "clean", d / "features.csv")
    (d / "injected.txt").write_text(str(info["injected"]))
    return d
