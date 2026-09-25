from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

LOADED = ["customers", "subscriptions", "usage", "payments", "contracts"]
EXPORTED = ["account_month", "cohort_retention", "mrr_bridge", "discount_dependence"]


def build_sql_marts(clean_dir: Path, sql_dir: Path, output_dir: Path) -> None:
    """Load cleaned tables into SQLite, run every script in sql/ in order, export the marts."""
    output_dir.mkdir(parents=True, exist_ok=True)
    db = output_dir / "analytics.db"
    db.unlink(missing_ok=True)
    con = sqlite3.connect(db)
    for table in LOADED:
        pd.read_csv(clean_dir / f"{table}_clean.csv").to_sql(f"{table}_clean", con, index=False)
    for script in sorted(sql_dir.glob("*.sql")):
        con.executescript(script.read_text(encoding="utf-8"))
    for table in EXPORTED:
        pd.read_sql_query(f"SELECT * FROM {table}", con).to_csv(output_dir / f"{table}.csv", index=False)
    con.close()
