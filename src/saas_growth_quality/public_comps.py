from __future__ import annotations

import json
import gzip
from pathlib import Path

import pandas as pd


METRIC_TAGS = {
    "revenue": [
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "SalesRevenueNet",
        "Revenues",
    ],
    "cost_of_revenue": ["CostOfRevenue", "CostOfGoodsAndServicesSold"],
    "operating_income": ["OperatingIncomeLoss"],
    "research_and_development": ["ResearchAndDevelopmentExpense"],
    "cash_from_operations": ["NetCashProvidedByUsedInOperatingActivities"],
    "capital_expenditure": ["PaymentsToAcquirePropertyPlantAndEquipment"],
}


def build_public_comps_panel(raw_dir: Path, universe_path: Path, output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build auditable company-quarter observations from SEC Company Facts snapshots."""
    output_dir.mkdir(parents=True, exist_ok=True)
    universe = pd.read_csv(universe_path, dtype={"cik": str})
    observations: list[pd.DataFrame] = []
    for company in universe.itertuples(index=False):
        snapshot = raw_dir / f"CIK{str(company.cik).zfill(10)}.json"
        if not snapshot.exists():
            continue
        raw_bytes = snapshot.read_bytes()
        if raw_bytes[:2] == b"\x1f\x8b":
            raw_bytes = gzip.decompress(raw_bytes)
        facts = json.loads(raw_bytes.decode("utf-8"))
        for metric, tags in METRIC_TAGS.items():
            extracted = _extract_metric(facts, tags, metric)
            if extracted.empty:
                continue
            extracted["ticker"] = company.ticker
            extracted["cik"] = str(company.cik).zfill(10)
            extracted["company_name"] = company.company_name
            extracted["source_url"] = company.company_facts_url
            observations.append(extracted)

    long = pd.concat(observations, ignore_index=True) if observations else _empty_long_frame()
    long = long.sort_values(["ticker", "metric", "period_end", "filing_date"])
    long = long.drop_duplicates(["ticker", "metric", "period_start", "period_end"], keep="last")
    long.to_csv(output_dir / "sec_financials_long.csv", index=False)
    quarterly = _build_quarterly_wide(long)
    quarterly.to_csv(output_dir / "quarterly_financials.csv", index=False)
    return long, quarterly


def _extract_metric(facts: dict[str, object], tags: list[str], metric: str) -> pd.DataFrame:
    """All 10-Q/10-K duration facts for one metric, merged across US-GAAP tags in priority order.

    Companies switch tags over time (for example pre- and post-ASC 606 revenue), so a tag
    is a fallback per period, not per company. Cumulative fiscal-year-to-date facts are kept.
    """
    taxonomy = facts.get("facts", {}).get("us-gaap", {})
    frames = []
    for rank, tag in enumerate(tags):
        frame = pd.DataFrame(taxonomy.get(tag, {}).get("units", {}).get("USD", []))
        if frame.empty or "start" not in frame:
            continue
        frame = frame.loc[frame["form"].isin(["10-Q", "10-K"])].copy()
        frame["tag_rank"], frame["source_tag"] = rank, tag
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    merged = pd.concat(frames, ignore_index=True)
    for column in ["start", "end", "filed"]:
        merged[column] = pd.to_datetime(merged[column], errors="coerce")
    days = (merged["end"] - merged["start"]).dt.days
    merged = merged.loc[days.between(75, 120) | days.between(150, 290) | days.between(330, 380)].copy()
    merged = merged.sort_values("filed").drop_duplicates(["tag_rank", "start", "end"], keep="last")  # latest restatement
    merged = merged.sort_values("tag_rank").drop_duplicates(["start", "end"], keep="first")  # preferred tag
    merged["metric"] = metric
    merged = merged.rename(
        columns={
            "fy": "fiscal_year", "fp": "fiscal_period", "start": "period_start", "end": "period_end",
            "val": "value_usd", "filed": "filing_date", "accn": "accession_number", "form": "form_type",
        }
    )
    return merged[
        ["metric", "source_tag", "fiscal_year", "fiscal_period", "period_start", "period_end",
         "value_usd", "filing_date", "accession_number", "form_type"]
    ]


def _discrete_quarters(facts: pd.DataFrame) -> pd.DataFrame:
    """Discrete quarters for one company-metric, identified by date windows.

    SEC fy/fp labels describe the filing, so a prior-year comparative column carries the
    current label; they are not used here. Reported three-month facts are kept. Missing
    quarters come from differencing facts that share a fiscal-year start (H1 - Q1, 9M - H1,
    FY - 9M). Cash-flow statements report only cumulative values, so they need this path.
    """
    facts = facts.assign(days=(facts["period_end"] - facts["period_start"]).dt.days)
    rows = [facts.loc[facts["days"].between(75, 120), ["period_start", "period_end", "value_usd"]]]
    for _, chain in facts.groupby("period_start"):
        chain = chain.sort_values("period_end")
        for prev, cur in zip(chain.itertuples(), chain.iloc[1:].itertuples()):
            if 75 <= (cur.period_end - prev.period_end).days <= 120:
                rows.append(
                    pd.DataFrame(
                        [{"period_start": prev.period_end + pd.Timedelta(days=1), "period_end": cur.period_end,
                          "value_usd": cur.value_usd - prev.value_usd}]
                    )
                )
    quarters = pd.concat(rows, ignore_index=True)
    return quarters.drop_duplicates("period_end", keep="first")  # reported values win over derived ones


def _build_quarterly_wide(long: pd.DataFrame) -> pd.DataFrame:
    if long.empty:
        return pd.DataFrame()
    pieces = []
    for (ticker, metric), group in long.groupby(["ticker", "metric"]):
        quarters = _discrete_quarters(group)
        pieces.append(quarters.assign(ticker=ticker, cik=group["cik"].iloc[0], company_name=group["company_name"].iloc[0], metric=metric))
    quarterly = pd.concat(pieces, ignore_index=True)
    wide = quarterly.pivot_table(
        index=["ticker", "cik", "company_name", "period_end"], columns="metric", values="value_usd", aggfunc="last"
    ).reset_index()
    wide.columns.name = None
    if {"revenue", "cost_of_revenue"}.issubset(wide.columns):
        wide["gross_margin"] = 1 - wide["cost_of_revenue"] / wide["revenue"]
    if {"revenue", "operating_income"}.issubset(wide.columns):
        wide["operating_margin"] = wide["operating_income"] / wide["revenue"]
    if {"cash_from_operations", "capital_expenditure"}.issubset(wide.columns):
        wide["free_cash_flow"] = wide["cash_from_operations"] - wide["capital_expenditure"].abs()
        wide["free_cash_flow_margin"] = wide["free_cash_flow"] / wide["revenue"]
    if {"revenue", "research_and_development"}.issubset(wide.columns):
        wide["research_and_development_pct_revenue"] = wide["research_and_development"] / wide["revenue"]
    return wide.sort_values(["ticker", "period_end"]).reset_index(drop=True)


def _empty_long_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "ticker",
            "cik",
            "company_name",
            "metric",
            "source_tag",
            "fiscal_year",
            "fiscal_period",
            "period_start",
            "period_end",
            "value_usd",
            "filing_date",
            "accession_number",
            "form_type",
            "source_url",
        ]
    )
