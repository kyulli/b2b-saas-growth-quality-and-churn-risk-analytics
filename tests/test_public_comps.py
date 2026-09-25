import pandas as pd

from saas_growth_quality.public_comps import _build_quarterly_wide, _extract_metric


def _fact(end: str, value: float, start: str = "2024-01-01", form: str = "10-Q") -> dict[str, object]:
    return {"start": start, "end": end, "val": value, "form": form, "fy": 2024, "fp": "X", "filed": "2025-02-01", "accn": "1"}


def test_cumulative_cash_flow_is_differenced_into_discrete_quarters() -> None:
    ytd = [("2024-03-31", 10.0), ("2024-06-30", 22.0), ("2024-09-30", 36.0), ("2024-12-31", 60.0)]
    long = pd.DataFrame(
        [
            {"ticker": "T", "cik": "1", "company_name": "Test Co.", "metric": "cash_from_operations",
             "period_start": pd.Timestamp("2024-01-01"), "period_end": pd.Timestamp(end), "value_usd": value}
            for end, value in ytd
        ]
    )
    result = _build_quarterly_wide(long).set_index("period_end")["cash_from_operations"]

    assert result.tolist() == [10.0, 12.0, 14.0, 24.0]


def test_extract_metric_merges_tags_across_periods() -> None:
    facts = {
        "facts": {
            "us-gaap": {
                "SalesRevenueNet": {"units": {"USD": [_fact("2016-03-31", 5.0, "2016-01-01")]}},
                "Revenues": {"units": {"USD": [_fact("2024-03-31", 9.0)]}},
            }
        }
    }
    result = _extract_metric(facts, ["Revenues", "SalesRevenueNet"], "revenue")

    assert sorted(result["value_usd"]) == [5.0, 9.0]
