"""Acquisition view: diligence at close (Dec 2024), underwriting forecast, and post-close monitoring.

Diligence window: Dec 2023 to Dec 2024, the same window SaaS Capital's 2025 retention survey measured.
Anything after close is out of sample: the forecast is made with pre-close data only and compared with
what the book actually did from Jan 2025 to Jul 2026.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from saas_growth_quality.forecast import bridge_rates, fan_paths, project

CLOSE = "2024-12-01"
DILIGENCE_START = "2023-12-01"
LATEST = "2026-07-01"
ACV_BANDS = [0, 12e3, 25e3, 50e3, 100e3, 250e3, np.inf]
ACV_LABELS = ["ACV <$12k", "ACV $12k-$25k", "ACV $25k-$50k", "ACV $50k-$100k", "ACV $100k-$250k", "ACV >$250k"]
TERM_LABELS = {1: "Month-to-month contracts", 12: "Annual contracts", 24: "Multi-year contracts", 36: "Multi-year contracts"}


def cohort_frame(am: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    act = am[am["status"] == "active"]
    s = act[act["month"] == start].set_index("customer_id")
    e = act[act["month"] == end].set_index("customer_id")["list_mrr"].reindex(s.index).fillna(0)
    d = pd.DataFrame({"start": s["list_mrr"], "end": e, "segment": s["segment"], "term": s["contract_term_months"],
                      "pricing_model": s["pricing_model"], "channel": s["channel"], "acv": s["billed_mrr"] * 12})
    d["acv_band"] = pd.cut(d["acv"], ACV_BANDS, labels=ACV_LABELS, right=False)
    d["term_band"] = d["term"].map(TERM_LABELS)
    return d


def _ret(x: pd.DataFrame) -> pd.Series:
    return pd.Series({"accounts": len(x), "start_arr": x["start"].sum() * 12,
                      "grr": np.minimum(x["end"], x["start"]).sum() / x["start"].sum(),
                      "nrr": x["end"].sum() / x["start"].sum(), "logo_churn": (x["end"] == 0).mean(),
                      "churn_loss": x.loc[x["end"] == 0, "start"].sum() / x["start"].sum(),
                      "contraction_loss": (x["start"] - x["end"]).clip(lower=0)[x["end"] > 0].sum() / x["start"].sum(),
                      "expansion": (x["end"] - x["start"]).clip(lower=0).sum() / x["start"].sum()})


def retention_by(d: pd.DataFrame, by: str | None) -> pd.DataFrame:
    if by is None:
        return _ret(d).to_frame("Portfolio").T
    return d.groupby(by, observed=True).apply(_ret, include_groups=False)


def cohort_ci(d: pd.DataFrame, n: int = 1000, seed: int = 5) -> dict[str, float]:
    """Customer bootstrap interval for cohort GRR and NRR: how much a single year's figure moves by chance."""
    rng = np.random.default_rng(seed)
    s, e = d["start"].to_numpy(), d["end"].to_numpy()
    g, r = [], []
    for _ in range(n):
        i = rng.integers(0, len(d), len(d))
        g.append(np.minimum(e[i], s[i]).sum() / s[i].sum()); r.append(e[i].sum() / s[i].sum())
    return {"grr_lo": np.quantile(g, 0.025), "grr_hi": np.quantile(g, 0.975), "nrr_lo": np.quantile(r, 0.025), "nrr_hi": np.quantile(r, 0.975)}


def benchmark_table(am: pd.DataFrame, bench: pd.DataFrame, start: str = DILIGENCE_START, end: str = CLOSE) -> pd.DataFrame:
    """Our cohort vs SaaS Capital 2025 medians on matched cuts (ACV band, contract term, funding, overall)."""
    d = cohort_frame(am, start, end)
    b = bench[bench["source"] == "SaaS Capital 2025 retention"].pivot_table(index="cut", columns="metric", values="value") / 100
    rows = []
    port = retention_by(d, None).iloc[0]
    for cut in ["All private B2B SaaS >$1M ARR", "Bootstrapped", "Equity-backed"]:
        rows.append({"cut": cut, "ours_accounts": port["accounts"], "ours_grr": port["grr"], "ours_nrr": port["nrr"],
                     "bench_grr": b.loc[cut, "GRR_median"], "bench_nrr": b.loc[cut, "NRR_median"], "match": "portfolio"})
    for col in ["acv_band", "term_band"]:
        for cut, r in retention_by(d, col).iterrows():
            rows.append({"cut": cut, "ours_accounts": r["accounts"], "ours_grr": r["grr"], "ours_nrr": r["nrr"],
                         "bench_grr": b.loc[cut, "GRR_median"], "bench_nrr": b.loc[cut, "NRR_median"], "match": col})
    t = pd.DataFrame(rows)
    t["small_sample"] = t["ours_accounts"] < 30
    t["grr_gap"] = t["ours_grr"] - t["bench_grr"]
    t["nrr_gap"] = t["ours_nrr"] - t["bench_nrr"]
    return t


def calendar_2025_check(am: pd.DataFrame, bench: pd.DataFrame) -> pd.DataFrame:
    """Out-of-sample year (Dec 2024 to Dec 2025) vs the benchmarks published in 2026 for calendar 2025."""
    d25 = retention_by(cohort_frame(am, CLOSE, "2025-12-01"), None).iloc[0]
    d24 = retention_by(cohort_frame(am, DILIGENCE_START, CLOSE), None).iloc[0]
    act = am[am["status"] == "active"].groupby("month")["list_mrr"].sum()
    g24 = act["2024-12-01"] / act["2023-12-01"] - 1
    g25 = act["2025-12-01"] / act["2024-12-01"] - 1
    v = lambda src, metric, cut: float(bench[(bench.source == src) & (bench.metric == metric) & (bench.cut == cut)]["value"].iloc[0]) / 100
    return pd.DataFrame([
        {"metric": "GRR", "ours_2024": d24["grr"], "ours_2025": d25["grr"], "bench_2024": v("SaaS Capital 2025 retention", "GRR_median", "Bootstrapped"),
         "bench_2025": v("SaaS Capital 2026 bootstrapped", "GRR_median", "Bootstrapped $3M-$20M ARR"), "bench_note": "SaaS Capital bootstrapped; 2025 figure covers $3M-$20M ARR only"},
        {"metric": "NRR", "ours_2024": d24["nrr"], "ours_2025": d25["nrr"], "bench_2024": v("SaaS Capital 2025 retention", "NRR_median", "Bootstrapped"),
         "bench_2025": v("SaaS Capital 2026 bootstrapped", "NRR_median", "Bootstrapped $3M-$20M ARR"), "bench_note": "Same"},
        {"metric": "ARR growth", "ours_2024": g24, "ours_2025": g25, "bench_2024": np.nan,
         "bench_2025": v("SaaS Capital 2026 growth", "revenue_growth_median", "Equity-backed (all sizes)"), "bench_note": "Equity-backed revenue growth, all sizes (bootstrapped 20%)"},
    ])


def underwrite(bridge: pd.DataFrame, close: str = CLOSE, end: str = LATEST, lookback: int = 12, pool: int = 24,
               n: int = 4000, seed: int = 42) -> pd.DataFrame:
    """Forecast list MRR from close to end using only pre-close months, then attach actuals.
    Base case = trailing 12-month average rates; band = forecast.fan_paths on the same pre-close months."""
    rates = bridge_rates(bridge)
    pre = rates[rates.index <= pd.Timestamp(close)]
    idx = rates.index[(rates.index > pd.Timestamp(close)) & (rates.index <= pd.Timestamp(end))]
    h = len(idx)
    base = pre.iloc[-lookback:][["new", "expansion", "contraction", "churn"]].mean()
    base_path = project(pre["closing"].iloc[-1], *(np.full(h, base[k]) for k in ["new", "expansion", "contraction", "churn"]))
    q = np.quantile(fan_paths(pre, h, pool, lookback, n, seed), [0.05, 0.25, 0.5, 0.75, 0.95], axis=0)
    out = pd.DataFrame({"base": base_path, "p05": q[0], "p25": q[1], "p50": q[2], "p75": q[3], "p95": q[4],
                        "actual": rates.loc[idx, "closing"].to_numpy()}, index=idx)
    out["error"] = out["base"] / out["actual"] - 1
    out["in_90_band"] = out["actual"].between(out["p05"], out["p95"])
    out["in_50_band"] = out["actual"].between(out["p25"], out["p75"])
    out.index.name = "month"
    return out


def rolling_coverage(bridge: pd.DataFrame, horizon: int = 12, pool: int = 24, n: int = 1000, seed: int = 3) -> pd.DataFrame:
    """Out-of-sample interval coverage of the same fan method: at every origin, build the band from prior months only
    and record whether each of the next `horizon` actuals falls inside it."""
    rates = bridge_rates(bridge)
    rows = []
    for i in range(pool, len(rates) - 1):
        pre = rates.iloc[:i]
        hh = min(horizon, len(rates) - i)
        paths = fan_paths(pre, hh, pool, n=n, seed=seed + i)
        lo, hi = np.quantile(paths, [0.05, 0.95], axis=0)
        act = rates["closing"].iloc[i:i + hh].to_numpy()
        for k in range(hh):
            rows.append({"origin": pre.index[-1], "horizon": k + 1, "inside_90": bool(lo[k] <= act[k] <= hi[k]),
                         "rel_width": (hi[k] - lo[k]) / act[k], "error_of_median": np.median(paths[:, k]) / act[k] - 1})
    return pd.DataFrame(rows)


def retention_plan_vs_actual(am: pd.DataFrame) -> pd.DataFrame:
    """Plan = diligence-window GRR/NRR held flat; actual = rolling 12-month cohort measured at each month-end after close."""
    plan = retention_by(cohort_frame(am, DILIGENCE_START, CLOSE), None).iloc[0]
    rows = []
    for end in pd.date_range("2025-12-01", LATEST, freq="MS"):
        start = (end - pd.DateOffset(months=12)).strftime("%Y-%m-%d")
        r = retention_by(cohort_frame(am, start, end.strftime("%Y-%m-%d")), None).iloc[0]
        rows.append({"period_end": end.date(), "plan_grr": plan["grr"], "actual_grr": r["grr"], "plan_nrr": plan["nrr"], "actual_nrr": r["nrr"]})
    return pd.DataFrame(rows)


def build(root: Path) -> dict:
    marts = root / "outputs" / "marts"
    am = pd.read_csv(marts / "account_month.csv")
    bridge = pd.read_csv(marts / "mrr_bridge.csv")
    bench = pd.read_csv(root / "data" / "reference" / "saas_benchmarks.csv")
    dil = cohort_frame(am, DILIGENCE_START, CLOSE)
    mon = cohort_frame(am, "2025-07-01", LATEST)
    return {
        "benchmark_table": benchmark_table(am, bench),
        "diligence_by": {g: retention_by(dil, g) for g in [None, "segment", "acv_band", "term_band", "pricing_model", "channel"]},
        "monitoring_by": {g: retention_by(mon, g) for g in [None, "segment", "acv_band", "term_band", "pricing_model", "channel"]},
        "calendar_2025": calendar_2025_check(am, bench),
        "ci_2024": cohort_ci(dil), "ci_2025": cohort_ci(cohort_frame(am, CLOSE, "2025-12-01")),
        "underwrite": underwrite(bridge),
        "coverage": rolling_coverage(bridge),
        "retention_plan": retention_plan_vs_actual(am),
    }
