"""Assemble results from pipeline outputs and write the brief, the dashboard, the PDF and the workbook."""
from __future__ import annotations

import base64
import json
from pathlib import Path

import numpy as np
import pandas as pd

from saas_growth_quality import diligence as DL
from saas_growth_quality import figures as F
from saas_growth_quality.forecast import bootstrap_fan, bridge_rates, scenario_forecast
from saas_growth_quality.growth import bridge_check, discount_dependence, discounted_mrr_share, load_marts
from saas_growth_quality.intervention import break_even, pilot_size
from saas_growth_quality.revenue_quality import build as build_rq
from saas_growth_quality.revenue_quality import concentration, customer_arr

TARGET_LISTS = {"Random 10%": None, "Rule score": "rule_score_score", "Rule x renewal": "rule_renewal_score",
                "Logistic (L2)": "logistic_score", "Gradient boosted trees": "gbt_score", "Logistic, value-ranked": "value_logistic"}


def promotion_decision(metrics: pd.DataFrame, paired: pd.DataFrame, challenger: str, baseline: str = "rule_renewal") -> dict:
    """Pre-registered rule: a challenger replaces the operating score only if its paired AUC gain is at least 0.02 with a
    95% CI above zero, its Brier skill is no worse, and its monthly AUC std is no worse by more than 0.01."""
    m = metrics.set_index("model")
    p = paired[(paired["model"] == challenger) & (paired["metric"] == "roc_auc") & (paired["baseline"] == baseline)].iloc[0]
    bb = "rule_calibrated" if baseline == "rule_score" else baseline
    checks = {"auc_gain_ge_0.02": bool(p["difference"] >= 0.02), "auc_gain_ci_above_0": bool(p["ci_lower_95"] > 0),
              "brier_skill_not_worse": bool(m.loc[challenger, "brier_skill"] >= m.loc[bb, "brier_skill"]),
              "stability_not_worse": bool(m.loc[challenger, "monthly_auc_std"] <= m.loc[baseline, "monthly_auc_std"] + 0.01)}
    return {"challenger": challenger, "baseline": baseline, "auc_gain": p["difference"], "ci": [p["ci_lower_95"], p["ci_upper_95"]],
            "checks": checks, "promote": all(checks.values())}


def compute(root: Path) -> dict:
    out = root / "outputs"
    marts = load_marts(out / "marts")
    am, dd = marts["account_month"], marts["discount_dependence"]
    r: dict = {}
    r["quality"] = pd.read_csv(out / "quality" / "rule_summary.csv")
    r["manifest"] = json.loads((out / "quality" / "cleaning_manifest.json").read_text())
    r["bridge"] = marts["mrr_bridge"]
    r["bridge_gap"] = bridge_check(r["bridge"])
    r["share"] = discounted_mrr_share(am)
    r["cohort"] = marts["cohort_retention"]
    dd24 = dd[(dd["month"] >= "2024-01-01") & (dd["month"] <= DL.CLOSE)]
    ddl = dd[dd["month"] >= "2025-08-01"]
    r["dd_2024"], r["dd_latest"] = discount_dependence(dd24), discount_dependence(ddl)
    r["dd_channel_2024"], r["dd_term_2024"] = discount_dependence(dd24, "channel"), discount_dependence(dd24, "contract_term_months")
    r["dd_channel_latest"] = discount_dependence(ddl, "channel")
    r["dil"] = DL.build(root)
    r["conc_close"] = concentration(customer_arr(am, DL.CLOSE)["arr"])
    r["rq"] = build_rq(root)

    r["metrics"] = pd.read_csv(out / "model" / "model_comparison.csv")
    r["model_json"] = json.loads((out / "model" / "model_comparison.json").read_text())
    r["boot"] = pd.read_csv(out / "uncertainty" / "cluster_bootstrap_model_metrics.csv")
    r["paired"] = pd.read_csv(out / "uncertainty" / "cluster_bootstrap_paired_difference.csv")
    r["coefs"] = pd.read_csv(out / "model" / "logistic_coefficients.csv")
    r["stab"] = pd.read_csv(out / "model" / "penalty_coefficient_stability.csv")
    r["pen_diag"] = json.loads((out / "model" / "penalty_diagnostics.json").read_text())
    r["cells"] = pd.read_csv(out / "model" / "interaction_cells.csv")
    r["gbt_imp"] = pd.read_csv(out / "model" / "gbt_permutation_importance.csv")
    r["decisions"] = {c: promotion_decision(r["metrics"], r["paired"], c) for c in ["logistic", "logistic_interact", "gbt"]}
    pg = r["paired"]
    r["gbt_vs_logistic"] = pg[(pg["model"] == "gbt") & (pg["baseline"] == "logistic") & (pg["metric"] == "roc_auc")].iloc[0].to_dict()

    scored = pd.read_csv(out / "model" / "holdout_scored_accounts.csv")
    scored["value_logistic"] = scored["logistic_score"] * scored["mrr_amount"]
    r["scored"] = scored
    r["break_even"] = pd.DataFrame({name: break_even(scored, col) if col else break_even(scored, "x", seed=7) for name, col in TARGET_LISTS.items()}).T
    r["pilot_churn_rate"] = r["break_even"].loc["Logistic (L2)", "precision"]
    r["pilot"] = {s: pilot_size(r["pilot_churn_rate"], s) for s in (0.15, 0.20, 0.30)}
    r["new_per_month"] = r["break_even"].loc["Logistic (L2)", "targeted"] / scored["month"].nunique()

    rates = bridge_rates(r["bridge"])
    r["rates"] = rates
    r["scenarios"] = scenario_forecast(rates)
    r["fan"] = bootstrap_fan(rates)
    return r


def save_figures(r: dict, fig_dir: Path) -> dict[str, Path]:
    fig_dir.mkdir(parents=True, exist_ok=True)
    mon = r["dil"]["monitoring_by"]["segment"].reindex(["SMB", "Mid-Market", "Enterprise"])
    figs = {
        "quality": F.quality(r["quality"]),
        "mrr_bridge": F.mrr_bridge(r["bridge"]),
        "billed_vs_list": F.billed_vs_list(r["share"]),
        "discount_by_channel": F.discount_dependence(r["dd_channel_2024"], "Discount-dependent growth by channel, 2024"),
        "benchmark_gap": F.benchmark_gap(r["dil"]["benchmark_table"]),
        "plan_vs_actual": F.plan_vs_actual(r["rates"], r["dil"]["underwrite"]),
        "retention_segment": F.retention_vs_benchmark(mon, "Retention by segment, Jul 2025 to Jul 2026", {"GRR SaaS Capital 91": 91, "NRR SaaS Capital 101": 101}),
        "retention_pricing": F.retention_vs_benchmark(r["dil"]["monitoring_by"]["pricing_model"], "Retention by pricing model, Jul 2025 to Jul 2026", {"NRR Benchmarkit seat 98": 98, "NRR Benchmarkit usage 108": 108}),
        "cohort_curves": F.cohort_curves(r["cohort"]),
        "model_ladder": F.model_ladder(r["boot"]),
        "calibration": F.calibration(r["scored"], ["rule_renewal", "logistic", "gbt"]),
        "penalty_stability": F.penalty_stability(r["stab"]),
        "break_even": F.break_even(r["break_even"]),
        "forecast": F.forecast(r["rates"], r["scenarios"], r["fan"]),
        "renewal_exposure": F.renewal_exposure(r["rq"]["renewal_schedule"]),
        "lorenz": F.lorenz(r["rq"]["customer_arr"]["arr"]),
    }
    paths = {}
    for name, fig in figs.items():
        paths[name] = fig_dir / f"{name}.png"
        fig.savefig(paths[name], bbox_inches="tight")
        F.plt.close(fig)
    return paths


def headline(r: dict) -> dict[str, str]:
    d = r["dil"]["diligence_by"][None].iloc[0]
    uw = r["dil"]["underwrite"]
    m = r["metrics"].set_index("model")
    return {"ARR at close (Dec 2024)": f"${r['rates'].loc['2024-12-01', 'closing'] * 12 / 1e6:.0f}m",
            "ARR now (Jul 2026)": f"${r['rates']['closing'].iloc[-1] * 12 / 1e6:.0f}m",
            "Diligence GRR / NRR": f"{d['grr']:.1%} / {d['nrr']:.1%}",
            "Forecast error, Jul 2026": f"{uw['error'].iloc[-1]:+.1%}",
            "Holdout AUC, logistic": f"{m.loc['logistic', 'roc_auc']:.3f}",
            "Break-even save rate": f"{r['break_even'].loc['Logistic (L2)', 'break_even_save_rate']:.0%}"}


def brief_markdown(r: dict) -> str:
    dil = r["dil"]
    d, mo = dil["diligence_by"][None].iloc[0], dil["monitoring_by"][None].iloc[0]
    bt = dil["benchmark_table"].set_index("cut")
    mt = dil["benchmark_table"][(dil["benchmark_table"]["match"] != "portfolio") & ~dil["benchmark_table"]["small_sample"]].reset_index(drop=True)
    uw, cov, cal = dil["underwrite"], dil["coverage"], dil["calendar_2025"].set_index("metric")
    rp = dil["retention_plan"]
    m = r["metrics"].set_index("model").round(3)
    lc = r["model_json"]["logistic_choice"]
    pdg = r["pen_diag"]
    be = r["break_even"]["break_even_save_rate"]
    rc = r["rq"]["revenue_to_cash"].iloc[-1]
    sch, rates = r["rq"]["renewal_schedule"], r["rq"]["renewal_rates"]["non_renewal_rate"]
    fwd = [b for b in sch.index if b[:4].isdigit()]
    arr_close = r["rates"].loc["2024-12-01", "closing"] * 12
    g24 = cal.loc["ARR growth", "ours_2024"]
    unstable = pdg["unstable_selection_features_l1"]
    gl = r["gbt_vs_logistic"]
    ci24, ci25 = dil["ci_2024"], dil["ci_2025"]
    lines = [
        "# Portfolio Monitoring Brief: Acquisition of a B2B SaaS Book",
        "",
        "Scenario: a formerly bootstrapped B2B SaaS company acquired in December 2024. Diligence uses data to the close; everything after the close "
        "(Jan 2025 to Jul 2026) is out of sample. Customer data is synthetic; its Dec 2023 to Dec 2024 retention was calibrated to SaaS Capital's 2025 "
        "survey, which measured the same window, so agreement in that window is by design and is not evidence.",
        "",
        "## Diligence at close",
        "",
        f"- List ARR ${arr_close / 1e6:.0f}m at close, up {g24:.1%} in 2024. 12-month GRR {d['grr']:.1%} and NRR {d['nrr']:.1%} (customer-bootstrap 95% range {ci24['grr_lo']:.1%} to {ci24['grr_hi']:.1%} and {ci24['nrr_lo']:.1%} to {ci24['nrr_hi']:.1%}).",
        f"- Against SaaS Capital 2025 medians: all companies {bt.loc['All private B2B SaaS >$1M ARR', 'bench_grr']:.0%} / {bt.loc['All private B2B SaaS >$1M ARR', 'bench_nrr']:.0%}, bootstrapped {bt.loc['Bootstrapped', 'bench_grr']:.0%} / {bt.loc['Bootstrapped', 'bench_nrr']:.0%}. "
        f"Matched by ACV band and contract term (bands with 30+ accounts), GRR gaps run {mt['grr_gap'].min() * 100:+.1f} to {mt['grr_gap'].max() * 100:+.1f} points and NRR gaps {mt['nrr_gap'].min() * 100:+.1f} to {mt['nrr_gap'].max() * 100:+.1f}; "
        f"the largest GRR shortfall is {mt.loc[mt['grr_gap'].idxmin(), 'cut']}. The >$250k band has only {int(bt.loc['ACV >$250k', 'ours_accounts'])} accounts and is not compared.",
        f"- Loss is mostly logo churn ({d['churn_loss']:.1%} of starting ARR) rather than contraction ({d['contraction_loss']:.1%}); expansion added {d['expansion']:.1%}.",
        f"- {r['dd_2024']['new']:.1%} of 2024 new-logo list MRR and {r['dd_2024']['expansion']:.1%} of expansion were written above a 15% discount; by channel the new-logo share peaks in {r['dd_channel_2024']['new'].idxmax()} at {r['dd_channel_2024']['new'].max():.0%} "
        f"and the expansion share in {r['dd_channel_2024']['expansion'].idxmax()} at {r['dd_channel_2024']['expansion'].max():.0%}. Billed ARR grew {r['share'].loc['2024-12-01', 'billed_yoy']:.1%} in 2024 against {r['share'].loc['2024-12-01', 'list_yoy']:.1%} for list ARR, so discounting did not flatter growth.",
        f"- Concentration at close is low: top 10 customers {r['conc_close']['top10']:.1%} of ARR, HHI {r['conc_close']['hhi']:.0f}. This follows from the simulated size distribution.",
        "",
        "## Underwriting forecast vs actual",
        "",
        f"- Made at close from pre-close data only. By Jul 2026 the base case was {uw['error'].iloc[-1]:+.1%} off actual list ARR (${uw['base'].iloc[-1] * 12 / 1e6:.0f}m vs ${uw['actual'].iloc[-1] * 12 / 1e6:.0f}m). "
        f"Actual stayed inside the 90% band in {uw['in_90_band'].mean():.0%} of the 19 months and inside the 50% band in {uw['in_50_band'].mean():.0%}.",
        f"- Across {cov['origin'].nunique()} rolling origins, the same 90% band held the realised value {cov['inside_90'].mean():.1%} of the time, falling to {cov.loc[cov['horizon'] >= 10, 'inside_90'].mean():.0%} at 10 to 12 months: "
        "the band is too narrow at long horizons because trailing rates lag a rising new-logo trend. The windows overlap, so the effective sample is small.",
        f"- Retention plan (diligence GRR and NRR held flat) vs actual: GRR ran {(rp['actual_grr'].min() - rp['plan_grr'].iloc[0]) * 100:+.1f} to {(rp['actual_grr'].max() - rp['plan_grr'].iloc[0]) * 100:+.1f} points against plan, NRR {(rp['actual_nrr'].min() - rp['plan_nrr'].iloc[0]) * 100:+.1f} to {(rp['actual_nrr'].max() - rp['plan_nrr'].iloc[0]) * 100:+.1f} points.",
        f"- Calendar 2025 vs benchmarks published in 2026: our GRR {cal.loc['GRR', 'ours_2025']:.1%} and NRR {cal.loc['NRR', 'ours_2025']:.1%} vs SaaS Capital bootstrapped 91% and 103% ($3M to $20M ARR only). "
        f"Our GRR rose {(cal.loc['GRR', 'ours_2025'] - cal.loc['GRR', 'ours_2024']) * 100:+.1f} points while theirs moved about -1 point. The simulator has no market-level drift, so our move is sampling noise and cohort mix; "
        f"the customer-bootstrap ranges ({ci24['grr_lo']:.1%} to {ci24['grr_hi']:.1%} for 2024, {ci25['grr_lo']:.1%} to {ci25['grr_hi']:.1%} for 2025) show a single year's GRR for a book this size moves by roughly 1.5 to 2 points by chance.",
        f"- ARR growth in 2025 was {cal.loc['ARR growth', 'ours_2025']:.1%} vs SaaS Capital medians of 25% (equity-backed) and 20% (bootstrapped).",
        "",
        "## Quality of revenue now (Jul 2026)",
        "",
        f"- 12-month GRR {mo['grr']:.1%}, NRR {mo['nrr']:.1%}. By segment: " + "; ".join(f"{k} {v['grr']:.1%} / {v['nrr']:.1%}" for k, v in dil['monitoring_by']['segment'].reindex(['SMB', 'Mid-Market', 'Enterprise']).iterrows()) + ".",
        f"- Usage-priced accounts: NRR {dil['monitoring_by']['pricing_model'].loc['usage', 'nrr']:.1%} vs seat {dil['monitoring_by']['pricing_model'].loc['seat', 'nrr']:.1%}.",
        f"- Cash conversion {rc['cash_collected'] / rc['revenue']:.1%} in the latest quarter; {rc['failed_uncollected'] / rc['revenue']:.1%} of revenue never collected; DSO about {rc['ending_ar'] / rc['revenue'] * 91.25:.0f} days under a one-month terms assumption.",
        f"- ${sch.loc[fwd].sum().sum() / 1e6:.0f}m of committed ARR renews in the next 12 months ({sch.loc[fwd].sum().sum() / sch.sum().sum():.0%} of billed ARR).",
        "",
        "## Churn-risk models",
        "",
        f"- Trained on {r['model_json']['train_months'][0][:7]} to {r['model_json']['train_months'][1][:7]} (labels mature by close); holdout {r['model_json']['holdout_months'][0][:7]} to {r['model_json']['holdout_months'][1][:7]}, "
        f"{r['model_json']['holdout_rows']:,} account-months, 3-month churn {r['model_json']['holdout_churn_rate']:.1%}.",
        f"- Holdout ROC-AUC: hand-set rule {m.loc['rule_score', 'roc_auc']:.3f}, rule x renewal exposure {m.loc['rule_renewal', 'roc_auc']:.3f}, L2 logistic {m.loc['logistic', 'roc_auc']:.3f}, "
        f"logistic with renewal interactions {m.loc['logistic_interact', 'roc_auc']:.3f}, gradient boosted trees {m.loc['gbt', 'roc_auc']:.3f}, oracle {m.loc['oracle', 'roc_auc']:.3f}.",
        f"- Penalty choice: L1, L2 and Elastic Net score within {lc['one_se']:.3f} AUC of each other on forward folds, so the choice rests on stability. Usage and module adoption correlate at "
        f"{pdg['max_abs_correlation_pair'][2]:.2f} (VIF about {pdg['vif']['usage_ratio_3m']:.0f}); under L1, {len(unstable)} of 16 features switch in and out across customer resamples. "
        f"With about {pdg['events_per_variable']:.0f} churn events per feature, overfitting is not the binding risk. L2 was chosen (C = {lc['C']}).",
        f"- Gradient boosted trees did not beat the logistic (paired AUC difference {gl['difference']:+.3f}, 95% CI {gl['ci_lower_95']:+.3f} to {gl['ci_upper_95']:+.3f}), and explicit interaction terms added nothing. "
        "Accounts with falling usage, rising tickets and an upcoming renewal are not riskier than the additive model already predicts. The simulator's churn decision is close to additive on the log-odds scale, so this is not evidence about real data.",
    ]
    for c, x in r["decisions"].items():
        verdict = "passes" if x["promote"] else "fails " + ", ".join(k for k, v in x["checks"].items() if not v)
        lines.append(f"- Promotion test vs rule x renewal, {F.MODEL_LABEL[c]}: AUC gain {x['auc_gain']:+.3f} (95% CI {x['ci'][0]:+.3f} to {x['ci'][1]:+.3f}); {verdict}.")
    lines += [
        f"- Recommendation: replace the renewal rule with the L2 logistic; keep the trees as a monitored challenger. Predicted churn averaged {m.loc['logistic', 'mean_predicted']:.1%} against {m.loc['logistic', 'actual_rate']:.1%} actual.",
        "",
        "## Intervention economics",
        "",
        f"- Break-even save rate, top 10% list worked once per account ($400 outreach, 12 months of 75% margin, 10% concession): random {be['Random 10%']:.0%} (above 100% means it cannot pay back), rule {be['Rule score']:.0%}, rule x renewal {be['Rule x renewal']:.0%}, "
        f"L2 logistic {be['Logistic (L2)']:.0%}, trees {be['Gradient boosted trees']:.0%}, logistic ranked by expected MRR loss {be['Logistic, value-ranked']:.0%}.",
        f"- Detecting a 20% save rate at {r['pilot_churn_rate']:.1%} list churn needs about {r['pilot'][0.2]:,} accounts per arm; the list adds about {r['new_per_month']:.0f} new accounts a month, "
        f"so a single book needs about {2 * r['pilot'][0.2] / r['new_per_month']:.0f} months. A pooled pilot across portfolio companies is the realistic design.",
        "",
        "## Forecast, Aug 2026 to Jul 2027",
        "",
        f"- Base case list ARR ${r['scenarios']['Base'].iloc[-1] * 12 / 1e6:.0f}m by Jul 2027 (90% band ${r['fan']['p05'].iloc[-1] * 12 / 1e6:.0f}m to ${r['fan']['p95'].iloc[-1] * 12 / 1e6:.0f}m); "
        f"downside ${r['scenarios']['Downside'].iloc[-1] * 12 / 1e6:.0f}m; no new logos ${r['scenarios']['Run-off'].iloc[-1] * 12 / 1e6:.0f}m.",
        "",
        "## Limits",
        "",
        "- Effect sizes are simulation parameters. Benchmarks are calibration inputs in the diligence window and a check only afterwards.",
        "- SaaS Capital's samples are mostly far smaller than this book; no public source found reports private-company retention at about $150m ARR.",
        "- Discount dependence is descriptive; save offers create reverse causality. Save rates are unidentified, so economics are a hurdle, not an ROI.",
        "- No P&L is simulated, so Rule of 40 and valuation are out of scope.",
    ]
    return "\n".join(lines) + "\n"


def _img(path: Path) -> str:
    return f'<img src="data:image/png;base64,{base64.b64encode(path.read_bytes()).decode()}" alt="{path.stem}">'


def _table(df: pd.DataFrame, pct: list[str] = (), dec: list[str] = ()) -> str:
    d = df.copy()
    for c in d.columns:
        if c in pct:
            d[c] = d[c].map(lambda v: "" if pd.isna(v) else f"{v:.1%}")
        elif c in dec:
            d[c] = d[c].map(lambda v: "" if pd.isna(v) else f"{v:.3f}")
    return d.to_html(classes="t", border=0, na_rep="")


def dashboard_html(r: dict, figs: dict[str, Path]) -> str:
    h = headline(r)
    tiles = "".join(f'<div class="tile"><div class="k">{k}</div><div class="v">{v}</div></div>' for k, v in h.items() if not k.startswith("_"))
    m = r["metrics"].set_index("model")[["roc_auc", "average_precision", "brier_skill", "monthly_auc_mean", "monthly_auc_std", "top10_churn_rate", "recall_top10", "f1_top10"]]
    m.index = [F.MODEL_LABEL[i] for i in m.index]
    seg = r["dil"]["monitoring_by"]["segment"].reindex(["SMB", "Mid-Market", "Enterprise"])[["accounts", "grr", "nrr", "logo_churn"]]
    bt = r["dil"]["benchmark_table"][["cut", "ours_accounts", "ours_grr", "bench_grr", "ours_nrr", "bench_nrr"]].set_index("cut")
    be = r["break_even"][["targeted", "churners_reached", "precision", "break_even_save_rate"]]
    q = r["quality"].set_index("rule")
    brief = brief_markdown(r)
    import html as H
    brief_html = "".join(f"<h3>{H.escape(l[3:])}</h3>" if l.startswith("## ") else f"<li>{H.escape(l[2:])}</li>" if l.startswith("- ") else f"<p>{H.escape(l)}</p>" if l and not l.startswith("# ") else "" for l in brief.splitlines())
    sections = [
        ("Diligence at close", [figs["benchmark_gap"], figs["mrr_bridge"], figs["billed_vs_list"], figs["discount_by_channel"]],
         "<h4>Dec 2023 to Dec 2024 cohort vs SaaS Capital 2025</h4>" + _table(bt, pct=["ours_grr", "bench_grr", "ours_nrr", "bench_nrr"])),
        ("Underwriting vs actual", [figs["plan_vs_actual"]], ""),
        ("Quality of revenue now", [figs["lorenz"], figs["renewal_exposure"], figs["retention_segment"], figs["retention_pricing"], figs["cohort_curves"]],
         "<h4>By segment, Jul 2025 to Jul 2026</h4>" + _table(seg, pct=["grr", "nrr", "logo_churn"])),
        ("Churn-risk models", [figs["model_ladder"], figs["calibration"], figs["penalty_stability"]], _table(m, pct=["top10_churn_rate", "recall_top10"], dec=["roc_auc", "average_precision", "brier_skill", "monthly_auc_mean", "monthly_auc_std", "f1_top10"])),
        ("Intervention and forecast", [figs["break_even"], figs["forecast"]], _table(be, pct=["precision", "break_even_save_rate"])),
        ("Data quality", [figs["quality"]], _table(q)),
    ]
    body = "".join(f'<section><h2>{t}</h2><div class="grid">{"".join(_img(p) for p in ps)}</div>{extra}</section>' for t, ps, extra in sections)
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>SaaS Portfolio Monitor</title><style>
:root{{--bg:#fcfcfb;--ink:#0b0b0b;--muted:#52514e;--line:#e6e5e0;--head:#eef3fb}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font:14px/1.5 -apple-system,Segoe UI,Helvetica,Arial,sans-serif}}
main{{max-width:1180px;margin:0 auto;padding:24px 16px 64px}}h1{{font-size:24px;margin:0 0 4px}}h2{{font-size:18px;margin:36px 0 12px;padding-top:12px;border-top:1px solid var(--line)}}
h3{{font-size:14px;margin:18px 0 6px}}h4{{font-size:13px;margin:16px 0 6px;color:var(--muted)}}.sub{{color:var(--muted);margin:0 0 20px}}
.tiles{{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px}}.tile{{border:1px solid var(--line);border-radius:8px;padding:12px}}
.k{{color:var(--muted);font-size:12px}}.v{{font-size:22px;font-variant-numeric:tabular-nums}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(460px,1fr));gap:16px}}img{{width:100%;height:auto;border:1px solid var(--line);border-radius:8px;background:#fcfcfb}}
.t{{border-collapse:collapse;width:100%;font-size:12.5px;font-variant-numeric:tabular-nums;margin-top:6px;overflow-x:auto;display:block}}.t th,.t td{{padding:5px 8px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}}
.t thead th{{background:var(--head);font-weight:500}}.t th:first-child,.t td:first-child{{text-align:left}}.brief li{{margin:4px 0}}
@media (max-width:560px){{.grid{{grid-template-columns:1fr}}}}
</style></head><body><main>
<h1>SaaS Portfolio Monitor</h1><p class="sub">Synthetic B2B SaaS book acquired Dec 2024; data Jan 2022 to Jul 2026. Benchmarks are calibration inputs before close and a check after it.</p>
<div class="tiles">{tiles}</div>
<section class="brief"><h2>Brief</h2><ul>{brief_html}</ul></section>{body}
</main></body></html>"""


def pdf_report(r: dict, figs: dict[str, Path], path: Path) -> None:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import cm
    from reportlab.platypus import CondPageBreak, Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
    from xml.sax.saxutils import escape

    ss = getSampleStyleSheet()
    body = ParagraphStyle("b", parent=ss["BodyText"], fontSize=9.2, leading=12.5)
    h1, h2 = ParagraphStyle("h1", parent=ss["Title"], fontSize=17, alignment=0), ParagraphStyle("h2", parent=ss["Heading2"], fontSize=12.5, spaceBefore=10)
    W = A4[0] - 3.4 * cm
    story = []
    for line in brief_markdown(r).splitlines():
        if line.startswith("# "):
            story.append(Paragraph(escape(line[2:]), h1))
        elif line.startswith("## "):
            story.append(Paragraph(escape(line[3:]), h2))
        elif line.startswith("- "):
            story.append(Paragraph(escape(line[2:]), body, bulletText="•"))
        elif line:
            story.append(Paragraph(escape(line), body))
    h = headline(r)
    kpi = Table([[Paragraph(k, ParagraphStyle("k", fontSize=7.5, leading=9, alignment=1)) for k in h], [v for v in h.values()]], colWidths=[W / 6] * 6)
    kpi.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eef3fb")), ("FONTSIZE", (0, 0), (-1, -1), 8.5),
                             ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#e6e5e0")), ("ALIGN", (0, 0), (-1, -1), "CENTER")]))
    story.insert(2, kpi); story.insert(3, Spacer(1, 8))
    story.append(PageBreak())
    for title, names in [("Diligence at close", ["benchmark_gap", "billed_vs_list", "discount_by_channel"]),
                         ("Underwriting vs actual", ["plan_vs_actual"]),
                         ("Quality of revenue now", ["lorenz", "renewal_exposure", "retention_segment", "retention_pricing"]),
                         ("Churn-risk models", ["model_ladder", "calibration", "penalty_stability"]),
                         ("Intervention, forecast, data quality", ["break_even", "forecast", "quality"])]:
        story.append(CondPageBreak(9 * cm)); story.append(Paragraph(title, h2))
        for n in names:
            from PIL import Image as PI
            w, hh = PI.open(figs[n]).size
            scale = min(W / w, 8.5 * cm / hh)
            story.append(Image(str(figs[n]), width=w * scale, height=hh * scale)); story.append(Spacer(1, 6))
    SimpleDocTemplate(str(path), pagesize=A4, leftMargin=1.7 * cm, rightMargin=1.7 * cm, topMargin=1.5 * cm, bottomMargin=1.5 * cm,
                      title="Portfolio Monitoring Brief", author="Qiuli Lai").build(story)


def build_reports(root: Path) -> dict:
    r = compute(root)
    figs = save_figures(r, root / "outputs" / "figures")
    rep = root / "reports"
    rep.mkdir(exist_ok=True)
    (rep / "portfolio_monitoring_brief.md").write_text(brief_markdown(r), encoding="utf-8")
    (root / "dashboard").mkdir(exist_ok=True)
    (root / "dashboard" / "index.html").write_text(dashboard_html(r, figs), encoding="utf-8")
    pdf_report(r, figs, rep / "portfolio_monitoring_brief.pdf")
    from saas_growth_quality.qor_workbook import build_workbook
    build_workbook(root, rep / "quality_of_revenue.xlsx", r)
    return r
