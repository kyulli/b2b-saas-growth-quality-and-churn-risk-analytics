"""Shared figures for the notebook, dashboard, and PDF. Each function returns a matplotlib Figure."""
from __future__ import annotations

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e0"
plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 150, "font.size": 9.5, "axes.titlesize": 11, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "axes.edgecolor": GRID, "axes.labelcolor": MUTED, "xtick.color": MUTED,
    "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
    "axes.spines.right": False, "legend.frameon": False, "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
    "axes.prop_cycle": matplotlib.cycler(color=C)})
MODEL_LABEL = {"rule_score": "Rule score (hand-set)", "rule_calibrated": "Rule, Platt-calibrated", "rule_renewal": "Rule x renewal exposure",
               "logistic": "Logistic, L2 (16 features)", "logistic_interact": "Logistic + renewal interactions",
               "gbt": "Gradient boosted trees", "oracle": "Oracle (true hazard)"}


def mrr_bridge(bridge: pd.DataFrame) -> plt.Figure:
    b = bridge.assign(month=pd.to_datetime(bridge["month"])).set_index("month") / 1e3
    fig, ax = plt.subplots(figsize=(9, 3.6))
    x, w = b.index, 20
    ax.bar(x, b["new_list"], w, color=C[0], label="New")
    ax.bar(x, b["expansion_list"], w, bottom=b["new_list"], color=C[2], label="Expansion")
    ax.bar(x, b["contraction_list"], w, color=C[3], label="Contraction")
    ax.bar(x, b["churn_list"], w, bottom=b["contraction_list"], color=C[7], label="Churn")
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_title("Monthly list-MRR bridge ($k)")
    ax.legend(ncol=4, loc="upper left")
    return fig


def billed_vs_list(share: pd.DataFrame) -> plt.Figure:
    s = share.copy(); s.index = pd.to_datetime(s.index)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
    axes[0].plot(s.index, s["list_yoy"] * 100, color=C[0], lw=2, label="List MRR")
    axes[0].plot(s.index, s["billed_yoy"] * 100, color=C[1], lw=2, label="Billed MRR")
    axes[0].set_title("YoY MRR growth (%)"); axes[0].legend()
    axes[1].plot(s.index, s["discounted_mrr_share"] * 100, color=C[6], lw=2)
    axes[1].set_title("Discounted MRR share (% of list)"); axes[1].set_ylim(0, 10)
    import matplotlib.dates as mdates
    for ax in axes:
        ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1])); ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.tight_layout()
    return fig


def discount_dependence(table: pd.DataFrame, title: str) -> plt.Figure:
    t = table.sort_values("new") * 100
    fig, ax = plt.subplots(figsize=(6.5, 0.45 * len(t) + 1.2))
    y = np.arange(len(t))
    ax.barh(y - 0.2, t["new"], 0.38, color=C[0], label="New logo MRR")
    ax.barh(y + 0.2, t["expansion"], 0.38, color=C[2], label="Expansion MRR")
    ax.set_yticks(y, t.index.astype(str)); ax.set_xlabel("% of gross MRR written at >15% discount")
    ax.set_title(title); ax.legend(loc="lower right")
    return fig


def retention_vs_benchmark(ret: pd.DataFrame, title: str, bench: dict[str, float]) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(9, 0.4 * len(ret) + 1.6), sharey=True)
    y = np.arange(len(ret))
    for ax, col, colr in zip(axes, ["grr", "nrr"], [C[0], C[2]]):
        ax.barh(y, ret[col] * 100, 0.6, color=colr)
        for i, v in enumerate(ret[col] * 100):
            ax.text(v + 0.3, i, f"{v:.1f}", va="center", color=INK, fontsize=8.5)
        for k, (name, v) in enumerate((n, v) for n, v in bench.items() if n.startswith(col.upper())):
            ax.axvline(v, color=MUTED, lw=1, ls="--" if k == 0 else ":")
            ax.text(v + 0.3, -0.75 - 0.3 * k, name.split(" ", 1)[1], color=MUTED, fontsize=7.5, ha="left", va="center")
        ax.set_xlim(60 if col == "grr" else 80, 112); ax.set_ylim(-1.2, len(ret) - 0.5); ax.set_title(f"12-month {col.upper()} (%)")
    axes[0].set_yticks(y, ret.index.astype(str))
    fig.suptitle(title, x=0.01, ha="left", fontweight="bold", fontsize=11)
    fig.tight_layout()
    return fig


def cohort_curves(cohort: pd.DataFrame) -> plt.Figure:
    c = cohort.groupby(["segment", "months_since_signup"])[["cohort_customers", "retained_customers"]].sum()
    c = c[c["cohort_customers"] >= 100]
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    for i, seg in enumerate(["SMB", "Mid-Market", "Enterprise"]):
        s = c.loc[seg]
        ax.plot(s.index, s["retained_customers"] / s["cohort_customers"] * 100, color=C[i], lw=2, label=seg)
    ax.set_xlabel("Months since signup"); ax.set_title("Logo retention by signup cohort (%)"); ax.legend()
    return fig


def model_ladder(metrics: pd.DataFrame) -> plt.Figure:
    m = metrics[(metrics["metric"] == "roc_auc") & (metrics["model"] != "rule_calibrated")].sort_values("estimate")
    fig, ax = plt.subplots(figsize=(7, 3.2))
    y = np.arange(len(m))
    colors = [MUTED if n == "oracle" else C[0] if n.startswith("rule") else C[2] for n in m["model"]]
    ax.errorbar(m["estimate"], y, xerr=[m["estimate"] - m["ci_lower_95"], m["ci_upper_95"] - m["estimate"]],
                fmt="none", ecolor=MUTED, elinewidth=1.2, capsize=3)
    ax.scatter(m["estimate"], y, s=60, c=colors, zorder=3, edgecolor="#fcfcfb", linewidth=2)
    for i, v in enumerate(m["estimate"]):
        ax.text(v, i + 0.25, f"{v:.3f}", ha="center", fontsize=8, color=INK)
    ax.set_yticks(y, [MODEL_LABEL[n] for n in m["model"]]); ax.set_xlim(0.5, 0.88)
    ax.set_title("Holdout ROC-AUC, customer-cluster 95% CI")
    return fig


def calibration(scored: pd.DataFrame, models: list[str]) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(4.6, 4))
    top = 0
    for i, n in enumerate(models):
        q = pd.qcut(scored[f"{n}_score"].rank(method="first"), 10, labels=False)
        g = scored.groupby(q).agg(p=(f"{n}_score", "mean"), y=("churn_next_3m", "mean")) * 100
        ax.plot(g["p"], g["y"], "-o", ms=4, lw=1.6, color=C[i], label=MODEL_LABEL[n]); top = max(top, g.max().max())
    ax.plot([0, top], [0, top], color=MUTED, lw=1, ls="--")
    ax.set_xlabel("Predicted 3-month churn (%)"); ax.set_ylabel("Observed (%)"); ax.set_title("Calibration by decile"); ax.legend(fontsize=7.5)
    return fig


def break_even(be: pd.DataFrame) -> plt.Figure:
    b = be.sort_values("break_even_save_rate", ascending=False)
    fig, ax = plt.subplots(figsize=(6.5, 0.42 * len(b) + 1.2))
    ax.barh(np.arange(len(b)), b["break_even_save_rate"] * 100, 0.6, color=[MUTED if "Random" in i else C[0] for i in b.index])
    for i, v in enumerate(b["break_even_save_rate"] * 100):
        ax.text(v + 0.8, i, f"{v:.0f}%", va="center", fontsize=8.5)
    ax.set_yticks(np.arange(len(b)), b.index); ax.set_xlabel("Save rate needed to pay back outreach (%)")
    ax.set_title("Break-even save rate, top 10% list")
    return fig


def forecast(rates: pd.DataFrame, scen: pd.DataFrame, band: pd.DataFrame) -> plt.Figure:
    """List ARR (MRR x 12) actuals and scenarios, in $m, to match the report tables."""
    fig, ax = plt.subplots(figsize=(9, 3.6))
    k = 12 / 1e6
    hist = rates["closing"].iloc[-18:] * k
    ax.plot(hist.index, hist, color=INK, lw=2, label="Actual")
    ax.fill_between(band.index, band["p05"] * k, band["p95"] * k, color=C[0], alpha=0.15, lw=0, label="90% resampled-month band")
    for i, n in enumerate(["Upside", "Base", "Downside", "Run-off"]):
        ax.plot(scen.index, scen[n] * k, color=[C[2], C[0], C[3], C[7]][i], lw=2 if n == "Base" else 1.5, ls="-" if n == "Base" else "--", label=n)
    ax.set_title("List ARR scenarios, Aug 2026 to Jul 2027 ($m)"); ax.legend(ncol=3, fontsize=8, loc="upper left")
    return fig


def quality(summary: pd.DataFrame) -> plt.Figure:
    s = summary.set_index("rule")
    fig, ax = plt.subplots(figsize=(6.5, 3))
    y = np.arange(len(s))
    ax.barh(y - 0.2, s["injected"], 0.38, color=MUTED, label="Injected")
    ax.barh(y + 0.2, s["detected"], 0.38, color=C[0], label="Detected")
    ax.set_yticks(y, s.index); ax.set_title("Planted defects vs detected"); ax.legend(loc="lower right")
    return fig


def renewal_exposure(schedule: pd.DataFrame) -> plt.Figure:
    order = [b for b in schedule.index if b[:4].isdigit()] + ["Monthly (any month)", "Beyond 12 months"]
    s = schedule.reindex(order).fillna(0) / 1e6
    fig, ax = plt.subplots(figsize=(7.5, 3.4))
    bottom = np.zeros(len(s))
    for i, seg in enumerate(["Enterprise", "Mid-Market", "SMB"]):
        ax.bar(range(len(s)), s[seg], 0.6, bottom=bottom, color=C[i], label=seg, edgecolor="#fcfcfb", linewidth=1)
        bottom += s[seg].to_numpy()
    rename = {"Monthly (any month)": "Monthly", "Beyond 12 months": ">12m", "2026Q3": "2026Q3\n(Aug-Sep)", "2027Q3": "2027Q3\n(Jul)"}
    labels = [rename.get(b, b) for b in s.index]
    ax.set_axisbelow(True); ax.set_ylim(0, bottom.max() * 1.18)
    ax.set_xticks(range(len(s)), labels, fontsize=8.5)
    ax.set_title("Billed ARR by renewal window, from Jul 2026 ($m)"); ax.legend(ncol=3, loc="upper left")
    return fig


def lorenz(arr: pd.Series) -> plt.Figure:
    a = np.sort(arr.to_numpy())[::-1]
    cum = np.cumsum(a) / a.sum() * 100
    x = np.arange(1, len(a) + 1) / len(a) * 100
    fig, ax = plt.subplots(figsize=(4.6, 4))
    ax.plot(x, cum, color=C[0], lw=2)
    ax.plot([0, 100], [0, 100], color=MUTED, lw=1, ls="--")
    for p in (10, 20):
        k = int(np.ceil(p / 100 * len(a)))
        ax.scatter([p], [cum[k - 1]], color=C[0], s=30, zorder=3)
        ax.text(p + 2, cum[k - 1] - 4, f"top {p}%: {cum[k - 1]:.0f}% of ARR", fontsize=8)
    ax.set_xlabel("Customers, largest first (%)"); ax.set_ylabel("Cumulative billed ARR (%)"); ax.set_title("ARR concentration curve")
    return fig


def benchmark_gap(t: pd.DataFrame) -> plt.Figure:
    """Our Dec 2023 to Dec 2024 cohort vs SaaS Capital 2025 medians on matched cuts."""
    d = t[~t["small_sample"]].iloc[::-1].reset_index(drop=True)
    fig, axes = plt.subplots(1, 2, figsize=(9, 0.42 * len(d) + 1.4), sharey=True)
    y = np.arange(len(d))
    for ax, a, b, title in [(axes[0], "ours_grr", "bench_grr", "12-month GRR (%)"), (axes[1], "ours_nrr", "bench_nrr", "12-month NRR (%)")]:
        for k in range(len(d)):
            ax.plot([d[b][k] * 100, d[a][k] * 100], [k, k], color=GRID, lw=2, zorder=1)
        ax.scatter(d[b] * 100, y, s=46, color=MUTED, label="SaaS Capital 2025 median", zorder=2)
        ax.scatter(d[a] * 100, y, s=46, color=C[0], label="Our cohort", zorder=3, edgecolor="#fcfcfb", linewidth=1.5)
        ax.set_title(title)
    # Escape "$" so matplotlib does not read "$12k-$25k" as mathtext.
    axes[0].set_yticks(y, [c.replace("$", r"\$") for c in d["cut"]])
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", ncol=2, fontsize=8, frameon=False)
    fig.suptitle("Diligence benchmark, Dec 2023 to Dec 2024", x=0.01, ha="left", fontweight="bold", fontsize=11)
    fig.tight_layout()
    return fig


def plan_vs_actual(rates: pd.DataFrame, uw: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(9, 3.6))
    hist = rates["closing"][(rates.index >= "2023-07-01")] * 12 / 1e6
    ax.plot(hist.index, hist, color=INK, lw=2, label="Actual list ARR")
    ax.fill_between(uw.index, uw["p05"] * 12 / 1e6, uw["p95"] * 12 / 1e6, color=C[0], alpha=0.15, lw=0, label="90% band set at close")
    ax.plot(uw.index, uw["base"] * 12 / 1e6, color=C[0], lw=2, ls="--", label="Underwriting base case")
    ax.axvline(pd.Timestamp("2024-12-01"), color=MUTED, lw=1, ls=":")
    ax.text(pd.Timestamp("2024-12-10"), hist.min(), "close", color=MUTED, fontsize=8)
    ax.set_title("Underwriting forecast vs actual ($m ARR)"); ax.legend(fontsize=8, loc="upper left")
    return fig


def penalty_stability(stab: pd.DataFrame) -> plt.Figure:
    p = stab.pivot(index="feature", columns="penalty", values="selected_share")
    p = p.loc[p["l1"].sort_values().index]
    fig, ax = plt.subplots(figsize=(6.5, 0.32 * len(p) + 1.2))
    y = np.arange(len(p))
    for k, (col, lab, colr) in enumerate([("l1", "L1", C[1]), ("elasticnet", "Elastic Net", C[3]), ("l2", "L2", C[0])]):
        ax.barh(y + (k - 1) * 0.27, p[col] * 100, 0.26, color=colr, label=lab)
    ax.set_yticks(y, p.index, fontsize=8); ax.set_xlabel("Share of customer-bootstrap fits keeping the feature (%)")
    ax.set_title("Coefficient selection stability by penalty"); ax.legend(fontsize=8, loc="lower right")
    return fig
