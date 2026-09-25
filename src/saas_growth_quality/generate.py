from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


SEGMENTS = ["SMB", "Mid-Market", "Enterprise"]
INDUSTRIES = ["Fintech", "Healthcare", "Retail", "Manufacturing", "Cybersecurity", "Logistics", "Education"]
INDUSTRY_P = [0.17, 0.16, 0.14, 0.14, 0.12, 0.15, 0.12]
CHANNELS = ["Inbound", "Outbound", "Partner", "Paid", "Self-serve"]
REGIONS = ["North America", "EMEA", "APAC", "LATAM"]
REGION_P = [0.50, 0.27, 0.16, 0.07]
PRICING = ["seat", "usage", "hybrid"]
USAGE_WEIGHT = {"seat": 0.0, "hybrid": 0.5, "usage": 1.0}   # share of list MRR that moves with metered usage

# Segment structure (arrays are indexed by SEGMENTS order).
MIX_NEW, MIX_BOOK = [0.62, 0.30, 0.08], [0.55, 0.33, 0.12]
MRR_RANGE = [(300, 1_400), (1_500, 5_000), (6_000, 20_000)]
TERMS = [([1, 12], [0.55, 0.45]), ([12, 24], [0.65, 0.35]), ([12, 24, 36], [0.40, 0.35, 0.25])]
SEAT_PRICE = np.array([45.0, 60.0, 75.0])
MEAN_HEALTH = np.array([0.62, 0.68, 0.74])
CHANNEL_P = [[.25, .10, .10, .25, .30], [.30, .30, .15, .20, .05], [.20, .55, .20, .05, .00]]
PRICING_P = [[.65, .15, .20], [.50, .20, .30], [.35, .25, .40]]

# Churn hazard at a renewal decision, on the logit scale, at mean health. Calibrated so that the Dec 2023 to
# Dec 2024 cohort (the diligence window before a Dec 2024 acquisition) lands near SaaS Capital's 2025 retention
# medians by ACV band and contract term (survey measured Dec 2023 to Dec 2024). Parameters stay fixed afterwards,
# so Jan 2025 to Jul 2026 is out of sample for every forecast. Targets and realised values are listed in
# data/reference/simulation_assumptions.csv; they are calibration inputs, not validation.
CHURN_COMMITTED = np.array([-1.75, -1.85, -2.10])   # renewal of an annual or longer contract
CHURN_MONTHLY = np.array([-4.45, -4.45, -4.80])     # every month of a monthly contract is a renewal
K_HEALTH, K_DISCOUNT, K_LATE, K_SEV1, K_PROMO = 4.0, 1.2, 0.5, 0.4, 0.45
SAVE_PROB, SAVE_DISCOUNT = 0.35, 0.10               # share of would-be churners retained with a concession
EARLY_BASE, EARLY_SLOPE = 0.002, 0.03                # monthly early-termination hazard for committed contracts
EXP_REVIEW, EXP_REVIEW_SLOPE = 0.25, 0.30            # expansion probability at a commercial review
EXP_MID, EXP_MID_HEALTHY = 0.014, 0.018              # monthly mid-term expansion probability


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def generate_raw_data(
    raw_dir: Path,
    *,
    n_customers: int = 1_650,
    n_months: int = 55,
    start: str = "2022-01-01",
    monthly_arrivals: float = 44.0,
    seed: int = 42,
    defect_rate: float = 0.001,
) -> dict[str, object]:
    """Simulate a monthly B2B SaaS book (Jan 2022 to Jul 2026 by default) and inject known data defects.

    `n_customers` is the installed base at the start of the panel; new logos arrive every month after that.
    """
    raw_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    months = pd.date_range(start, periods=n_months, freq="MS")
    tables = _simulate(rng, months, n_customers, monthly_arrivals)
    oracle = tables.pop("_oracle")
    oracle_dir = raw_dir.parent / "oracle"
    oracle_dir.mkdir(parents=True, exist_ok=True)
    oracle.to_csv(oracle_dir / "oracle_churn_3m.csv", index=False)
    tables, injected = _inject_defects(tables, rng, defect_rate)
    (raw_dir / "injected_defects.json").write_text(json.dumps(injected, indent=2), encoding="utf-8")
    for name, table in tables.items():
        table.to_csv(raw_dir / f"{name}.csv", index=False)
    return {"rows": {name: len(table) for name, table in tables.items()}, "injected": injected}


def _simulate(rng: np.random.Generator, months: pd.DatetimeIndex, n_initial: int, monthly_arrivals: float) -> dict[str, pd.DataFrame]:
    T = len(months)
    trend = 1.0 + 0.10 * np.arange(T) / 12
    season = 1.0 + 0.15 * np.cos(2 * np.pi * (months.month.to_numpy() - 11) / 12)
    arrivals = rng.poisson(monthly_arrivals * trend * season)
    N = n_initial + int(arrivals.sum())
    is_book = np.arange(N) < n_initial
    signup = np.concatenate([-rng.integers(1, 37, n_initial), np.repeat(np.arange(T), arrivals)])   # month index; book is negative

    seg = np.where(is_book, rng.choice(3, N, p=MIX_BOOK), rng.choice(3, N, p=MIX_NEW))
    term, list0 = np.empty(N, dtype=int), np.empty(N)
    channel, pricing = np.empty(N, dtype=object), np.empty(N, dtype=object)
    for k in range(3):
        idx = np.flatnonzero(seg == k)
        term[idx] = rng.choice(TERMS[k][0], idx.size, p=TERMS[k][1])
        list0[idx] = np.exp(rng.uniform(np.log(MRR_RANGE[k][0]), np.log(MRR_RANGE[k][1]), idx.size)).round(2)
        channel[idx] = rng.choice(CHANNELS, idx.size, p=CHANNEL_P[k])
        pricing[idx] = rng.choice(PRICING, idx.size, p=PRICING_P[k])
    industry = rng.choice(INDUSTRIES, N, p=INDUSTRY_P)
    region = rng.choice(REGIONS, N, p=REGION_P)
    committed = term >= 12
    wu = np.array([USAGE_WEIGHT[p] for p in pricing])

    p_disc = 0.20 + 0.20 * np.isin(channel, ["Outbound", "Paid"]) + 0.10 * (term >= 24) + 0.10 * (seg == 2)
    disc0 = np.where(rng.random(N) < p_disc, rng.choice([.05, .10, .15, .20, .25, .30, .35], N, p=[.12, .28, .22, .16, .12, .07, .03]), 0.0)
    age0 = np.where(is_book, -signup, 0)
    promo = (disc0 > 0) & (rng.random(N) < 0.40) & (age0 < term)     # promotional discounts lapse at the first renewal
    price_sens = np.clip(rng.beta(2, 4, N) + 0.5 * disc0, 0, 1)
    mu = MEAN_HEALTH[seg]
    health = rng.beta(mu * 8, (1 - mu) * 8)

    cur_list, cur_disc = list0.copy(), disc0.copy()
    cur_seats = np.maximum(3.0, np.round(list0 / SEAT_PRICE[seg]))
    usage_mult = np.ones(N)
    churned, churn_idx = np.zeros(N, dtype=bool), np.full(N, -1)
    prev_late, prev_sev1 = np.zeros(N), np.zeros(N)
    prev_list = np.where(is_book, list0, 0.0)
    contract_no = np.zeros(N, dtype=int)
    contract_frames: list[pd.DataFrame] = []
    ids = np.array([f"C{i:05d}" for i in range(N)])

    sub_f, use_f, pay_f, evt_f, orc_f = [], [], [], [], []

    def add_events(mask, month, kind, reason, before, after, disc_before, disc_after):
        idx = np.flatnonzero(mask)
        if idx.size:
            evt_f.append(pd.DataFrame({"customer_id": ids[idx], "month": month, "event_type": kind if isinstance(kind, str) else kind[idx],
                                       "reason": reason if isinstance(reason, str) else reason[idx],
                                       "list_mrr_before": np.round(before[idx], 2), "list_mrr_after": np.round(after[idx], 2),
                                       "discount_before": disc_before[idx], "discount_after": disc_after[idx]}))

    def add_contracts(mask, month_start):
        idx = np.flatnonzero(mask)
        if not idx.size:
            return
        contract_no[idx] += 1
        start = np.asarray(month_start, dtype="datetime64[M]")
        start = np.broadcast_to(start, idx.shape) if np.ndim(start) == 0 else start[idx]
        end = (start + term[idx].astype("timedelta64[M]")).astype("datetime64[D]") - np.timedelta64(1, "D")
        contract_frames.append(pd.DataFrame({
            "contract_id": [f"K{ids[i][1:]}-{contract_no[i]}" for i in idx], "customer_id": ids[idx],
            "start_date": start.astype("datetime64[D]"), "end_date": end, "term_months": term[idx],
            "list_mrr": np.round(cur_list[idx], 2), "discount_pct": cur_disc[idx]}))

    for t, month in enumerate(months):
        msi = t - signup                                                    # months since signup
        started, new = signup <= t, signup == t
        cont = started & ~churned & ~new & (t > 0)
        renew = cont & (msi % term == 0)
        promo_roll = renew & promo

        # --- churn decision at the start of the month, from what was observable last month
        base = np.where(committed, CHURN_COMMITTED[seg] - 0.2 * (term / 12 - 1), CHURN_MONTHLY[seg])
        z = base + K_HEALTH * (0.55 - health) + K_DISCOUNT * cur_disc + K_LATE * (prev_late > 20) + K_SEV1 * (prev_sev1 > 0) + K_PROMO * promo_roll
        p_early = EARLY_BASE + EARLY_SLOPE * np.maximum(0.3 - health, 0)
        p = np.where(renew, _sigmoid(z), np.where(cont & committed, p_early, 0.0))
        would_churn = cont & (rng.random(N) < p)
        saved = would_churn & renew & (price_sens > 0.25) & (rng.random(N) < SAVE_PROB / 0.75)
        lost = would_churn & ~saved
        disc_before, list_before = cur_disc.copy(), prev_list.copy()
        cur_disc = np.where(saved, np.minimum(cur_disc + SAVE_DISCOUNT, 0.40), cur_disc)
        promo = np.where(saved, False, promo)
        health = np.where(saved, np.minimum(health + 0.10, 1.0), health)
        add_events(saved, month, "discount_change", "save_discount", cur_list, cur_list, disc_before, cur_disc)
        add_events(lost, month, "churn", np.where(renew, "non_renewal", "early_termination"), list_before, np.zeros(N), disc_before, np.zeros(N))
        churned |= lost
        churn_idx = np.where(lost, t, churn_idx)

        # --- survivors: renewal terms, mid-term changes, health, usage drift
        alive = started & ~churned
        surv = cont & ~lost
        ren_s = renew & surv
        expire = promo_roll & ~lost & ~saved
        d0 = cur_disc.copy()
        cur_disc = np.where(expire, np.where(rng.random(N) < 0.7, 0.0, cur_disc * 0.5), cur_disc)
        promo = np.where(expire, False, promo)
        add_events(expire, month, "discount_change", "promo_expiry", cur_list, cur_list, d0, cur_disc)

        pre_list = cur_list * (1 + wu * (usage_mult - 1))
        # Commercial reviews (price uplift, renewal-sized seat changes) happen at contract renewal for committed
        # accounts and on the 12-month anniversary for monthly accounts, not at every monthly auto-renewal.
        review = surv & np.where(committed, renew, (msi % 12 == 0))
        uplift = review & (rng.random(N) < 0.35)
        cur_list = np.where(uplift, cur_list * (1 + rng.uniform(0.03, 0.07, N) * np.where(health > 0.35, 1.0, 0.5)), cur_list)
        u1, u2 = rng.random(N), rng.random(N)
        p_exp = np.where(review, EXP_REVIEW + EXP_REVIEW_SLOPE * np.maximum(health - 0.55, 0), EXP_MID + EXP_MID_HEALTHY * (health > 0.6))
        p_con = np.where(review, 0.06 + 0.35 * np.maximum(0.45 - health, 0), 0.001 + 0.004 * (health < 0.4))
        grow = surv & (u1 < p_exp)
        shrink = surv & ~grow & (u2 < p_con)
        factor = np.where(grow, rng.uniform(1.08, 1.40, N), np.where(shrink, rng.uniform(0.65, 0.92, N), 1.0))
        factor = np.where(grow & ~review, rng.uniform(1.03, 1.15, N), np.where(shrink & ~review, rng.uniform(0.85, 0.97, N), factor))
        cur_list, cur_seats = cur_list * factor, np.maximum(1.0, cur_seats * factor)
        drift = wu * (0.006 * (2 * health - 1) + 0.05 * rng.standard_normal(N))
        usage_mult = np.where(surv, np.clip(usage_mult * np.exp(drift), 0.4, 3.0), usage_mult)
        now_list = np.where(alive, cur_list * (1 + wu * (usage_mult - 1)), 0.0)
        reason = np.select([grow | shrink, uplift], ["seat_change", "price_uplift"], default="usage_change")
        moved = surv & (np.abs(now_list / np.where(pre_list > 0, pre_list, 1) - 1) >= 0.03)
        add_events(moved, month, np.where(now_list > pre_list, "expansion", "contraction"), reason, pre_list, now_list, cur_disc, cur_disc)
        add_contracts(ren_s, month.to_datetime64())

        upd = surv | new
        shock = health + 0.06 * (MEAN_HEALTH[seg] - health) + 0.05 * rng.standard_normal(N) - (rng.random(N) < 0.006) * rng.uniform(0.15, 0.35, N)
        health = np.where(surv, np.clip(shock, 0.02, 1.0), health)

        # --- new logos this month
        if new.any():
            now_list = np.where(new, cur_list * (1 + wu * (usage_mult - 1)), now_list)
            add_events(new, month, "new_logo", "signup", np.zeros(N), now_list, np.zeros(N), cur_disc)
            add_contracts(new, month.to_datetime64())
        if t == 0:                                                           # installed base: contract phase inferred from age
            book = started & ~churned
            now_list = np.where(book, cur_list, now_list)
            add_contracts(book, (np.datetime64(month, "M") - (msi % term).astype("timedelta64[M]")))

        # --- observables for active accounts
        act = started & ~churned
        usage = np.clip(health + 0.07 * rng.standard_normal(N), 0.02, 1.0)
        tickets = rng.poisson((0.4 + 3.2 * (1 - usage)) * (1 + 0.4 * seg))
        sev1 = rng.poisson(0.02 + 0.6 * np.maximum(0.5 - health, 0))
        late = rng.random(N) < np.clip(0.03 + 0.30 * (1 - health) + 0.15 * cur_disc, 0.02, 0.6)
        days_late = np.where(late, rng.integers(5, 46, N), 0)
        failed = ~late & (rng.random(N) < 0.01 + 0.03 * (1 - health))
        nps = np.where((msi % 3 == 0) & (rng.random(N) < 0.35), np.round(10 * np.clip(health + 0.15 * rng.standard_normal(N), 0, 1)), np.nan)
        billed = np.round(now_list * (1 - cur_disc), 2)
        to_renewal = term - (msi % term)

        sub_f.append(pd.DataFrame({
            "customer_id": ids[started], "month": month,
            "mrr_amount": np.where(act, billed, 0.0)[started], "discount_pct": np.where(act, cur_disc, 0.0)[started],
            "status": np.where(act, "active", "churned")[started], "list_mrr": np.round(np.where(act, now_list, 0.0), 2)[started],
            "seats": np.where(act, np.round(cur_seats), 0)[started], "promo_discount": (promo & act)[started],
            "months_to_renewal": to_renewal[started], "is_renewal_month": renew[started]}))
        use_f.append(pd.DataFrame({
            "customer_id": ids[act], "month": month, "active_users": np.maximum(1, np.round(cur_seats * usage)).astype(int)[act],
            "usage_ratio": np.round(usage, 4)[act], "ticket_count": tickets[act], "sev1_tickets": sev1[act],
            "modules_adopted": np.round(np.clip(0.15 + 0.75 * health + 0.08 * rng.standard_normal(N), 0, 1), 3)[act], "nps_score": nps[act]}))
        pay_f.append(pd.DataFrame({
            "payment_id": [f"P{i[1:]}{month:%Y%m}" for i in ids[act]], "customer_id": ids[act], "month": month,
            "invoice_amount": billed[act], "days_late": days_late[act],
            "payment_status": np.where(late, "late", np.where(failed, "failed", "paid"))[act]}))
        prev_late, prev_sev1, prev_list = np.where(act, days_late, 0), np.where(act, sev1, 0), np.where(act, now_list, 0.0)
        # Oracle: 3-month churn probability from the true latent state and hazard, holding today's state fixed.
        # It is written outside data/raw and is used only as a performance ceiling, never as a feature.
        keep = 1.0 - (price_sens > 0.25) * SAVE_PROB / 0.75
        survive = np.ones(N)
        for k in (1, 2, 3):
            ren_k = (msi + k) % term == 0
            z_k = base + K_HEALTH * (0.55 - health) + K_DISCOUNT * cur_disc + K_LATE * (days_late > 20) * (k == 1) \
                + K_SEV1 * (sev1 > 0) * (k == 1) + K_PROMO * (ren_k & promo)
            p_k = np.where(ren_k, _sigmoid(z_k) * keep, np.where(committed, EARLY_BASE + EARLY_SLOPE * np.maximum(0.3 - health, 0), 0.0))
            survive *= 1 - p_k
        orc_f.append(pd.DataFrame({"customer_id": ids[act], "month": month, "oracle_churn_3m": np.round(1 - survive, 5)[act]}))

    customers = pd.DataFrame({
        "customer_id": ids, "signup_date": (months[0].to_datetime64().astype("datetime64[M]") + signup.astype("timedelta64[M]")).astype("datetime64[D]").astype(str),
        "segment": np.array(SEGMENTS)[seg], "industry": industry, "contract_term_months": term, "base_mrr": list0,
        "region": region, "channel": channel, "pricing_model": pricing, "initial_discount_pct": disc0})
    contracts = pd.concat(contract_frames, ignore_index=True).sort_values(["customer_id", "start_date"]).reset_index(drop=True)
    contracts["outcome"] = _contract_outcomes(contracts, months, churn_idx, ids)
    return {"customers": customers, "subscriptions": pd.concat(sub_f, ignore_index=True), "usage": pd.concat(use_f, ignore_index=True),
            "payments": pd.concat(pay_f, ignore_index=True), "events": pd.concat(evt_f, ignore_index=True), "contracts": contracts,
            "_oracle": pd.concat(orc_f, ignore_index=True)}


def _contract_outcomes(contracts: pd.DataFrame, months: pd.DatetimeIndex, churn_idx: np.ndarray, ids: np.ndarray) -> np.ndarray:
    """renewed / churned_at_renewal / ended_early / active, from the contract sequence and each customer's churn month."""
    churn_month = pd.Series(churn_idx, index=ids)[contracts["customer_id"]].to_numpy()
    last = ~contracts["customer_id"].duplicated(keep="last").to_numpy()
    end_next = (contracts["end_date"] + pd.Timedelta(days=1)).dt.to_period("M").dt.to_timestamp()
    end_idx = ((end_next.dt.year - months[0].year) * 12 + end_next.dt.month - months[0].month).to_numpy()
    outcome = np.where(~last, "renewed", np.where(churn_month < 0, "active", np.where(churn_month == end_idx, "churned_at_renewal", "ended_early")))
    return outcome


def _inject_defects(tables: dict[str, pd.DataFrame], rng: np.random.Generator, rate: float) -> tuple[dict[str, pd.DataFrame], dict[str, int]]:
    """Plant known defects at `rate` per customer (and rate/5 per subscription row) so detection can be scored."""
    t = {name: frame.copy() for name, frame in tables.items()}
    customers, subs, pays, contracts = t["customers"], t["subscriptions"], t["payments"], t["contracts"]
    k = max(1, round(rate * len(customers)))
    m = max(1, round(rate / 5 * len(subs)))
    pick = rng.permutation(len(customers))[: 2 * k]
    dup, miss = pick[:k], pick[k:]
    future = customers.iloc[:k].copy().assign(customer_id=[f"C_FUTURE{i}" for i in range(k)], signup_date="2028-01-01")
    customers.loc[customers.index[miss], "industry"] = np.nan
    t["customers"] = pd.concat([customers, customers.iloc[dup], future], ignore_index=True)

    active = np.flatnonzero((subs["mrr_amount"] > 0).to_numpy())
    rows = rng.permutation(active)[: 3 * m]
    neg, dup_rows, bad_disc = rows[:m], rows[m:2 * m], rows[2 * m:]
    subs.loc[subs.index[neg], "mrr_amount"] *= -1
    subs.loc[subs.index[bad_disc], "discount_pct"] = np.where(rng.random(len(bad_disc)) < 0.5, 1.4, -0.05)
    t["subscriptions"] = pd.concat([subs, subs.iloc[dup_rows]], ignore_index=True)

    orphans = pays.iloc[:k].assign(customer_id=[f"C_ORPHAN{i}" for i in range(k)], payment_id=[f"P_ORPHAN{i}" for i in range(k)])
    t["payments"] = pd.concat([pays, orphans], ignore_index=True)
    inverted = rng.permutation(len(contracts))[:k]
    contracts.loc[contracts.index[inverted], ["start_date", "end_date"]] = contracts.loc[contracts.index[inverted], ["end_date", "start_date"]].to_numpy()
    injected = {"duplicate_customer_id": k, "missing_industry": k, "future_signup": k, "negative_mrr": m, "orphan_foreign_key": k,
                "duplicate_subscription_month": m, "discount_out_of_range": m, "contract_date_inversion": k}
    return t, injected
