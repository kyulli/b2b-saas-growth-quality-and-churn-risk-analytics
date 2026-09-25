"""Retention-outreach economics, framed as a break-even save rate.

The synthetic panel cannot identify how many at-risk accounts an intervention saves, so the
honest output is the save rate at which targeting a list pays for itself, and the pilot size
needed to detect that save rate. A list is worth funding if a plausible save rate clears it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

GROSS_MARGIN = 0.75
VALUE_MONTHS = 12          # months of margin a save preserves
OUTREACH_COST = 400.0      # CSM time + exec sponsor per targeted account
CONCESSION = 0.10          # discount granted to a saved account for VALUE_MONTHS


def break_even(scored: pd.DataFrame, score: str, top_share: float = 0.10, seed: int | None = None) -> dict[str, float]:
    """Each holdout month, flag the top share by score (or at random). An account is worked once, in the first
    month it is flagged; it counts as a churner if it churns within 3 months of that first flag. Counting
    account-months instead would pay for and credit the same account up to six times."""
    flags = []
    for _, m in scored.groupby("month"):
        k = int(np.ceil(top_share * len(m)))
        flags.append(m.sample(k, random_state=seed) if seed is not None else m.nlargest(k, score))
    first = pd.concat(flags).sort_values("month").drop_duplicates("customer_id", keep="first")
    at_risk = (first["churn_next_3m"] * first["mrr_amount"] * GROSS_MARGIN * VALUE_MONTHS * (1 - CONCESSION)).sum()
    cost = len(first) * OUTREACH_COST
    return {"targeted": len(first), "churners_reached": int(first["churn_next_3m"].sum()), "precision": first["churn_next_3m"].mean(),
            "margin_at_risk": at_risk, "outreach_cost": cost, "break_even_save_rate": cost / at_risk}


def pilot_size(churn_rate: float, save_rate: float, alpha: float = 0.05, power: float = 0.8) -> int:
    """Accounts per arm to detect churn falling from p to p(1 - save_rate), two-sided two-proportion z-test."""
    p1, p2 = churn_rate, churn_rate * (1 - save_rate)
    pbar = (p1 + p2) / 2
    z = norm.ppf(1 - alpha / 2) * np.sqrt(2 * pbar * (1 - pbar)) + norm.ppf(power) * np.sqrt(p1 * (1 - p1) + p2 * (1 - p2))
    return int(np.ceil(z ** 2 / (p1 - p2) ** 2))
