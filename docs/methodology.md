# Methodology

## Scenario and evidence layers

A formerly bootstrapped B2B SaaS company is acquired in December 2024. The customer panel runs from January 2022
to July 2026. Everything before the close is diligence data; everything after is out of sample.

| Layer | Source | Used for | Not used for |
|---|---|---|---|
| Customer panel | `generate.py`, seed 42, 4,547 accounts, Jan 2022 to Jul 2026 | quality, retention, revenue quality, churn models, economics, forecasts | claims about real companies |
| Benchmarks | `data/reference/saas_benchmarks.csv` | calibration targets for Dec 2023 to Dec 2024; checks after close | validating models |
| Public comps | SEC Company Facts, `data/public_comps/` | company-level context (notebook 02) | churn-model training |

The generator is calibrated so the Dec 2023 to Dec 2024 cohort lands near SaaS Capital's 2025 retention medians,
which were measured over the same window, by ACV band, contract term and funding type. Parameters are then held
fixed. `data/reference/simulation_assumptions.csv` lists every target, its evidence type and the realised value.

## Data quality

Eight defect classes are planted at a known rate and detection is scored against them. A value is repaired only
when billed = list x (1 - discount) reconstructs it; otherwise the record is quarantined. Post-clean assertions:
no identity violations, no duplicate keys, no orphans.

## Revenue metrics

Retention is measured on list-price MRR, per customer, over twelve months, on accounts active at the start of the
window (the SaaS Capital definition). GRR caps each customer at its starting MRR; NRR does not. The SQL bridge must
reconcile to the cent and match a pandas rebuild. Discount dependence is descriptive.

## Diligence and underwriting

Benchmarks are matched by customer ACV band and contract term, not by segment label. The underwriting forecast is
made at close from pre-close months only: trailing twelve-month bridge rates for the base case and a band from
resampled pre-close months. It is compared with the nineteen post-close months, and the same band method is tested
for coverage at every rolling origin.

## Churn models

Features use months t-2 to t; the label is first churn in t+1 to t+3. Training uses feature months whose labels
matured by the close; the holdout is every feature month after the close. Hyperparameters are chosen on four
forward folds with the same three-month purge.

Models: hand-set rule; rule scaled by renewal exposure; regularised logistic regression on sixteen features (L1,
L2 and Elastic Net compared on forward-fold AUC and on coefficient stability across customer resamples, with the
one-standard-error rule favouring the more stable penalty); the same logistic with renewal-exposure interaction
terms; histogram gradient boosted trees; and an oracle using the simulator's true hazard.

Promotion rule (`reporting.promotion_decision`): against the operating score, a paired AUC gain of at least 0.02
with a 95% CI above zero, Brier skill no worse, and monthly AUC standard deviation no worse by more than 0.01.
Uncertainty uses a customer-cluster bootstrap with paired draws.

## Economics and forecast

Break-even save rate = outreach cost / margin at risk among targeted churners, each account counted once at its
first flag. Pilot size from a two-proportion z-test. The forward forecast uses the same rate and band method whose
coverage was tested.
