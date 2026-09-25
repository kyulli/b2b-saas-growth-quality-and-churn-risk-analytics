# Portfolio Monitoring Brief: Acquisition of a B2B SaaS Book

Scenario: a formerly bootstrapped B2B SaaS company acquired in December 2024. Diligence uses data to the close; everything after the close (Jan 2025 to Jul 2026) is out of sample. Customer data is synthetic; its Dec 2023 to Dec 2024 retention was calibrated to SaaS Capital's 2025 survey, which measured the same window, so agreement in that window is by design and is not evidence.

## Diligence at close

- List ARR $113m at close, up 21.2% in 2024. 12-month GRR 90.9% and NRR 102.7% (customer-bootstrap 95% range 89.2% to 92.7% and 100.5% to 105.0%).
- Against SaaS Capital 2025 medians: all companies 91% / 101%, bootstrapped 92% / 104%. Matched by ACV band and contract term (bands with 30+ accounts), GRR gaps run -2.4 to +1.9 points and NRR gaps -2.9 to +3.2; the largest GRR shortfall is Multi-year contracts. The >$250k band has only 25 accounts and is not compared.
- Loss is mostly logo churn (7.6% of starting ARR) rather than contraction (1.4%); expansion added 11.8%.
- 13.2% of 2024 new-logo list MRR and 13.0% of expansion were written above a 15% discount; by channel the new-logo share peaks in Outbound at 15% and the expansion share in Partner at 22%. Billed ARR grew 21.6% in 2024 against 21.2% for list ARR, so discounting did not flatter growth.
- Concentration at close is low: top 10 customers 3.7% of ARR, HHI 11. This follows from the simulated size distribution.

## Underwriting forecast vs actual

- Made at close from pre-close data only. By Jul 2026 the base case was -1.0% off actual list ARR ($145m vs $147m). Actual stayed inside the 90% band in 100% of the 19 months and inside the 50% band in 42%.
- Across 29 rolling origins, the same 90% band held the realised value 84.0% of the time, falling to 73% at 10 to 12 months: the band is too narrow at long horizons because trailing rates lag a rising new-logo trend. The windows overlap, so the effective sample is small.
- Retention plan (diligence GRR and NRR held flat) vs actual: GRR ran +1.2 to +2.2 points against plan, NRR +0.4 to +1.6 points.
- Calendar 2025 vs benchmarks published in 2026: our GRR 92.5% and NRR 103.2% vs SaaS Capital bootstrapped 91% and 103% ($3M to $20M ARR only). Our GRR rose +1.6 points while theirs moved about -1 point. The simulator has no market-level drift, so our move is sampling noise and cohort mix; the customer-bootstrap ranges (89.2% to 92.7% for 2024, 91.1% to 93.7% for 2025) show a single year's GRR for a book this size moves by roughly 1.5 to 2 points by chance.
- ARR growth in 2025 was 19.9% vs SaaS Capital medians of 25% (equity-backed) and 20% (bootstrapped).

## Quality of revenue now (Jul 2026)

- 12-month GRR 92.1%, NRR 103.1%. By segment: SMB 88.5% / 99.7%; Mid-Market 91.5% / 102.9%; Enterprise 93.6% / 104.2%.
- Usage-priced accounts: NRR 105.5% vs seat 101.1%.
- Cash conversion 97.2% in the latest quarter; 1.4% of revenue never collected; DSO about 32 days under a one-month terms assumption.
- $92m of committed ARR renews in the next 12 months (66% of billed ARR).

## Churn-risk models

- Trained on 2022-03 to 2024-09 (labels mature by close); holdout 2025-01 to 2026-04, 48,800 account-months, 3-month churn 2.2%.
- Holdout ROC-AUC: hand-set rule 0.603, rule x renewal exposure 0.754, L2 logistic 0.809, logistic with renewal interactions 0.809, gradient boosted trees 0.800, oracle 0.816.
- Penalty choice: L1, L2 and Elastic Net score within 0.005 AUC of each other on forward folds, so the choice rests on stability. Usage and module adoption correlate at 0.90 (VIF about 6); under L1, 8 of 16 features switch in and out across customer resamples. With about 96 churn events per feature, overfitting is not the binding risk. L2 was chosen (C = 0.01).
- Gradient boosted trees did not beat the logistic (paired AUC difference -0.009, 95% CI -0.017 to -0.001), and explicit interaction terms added nothing. Accounts with falling usage, rising tickets and an upcoming renewal are not riskier than the additive model already predicts. The simulator's churn decision is close to additive on the log-odds scale, so this is not evidence about real data.
- Promotion test vs rule x renewal, Logistic, L2 (16 features): AUC gain +0.054 (95% CI +0.039 to +0.068); passes.
- Promotion test vs rule x renewal, Logistic + renewal interactions: AUC gain +0.055 (95% CI +0.039 to +0.071); passes.
- Promotion test vs rule x renewal, Gradient boosted trees: AUC gain +0.045 (95% CI +0.030 to +0.060); passes.
- Recommendation: replace the renewal rule with the L2 logistic; keep the trees as a monitored challenger. Predicted churn averaged 2.4% against 2.2% actual.

## Intervention economics

- Break-even save rate, top 10% list worked once per account ($400 outreach, 12 months of 75% margin, 10% concession): random 137% (above 100% means it cannot pay back), rule 67%, rule x renewal 37%, L2 logistic 21%, trees 25%, logistic ranked by expected MRR loss 17%.
- Detecting a 20% save rate at 9.3% list churn needs about 3,474 accounts per arm; the list adds about 103 new accounts a month, so a single book needs about 68 months. A pooled pilot across portfolio companies is the realistic design.

## Forecast, Aug 2026 to Jul 2027

- Base case list ARR $170m by Jul 2027 (90% band $165m to $173m); downside $154m; no new logos $151m.

## Limits

- Effect sizes are simulation parameters. Benchmarks are calibration inputs in the diligence window and a check only afterwards.
- SaaS Capital's samples are mostly far smaller than this book; no public source found reports private-company retention at about $150m ARR.
- Discount dependence is descriptive; save offers create reverse causality. Save rates are unidentified, so economics are a hurdle, not an ROI.
- No P&L is simulated, so Rule of 40 and valuation are out of scope.
