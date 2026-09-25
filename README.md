<div align="center">

# B2B SaaS Growth Quality and Churn Risk

A buy-side case study: underwriting a $113m-ARR software acquisition, testing the underwriting against
nineteen months of post-close results, and deciding where to spend retention effort.

[Full report](reports/project_report.docx) · [Quality-of-revenue workbook (Excel)](reports/quality_of_revenue.xlsx) · [Investment committee brief (PDF)](reports/portfolio_monitoring_brief.pdf) · [Dashboard](dashboard/index.html)

</div>

---

## The situation

A formerly bootstrapped B2B software company selling to SMB, Mid-Market and Enterprise customers is acquired in
December 2024 at $113m of annual recurring revenue (ARR). By July 2026 it has grown to $147m. The case works through
the questions an owner asks in order.

1. At close, is the revenue as durable as peers', and was growth bought with discounts?
2. Did the plan written at close hold over the following nineteen months?
3. Today, where is revenue at risk, and is a retention programme worth funding?

## Key takeaways

- **Retention at close was at the peer median.** Gross revenue retention (GRR) of 90.9% and net revenue retention
  (NRR) of 102.7% match SaaS Capital's private-company medians for the same period. One year's GRR for a book this
  size moves about two points by chance, so it should be underwritten as a range, not a point.
- **Growth was earned, not discounted.** Billed revenue grew as fast as list revenue in 2024 and the discounted
  share of revenue edged down.
- **Growth is slowing by arithmetic.** New-customer revenue has been flat at $17m to $19m a year while the base
  grows, so growth fell from 27% (2023) to 21% (2024) to 20% (2025). The plan should be built from its drivers,
  not from last year's growth rate.
- **The underwriting held.** The base case written at close finished 1.0% below actual ARR after nineteen months
  and was never more than 1.7% off. Its risk range, however, was too narrow and should be widened before it is
  used for a plan or a covenant.
- **Risk sits on the renewal calendar.** $92m of ARR renews in the next twelve months, and 88% of lost customers
  left either within three months of a contract end date or from a month-to-month contract. SMB losses are lost customers rather than smaller contracts, and SMB
  generates about eleven times more support work per dollar of revenue than Enterprise.
- **A retention programme pays only if it is targeted.** Working the right accounts breaks even if outreach saves
  about one in five at-risk customers (one in six when ranked by revenue at stake), against more than one in three
  for the current rule of thumb. One point of GRR is worth about $1.5m of ARR, or $1.1m of annual gross profit.

## Scorecard

| Metric | At close (Dec 2024) | Today (Jul 2026) | Peer reference |
|---|---|---|---|
| ARR | $113m | $147m | |
| ARR growth, past 12 months | 21% | 18% | SaaS Capital median 20% to 25% |
| Gross revenue retention | 90.9% | 92.1% | SaaS Capital median 91% |
| Net revenue retention | 102.7% | 103.1% | SaaS Capital medians 101% to 104% |
| Customers | 2,875 | 3,505 | 391 Enterprise today |
| Largest 10 customers, share of ARR | 3.7% | 3.7% | |
| Cash collected, share of revenue | | 97% | 1.4% never collected |
| ARR renewing in next 12 months | | $92m | at least $6.4m expected to be lost |

## Recommendations to the owner

1. Underwrite retention at the peer median with a band of about two points either way.
2. Build the growth plan from new-customer revenue, expansion and losses, and expect continued deceleration.
3. Run retention outreach from a ranked list tied to the renewal calendar, prioritised by revenue at stake.
4. Before shrinking SMB, obtain customer acquisition cost, payback and cost to serve by segment. Retention alone
   does not justify it.
5. Test the retention programme with a controlled pilot pooled across portfolio companies; one company alone would
   need more than five years to measure the save rate.

<p align="center">
  <img src="outputs/figures/plan_vs_actual.png" alt="Underwriting forecast vs actual ARR" width="48%" />
  <img src="outputs/figures/benchmark_gap.png" alt="Retention at close vs peer medians" width="48%" />
</p>

<p align="center">
  <img src="outputs/figures/renewal_exposure.png" alt="ARR by renewal window" width="48%" />
  <img src="outputs/figures/break_even.png" alt="Save rate needed for retention outreach to pay back" width="48%" />
</p>

## What is in this repository

| Deliverable | What it is for |
|---|---|
| [Full report](reports/project_report.docx) | Executive summary, plain definitions of every metric, and the full analysis with a reflection on each step |
| [Quality-of-revenue workbook](reports/quality_of_revenue.xlsx) | Excel model with live formulas: ARR bridge, retention, concentration, revenue to cash, renewals, sensitivity |
| [Investment committee brief](reports/portfolio_monitoring_brief.pdf) | Short brief of findings and actions |
| [Dashboard](dashboard/index.html) | One-page view of the charts |
| [Benchmarks and assumptions](data/reference/) | Every peer figure with its source and period, and every modelling assumption |

## About the data

Customer-level billing, usage and support data is never public, so the company is simulated from written rules
covering 4,547 customers over 55 months. Its retention in the year before close was set to match SaaS Capital's
published medians, so agreement in that year is by design; everything after the close is tested out of sample. The
numbers illustrate a diligence and monitoring method, not a real company. Peer figures come from SaaS Capital,
Benchmarkit, SBI and the SEC filings of eleven listed software companies.

Analysis code (Python and SQL) is private and available on request.
