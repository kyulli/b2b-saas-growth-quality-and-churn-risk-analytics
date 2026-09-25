# Public SaaS Company Panel

## Purpose

This layer provides real company-quarter evidence about traditional B2B SaaS
financial performance. It is deliberately separate from the synthetic
account-level panel. Public filings describe firm-level outcomes; they cannot
reveal individual account usage, discounting, tickets, or churn.

## Sources and provenance

`scripts/fetch_sec_companyfacts.py` downloads an immutable local snapshot of
the SEC Company Facts JSON for every company in `company_universe.csv`. The
processed long table retains the source URL, US-GAAP tag, form type, filing
date, and accession number for every extracted observation. Quarterly metrics
are then built only from 10-Q and 10-K facts; fourth-quarter values are
derived from annual less first through third fiscal-quarter values when all
components exist.

The SEC download command requires a contactable `SEC_USER_AGENT`; use
`SEC_USER_AGENT="Your Name your_email@example.com" make public-comps` when
refreshing source snapshots.

The SEC does not standardize NRR, GRR, customer counts, or pricing changes in
XBRL. `reported_saas_metrics.csv` and `pricing_and_ai_events.csv` therefore
accept only manually verified disclosures with a direct investor-relations or
SEC filing URL. Empty cells mean "not publicly disclosed", never zero.

## Scope

The universe contains 11 listed software companies. It is a transparent
research sample, not a claim that the companies are identical peers. Fiscal
calendars differ, so the panel preserves each company's reported fiscal period
end rather than forcing a calendar-quarter alignment.

## Permitted claims

- The public panel supports descriptive comparisons of reported revenue,
  gross margin, operating margin, free cash flow, and R&D intensity over time.
- It can provide external context for synthetic simulation assumptions.
- It cannot identify account-level churn or prove that AI announcements caused
  a change in retention, pricing, or margins.
