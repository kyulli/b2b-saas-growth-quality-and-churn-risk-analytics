-- Logo and list-MRR retention by signup cohort (quarter) and months since signup.
-- Only cohorts that sign up inside the panel, so month 0 is observed.
DROP TABLE IF EXISTS cohort_retention;
CREATE TABLE cohort_retention AS
WITH first_month AS (
    SELECT customer_id, MIN(month) AS m0 FROM account_month GROUP BY customer_id
),
base AS (
    SELECT a.*, f.m0,
           substr(f.m0, 1, 4) || '-Q' || ((CAST(substr(f.m0, 6, 2) AS INTEGER) + 2) / 3) AS cohort_quarter,
           (CAST(substr(a.month, 1, 4) AS INTEGER) - CAST(substr(f.m0, 1, 4) AS INTEGER)) * 12
           + CAST(substr(a.month, 6, 2) AS INTEGER) - CAST(substr(f.m0, 6, 2) AS INTEGER) AS months_since_signup
    FROM account_month AS a JOIN first_month AS f USING (customer_id)
    WHERE f.m0 = a.signup_month
),
start_mrr AS (
    SELECT customer_id, list_mrr AS m0_list FROM base WHERE months_since_signup = 0
)
SELECT cohort_quarter, segment, industry, contract_term_months, months_since_signup,
       COUNT(*) AS cohort_customers,
       SUM(status = 'active') AS retained_customers,
       ROUND(1.0 * SUM(status = 'active') / COUNT(*), 4) AS logo_retention,
       ROUND(SUM(b.list_mrr) / SUM(s.m0_list), 4) AS list_mrr_retention
FROM base AS b JOIN start_mrr AS s USING (customer_id)
GROUP BY cohort_quarter, segment, industry, contract_term_months, months_since_signup;
