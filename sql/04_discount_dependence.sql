-- Gross new and expansion list MRR written by discount band, by month and cohort attributes.
-- Descriptive: it shows how much growth arrives with a material discount, not that discounts caused it.
DROP TABLE IF EXISTS discount_dependence;
CREATE TABLE discount_dependence AS
SELECT month, segment, industry, contract_term_months, pricing_model, channel,
       CASE WHEN list_prev = 0 THEN 'new' ELSE 'expansion' END AS motion,
       discount_pct > 0.15 AS deep_discount,
       COUNT(*) AS accounts,
       SUM(CASE WHEN list_prev = 0 THEN list_now ELSE list_now - list_prev END) AS gross_list_mrr,
       SUM(CASE WHEN list_prev = 0 THEN list_now ELSE list_now - list_prev END * discount_pct) AS discount_dollars
FROM mrr_bridge_account
WHERE month > (SELECT MIN(month) FROM account_month)
  AND ((list_prev = 0 AND list_now > 0 AND first_row) OR (list_prev > 0 AND list_now > list_prev))
GROUP BY month, segment, industry, contract_term_months, pricing_model, channel, motion, deep_discount;
