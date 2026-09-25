-- Monthly list-price MRR bridge. Opening = prior month list MRR of active accounts.
-- new: first billed month; churn: active last month, not this month;
-- expansion / contraction: list MRR change of continuing accounts.
-- The billed columns repeat the bridge on billed MRR, where discount changes also move MRR.
DROP TABLE IF EXISTS mrr_bridge_account;
CREATE TABLE mrr_bridge_account AS
SELECT customer_id, month, segment, industry, contract_term_months, pricing_model, channel,
       discount_pct,
       CASE WHEN status = 'active' THEN list_mrr ELSE 0 END AS list_now,
       COALESCE(LAG(CASE WHEN status = 'active' THEN list_mrr ELSE 0 END) OVER w, 0) AS list_prev,
       CASE WHEN status = 'active' THEN billed_mrr ELSE 0 END AS billed_now,
       COALESCE(LAG(CASE WHEN status = 'active' THEN billed_mrr ELSE 0 END) OVER w, 0) AS billed_prev,
       LAG(month) OVER w IS NULL AS first_row
FROM account_month
WINDOW w AS (PARTITION BY customer_id ORDER BY month);

DROP TABLE IF EXISTS mrr_bridge;
CREATE TABLE mrr_bridge AS
SELECT month,
       SUM(list_prev) AS opening_list,
       SUM(CASE WHEN first_row AND month > (SELECT MIN(month) FROM account_month) THEN list_now ELSE 0 END) AS new_list,
       SUM(CASE WHEN list_prev > 0 AND list_now > list_prev THEN list_now - list_prev ELSE 0 END) AS expansion_list,
       -SUM(CASE WHEN list_prev > 0 AND list_now > 0 AND list_now < list_prev THEN list_prev - list_now ELSE 0 END) AS contraction_list,
       -SUM(CASE WHEN list_prev > 0 AND list_now = 0 THEN list_prev ELSE 0 END) AS churn_list,
       SUM(list_now) AS closing_list,
       SUM(billed_prev) AS opening_billed,
       SUM(billed_now) AS closing_billed
FROM mrr_bridge_account
WHERE month > (SELECT MIN(month) FROM account_month)
GROUP BY month ORDER BY month;
