-- One row per customer-month with commercial, usage, and billing state.
DROP TABLE IF EXISTS account_month;
CREATE TABLE account_month AS
SELECT
    s.customer_id, s.month, s.status,
    s.mrr_amount AS billed_mrr, s.list_mrr, s.discount_pct, s.seats,
    s.promo_discount, s.months_to_renewal, s.is_renewal_month,
    c.segment, c.industry, c.contract_term_months, c.channel, c.pricing_model, c.region,
    substr(c.signup_date, 1, 7) || '-01' AS signup_month,
    u.usage_ratio, u.ticket_count, u.sev1_tickets, u.modules_adopted, u.nps_score,
    p.days_late, p.payment_status
FROM subscriptions_clean AS s
JOIN customers_clean AS c USING (customer_id)
LEFT JOIN usage_clean AS u USING (customer_id, month)
LEFT JOIN payments_clean AS p USING (customer_id, month);
