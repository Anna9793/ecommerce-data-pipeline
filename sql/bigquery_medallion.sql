-- ==============================================================================
-- BigQuery Medallion Architecture: Bronze -> Silver -> Gold
-- Project: anna-ml-pipeline | Dataset: retail_data
-- ==============================================================================

-- ------------------------------------------------------------------------------
-- 1. SILVER LAYER: Curated & Sanitized Transactions
-- Source: retail_data.transactions (Bronze)
-- Guarantees:
--   - CustomerID is non-null, trimmed, and not 'GUEST'
--   - Strictly positive UnitPrice (> 0)
--   - Valid purchases (Quantity > 0) or valid cancellations (Quantity < 0 & InvoiceNo starts with 'C')
--   - Pre-computed order_value (Quantity * UnitPrice)
-- ------------------------------------------------------------------------------
CREATE OR REPLACE VIEW `retail_data.clean_transactions` AS
SELECT
  InvoiceNo AS invoice_no,
  StockCode AS stock_code,
  Description AS description,
  CAST(Quantity AS INT64) AS quantity,
  CAST(InvoiceDate AS TIMESTAMP) AS invoice_date,
  CAST(UnitPrice AS FLOAT64) AS unit_price,
  CAST(CustomerID AS STRING) AS customer_id,
  Country AS country,
  ROUND(CAST(Quantity AS FLOAT64) * CAST(UnitPrice AS FLOAT64), 2) AS order_value,
  EXTRACT(YEAR FROM InvoiceDate) AS invoice_year,
  EXTRACT(MONTH FROM InvoiceDate) AS invoice_month,
  EXTRACT(HOUR FROM InvoiceDate) AS invoice_hour
FROM `retail_data.transactions`
WHERE CustomerID IS NOT NULL
  AND TRIM(CustomerID) NOT IN ('', 'GUEST', 'nan', 'None')
  AND UnitPrice > 0
  AND (
    Quantity > 0 
    OR (Quantity < 0 AND STARTS_WITH(InvoiceNo, 'C'))
  );


-- ------------------------------------------------------------------------------
-- 2. GOLD LAYER: Analytical Feature Store for ML & Segmentation
-- Source: retail_data.clean_transactions (Silver)
-- Guarantees:
--   - Exactly 0 NaNs / NULL values
--   - SAFE_DIVIDE and COALESCE on all ratios and velocity metrics
--   - frequency > 0 (filters customers with only cancellations)
-- ------------------------------------------------------------------------------
CREATE OR REPLACE VIEW `retail_data.rfm_features` AS
WITH reference_date AS (
  SELECT MAX(invoice_date) AS max_date
  FROM `retail_data.clean_transactions`
),
windows AS (
  SELECT
    max_date,
    TIMESTAMP_SUB(max_date, INTERVAL 30 DAY) AS date_30,
    TIMESTAMP_SUB(max_date, INTERVAL 90 DAY) AS date_90
  FROM reference_date
),
valid_purchases AS (
  SELECT *
  FROM `retail_data.clean_transactions`
  WHERE quantity > 0
),
customer_sales AS (
  SELECT
    p.customer_id,
    COUNT(DISTINCT p.invoice_no) AS frequency,
    ROUND(SUM(p.order_value), 2) AS monetary,
    TIMESTAMP_DIFF((SELECT max_date FROM windows), MAX(p.invoice_date), DAY) AS recency
  FROM valid_purchases p
  GROUP BY p.customer_id
),
customer_velocity AS (
  SELECT
    p.customer_id,
    COALESCE(SUM(CASE WHEN p.invoice_date >= (SELECT date_30 FROM windows) THEN p.order_value ELSE 0 END), 0.0) AS monetary_30,
    COALESCE(SUM(CASE WHEN p.invoice_date >= (SELECT date_90 FROM windows) THEN p.order_value ELSE 0 END), 0.0) AS monetary_90
  FROM valid_purchases p
  GROUP BY p.customer_id
),
customer_cancellations AS (
  SELECT
    t.customer_id,
    COUNT(DISTINCT CASE WHEN STARTS_WITH(t.invoice_no, 'C') THEN t.invoice_no END) AS cancelled_orders,
    COUNT(DISTINCT t.invoice_no) AS total_orders
  FROM `retail_data.clean_transactions` t
  GROUP BY t.customer_id
),
customer_hours AS (
  SELECT
    customer_id,
    -- Default to 12 if mode calculation is empty
    COALESCE(APPROX_TOP_COUNT(invoice_hour, 1)[OFFSET(0)].value, 12) AS preferred_shopping_hour
  FROM `retail_data.clean_transactions`
  GROUP BY customer_id
)
SELECT
  s.customer_id,
  s.recency,
  s.frequency,
  COALESCE(ROUND(SAFE_DIVIDE(s.monetary, s.frequency), 2), 0.0) AS avg_order_value,
  -- Spending Velocity: monetary_30 / (monetary_90 / 3.0), protected against 0 division
  COALESCE(
    ROUND(SAFE_DIVIDE(v.monetary_30, SAFE_DIVIDE(v.monetary_90, 3.0)), 2),
    1.0
  ) AS spending_velocity,
  -- Cancellation Rate: cancelled_orders / total_orders
  COALESCE(
    ROUND(SAFE_DIVIDE(c.cancelled_orders, c.total_orders), 4),
    0.0
  ) AS cancellation_rate,
  h.preferred_shopping_hour
FROM customer_sales s
LEFT JOIN customer_velocity v ON s.customer_id = v.customer_id
LEFT JOIN customer_cancellations c ON s.customer_id = c.customer_id
LEFT JOIN customer_hours h ON s.customer_id = h.customer_id
WHERE s.frequency > 0;
