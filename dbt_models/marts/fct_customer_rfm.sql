/*
  Marts model: fct_customer_rfm
  Final production analytical mart for RFM customer segmentation and behavioral scoring.
*/

{{ config(
    materialized='table',
    tags=['marts', 'daily', 'rfm']
) }}

WITH rollup_data AS (
    SELECT * FROM {{ ref('int_customer_orders_rollup') }}
),

max_date_reference AS (
    SELECT MAX(last_order_timestamp) AS max_snapshot_timestamp
    FROM rollup_data
),

rfm_calculated AS (
    SELECT
        r.customer_id,
        DATE_DIFF(DATE(ref.max_snapshot_timestamp), DATE(r.last_order_timestamp), DAY) AS recency,
        r.total_valid_orders AS frequency,
        r.total_monetary_spend AS monetary,
        ROUND(
            r.total_monetary_spend / NULLIF(r.total_valid_orders, 0), 
            2
        ) AS avg_order_value,
        ROUND(
            CAST(r.total_cancelled_orders AS FLOAT64) / NULLIF(r.total_orders_all, 0), 
            4
        ) AS cancellation_rate,
        r.preferred_shopping_hour,
        DATE_DIFF(DATE(r.last_order_timestamp), DATE(r.first_order_timestamp), DAY) AS customer_tenure_days,
        r.total_items_purchased
    FROM rollup_data r
    CROSS JOIN max_date_reference ref
)

SELECT
    customer_id,
    COALESCE(recency, 0) AS recency,
    COALESCE(frequency, 0) AS frequency,
    COALESCE(monetary, 0.0) AS monetary,
    COALESCE(avg_order_value, 0.0) AS avg_order_value,
    COALESCE(cancellation_rate, 0.0) AS cancellation_rate,
    preferred_shopping_hour,
    customer_tenure_days,
    total_items_purchased,
    -- Semantic RFM Classification Tier for Looker / BI Dashboards
    CASE
        WHEN recency <= 30 AND frequency >= 5 THEN 'Champions'
        WHEN recency <= 60 AND frequency >= 3 THEN 'Loyal Customers'
        WHEN recency <= 90 AND monetary >= 500.0 THEN 'Potential Loyalists'
        WHEN recency <= 90 THEN 'Active Shoppers'
        WHEN recency > 180 THEN 'Lost Customers'
        ELSE 'At Risk'
    END AS rfm_segment_tier,
    CURRENT_TIMESTAMP() AS dbt_updated_at
FROM rfm_calculated
