/*
  Intermediate model: int_customer_orders_rollup
  Aggregates cleaned transactions into customer-level order summaries.
*/

{{ config(materialized='ephemeral') }}

WITH transactions AS (
    SELECT * FROM {{ ref('stg_transactions') }}
),

customer_aggregates AS (
    SELECT
        customer_id,
        MIN(invoice_date) AS first_order_timestamp,
        MAX(invoice_date) AS last_order_timestamp,
        COUNT(DISTINCT CASE WHEN NOT is_cancellation THEN invoice_no END) AS total_valid_orders,
        COUNT(DISTINCT CASE WHEN is_cancellation THEN invoice_no END) AS total_cancelled_orders,
        COUNT(DISTINCT invoice_no) AS total_orders_all,
        ROUND(SUM(CASE WHEN NOT is_cancellation THEN line_item_amount ELSE 0.0 END), 2) AS total_monetary_spend,
        ROUND(SUM(CASE WHEN is_cancellation THEN ABS(line_item_amount) ELSE 0.0 END), 2) AS total_refunded_amount,
        SUM(CASE WHEN NOT is_cancellation THEN quantity ELSE 0 END) AS total_items_purchased
    FROM transactions
    GROUP BY customer_id
),

modal_hour_ranking AS (
    SELECT 
        customer_id,
        order_hour,
        COUNT(1) AS hour_order_frequency,
        ROW_NUMBER() OVER (PARTITION BY customer_id ORDER BY COUNT(1) DESC, order_hour ASC) AS rank_order
    FROM transactions
    GROUP BY customer_id, order_hour
),

top_hour AS (
    SELECT 
        customer_id,
        order_hour AS preferred_shopping_hour
    FROM modal_hour_ranking
    WHERE rank_order = 1
)

SELECT
    agg.customer_id,
    agg.first_order_timestamp,
    agg.last_order_timestamp,
    agg.total_valid_orders,
    agg.total_cancelled_orders,
    agg.total_orders_all,
    agg.total_monetary_spend,
    agg.total_refunded_amount,
    agg.total_items_purchased,
    COALESCE(th.preferred_shopping_hour, 12) AS preferred_shopping_hour
FROM customer_aggregates agg
LEFT JOIN top_hour th ON agg.customer_id = th.customer_id
