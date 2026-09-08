/*
  Staging model: stg_transactions
  Cleans, normalizes, and validates raw incoming transactions from BigQuery.
*/

{{ config(materialized='view') }}

WITH source_data AS (
    SELECT 
        CAST(InvoiceNo AS STRING) AS invoice_no,
        CAST(StockCode AS STRING) AS stock_code,
        TRIM(CAST(Description AS STRING)) AS description,
        CAST(Quantity AS INT64) AS quantity,
        CAST(InvoiceDate AS TIMESTAMP) AS invoice_date,
        CAST(UnitPrice AS FLOAT64) AS unit_price,
        CAST(CustomerID AS STRING) AS customer_id,
        TRIM(CAST(Country AS STRING)) AS country
    FROM `anna-ml-pipeline.retail_data.transactions`
    WHERE CustomerID IS NOT NULL
      AND TRIM(CAST(CustomerID AS STRING)) != ''
      AND CAST(CustomerID AS STRING) != 'nan'
)

SELECT
    invoice_no,
    stock_code,
    description,
    quantity,
    invoice_date,
    unit_price,
    ROUND(quantity * unit_price, 2) AS line_item_amount,
    customer_id,
    country,
    -- Flag cancellations (Invoice starting with 'C' or negative quantities)
    CASE 
        WHEN STARTS_WITH(invoice_no, 'C') OR quantity < 0 THEN TRUE 
        ELSE FALSE 
    END AS is_cancellation,
    -- Extract temporal attributes for analytical slicing
    EXTRACT(HOUR FROM invoice_date) AS order_hour,
    EXTRACT(DAYOFWEEK FROM invoice_date) AS order_day_of_week,
    DATE(invoice_date) AS order_date
FROM source_data
