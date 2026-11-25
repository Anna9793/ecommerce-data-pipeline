"""BigQuery SQL query templates and builders for Medallion architecture and BQML."""

from typing import List, Optional

# Standard Category Taxonomy for E-Commerce Platform
STANDARD_CATEGORIES: List[str] = [
    "Kitchen & Dining",
    "Home Decor & Lighting",
    "Holiday & Seasonal",
    "Smart Audio & Acoustics",
    "Performance Activewear",
    "Ergonomic Workspace",
    "Storage & Accessories",
    "Party & Celebration",
    "Kids & Toys",
    "Gifts & Living",
]


# ==============================================================================
# 1. Medallion Architecture DDL Queries (Silver & Gold Views)
# ==============================================================================

def get_clean_transactions_view_query(project_id: str, dataset_id: str = "retail_data") -> str:
    """Silver Layer: Curated, typed, and sanitized transactions view query."""
    return f"""
    CREATE OR REPLACE VIEW `{project_id}.{dataset_id}.clean_transactions` AS
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
    FROM `{project_id}.{dataset_id}.transactions`
    WHERE CustomerID IS NOT NULL
      AND UnitPrice > 0
      AND (
        Quantity > 0 
        OR (Quantity < 0 AND STARTS_WITH(InvoiceNo, 'C'))
      );
    """.strip()


def get_raw_products_view_query(project_id: str, dataset_id: str = "retail_data") -> str:
    """Silver Layer: Deduplicated product catalog view query."""
    return f"""
    CREATE OR REPLACE VIEW `{project_id}.{dataset_id}.raw_products` AS
    SELECT
      stock_code,
      description,
      ROUND(AVG(unit_price), 2) AS unit_price,
      COUNT(*) AS sales_volume
    FROM `{project_id}.{dataset_id}.clean_transactions`
    WHERE description IS NOT NULL AND TRIM(description) != ''
    GROUP BY stock_code, description;
    """.strip()


def get_customer_features_view_query(project_id: str, dataset_id: str = "retail_data") -> str:
    """Gold Layer: Behavioral RFM and ML analytical feature store view query."""
    return f"""
    CREATE OR REPLACE VIEW `{project_id}.{dataset_id}.customer_features` AS
    WITH reference_date AS (
      SELECT MAX(invoice_date) AS max_date
      FROM `{project_id}.{dataset_id}.clean_transactions`
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
      FROM `{project_id}.{dataset_id}.clean_transactions`
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
      FROM `{project_id}.{dataset_id}.clean_transactions` t
      GROUP BY t.customer_id
    ),
    customer_hours AS (
      SELECT
        customer_id,
        COALESCE(APPROX_TOP_COUNT(invoice_hour, 1)[OFFSET(0)].value, 12) AS preferred_shopping_hour
      FROM `{project_id}.{dataset_id}.clean_transactions`
      GROUP BY customer_id
    )
    SELECT
      s.customer_id,
      s.recency,
      s.frequency,
      COALESCE(ROUND(SAFE_DIVIDE(s.monetary, s.frequency), 2), 0.0) AS avg_order_value,
      COALESCE(
        ROUND(SAFE_DIVIDE(v.monetary_30, SAFE_DIVIDE(v.monetary_90, 3.0)), 2),
        1.0
      ) AS spending_velocity,
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
    """.strip()


def get_medallion_ddl_script(project_id: str, dataset_id: str = "retail_data") -> str:
    """Assembles all Medallion DDL statements (Silver + Gold) into a single executable script."""
    return f"""
{get_clean_transactions_view_query(project_id, dataset_id)}

{get_raw_products_view_query(project_id, dataset_id)}

{get_customer_features_view_query(project_id, dataset_id)}
    """.strip()


# ==============================================================================
# 2. BigQuery ML Remote Models & GenAI Catalog Enrichment Queries
# ==============================================================================

def get_bqml_remote_models_setup_query(
    project_id: str,
    dataset_id: str = "retail_data",
    connection_id: str = "us-central1.vertex_ai_connection",
) -> str:
    """DDL query to create the remote Gemini 1.5 Flash and text-embedding-004 models."""
    return f"""
    CREATE OR REPLACE MODEL `{project_id}.{dataset_id}.gemini_flash_remote`
    REMOTE WITH CONNECTION `{connection_id}`
    OPTIONS(ENDPOINT = 'gemini-1.5-flash');

    CREATE OR REPLACE MODEL `{project_id}.{dataset_id}.text_embedding_remote`
    REMOTE WITH CONNECTION `{connection_id}`
    OPTIONS(ENDPOINT = 'text-embedding-004');
    """.strip()


def build_text_enrichment_query(
    project_id: str,
    dataset_id: str = "retail_data",
    source_table: str = "raw_products",
    target_table: str = "dim_products_enriched",
    categories: Optional[List[str]] = None,
) -> str:
    """Builds the BigQuery ML SQL query executing ML.GENERATE_TEXT with Gemini 1.5 Flash."""
    valid_categories = categories or STANDARD_CATEGORIES
    categories_str = ", ".join(valid_categories)
    return f"""
    CREATE OR REPLACE TABLE `{project_id}.{dataset_id}.{target_table}` AS
    WITH raw_prompts AS (
      SELECT 
        stock_code,
        description,
        unit_price,
        CONCAT(
          'You are an expert e-commerce catalog taxonomist. Analyze this product description: "', 
          description, 
          '". Return a JSON object with "category" (choose one of: {categories_str}) and "tags" (array of 3-5 search keywords). JSON only.'
        ) AS prompt
      FROM `{project_id}.{dataset_id}.{source_table}`
      WHERE description IS NOT NULL AND TRIM(description) != ''
    ),
    llm_responses AS (
      SELECT 
        stock_code,
        description,
        unit_price,
        ml_generate_text_result
      FROM ML.GENERATE_TEXT(
        MODEL `{project_id}.{dataset_id}.gemini_flash_remote`,
        TABLE raw_prompts,
        STRUCT(
          0.1 AS temperature,
          128 AS max_output_tokens,
          'application/json' AS response_mime_type,
          TRUE AS flatten_json_output
        )
      )
    )
    SELECT
      stock_code,
      description,
      COALESCE(JSON_EXTRACT_SCALAR(ml_generate_text_result, '$.category'), 'Gifts & Living') AS category,
      COALESCE(JSON_EXTRACT_STRING_ARRAY(ml_generate_text_result, '$.tags'), ARRAY<STRING>['lifestyle', 'general']) AS tags,
      unit_price,
      CONCAT(
        'Product: ', description, 
        ' | Category: ', COALESCE(JSON_EXTRACT_SCALAR(ml_generate_text_result, '$.category'), 'Gifts & Living'), 
        ' | Price: $', CAST(ROUND(unit_price, 2) AS STRING),
        ' | Tags: ', ARRAY_TO_STRING(COALESCE(JSON_EXTRACT_STRING_ARRAY(ml_generate_text_result, '$.tags'), ARRAY<STRING>['gift', 'lifestyle']), ', ')
      ) AS document_text,
      CURRENT_TIMESTAMP() AS enriched_at
    FROM llm_responses;
    """.strip()


def build_vector_embedding_query(
    project_id: str,
    dataset_id: str = "retail_data",
    source_table: str = "dim_products_enriched",
    target_table: str = "product_catalog_vectors",
) -> str:
    """Builds the BigQuery ML SQL query executing ML.GENERATE_EMBEDDING with text-embedding-004."""
    return f"""
    CREATE OR REPLACE TABLE `{project_id}.{dataset_id}.{target_table}` AS
    SELECT
      stock_code,
      description,
      category,
      unit_price,
      document_text,
      ml_generate_embedding_result AS embedding,
      CURRENT_TIMESTAMP() AS vectorized_at
    FROM ML.GENERATE_EMBEDDING(
      MODEL `{project_id}.{dataset_id}.text_embedding_remote`,
      (
        SELECT 
          stock_code,
          description,
          category,
          unit_price,
          document_text,
          document_text AS content
        FROM `{project_id}.{dataset_id}.{source_table}`
      ),
      STRUCT(
        'RETRIEVAL_DOCUMENT' AS task_type,
        768 AS output_dimensionality
      )
    );
    """.strip()
