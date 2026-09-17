-- ==============================================================================
-- BigQuery ML: In-Database AI Catalog Enrichment & Vectorization
-- Models: Gemini 1.5 Flash (ML.GENERATE_TEXT) & Text-Embedding-004 (ML.GENERATE_EMBEDDING)
-- ==============================================================================

-- 1. Create Cloud Resource Connection to Vertex AI
-- (Executed once per GCP Project / Region)
-- In Cloud Shell / Terraform:
-- gcloud bigquery connections create --connection_type=CLOUD_RESOURCE --location=US vertex_ai_connection

-- 2. Define Remote Gemini 1.5 Flash Model for Automated Taxonomy & Tags
CREATE OR REPLACE MODEL `retail_data.gemini_flash_remote`
REMOTE WITH CONNECTION `us.vertex_ai_connection`
OPTIONS(
  ENDPOINT = 'gemini-1.5-flash'
);

-- 3. Define Remote Text Embedding Model (768-dimensional)
CREATE OR REPLACE MODEL `retail_data.text_embedding_remote`
REMOTE WITH CONNECTION `us.vertex_ai_connection`
OPTIONS(
  ENDPOINT = 'text-embedding-004'
);

-- 4. Batch AI Transformation: Enriched Products with Gemini Taxonomy (ML.GENERATE_TEXT)
CREATE OR REPLACE TABLE `retail_data.dim_products_enriched` AS
WITH raw_prompts AS (
  SELECT 
    stock_code,
    description,
    unit_price,
    tenant_id,
    CONCAT(
      'You are an expert e-commerce catalog taxonomist. Analyze this product description: "', 
      description, 
      '". Return a JSON object with: ',
      '1. "category": a standard taxonomy category (e.g., Kitchen & Dining, Home Decor & Lighting, Holiday & Seasonal, Smart Audio & Acoustics, Performance Activewear, Ergonomic Workspace, Storage & Accessories, Kids & Toys). ',
      '2. "tags": an array of 3-5 descriptive lifestyle, styling, or functional search keywords. ',
      'JSON Output only.'
    ) AS prompt
  FROM `retail_data.raw_products`
  WHERE description IS NOT NULL AND TRIM(description) != ''
),
llm_responses AS (
  SELECT 
    stock_code,
    description,
    unit_price,
    tenant_id,
    ml_generate_text_result
  FROM ML.GENERATE_TEXT(
    MODEL `retail_data.gemini_flash_remote`,
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
  tenant_id,
  description,
  COALESCE(JSON_EXTRACT_SCALAR(ml_generate_text_result, '$.category'), 'General Merchandise') AS category,
  COALESCE(JSON_EXTRACT_STRING_ARRAY(ml_generate_text_result, '$.tags'), ARRAY<STRING>['lifestyle', 'general']) AS tags,
  unit_price,
  CONCAT(
    'Product: ', description, 
    ' | Category: ', COALESCE(JSON_EXTRACT_SCALAR(ml_generate_text_result, '$.category'), 'General Merchandise'), 
    ' | Price: $', CAST(ROUND(unit_price, 2) AS STRING),
    ' | Tags: ', ARRAY_TO_STRING(COALESCE(JSON_EXTRACT_STRING_ARRAY(ml_generate_text_result, '$.tags'), ARRAY<STRING>['gift', 'lifestyle']), ', ')
  ) AS document_text,
  CURRENT_TIMESTAMP() AS enriched_at
FROM llm_responses;

-- 5. Batch Vectorization: Generate 768d Dense Vector Embeddings (ML.GENERATE_EMBEDDING)
CREATE OR REPLACE TABLE `retail_data.product_catalog_vectors` AS
SELECT
  stock_code,
  tenant_id,
  description,
  category,
  unit_price,
  document_text,
  ml_generate_embedding_result AS embedding,
  CURRENT_TIMESTAMP() AS vectorized_at
FROM ML.GENERATE_EMBEDDING(
  MODEL `retail_data.text_embedding_remote`,
  (
    SELECT 
      stock_code,
      tenant_id,
      description,
      category,
      unit_price,
      document_text,
      document_text AS content
    FROM `retail_data.dim_products_enriched`
  ),
  STRUCT(
    'RETRIEVAL_DOCUMENT' AS task_type,
    768 AS output_dimensionality
  )
);
