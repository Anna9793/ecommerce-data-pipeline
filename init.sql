CREATE TABLE IF NOT EXISTS predictions (
    request_id VARCHAR(50) PRIMARY KEY,
    customer_id VARCHAR(50),
    recency DOUBLE PRECISION,
    frequency INT,
    avg_order_value DOUBLE PRECISION,
    cluster INT,
    label VARCHAR(100),
    model_version VARCHAR(20),
    feature_version VARCHAR(20),
    response_time_ms DOUBLE PRECISION,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS churn_predictions (
    request_id VARCHAR(50) PRIMARY KEY,
    customer_id VARCHAR(50),
    recency DOUBLE PRECISION,
    frequency INT,
    avg_order_value DOUBLE PRECISION,
    spending_velocity DOUBLE PRECISION,
    cancellation_rate DOUBLE PRECISION,
    preferred_shopping_hour INT,
    churn_probability DOUBLE PRECISION,
    is_churn INT,
    model_version VARCHAR(20),
    feature_version VARCHAR(20),
    response_time_ms DOUBLE PRECISION,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS online_customer_features (
    customer_id VARCHAR(50) PRIMARY KEY,
    recency DOUBLE PRECISION,
    frequency INT,
    avg_order_value DOUBLE PRECISION,
    spending_velocity DOUBLE PRECISION,
    cancellation_rate DOUBLE PRECISION,
    preferred_shopping_hour INT,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS product_catalog_vectors (
    stock_code VARCHAR(50) PRIMARY KEY,
    description TEXT NOT NULL,
    category VARCHAR(100),
    unit_price DOUBLE PRECISION NOT NULL,
    document_text TEXT NOT NULL,
    embedding vector(768),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS product_vector_idx 
ON product_catalog_vectors 
USING hnsw (embedding vector_cosine_ops);
