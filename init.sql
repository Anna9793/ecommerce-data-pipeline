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
    churn_probability DOUBLE PRECISION,
    is_churn INT,
    model_version VARCHAR(20),
    feature_version VARCHAR(20),
    response_time_ms DOUBLE PRECISION,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
