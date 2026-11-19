import os
import logging
from contextlib import contextmanager
import psycopg2
from psycopg2 import pool
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

_connection_pool = None

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://postgres:postgres@localhost:5432/postgres",
)

def get_pool():
    global _connection_pool
    if _connection_pool is None:
        try:
            _connection_pool = pool.ThreadedConnectionPool(
                minconn=1,
                maxconn=20,
                dsn=DATABASE_URL,
            )
            logger.info("ThreadedConnectionPool initialized successfully (minconn=1, maxconn=20).")
        except Exception as e:
            logger.warning("Could not initialize ThreadedConnectionPool: %s. Using direct connections.", e)
            _connection_pool = None
    return _connection_pool

def close_pool():
    """Gracefully closes all connections in the PostgreSQL ThreadedConnectionPool (12-Factor App Factor IX: Disposability)."""
    global _connection_pool
    if _connection_pool is not None:
        try:
            _connection_pool.closeall()
            logger.info("PostgreSQL ThreadedConnectionPool closed gracefully.")
        except Exception as e:
            logger.warning("Error closing ThreadedConnectionPool: %s", e)
        finally:
            _connection_pool = None

def get_connection():
    """Fetches an active, pre-established connection from the thread-safe connection pool."""
    pool_inst = get_pool()
    if pool_inst:
        try:
            return pool_inst.getconn()
        except Exception as e:
            logger.warning("Failed to get connection from pool: %s. Falling back to direct connect.", e)
    return psycopg2.connect(DATABASE_URL)

def release_connection(conn):
    """Releases an open connection back to the ThreadedConnectionPool for reuse."""
    if conn:
        pool_inst = get_pool()
        if pool_inst and hasattr(pool_inst, "putconn"):
            try:
                pool_inst.putconn(conn)
                return
            except Exception:
                pass
        try:
            conn.close()
        except Exception:
            pass

@contextmanager
def get_db_cursor(commit: bool = False):
    """
    Context manager for thread-safe PostgreSQL connection checkout, cursor handling,
    and guaranteed return to the ThreadedConnectionPool.
    """
    conn = get_connection()
    cursor = conn.cursor()
    try:
        yield cursor
        if commit:
            conn.commit()
    except Exception:
        if commit:
            try:
                conn.rollback()
            except Exception:
                pass
        raise
    finally:
        try:
            cursor.close()
        except Exception:
            pass
        release_connection(conn)

def insert_prediction(record):
    if os.getenv("USE_BIGQUERY", "false").lower() == "true":
        from app.db_bigquery import insert_prediction_bigquery
        return insert_prediction_bigquery(record)

    try:
        with get_db_cursor(commit=True) as cursor:
            cursor.execute("""
                INSERT INTO predictions (
                    request_id,
                    customer_id,
                    recency,
                    frequency,
                    avg_order_value,
                    cluster,
                    label,
                    model_version,
                    feature_version,
                    response_time_ms
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                record["request_id"],
                record["customer_id"],
                record["recency"],
                record["frequency"],
                record["avg_order_value"],
                record["cluster"],
                record["label"],
                record["model_version"],
                record["feature_version"],
                record.get("response_time_ms", None),
            ))
    except Exception:
        logger.exception("Error inserting prediction")
        raise

def insert_churn_prediction(record):
    if os.getenv("USE_BIGQUERY", "false").lower() == "true":
        from app.db_bigquery import insert_churn_prediction_bigquery
        return insert_churn_prediction_bigquery(record)

    try:
        with get_db_cursor(commit=True) as cursor:
            cursor.execute("""
                INSERT INTO churn_predictions (
                    request_id,
                    customer_id,
                    recency,
                    frequency,
                    avg_order_value,
                    spending_velocity,
                    cancellation_rate,
                    preferred_shopping_hour,
                    churn_probability,
                    is_churn,
                    model_version,
                    feature_version,
                    response_time_ms
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                record["request_id"],
                record["customer_id"],
                record["recency"],
                record["frequency"],
                record["avg_order_value"],
                record["spending_velocity"],
                record["cancellation_rate"],
                record["preferred_shopping_hour"],
                record["churn_probability"],
                record["is_churn"],
                record["model_version"],
                record["feature_version"],
                record.get("response_time_ms", None),
            ))
    except Exception:
        logger.exception("Error inserting churn prediction")
        raise

def get_online_features(customer_id: str) -> dict:
    if os.getenv("USE_BIGQUERY", "false").lower() == "true":
        from app.db_bigquery import get_cloud_customer_features
        return get_cloud_customer_features(customer_id)

    # Local Mode: Query Postgres online_customer_features table
    try:
        with get_db_cursor() as cursor:
            cursor.execute("""
                SELECT recency, frequency, avg_order_value, spending_velocity, cancellation_rate, preferred_shopping_hour
                FROM online_customer_features
                WHERE customer_id = %s
            """, (str(customer_id),))
            row = cursor.fetchone()
            if row:
                logger.info("Successfully fetched features for customer %s from PostgreSQL online features table", customer_id)
                return {
                    "customer_id": customer_id,
                    "recency": row[0],
                    "frequency": row[1],
                    "avg_order_value": row[2],
                    "spending_velocity": row[3],
                    "cancellation_rate": row[4],
                    "preferred_shopping_hour": row[5]
                }
    except Exception as e:
        logger.error("Failed to query features from PostgreSQL: %s", e)
                
    return None

def init_pgvector_table():
    """Initializes pgvector extension and product_catalog_vectors table in PostgreSQL."""
    try:
        with get_db_cursor(commit=True) as cursor:
            cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS product_catalog_vectors (
                    stock_code VARCHAR(50) NOT NULL,
                    tenant_id VARCHAR(50) NOT NULL DEFAULT 'giftshop_uk',
                    description TEXT NOT NULL,
                    category VARCHAR(100),
                    unit_price DOUBLE PRECISION NOT NULL,
                    document_text TEXT NOT NULL,
                    embedding vector(768),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (stock_code, tenant_id)
                );
                CREATE INDEX IF NOT EXISTS product_vector_idx 
                ON product_catalog_vectors 
                USING hnsw (embedding vector_cosine_ops);
                CREATE INDEX IF NOT EXISTS idx_product_catalog_tenant 
                ON product_catalog_vectors (tenant_id);
            """)
        logger.info("pgvector schema and HNSW index successfully initialized.")
    except Exception as e:
        logger.warning("pgvector initialization failed: %s", e)

def search_product_catalog_pgvector(query_vector: list, budget_max: float = None, top_k: int = 4) -> list:
    """
    Executes a sub-millisecond similarity search using pgvector in PostgreSQL
    with optional budget price filtering.
    """
    try:
        with get_db_cursor() as cursor:
            # Format vector as string literal for pgvector: '[0.12, 0.34, ...]'
            vector_str = "[" + ",".join(map(str, query_vector)) + "]"
            
            if budget_max is not None and budget_max > 0:
                query = """
                    SELECT stock_code, description, category, unit_price, document_text,
                           1 - (embedding <=> %s::vector) AS similarity
                    FROM product_catalog_vectors
                    WHERE unit_price <= %s
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s;
                """
                cursor.execute(query, (vector_str, float(budget_max), vector_str, int(top_k)))
            else:
                query = """
                    SELECT stock_code, description, category, unit_price, document_text,
                           1 - (embedding <=> %s::vector) AS similarity
                    FROM product_catalog_vectors
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s;
                """
                cursor.execute(query, (vector_str, vector_str, int(top_k)))
                
            rows = cursor.fetchall()
            results = []
            for r in rows:
                results.append({
                    "stock_code": r[0],
                    "description": r[1],
                    "category": r[2] if r[2] else "General Merchandise",
                    "unit_price": float(r[3]),
                    "document_text": r[4],
                    "similarity": round(float(r[5]), 4)
                })
            return results
    except Exception as e:
        logger.warning("pgvector search failed: %s. Returning empty results.", e)
        return []



    