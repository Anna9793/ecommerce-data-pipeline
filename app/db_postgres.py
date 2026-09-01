import os
import logging
import psycopg2
from dotenv import load_dotenv

load_dotenv()

def get_connection():
    return psycopg2.connect(
        dbname=os.getenv("POSTGRES_DB"),
        user=os.getenv("POSTGRES_USER"),
        password=os.getenv("POSTGRES_PASSWORD"),
        host=os.getenv("POSTGRES_HOST"),
        port=os.getenv("POSTGRES_PORT"),
    )

def insert_prediction(record):
    if os.getenv("USE_BIGQUERY", "false").lower() == "true":
        return insert_prediction_bigquery(record)

    conn = None

    cursor = None
    
    try:
        conn = get_connection()
        cursor = conn.cursor()

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
            record["response_time_ms"],
        ))

        conn.commit()



    except Exception:
        logging.exception(
            "Error inserting prediction")
        raise

    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()

def insert_churn_prediction(record):
    if os.getenv("USE_BIGQUERY", "false").lower() == "true":
        return insert_churn_prediction_bigquery(record)

    conn = None

    cursor = None
    try:
        conn = get_connection()
        cursor = conn.cursor()
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
            record["response_time_ms"],
        ))
        conn.commit()
    except Exception:
        logging.exception("Error inserting churn prediction")
        raise
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()

def insert_prediction_bigquery(record):
    from google.cloud import bigquery
    from datetime import datetime, timezone
    
    logging.info("Logging segmentation prediction to BigQuery predictions_log table")
    record = record.copy()
    record["created_at"] = datetime.now(timezone.utc).isoformat()
    
    if record.get("customer_id") is None:
        record["customer_id"] = ""
    else:
        record["customer_id"] = str(record["customer_id"])

    project_id = os.getenv("GCP_PROJECT", "anna-ml-pipeline")
    client = bigquery.Client(project=project_id)
    table_id = f"{project_id}.retail_data.predictions_log"

    
    errors = client.insert_rows_json(table_id, [record])
    if errors:
        logging.error("BigQuery insert errors: %s", errors)
        raise RuntimeError(f"Errors inserting rows to BigQuery: {errors}")

def insert_churn_prediction_bigquery(record):
    from google.cloud import bigquery
    from datetime import datetime, timezone
    
    logging.info("Logging churn prediction to BigQuery churn_predictions_log table")
    record = record.copy()
    record["created_at"] = datetime.now(timezone.utc).isoformat()
    
    if record.get("customer_id") is None:
        record["customer_id"] = ""
    else:
        record["customer_id"] = str(record["customer_id"])

    project_id = os.getenv("GCP_PROJECT", "anna-ml-pipeline")
    client = bigquery.Client(project=project_id)
    table_id = f"{project_id}.retail_data.churn_predictions_log"

    
    errors = client.insert_rows_json(table_id, [record])
    if errors:
        logging.error("BigQuery churn insert errors: %s", errors)
        raise RuntimeError(f"Errors inserting churn rows to BigQuery: {errors}")

def get_online_features(customer_id: str) -> dict:
    use_bigquery = os.getenv("USE_BIGQUERY", "false").lower() == "true"
    
    if use_bigquery:
        # Try Firestore first (our Online Feature Store)
        try:
            from google.cloud import firestore
            project_id = os.getenv("GCP_PROJECT", "anna-ml-pipeline")
            db = firestore.Client(project=project_id)
            doc_ref = db.collection("online_customer_features").document(str(customer_id))
            doc = doc_ref.get()
            if doc.exists:
                logging.info("Successfully fetched features for customer %s from Firestore", customer_id)
                return doc.to_dict()
        except Exception as e:
            logging.warning("Failed to fetch features from Firestore: %s. Trying BigQuery fallback.", e)

        # Fallback to BigQuery view
        try:
            from google.cloud import bigquery
            project_id = os.getenv("GCP_PROJECT", "anna-ml-pipeline")
            client = bigquery.Client(project=project_id)
            query = f"""
                SELECT recency, frequency, avg_order_value, spending_velocity, cancellation_rate, preferred_shopping_hour 
                FROM `{project_id}.retail_data.rfm_features` 
                WHERE CAST(customer_id AS STRING) = '{customer_id}' 
                LIMIT 1
            """
            query_job = client.query(query)
            results = list(query_job.result())
            if results:
                row = results[0]
                logging.info("Successfully fetched features for customer %s from BigQuery view fallback", customer_id)
                return {
                    "customer_id": customer_id,
                    "recency": float(row.recency),
                    "frequency": int(row.frequency),
                    "avg_order_value": float(row.avg_order_value),
                    "spending_velocity": float(row.spending_velocity) if "spending_velocity" in row else 1.0,
                    "cancellation_rate": float(row.cancellation_rate) if "cancellation_rate" in row else 0.0,
                    "preferred_shopping_hour": int(row.preferred_shopping_hour) if "preferred_shopping_hour" in row else 12
                }
        except Exception as e:
            logging.error("Failed to fetch features from BigQuery fallback: %s", e)
            
    else:
        # Local Mode: Query Postgres online_customer_features table
        conn = None
        cursor = None
        try:
            conn = get_connection()
            cursor = conn.cursor()
            cursor.execute("""
                SELECT recency, frequency, avg_order_value, spending_velocity, cancellation_rate, preferred_shopping_hour
                FROM online_customer_features
                WHERE customer_id = %s
            """, (str(customer_id),))
            row = cursor.fetchone()
            if row:
                logging.info("Successfully fetched features for customer %s from PostgreSQL online features table", customer_id)
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
            logging.error("Failed to query features from PostgreSQL: %s", e)
        finally:
            if cursor:
                cursor.close()
            if conn:
                conn.close()
                
    return None

def init_pgvector_table():
    """Initializes pgvector extension and product_catalog_vectors table in PostgreSQL."""
    conn = None
    cursor = None
    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        cursor.execute("""
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
        """)
        conn.commit()
        logging.info("pgvector schema and HNSW index successfully initialized.")
    except Exception as e:
        logging.warning("pgvector initialization failed: %s", e)
        if conn:
            conn.rollback()
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()

def search_product_catalog_pgvector(query_vector: list, budget_max: float = None, top_k: int = 4) -> list:
    """
    Executes a sub-millisecond similarity search using pgvector in PostgreSQL
    with optional budget price filtering.
    """
    conn = None
    cursor = None
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
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
        logging.warning("pgvector search failed: %s. Returning empty results.", e)
        return []
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()



    