import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import logging
import pandas as pd
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
load_dotenv()

def sync_to_firestore(df, project_id):
    try:
        from google.cloud import firestore
        logging.info("Initializing Firestore client for project: %s...", project_id)
        db = firestore.Client(project=project_id)
        
        collection_ref = db.collection("online_customer_features")
        
        logging.info("Syncing %d customer feature profiles to Firestore in batches...", len(df))
        
        # Firestore batch supports up to 500 operations
        batch = db.batch()
        batch_count = 0
        total_synced = 0
        
        for idx, row in df.iterrows():
            customer_id = str(row["customer_id"])
            if not customer_id or customer_id == "nan":
                continue
                
            doc_ref = collection_ref.document(customer_id)
            doc_data = {
                "customer_id": customer_id,
                "recency": float(row["recency"]),
                "frequency": int(row["frequency"]),
                "avg_order_value": float(row["avg_order_value"]),
                "spending_velocity": float(row["spending_velocity"]),
                "cancellation_rate": float(row["cancellation_rate"]),
                "preferred_shopping_hour": int(row["preferred_shopping_hour"])
            }
            
            batch.set(doc_ref, doc_data)
            batch_count += 1
            
            if batch_count >= 500:
                batch.commit()
                total_synced += batch_count
                logging.info("Committed batch: %d profiles synced so far...", total_synced)
                batch = db.batch()
                batch_count = 0
                
        if batch_count > 0:
            batch.commit()
            total_synced += batch_count
            
        logging.info("Firestore sync complete! Total synced: %d customer profiles.", total_synced)
        return True
    except Exception as e:
        logging.exception("Failed to sync customer feature profiles to Firestore: %s", e)
        return False

def sync_to_postgres(df):
    try:
        from app.db_postgres import get_connection, release_connection
        logging.info("Connecting to local PostgreSQL database...")
        conn = get_connection()
        if not conn:
            logging.error("Could not obtain database connection for PostgreSQL sync.")
            return False
            
        try:
            cursor = conn.cursor()
            logging.info("Upserting %d customer feature profiles to PostgreSQL...", len(df))
            
            # Prepare list of tuples
            data_tuples = []
            for idx, row in df.iterrows():
                customer_id = str(row["customer_id"])
                if not customer_id or customer_id == "nan":
                    continue
                data_tuples.append((
                    customer_id,
                    float(row["recency"]),
                    int(row["frequency"]),
                    float(row["avg_order_value"]),
                    float(row["spending_velocity"]),
                    float(row["cancellation_rate"]),
                    int(row["preferred_shopping_hour"])
                ))
                
            # Execute batched upsert
            query = """
                INSERT INTO online_customer_features (
                    customer_id, recency, frequency, avg_order_value, spending_velocity, cancellation_rate, preferred_shopping_hour
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (customer_id) DO UPDATE SET
                    recency = EXCLUDED.recency,
                    frequency = EXCLUDED.frequency,
                    avg_order_value = EXCLUDED.avg_order_value,
                    spending_velocity = EXCLUDED.spending_velocity,
                    cancellation_rate = EXCLUDED.cancellation_rate,
                    preferred_shopping_hour = EXCLUDED.preferred_shopping_hour,
                    updated_at = CURRENT_TIMESTAMP
            """
            
            from psycopg2.extras import execute_batch
            execute_batch(cursor, query, data_tuples)
            conn.commit()
            cursor.close()
            logging.info("PostgreSQL sync complete! Synced %d profiles.", len(data_tuples))
            return True
        finally:
            release_connection(conn)
    except Exception as e:
        logging.exception("Failed to sync customer feature profiles to PostgreSQL: %s", e)
        return False

def sync_feature_store():
    use_bigquery = os.getenv("USE_BIGQUERY", "false").lower() == "true"
    project_id = os.getenv("GCP_PROJECT", "anna-ml-pipeline")
    
    # 1. Load latest aggregated features
    from src.gold import load_gold
    if use_bigquery:
        logging.info("Cloud mode active: Loading RFM features from BigQuery...")
        try:
            df = load_gold(use_bigquery=True)
        except Exception as e:
            logging.error("Failed to load RFM features from BigQuery in cloud mode: %s", e)
            raise RuntimeError(f"BigQuery Gold RFM load failed: {e}") from e
    else:
        logging.info("Local mode active: Loading RFM features from local Gold layer...")
        df = load_gold(use_bigquery=False)
        
    # Validate Feature Store data contract (fail-fast on schema violation)
    required_cols = [
        "customer_id", "recency", "frequency", "avg_order_value",
        "spending_velocity", "cancellation_rate", "preferred_shopping_hour"
    ]
    missing = set(required_cols) - set(df.columns)
    if missing:
        raise ValueError(f"Gold layer schema violation! Missing required feature columns: {missing}")
                
    df["customer_id"] = df["customer_id"].astype(str)
    
    # 2. Sync to appropriate Online Feature Store
    if use_bigquery:
        success = sync_to_firestore(df, project_id)
    else:
        success = sync_to_postgres(df)
        
    if not success:
        logging.error("Feature store synchronization failed.")
        raise RuntimeError("Feature store synchronization failed.")
        
if __name__ == "__main__":
    sync_feature_store()
