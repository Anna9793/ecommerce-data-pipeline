import os
import logging
import pandas as pd
import numpy as np
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
        from app.db_postgres import get_connection
        logging.info("Connecting to local PostgreSQL database...")
        conn = get_connection()
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
        
        # We can execute batch execution
        from psycopg2.extras import execute_batch
        execute_batch(cursor, query, data_tuples)
        conn.commit()
        
        cursor.close()
        conn.close()
        logging.info("PostgreSQL sync complete! Synced %d profiles.", len(data_tuples))
        return True
    except Exception as e:
        logging.exception("Failed to sync customer feature profiles to PostgreSQL: %s", e)
        return False

def sync_feature_store():
    use_bigquery = os.getenv("USE_BIGQUERY", "false").lower() == "true"
    project_id = os.getenv("GCP_PROJECT", "anna-ml-pipeline")
    
    # 1. Load latest aggregated features
    from src.rfm_features import load_rfm
    try:
        df = load_rfm(use_bigquery=use_bigquery)
    except Exception as e:
        logging.error("Failed to load RFM features for syncing: %s. Reading local fallback.", e)
        df = pd.read_csv("data/processed/rfm_customers.csv")
        
    # Ensure columns exist
    required_cols = ["customer_id", "recency", "frequency", "avg_order_value", "spending_velocity", "cancellation_rate", "preferred_shopping_hour"]
    for col in required_cols:
        if col not in df.columns:
            # Inject defaults if missing
            if col == "spending_velocity":
                df[col] = 1.0
            elif col == "cancellation_rate":
                df[col] = 0.0
            elif col == "preferred_shopping_hour":
                df[col] = 12
            else:
                df[col] = 0.0
                
    # Clean customer_id formatting (prevent floats like 12345.0, cast to clean strings)
    df["customer_id"] = df["customer_id"].apply(lambda x: str(int(float(x))) if pd.notnull(x) and str(x) != "nan" else "")
    df = df[df["customer_id"] != ""]
    
    # 2. Sync to appropriate Online Feature Store
    if use_bigquery:
        success = sync_to_firestore(df, project_id)
    else:
        success = sync_to_postgres(df)
        
    if not success:
        logging.error("Feature store synchronization failed.")
        
if __name__ == "__main__":
    sync_feature_store()
