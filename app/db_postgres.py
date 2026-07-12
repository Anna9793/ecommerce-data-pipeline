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
                churn_probability,
                is_churn,
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


    