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


    except Exception:
        logging.exception(
            "Error inserting prediction")
        raise

    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()
    