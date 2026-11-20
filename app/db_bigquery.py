import os
import logging
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

_bigquery_client = None


def get_bigquery_client():
    """Returns a reusable, thread-safe BigQuery client instance (lazy singleton)."""
    global _bigquery_client
    if _bigquery_client is None:
        from google.cloud import bigquery
        project_id = os.getenv("GCP_PROJECT", "anna-ml-pipeline")
        _bigquery_client = bigquery.Client(project=project_id)
    return _bigquery_client


def insert_clustering_prediction_bigquery(record: dict) -> None:
    """Logs clustering prediction to BigQuery clustering_predictions_log table."""
    logger.info("Logging clustering prediction to BigQuery clustering_predictions_log table")
    record = record.copy()
    record["created_at"] = datetime.now(timezone.utc).isoformat()
    
    if record.get("customer_id") is None:
        record["customer_id"] = ""
    else:
        record["customer_id"] = str(record["customer_id"])

    project_id = os.getenv("GCP_PROJECT", "anna-ml-pipeline")
    client = get_bigquery_client()
    table_id = f"{project_id}.retail_data.clustering_predictions_log"

    errors = client.insert_rows_json(table_id, [record])
    if errors:
        logger.error("BigQuery insert errors: %s", errors)
        raise RuntimeError(f"Errors inserting rows to BigQuery: {errors}")


def insert_churn_prediction_bigquery(record: dict) -> None:
    """Logs churn prediction to BigQuery churn_predictions_log table."""
    logger.info("Logging churn prediction to BigQuery churn_predictions_log table")
    record = record.copy()
    record["created_at"] = datetime.now(timezone.utc).isoformat()
    
    if record.get("customer_id") is None:
        record["customer_id"] = ""
    else:
        record["customer_id"] = str(record["customer_id"])

    project_id = os.getenv("GCP_PROJECT", "anna-ml-pipeline")
    client = get_bigquery_client()
    table_id = f"{project_id}.retail_data.churn_predictions_log"

    errors = client.insert_rows_json(table_id, [record])
    if errors:
        logger.error("BigQuery churn insert errors: %s", errors)
        raise RuntimeError(f"Errors inserting churn rows to BigQuery: {errors}")


def get_cloud_customer_features(customer_id: str) -> dict:
    """
    Fetches online customer features from Google Cloud:
    1. Attempts low-latency lookup from Google Cloud Firestore (Feature Store).
    2. Falls back to BigQuery view rfm_features if Firestore misses or fails.
    """
    # 1. Try Firestore first (Online Feature Store)
    try:
        from google.cloud import firestore
        project_id = os.getenv("GCP_PROJECT", "anna-ml-pipeline")
        db = firestore.Client(project=project_id)
        doc_ref = db.collection("online_customer_features").document(str(customer_id))
        doc = doc_ref.get()
        if doc.exists:
            logger.info("Successfully fetched features for customer %s from Firestore", customer_id)
            return doc.to_dict()
    except Exception as e:
        logger.warning("Failed to fetch features from Firestore: %s. Trying BigQuery fallback.", e)

    # 2. Fallback to BigQuery view
    try:
        from google.cloud import bigquery
        project_id = os.getenv("GCP_PROJECT", "anna-ml-pipeline")
        client = get_bigquery_client()
        query = f"""
            SELECT recency, frequency, avg_order_value, spending_velocity, cancellation_rate, preferred_shopping_hour 
            FROM `{project_id}.retail_data.customer_features` 
            WHERE CAST(customer_id AS STRING) = @customer_id 
            LIMIT 1
        """
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("customer_id", "STRING", str(customer_id))
            ]
        )
        query_job = client.query(query, job_config=job_config)
        results = list(query_job.result())
        if results:
            row = results[0]
            logger.info("Successfully fetched features for customer %s from BigQuery view fallback", customer_id)
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
        logger.error("Failed to fetch features from BigQuery fallback: %s", e)

    return None
