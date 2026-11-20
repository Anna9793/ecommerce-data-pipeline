import os
import logging
from typing import Optional
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


def get_cloud_customer_features(customer_id: str) -> Optional[dict]:
    """
    Fetches online customer features from Google Cloud Firestore (Low-latency Online Feature Store).
    Returns None if customer does not exist or lookup fails.
    """
    try:
        from google.cloud import firestore
        project_id = os.getenv("GCP_PROJECT", "anna-ml-pipeline")
        db = firestore.Client(project=project_id)
        doc_ref = db.collection("online_customer_features").document(str(customer_id))
        doc = doc_ref.get()
        if doc.exists:
            logger.info("Successfully fetched features for customer %s from Firestore", customer_id)
            return doc.to_dict()
        logger.info("Customer %s not found in Firestore Online Feature Store", customer_id)
    except Exception as e:
        logger.error("Failed to fetch features from Firestore: %s", e)

    return None
