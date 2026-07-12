import os
import logging
import pandas as pd
from app.db_postgres import get_connection
from app.db_postgres import insert_prediction

def query_bigquery(query_str):
    from google.cloud import bigquery
    project_id = os.getenv("GCP_PROJECT", "anna-ml-pipeline")
    client = bigquery.Client(project=project_id)
    query_job = client.query(query_str)
    return query_job.to_dataframe()

def query_bigquery_scalar(query_str):
    from google.cloud import bigquery
    project_id = os.getenv("GCP_PROJECT", "anna-ml-pipeline")
    client = bigquery.Client(project=project_id)
    query_job = client.query(query_str)
    results = query_job.result()
    for row in results:
        return row[0]
    return None

def get_total_predictions():
    if os.getenv("USE_BIGQUERY", "false").lower() == "true":
        project_id = os.getenv("GCP_PROJECT", "anna-ml-pipeline")
        val = query_bigquery_scalar(f"SELECT COUNT(*) FROM `{project_id}.retail_data.predictions_log`")
        return int(val) if val is not None else 0

    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT COUNT(*) FROM predictions;")
        count = cursor.fetchone()[0]
        return int(count)
    finally:
        cursor.close()
        conn.close()

def get_average_response_time():
    if os.getenv("USE_BIGQUERY", "false").lower() == "true":
        project_id = os.getenv("GCP_PROJECT", "anna-ml-pipeline")
        val = query_bigquery_scalar(f"SELECT AVG(response_time_ms) FROM `{project_id}.retail_data.predictions_log`")
        return int(val) if val is not None else 0

    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT AVG(response_time_ms) FROM predictions;")
        avg_response = cursor.fetchone()[0]
        return int(avg_response) if avg_response is not None else 0
    finally:
        cursor.close()
        conn.close()

def get_predictions_by_model_version():
    if os.getenv("USE_BIGQUERY", "false").lower() == "true":
        project_id = os.getenv("GCP_PROJECT", "anna-ml-pipeline")
        query = f"""
        SELECT
            model_version,
            COUNT(*) as predictions
        FROM `{project_id}.retail_data.predictions_log`
        GROUP BY model_version
        ORDER BY CAST(model_version AS INT64);
        """
        return query_bigquery(query)

    conn = get_connection()
    version_query = """
    SELECT
        model_version,
        COUNT (*) as predictions
    FROM predictions
    GROUP BY model_version
    ORDER BY CAST(model_version AS INTEGER);
    """
    version_df = pd.read_sql(version_query, conn)
    conn.close()
    return version_df

def get_latest_predictions(limit=10):
    if os.getenv("USE_BIGQUERY", "false").lower() == "true":
        project_id = os.getenv("GCP_PROJECT", "anna-ml-pipeline")
        query = f"""
        SELECT *
        FROM `{project_id}.retail_data.predictions_log`
        ORDER BY created_at DESC
        LIMIT {limit};
        """
        return query_bigquery(query)

    conn = get_connection()
    latest_query = f"""
    SELECT *
    FROM predictions
    ORDER BY created_at DESC
    LIMIT {limit};
    """
    latest_df = pd.read_sql(latest_query, conn)
    conn.close()
    return latest_df

def get_segment_distribution():
    if os.getenv("USE_BIGQUERY", "false").lower() == "true":
        project_id = os.getenv("GCP_PROJECT", "anna-ml-pipeline")
        query = f"""
        SELECT
            label,
            COUNT(*) as count
        FROM `{project_id}.retail_data.predictions_log`
        GROUP BY label
        ORDER BY count DESC;
        """
        return query_bigquery(query)

    conn = get_connection()
    segment_query = """
    SELECT
        label,
        COUNT(*) as count
    FROM predictions
    GROUP BY label
    ORDER BY count DESC;
    """
    segment_df = pd.read_sql(segment_query, conn)
    conn.close()
    return segment_df

def save_prediction(record):
    insert_prediction(record)

def get_total_churn_predictions():
    if os.getenv("USE_BIGQUERY", "false").lower() == "true":
        project_id = os.getenv("GCP_PROJECT", "anna-ml-pipeline")
        val = query_bigquery_scalar(f"SELECT COUNT(*) FROM `{project_id}.retail_data.churn_predictions_log`")
        return int(val) if val is not None else 0

    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT COUNT(*) FROM churn_predictions;")
        count = cursor.fetchone()[0]
        return int(count)
    finally:
        cursor.close()
        conn.close()

def get_average_churn_probability():
    if os.getenv("USE_BIGQUERY", "false").lower() == "true":
        project_id = os.getenv("GCP_PROJECT", "anna-ml-pipeline")
        val = query_bigquery_scalar(f"SELECT AVG(churn_probability) FROM `{project_id}.retail_data.churn_predictions_log`")
        return float(val) if val is not None else 0.0

    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT AVG(churn_probability) FROM churn_predictions;")
        avg = cursor.fetchone()[0]
        return float(avg) if avg is not None else 0.0
    finally:
        cursor.close()
        conn.close()

def get_latest_churn_predictions(limit=10):
    if os.getenv("USE_BIGQUERY", "false").lower() == "true":
        project_id = os.getenv("GCP_PROJECT", "anna-ml-pipeline")
        query = f"""
        SELECT *
        FROM `{project_id}.retail_data.churn_predictions_log`
        ORDER BY created_at DESC
        LIMIT {limit};
        """
        return query_bigquery(query)

    conn = get_connection()
    latest_query = f"""
    SELECT *
    FROM churn_predictions
    ORDER BY created_at DESC
    LIMIT {limit};
    """
    latest_df = pd.read_sql(latest_query, conn)
    conn.close()
    return latest_df