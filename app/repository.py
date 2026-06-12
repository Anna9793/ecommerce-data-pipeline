import pandas as pd
from app.db_postgres import get_connection
from app.db_postgres import insert_prediction

def get_total_predictions():

    conn = get_connection()
    cursor = conn.cursor()

    try:
        cursor.execute("""
        SELECT COUNT(*)
        FROM predictions;
        """)

        count = cursor.fetchone()[0]

        return int(count)

    finally:
        cursor.close()
        conn.close()

def get_average_response_time():

    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        cursor.execute("""
        SELECT AVG(response_time_ms)
        FROM predictions;
        """)

        avg_response = cursor.fetchone()[0]

        return int(avg_response)

    finally:
        cursor.close()
        conn.close()

def get_predictions_by_model_version():

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