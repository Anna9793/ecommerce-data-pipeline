"""
Medallion Architecture - Bronze Layer (Raw Ingestion)
Responsible for raw, immutable data access:
- Local Mode: data/raw/online_retail.csv
- Cloud Mode: BigQuery table retail_data.transactions
"""

import os
import logging
import pandas as pd
from dotenv import load_dotenv
from config.paths import BRONZE_RETAIL

load_dotenv()


def load_bronze(path=BRONZE_RETAIL, use_bigquery=None):
    """
    Loads raw transactional dataset from the Bronze layer.
    
    Parameters
    ----------
    path : Path or str, default BRONZE_RETAIL
        Path to local raw CSV file.
    use_bigquery : bool, optional
        If True, loads from BigQuery retail_data.transactions.
        If False, loads from local CSV.
        If None, falls back to USE_BIGQUERY environment variable.
    """
    if use_bigquery is None:
        use_bigquery = os.getenv("USE_BIGQUERY", "false").lower() == "true"

    if use_bigquery:
        project_id = os.getenv("GCP_PROJECT", "anna-ml-pipeline")
        logging.info("Bronze Layer [Cloud]: Loading raw transactions from BigQuery: %s.retail_data.transactions", project_id)
        import pandas_gbq
        query = f"SELECT * FROM `{project_id}.retail_data.transactions`"
        df = pandas_gbq.read_gbq(query, project_id=project_id)
        return df

    logging.info("Bronze Layer [Local]: Loading raw transactions from %s", path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Bronze raw file not found at {path}")
    df = pd.read_csv(path)
    return df


def ingest_raw_to_bigquery(path=BRONZE_RETAIL, project_id=None, if_exists="replace"):
    """
    Utility: Uploads raw local CSV to populate or refresh BigQuery Bronze table retail_data.transactions.
    """
    if project_id is None:
        project_id = os.getenv("GCP_PROJECT", "anna-ml-pipeline")
        
    logging.info("Bronze Layer: Ingesting raw CSV (%s) to BigQuery table retail_data.transactions", path)
    import pandas_gbq
    df = pd.read_csv(path)
    destination_table = "retail_data.transactions"
    pandas_gbq.to_gbq(df, destination_table=destination_table, project_id=project_id, if_exists=if_exists)
    logging.info("Bronze Layer: Ingestion to %s completed (%d rows).", destination_table, len(df))


def run_bronze():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    df = load_bronze()
    logging.info("Bronze Layer: Verified %d raw transactions loaded successfully.", len(df))
    return df


if __name__ == "__main__":
    run_bronze()
