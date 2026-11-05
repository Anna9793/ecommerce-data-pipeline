"""
Medallion Architecture - Bronze Layer (Raw Ingestion & Landing Zone)
Responsible for raw, immutable data access and ingestion:
- Reads the raw source dataset (RAW_EXCEL from config.paths)
- Lands raw immutable transactions into BRONZE_RETAIL (CSV)
- Provides load_bronze() for downstream consumption (by Silver layer)
- Provides optional utility to upload raw data to BigQuery
"""

import os
import logging
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv

from config.paths import BRONZE_RETAIL, RAW_EXCEL

load_dotenv()


def ingest_raw_source(source_path=RAW_EXCEL, target_path=BRONZE_RETAIL):
    """
    Ingests the raw source data (e.g. Online_Retail.xlsx) and persists
    it as an untransformed, immutable Bronze CSV.
    """
    source_path = Path(source_path)
    target_path = Path(target_path)

    if not source_path.exists():
        raise FileNotFoundError(f"Source raw file not found at: {source_path}")

    logging.info("Bronze Layer: Ingesting raw data from %s -> %s", source_path, target_path)
    if source_path.suffix.lower() in [".xlsx", ".xls"]:
        df = pd.read_excel(source_path)
    else:
        df = pd.read_csv(source_path)

    target_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(target_path, index=False)
    logging.info("Bronze Layer: Saved %d raw records to %s", len(df), target_path)
    return df


def load_bronze(path=BRONZE_RETAIL):
    """
    Loads raw untransformed transactional data from the Bronze layer.
    If the Bronze CSV does not exist but the source Excel exists, it ingests it automatically.
    """
    path = Path(path)
    if not path.exists():
        logging.warning("Bronze file not found at %s. Ingesting from raw source...", path)
        if Path(RAW_EXCEL).exists():
            return ingest_raw_source(RAW_EXCEL, path)
        raise FileNotFoundError(f"Neither Bronze CSV ({path}) nor Raw Source ({RAW_EXCEL}) found.")

    logging.info("Bronze Layer: Loading raw transactions from %s", path)
    return pd.read_csv(path)


def upload_bronze_to_bigquery(path=BRONZE_RETAIL, project_id=None, if_exists="replace"):
    """
    Utility: Uploads raw local Bronze CSV to BigQuery table retail_data.transactions.
    """
    if project_id is None:
        project_id = os.getenv("GCP_PROJECT", "anna-ml-pipeline")
        
    logging.info("Bronze Layer: Uploading raw Bronze data (%s) to BigQuery table %s.retail_data.transactions", path, project_id)
    import pandas_gbq
    df = pd.read_csv(path)
    destination_table = "retail_data.transactions"
    pandas_gbq.to_gbq(df, destination_table=destination_table, project_id=project_id, if_exists=if_exists)
    logging.info("Bronze Layer: Upload completed successfully (%d rows).", len(df))


def run_bronze():
    """
    Executes the Bronze ingestion step.
    """
    logging.info("Starting Bronze Layer ingestion...")
    return load_bronze()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    run_bronze()
