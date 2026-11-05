"""
Medallion Architecture - Bronze Layer (Raw Ingestion & Landing Zone)
Responsible for raw, immutable data access and ingestion:
- Ingests raw Excel source (RAW_EXCEL from config.paths)
- Lands raw immutable transactions into BRONZE_RETAIL (CSV)
"""

import logging
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv

from config.paths import BRONZE_RETAIL, RAW_EXCEL

load_dotenv()


def ingest_raw_source(source_path=RAW_EXCEL, target_path=BRONZE_RETAIL):
    """
    Ingests the raw Excel source (Online_Retail.xlsx) and persists
    it as an untransformed, immutable Bronze CSV.
    """
    source_path = Path(source_path)
    target_path = Path(target_path)

    if not source_path.exists():
        raise FileNotFoundError(f"Source raw Excel file not found at: {source_path}")

    logging.info("Bronze Layer: Ingesting raw Excel %s -> %s", source_path, target_path)
    df = pd.read_excel(source_path)

    target_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(target_path, index=False)
    logging.info("Bronze Layer: Saved %d raw records to %s", len(df), target_path)
    return df


def run_bronze():
    """
    Executes the Bronze ingestion step:
    Ensures BRONZE_RETAIL exists; if not, ingests from RAW_EXCEL.
    """
    logging.info("Starting Bronze Layer ingestion...")
    if not BRONZE_RETAIL.exists():
        return ingest_raw_source(RAW_EXCEL, BRONZE_RETAIL)
    logging.info("Bronze file already exists at %s", BRONZE_RETAIL)
    return pd.read_csv(BRONZE_RETAIL)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    run_bronze()
