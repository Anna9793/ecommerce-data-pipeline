"""
Medallion Architecture - Silver Layer (Cleaned & Curated Transactions)
Responsible for row-level sanitization, typing, and transactional enrichment:
- Eliminates duplicate rows and invalid customer IDs
- Filters strictly valid transactions (unit_price > 0, purchases or valid cancellations)
- Computes transactional enrichments: order_value (quantity * unit_price) and timestamp features
- Transforms Bronze raw data into the Silver curated dataset in a single, high-efficiency pass
"""

import os
import logging
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv

from config.paths import SILVER_RETAIL
from src.bronze import load_bronze

load_dotenv()


def standardize_column_names(df):
    df.columns = (
        df.columns
        .str.replace(r'(?<=[a-z])(?=[A-Z])', '_', regex=True)
        .str.lower()
    )
    return df


def remove_duplicates(df):
    return df.drop_duplicates()


def remove_missing_customer_ids(df):
    df = df.dropna(subset=["customer_id"])
    if "customer_id" in df.columns:
        invalid_ids = {"nan", "none", "guest", ""}
        df = df[~df["customer_id"].astype(str).str.lower().isin(invalid_ids)]
    return df


def cast_data_types(df):
    if "invoice_date" in df.columns and not pd.api.types.is_datetime64_any_dtype(df["invoice_date"]):
        df["invoice_date"] = pd.to_datetime(df["invoice_date"], errors="coerce")
    if "customer_id" in df.columns:
        df["customer_id"] = df["customer_id"].astype(str).str.replace(r"\.0$", "", regex=True)
    if "invoice_no" in df.columns:
        df["invoice_no"] = df["invoice_no"].astype(str)
    if "quantity" in df.columns:
        df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce")
    if "unit_price" in df.columns:
        df["unit_price"] = pd.to_numeric(df["unit_price"], errors="coerce")
    return df


def filter_valid_transactions(df):
    is_valid_purchase = df["quantity"] > 0
    is_valid_cancellation = (df["quantity"] < 0) & df["invoice_no"].str.startswith("C", na=False)
    return df[(df["unit_price"] > 0) & (is_valid_purchase | is_valid_cancellation)]


def add_transaction_enrichments(df):
    df["order_value"] = (df["quantity"] * df["unit_price"]).round(2)
    if "invoice_date" in df.columns and pd.api.types.is_datetime64_any_dtype(df["invoice_date"]):
        df["invoice_year"] = df["invoice_date"].dt.year
        df["invoice_month"] = df["invoice_date"].dt.month
        df["invoice_hour"] = df["invoice_date"].dt.hour
    return df


def clean_and_enrich_transactions(df):
    """
    Executes the end-to-end Silver transformation pipeline in memory.
    """
    df = standardize_column_names(df.copy())
    df = remove_duplicates(df)
    df = remove_missing_customer_ids(df)
    df = cast_data_types(df)
    df = filter_valid_transactions(df)
    df = add_transaction_enrichments(df)
    return df


def save_silver(df, path=SILVER_RETAIL):
    logging.info("Silver Layer: Saving curated transactions to %s", path)
    df.to_csv(path, index=False)


def load_silver(path=SILVER_RETAIL, **kwargs):
    """
    Loads curated transactions from the Silver layer.
    """
    logging.info("Silver Layer: Loading clean transactions from %s", path)
    if not os.path.exists(path):
        logging.warning("Silver file not found at %s. Running Silver pipeline to generate it.", path)
        return run_silver()

    df = pd.read_csv(
        path,
        parse_dates=["invoice_date"],
        dtype={"customer_id": str, "invoice_no": str}
    )
    return df


def run_silver(**kwargs):
    """
    Executes the Silver pipeline locally:
    Loads Bronze (raw) -> Cleans & Enriches in memory -> Persists Silver.
    """
    logging.info("Starting Silver Layer processing...")
    raw_df = load_bronze()
    logging.info("Bronze rows received: %d", len(raw_df))
    silver_df = clean_and_enrich_transactions(raw_df)
    logging.info("Silver curated rows produced: %d", len(silver_df))

    save_silver(silver_df, SILVER_RETAIL)
    return silver_df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    run_silver()
