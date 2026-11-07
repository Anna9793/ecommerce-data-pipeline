"""Medallion Architecture - Silver Layer: transactional cleaning, typing, and enrichment."""

import os
import logging
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv

from config.paths import BRONZE_RETAIL, SILVER_RETAIL
from src.bronze import run_bronze

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
    return df.dropna(subset=["customer_id"])


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


def run_silver():
    """
    Executes the Silver pipeline locally:
    Reads Bronze CSV -> Cleans & Enriches in memory -> Persists Silver CSV.
    """
    logging.info("Starting Silver Layer processing...")
    if not BRONZE_RETAIL.exists():
        run_bronze()

    raw_df = pd.read_csv(BRONZE_RETAIL)
    logging.info("Bronze rows received: %d", len(raw_df))
    silver_df = clean_and_enrich_transactions(raw_df)
    logging.info("Silver curated rows produced: %d", len(silver_df))

    silver_df.to_csv(SILVER_RETAIL, index=False)
    logging.info("Silver Layer: Saved curated transactions to %s", SILVER_RETAIL)
    return silver_df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    run_silver()
