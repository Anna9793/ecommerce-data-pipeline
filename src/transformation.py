"""
Legacy transformation module - Adapted to Medallion Architecture (Silver Layer)
Maintained for backward compatibility. Direct calls should prefer src.silver.
"""

import os
import logging
import pandas as pd
from config.paths import SILVER_RETAIL
from src.silver import run_silver, add_transaction_enrichments


def load_clean_data(path=SILVER_RETAIL, **kwargs):
    """
    Backward-compatible loader.
    """
    if not os.path.exists(path):
        run_silver()
    return pd.read_csv(path, parse_dates=["invoice_date"])


def add_order_value(df):
    logging.info("Creating order_value feature")
    df["order_value"] = (df["quantity"] * df["unit_price"]).round(2)
    return df


def add_time_features(df):
    logging.info("Extracting time features")
    if "invoice_date" in df.columns:
        if not pd.api.types.is_datetime64_any_dtype(df["invoice_date"]):
            df["invoice_date"] = pd.to_datetime(df["invoice_date"])
        df["invoice_year"] = df["invoice_date"].dt.year
        df["invoice_month"] = df["invoice_date"].dt.month
        df["invoice_hour"] = df["invoice_date"].dt.hour
    return df


def transform_data(df):
    return add_transaction_enrichments(df)


def save_transformed_data(df, path=SILVER_RETAIL):
    df.to_csv(path, index=False)


def run_transformation():
    return run_silver()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    run_transformation()