"""
Legacy RFM module - Adapted to Medallion Architecture (Gold Layer)
Maintained for backward compatibility. Direct calls should prefer src.gold.
"""

import os
import logging
import pandas as pd
from dotenv import load_dotenv
from config.paths import GOLD_RFM, RFM_CUSTOMERS, SILVER_RETAIL, FEATURE_RETAIL
from src.silver import load_silver
from src.gold import compute_gold_features, save_gold, load_gold, run_gold

load_dotenv()


def load_transactions(path=SILVER_RETAIL):
    return load_silver(path=path, use_bigquery=False)


def compute_rfm(df):
    return compute_gold_features(df)


def save_rfm(df, path=GOLD_RFM):
    save_gold(df, path=path)


def load_rfm(path=GOLD_RFM, use_bigquery=None):
    """
    Backward-compatible loader delegating to Gold layer.
    Prioritizes explicit use_bigquery argument over environment variable.
    """
    return load_gold(path=path, use_bigquery=use_bigquery)


def run_rfm_features():
    return run_gold(use_bigquery=False)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    run_rfm_features()