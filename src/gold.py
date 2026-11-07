"""Medallion Architecture - Gold Layer: customer-level behavioral aggregations and ML feature store."""

import os
import logging
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from config.paths import GOLD_RFM, SILVER_RETAIL
from src.silver import run_silver

load_dotenv()


def compute_gold_features(silver_df):
    """Computes customer-level RFM and behavioral features for active purchasers."""
    df = silver_df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df["invoice_date"]):
        df["invoice_date"] = pd.to_datetime(df["invoice_date"])
        
    max_date = df["invoice_date"].max()
    valid_sales = df[df["quantity"] > 0]

    # 1. Base RFM (Recency, Frequency, Monetary computed exclusively on valid purchases)
    rfm = (
        valid_sales.groupby("customer_id")
        .agg(
            recency=("invoice_date", lambda x: (max_date - x.max()).days),
            frequency=("invoice_no", "nunique"),
            monetary=("order_value", "sum")
        )
        .reset_index()
    )
    rfm["monetary"] = rfm["monetary"].round(2)

    # 2. Avg Order Value
    rfm["avg_order_value"] = (
        rfm["monetary"] / rfm["frequency"].replace(0, 1)
    ).round(2)

    # 3. Spending Velocity (30-day pace vs 90-day normalized pace)
    date_30 = max_date - pd.Timedelta(days=30)
    date_90 = max_date - pd.Timedelta(days=90)

    monetary_30 = (
        valid_sales[valid_sales["invoice_date"] >= date_30]
        .groupby("customer_id")
        .agg(monetary_30=("order_value", "sum"))
        .reset_index()
    )

    monetary_90 = (
        valid_sales[valid_sales["invoice_date"] >= date_90]
        .groupby("customer_id")
        .agg(monetary_90=("order_value", "sum"))
        .reset_index()
    )

    rfm = rfm.merge(monetary_30, on="customer_id", how="left")
    rfm = rfm.merge(monetary_90, on="customer_id", how="left")
    rfm["monetary_30"] = rfm["monetary_30"].fillna(0.0)

    # Velocity: monetary_30 / (monetary_90 / 3.0), default 1.0 (neutral velocity)
    denom = rfm["monetary_90"] / 3.0
    rfm["spending_velocity"] = (rfm["monetary_30"] / denom).fillna(1.0).round(2)

    # 4. Cancellation Rate
    cancelled = (
        df[df["invoice_no"].str.startswith("C", na=False)]
        .groupby("customer_id")
        .agg(cancelled_orders=("invoice_no", "nunique"))
        .reset_index()
    )

    total_orders = (
        df.groupby("customer_id")
        .agg(total_orders=("invoice_no", "nunique"))
        .reset_index()
    )

    rfm = rfm.merge(cancelled, on="customer_id", how="left")
    rfm = rfm.merge(total_orders, on="customer_id", how="left")
    rfm["cancelled_orders"] = rfm["cancelled_orders"].fillna(0)
    rfm["total_orders"] = rfm["total_orders"].fillna(1)

    rfm["cancellation_rate"] = (
        rfm["cancelled_orders"] / rfm["total_orders"].replace(0, 1)
    ).round(4)

    # 5. Preferred Shopping Hour
    if "invoice_hour" not in df.columns:
        df["invoice_hour"] = df["invoice_date"].dt.hour

    def get_mode_hour(x):
        mode_series = x.mode()
        return int(mode_series.iloc[0]) if not mode_series.empty else 12

    pref_hour = (
        df.groupby("customer_id")
        .agg(preferred_shopping_hour=("invoice_hour", get_mode_hour))
        .reset_index()
    )

    rfm = rfm.merge(pref_hour, on="customer_id", how="left")
    rfm["preferred_shopping_hour"] = rfm["preferred_shopping_hour"].fillna(12).astype(int)

    # 6. Filter to valid buyers (frequency > 0)
    rfm = rfm[rfm["frequency"] > 0]

    feature_cols = [
        "customer_id",
        "recency",
        "frequency",
        "avg_order_value",
        "spending_velocity",
        "cancellation_rate",
        "preferred_shopping_hour"
    ]
    return rfm[feature_cols].reset_index(drop=True)


def save_gold(df, path=GOLD_RFM):
    logging.info("Gold Layer: Saving customer feature store to %s", path)
    df.to_csv(path, index=False)


def load_gold(path=GOLD_RFM, use_bigquery=None):
    """
    Loads customer feature store from the Gold layer.
    
    Parameters
    ----------
    path : Path or str, default GOLD_RFM
        Path to local Gold CSV file.
    use_bigquery : bool, optional
        If True, loads from BigQuery view retail_data.rfm_features.
        If False, loads from local CSV file.
        If None, defaults to USE_BIGQUERY environment variable.
    """
    if use_bigquery is None:
        use_bigquery = os.getenv("USE_BIGQUERY", "false").lower() == "true"

    if use_bigquery:
        project_id = os.getenv("GCP_PROJECT", "anna-ml-pipeline")
        logging.info("Gold Layer [Cloud]: Loading customer features from BigQuery view: %s.retail_data.rfm_features", project_id)
        import pandas_gbq
        query = f"SELECT * FROM `{project_id}.retail_data.rfm_features`"
        df = pandas_gbq.read_gbq(query, project_id=project_id)
        # Drop residual nulls if view in BQ has any anomalies
        df = df.dropna(subset=["recency", "frequency", "avg_order_value"])
        if "customer_id" in df.columns:
            df["customer_id"] = df["customer_id"].astype(str)
        return df

    logging.info("Gold Layer [Local]: Loading customer features from %s", path)
    if not os.path.exists(path):
        logging.warning("Gold file not found at %s. Running Gold pipeline to generate it.", path)
        return run_gold()

    df = pd.read_csv(path, dtype={"customer_id": str})
    return df


def run_gold(**kwargs):
    """
    Executes the Gold pipeline locally:
    Reads Silver CSV -> Computes RFM & Behavioral Features -> Persists Gold CSV.
    """
    logging.info("Starting Gold Layer processing...")
    if not SILVER_RETAIL.exists():
        run_silver()

    silver_df = pd.read_csv(SILVER_RETAIL, parse_dates=["invoice_date"])
    logging.info("Silver input rows received: %d", len(silver_df))

    gold_df = compute_gold_features(silver_df)
    logging.info("Gold customer profiles produced: %d (0 NaNs guaranteed)", len(gold_df))

    save_gold(gold_df, GOLD_RFM)
    return gold_df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    run_gold()
