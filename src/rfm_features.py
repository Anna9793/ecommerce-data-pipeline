import os
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from dotenv import load_dotenv
from config.paths import FEATURE_RETAIL, RFM_CUSTOMERS

load_dotenv()


def load_transactions(path):
    logging.info("Loading transaction dataset from %s", path)
    df = pd.read_csv(path)
    df["invoice_date"] = pd.to_datetime(df["invoice_date"])
    return df


def compute_rfm(df):
    df = df.copy()
    df["invoice_date"] = pd.to_datetime(df["invoice_date"])
    df["invoice_no"] = df["invoice_no"].astype(str)
    df["order_value"] = pd.to_numeric(df["order_value"], errors="coerce").fillna(0.0)
    
    max_date = df["invoice_date"].max()
    
    # 1. Base RFM metrics (only using positive quantities for frequency/monetary)
    valid_sales = df[df["quantity"] > 0]
    
    rfm = (
        df.groupby("customer_id")
        .agg(
            recency=("invoice_date", lambda x: (max_date - x.max()).days)
        )
        .reset_index()
    )
    
    sales_metrics = (
        valid_sales.groupby("customer_id")
        .agg(
            frequency=("invoice_no", "nunique"),
            monetary=("order_value", "sum")
        )
        .reset_index()
    )
    
    rfm = rfm.merge(sales_metrics, on="customer_id", how="left")
    rfm["frequency"] = rfm["frequency"].fillna(0).astype(int)
    rfm["monetary"] = rfm["monetary"].fillna(0.0)
    
    rfm["avg_order_value"] = (
        rfm["monetary"] / rfm["frequency"].replace(0, 1)
    ).round(2)
    
    # 2. Spending Velocity
    date_30 = max_date - pd.Timedelta(days=30)
    date_90 = max_date - pd.Timedelta(days=90)
    
    monetary_30 = (
        valid_sales[valid_sales["invoice_date"] >= date_30]
        .groupby("customer_id")["order_value"]
        .sum()
        .rename("monetary_30")
    )
    
    monetary_90 = (
        valid_sales[valid_sales["invoice_date"] >= date_90]
        .groupby("customer_id")["order_value"]
        .sum()
        .rename("monetary_90")
    )
    
    rfm = rfm.join(monetary_30, on="customer_id", how="left")
    rfm = rfm.join(monetary_90, on="customer_id", how="left")
    rfm["monetary_30"] = rfm["monetary_30"].fillna(0.0)
    rfm["monetary_90"] = rfm["monetary_90"].fillna(0.0)
    
    # Velocity: monetary_30 / (monetary_90 / 3.0)
    denom = rfm["monetary_90"] / 3.0
    rfm["spending_velocity"] = (rfm["monetary_30"] / denom.replace(0.0, np.nan)).fillna(1.0).round(2)
    
    # 3. Cancellation Rate
    cancelled = (
        df[df["invoice_no"].str.startswith("C", na=False)]
        .groupby("customer_id")["invoice_no"]
        .nunique()
        .rename("cancelled_orders")
    )
    
    total_orders = (
        df.groupby("customer_id")["invoice_no"]
        .nunique()
        .rename("total_orders")
    )
    
    rfm = rfm.join(cancelled, on="customer_id", how="left")
    rfm = rfm.join(total_orders, on="customer_id", how="left")
    rfm["cancelled_orders"] = rfm["cancelled_orders"].fillna(0)
    rfm["total_orders"] = rfm["total_orders"].fillna(1)
    
    rfm["cancellation_rate"] = (rfm["cancelled_orders"] / rfm["total_orders"]).round(4)
    
    # 4. Preferred Shopping Hour
    df["hour"] = df["invoice_date"].dt.hour
    
    def get_mode_hour(x):
        mode_series = x.mode()
        return int(mode_series.iloc[0]) if not mode_series.empty else 12
        
    pref_hour = (
        df.groupby("customer_id")["hour"]
        .agg(get_mode_hour)
        .rename("preferred_shopping_hour")
    )
    
    rfm = rfm.join(pref_hour, on="customer_id", how="left")
    rfm["preferred_shopping_hour"] = rfm["preferred_shopping_hour"].fillna(12).astype(int)
    
    rfm = rfm[[
        "customer_id", 
        "recency", 
        "frequency", 
        "avg_order_value", 
        "spending_velocity", 
        "cancellation_rate", 
        "preferred_shopping_hour"
    ]]
    
    return rfm


def save_rfm(df, path):
    logging.info("Saving RFM dataset to %s", path)
    df.to_csv(path, index=False)

def load_rfm(path=RFM_CUSTOMERS, use_bigquery=False):
    if use_bigquery or os.getenv("USE_BIGQUERY", "false").lower() == "true":
        logging.info("Loading RFM dataset from BigQuery view: retail_data.rfm_features")
        project_id = os.getenv("GCP_PROJECT", "anna-ml-pipeline")
        query = "SELECT * FROM `retail_data.rfm_features`"
        df = pd.read_gbq(query, project_id=project_id)
        # Ensure customer_id column type matches (e.g. string or float depending on needs)
        # BigQuery will load as string or float depending on casting, let's keep it clean
        return df

    logging.info("Loading RFM dataset from %s", path)
    return pd.read_csv(path)


def run_rfm_features():

    df = load_transactions(FEATURE_RETAIL)

    rfm = compute_rfm(df)

    save_rfm(rfm, RFM_CUSTOMERS)

if __name__ == "__main__":
    run_rfm_features()