import pandas as pd
import logging
from pathlib import Path
from config.paths import FEATURE_RETAIL, RFM_CUSTOMERS

def load_transactions(path):
    logging.info("Loading transaction dataset from %s", path)
    df = pd.read_csv(path)
    df["invoice_date"] = pd.to_datetime(df["invoice_date"])
    return df


def compute_rfm(df):
    
    rfm = (
        df.groupby("customer_id")
        .agg(
            recency=("invoice_date",lambda x: (df["invoice_date"].max() - x.max()).days),
            frequency=("invoice_no", "nunique"),
            monetary=("order_value", "sum")
        )
        .reset_index()
    )

    rfm["avg_order_value"] = (
        rfm["monetary"] / rfm["frequency"].replace(0, 1)
        ).round(2)

    rfm = rfm [["customer_id", "recency", "frequency", "avg_order_value"]]
    
    return rfm

def save_rfm(df, path):
    logging.info("Saving RFM dataset to %s", path)
    df.to_csv(path, index=False)

def load_rfm(path=RFM_CUSTOMERS):

    logging.info(
        "Loading RFM dataset from %s",
        path
    )

    return pd.read_csv(path)

def run_rfm_features():

    df = load_transactions(FEATURE_RETAIL)

    rfm = compute_rfm(df)

    save_rfm(rfm, RFM_CUSTOMERS)

if __name__ == "__main__":
    run_rfm_features()