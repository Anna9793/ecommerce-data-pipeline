import pandas as pd
import logging
from config.paths import ONLINE_RETAIL_CSV, CLEAN_RETAIL

def load_raw_data(path):
    logging.info("Loading raw dataset from %s", path)
    df = pd.read_csv(path)
    return df

def remove_duplicates(df):
    logging.info("Removing duplicate rows")
    return df.drop_duplicates()

def standardize_column_names(df):
    logging.info("Standardizing column names")
    df.columns = (
        df.columns
        .str.replace(r'(?<=[a-z])(?=[A-Z])', '_', regex=True)
        .str.lower()
    )
    return df

def remove_missing_customer_ids(df):
    logging.info("Step 3: Removing rows with missing CustomerID")
    return df.dropna(subset=["customer_id"])

def cast_data_types(df):
    logging.info("Step 4: Casting data types (dates, IDs, numerics)")
    if "invoice_date" in df.columns:
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
    logging.info("Step 5: Filtering valid transactions (positive price, purchases or 'C' cancellations)")
    is_valid_purchase = df["quantity"] > 0
    is_valid_cancellation = (df["quantity"] < 0) & df["invoice_no"].str.startswith("C", na=False)
    return df[(df["unit_price"] > 0) & (is_valid_purchase | is_valid_cancellation)]

def clean_data(df):
    df = standardize_column_names(df)
    df = remove_duplicates(df)
    df = remove_missing_customer_ids(df)
    df = cast_data_types(df)
    df = filter_valid_transactions(df)
    return df

def save_clean_data(df, path):
    logging.info("Saving cleaned dataset to %s", path)
    df.to_csv(path, index=False)

def run_cleaning():
    df = load_raw_data(ONLINE_RETAIL_CSV)
    logging.info("Rows loaded: %s", len(df))
    df = clean_data(df)
    logging.info("Cleaned rows remaining: %s", len(df))
    save_clean_data(df, CLEAN_RETAIL)

if __name__ == "__main__":
    run_cleaning()
