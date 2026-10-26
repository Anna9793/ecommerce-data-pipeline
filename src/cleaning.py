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
    logging.info("Removing rows with missing CustomerID")
    return df.dropna(subset=["customer_id"])

def filter_valid_transactions(df):
    logging.info("Filtering transactions with positive quantity and unit_price using query()")
    return df.query("quantity > 0 and unit_price > 0")

def clean_data(df):
    df = remove_duplicates(df)
    df = standardize_column_names(df)
    df = remove_missing_customer_ids(df)
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
