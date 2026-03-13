import pandas as pd
from pathlib import Path
import logging

from config.config import ONLINE_RETAIL_XLSX, ONLINE_RETAIL_CSV

def load_dataset(path):

    logging.info("Loading dataset from %s", path)

    df = pd.read_excel(path)

    logging.info("Rows loaded: %s", len(df))

    return df

def save_raw_dataset(df, path):

    logging.info("Saving dataset to %s", path)

    df.to_csv(path, index=False)

    logging.info("Dataset saved successfully")

def run_ingestion():
    
    df = load_dataset(ONLINE_RETAIL_XLSX)

    save_raw_dataset(df, ONLINE_RETAIL_CSV)

if __name__ == "__main__":

    run_ingestion()

