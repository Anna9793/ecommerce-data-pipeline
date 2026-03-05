import pandas as pd
from pathlib import Path
import logging

from config.config import INPUT_DATA_FILE, OUTPUT_DATA_FILE

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def load_dataset(path):
    logging.info(f"Loading dataset from {path}")
    df = pd.read_excel(path)
    logging.info(f"Loaded {len(df)} rows")
    return df

def save_raw_dataset(df, path):
    logging.info(f"Saving dataset to {path}")
    df.to_csv(path, index=False)
    logging.info("Dataset saved successfully")

def run_ingestion():
    input_path = INPUT_DATA_FILE
    output_path = OUTPUT_DATA_FILE

    df = load_dataset(input_path)
    save_raw_dataset(df,output_path)

if __name__ == "__main__":
    run_ingestion()

