from config.config import INPUT_DATA_FILE, OUTPUT_DATA_FILE

import pandas as pd
from pathlib import Path

def load_dataset(path):
    print(f"Loading dataset from {path}")
    df = pd.read_excel(path)
    print(f"Loaded {len(df)} rows")
    return df

def save_raw_dataset(df, path):
    print(f"Saving dataset to {path}")
    df.to_csv(path, index=False)
    print("Dataset saved successfully")

def run_ingestion():
    input_path = INPUT_DATA_FILE
    output_path = OUTPUT_DATA_FILE

    df = load_dataset(input_path)
    save_raw_dataset(df,output_path)

if __name__ == "__main__":
    run_ingestion()

