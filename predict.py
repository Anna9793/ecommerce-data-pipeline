import pandas as pd
import joblib
import logging
import argparse

from src.cleaning import clean_data
from src.transformation import transform_data
from src.rfm_features import compute_rfm
from src.utils.config_loader import load_config
from config.paths import ONLINE_RETAIL_CSV, BEST_MODEL_PATH, CUSTOMER_CLUSTERS
from src.utils.data_validation import validate_columns

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def run_prediction(input_path, model_path, config_path="config/experiment.yaml"):

    logging.info("Loading config")
    config = load_config(config_path)

    FEATURE_COLUMNS = config.features.columns

    logging.info ("Loading raw data from %s", input_path)
    df = pd.read_csv(input_path)

    logging.info("Cleaning data")
    df = clean_data(df)

    validate_columns(
        df,
        ["invoice_no",
        "stock_code",
        "quantity",
        "unit_price",
        "customer_id",
        "invoice_date",
        ]
    )

    logging.info("Transforming data")
    df = transform_data(df)

    logging.info("Computing RFM features")
    rfm = compute_rfm(df)

    logging.info("Loading trained model from %s", BEST_MODEL_PATH)
    model = joblib.load(BEST_MODEL_PATH)

    logging.info("Predicting clusters")
    rfm["cluster"] = model.predict(rfm[FEATURE_COLUMNS])

    return rfm

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Run customer segmentation predictions"
    )

    parser.add_argument(
        "--input",
        type=str,
        default=ONLINE_RETAIL_CSV,
        help="Path to input transaction dataset"
    )

    parser.add_argument(
        "--output",
        type=str,
        default=CUSTOMER_CLUSTERS,
        help="Path where predictions will be saved"
    )

    args = parser.parse_args()

    model_file = BEST_MODEL_PATH

    predictions = run_prediction(args.input, model_file)

    logging.info("Saving predictions to %s", args.output)

    predictions.to_csv(args.output, index = False)


