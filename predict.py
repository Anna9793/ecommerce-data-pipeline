import pandas as pd
import joblib
import logging

from src.cleaning import clean_data
from src.transformation import transform_data
from src.rfm_features import compute_rfm
from src.utils.config_loader import load_config
from config.paths import ONLINE_RETAIL_CSV, BEST_MODEL_PATH, CUSTOMER_CLUSTERS

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

    input_file = ONLINE_RETAIL_CSV
    model_file = BEST_MODEL_PATH

    predictions = run_prediction(input_file, model_file)

    output_path = CUSTOMER_CLUSTERS

    logging.info("Saving predictions to %s", output_path)

    predictions.to_csv(output_path, index = False)


