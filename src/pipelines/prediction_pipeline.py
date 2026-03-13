import logging
import pandas as pd 
import joblib

from src.cleaning import clean_data
from src.transformation import transform_data
from src.rfm_features import compute_rfm
from src.utils.data_validation import validate_columns

from config.paths import BEST_MODEL_PATH

def run_prediction_pipeline(input_path):

    logging.info("Loading dataset")

    df = pd.read_csv(input_path)

    df = clean_data(df)

    validate_columns(
        df,
        [
            "invoice_no",
            "stock_code",
            "quantity",
            "unit_price",
            "customer_id",
            "invoice_date",
        ]
    )
    
    df = transform_data(df)

    rfm = compute_rfm(df)

    logging.info("Loading model")

    model = joblib.load(BEST_MODEL_PATH)

    rfm["cluster"] = model.predict(rfm[["recency", "frequency", "monetary"]])

    return rfm

