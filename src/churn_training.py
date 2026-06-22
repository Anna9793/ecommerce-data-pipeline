import pandas as pd 
import logging

from src.transformation import(
    load_clean_data,
    transform_data
)

from src.rfm_features import compute_rfm

from src.churn_target import create_churn_target

from config.paths import CLEAN_RETAIL


def run_churn_training():

    logging.info("Loading and transforming data")

    df = load_clean_data(CLEAN_RETAIL)

    df = transform_data(df)

    max_date = df["invoice_date"].max()

    cutoff_date = max_date - pd.Timedelta(days=90)

    logging.info(
        "Cutoff date: %s",
        cutoff_date
    )

    feature_df = df[
    df["invoice_date"] < cutoff_date
]
    logging.info(
        "Feature rows: %s",
        len(feature_df)
    )

    future_df = df[
    df["invoice_date"] >= cutoff_date
]
    logging.info(
        "Future rows: %s",
        len(future_df)
    )

    rfm_df = compute_rfm(feature_df)

    target_df = create_churn_target(
    feature_df,
    future_df
)
    logging.info(
        "Target rows: %s",
        len(target_df)
    )

    training_df = rfm_df.merge(
    target_df,
    on="customer_id",
    how="inner"
)
    

    print(training_df.head())
    print(training_df.shape)

if __name__ == "__main__":
    run_churn_training()