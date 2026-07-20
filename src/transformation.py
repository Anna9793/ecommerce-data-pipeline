import os
import pandas as pd
import logging
from config.paths import CLEAN_RETAIL, FEATURE_RETAIL

def load_clean_data(path):
    use_bigquery = os.getenv("USE_BIGQUERY", "false").lower() == "true"
    
    if use_bigquery:
        logging.info("Cloud mode active: loading cleaned dataset from BigQuery table")
        project_id = os.getenv("GCP_PROJECT", "anna-ml-pipeline")
        query = f"""
        SELECT 
            InvoiceNo AS invoice_no,
            StockCode AS stock_code,
            Description AS description,
            CAST(Quantity AS INT64) AS quantity,
            InvoiceDate AS invoice_date,
            CAST(UnitPrice AS FLOAT64) AS unit_price,
            CAST(CustomerID AS STRING) AS customer_id,
            Country AS country
        FROM `{project_id}.retail_data.transactions`
        """
        import pandas_gbq
        df = pandas_gbq.read_gbq(query, project_id=project_id)
        return df

    logging.info("Loading cleaned dataset from path: %s", path)
    df = pd.read_csv(path)
    return df

def add_order_value(df):

    logging.info("Creating order_value feature")

    df["order_value"] = df["quantity"] * df["unit_price"]

    return df

def convert_invoice_date(df):

    logging.info("Converting invoice_date to datetime")

    df["invoice_date"] = pd.to_datetime(df["invoice_date"])

    return df

def add_time_features(df):

    logging.info("Extracting time features")

    df["invoice_year"] = df["invoice_date"].dt.year

    df["invoice_month"] = df["invoice_date"].dt.month

    df["invoice_hour"] = df["invoice_date"].dt.hour

    return df

def transform_data(df):

    df = convert_invoice_date(df)
    df = add_order_value(df)
    df = add_time_features(df)

    return df


def save_transformed_data(df, path):

    logging.info("Saving transformed data to %s" ,path)

    df.to_csv(path, index=False)


def run_transformation():

    df = load_clean_data(CLEAN_RETAIL)

    df = transform_data(df)

    save_transformed_data(df, FEATURE_RETAIL)

if __name__ == "__main__":
    run_transformation()