"""
Distributed Big Data Feature Engineering with PySpark on Google Cloud Dataproc.

This module computes scalable customer RFM, rolling spending velocity (30d/90d),
cancellation rates, and shopping hour patterns using PySpark DataFrame APIs
and distributed Window functions.
"""

import os
import sys
import logging
from typing import Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("pyspark_feature_engineering")


def get_spark_session(app_name: str = "EcommerceFeatureEngineering", master: Optional[str] = None):
    """
    Initializes or retrieves an active SparkSession.
    Configured for Dataproc with GCS and BigQuery connector support.
    """
    from pyspark.sql import SparkSession

    builder = SparkSession.builder.appName(app_name)
    
    if master:
        builder = builder.master(master)
        
    # BigQuery & GCS connector configuration for Dataproc / Spark
    builder = (
        builder
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
        .config("views.enabled", "true")
        .config("materializationDataset", "retail_data")
    )
    return builder.getOrCreate()


def compute_pyspark_rfm_features(df):
    """
    Computes distributed customer RFM and behavioral features using PySpark.
    
    Features generated per customer:
      - recency: Days since last purchase relative to max dataset date.
      - frequency: Distinct count of valid purchase invoices.
      - monetary: Total monetary spend across valid purchases.
      - avg_order_value: monetary / frequency.
      - spending_velocity: monetary_30d / (monetary_90d / 3.0).
      - cancellation_rate: Count of cancelled orders ('C%') / total orders.
      - preferred_shopping_hour: Modal purchase hour.
      
    Args:
        df: PySpark DataFrame containing raw transaction records.
        
    Returns:
        PySpark DataFrame with one row per customer and calculated features.
    """
    from pyspark.sql import functions as F
    from pyspark.sql.window import Window

    # Ensure required columns are typed appropriately
    df_clean = (
        df
        .withColumn("customer_id", F.col("customer_id").cast("string"))
        .withColumn("invoice_no", F.col("invoice_no").cast("string"))
        .withColumn("invoice_date", F.to_timestamp(F.col("invoice_date")))
        .withColumn("quantity", F.col("quantity").cast("double"))
        .withColumn("unit_price", F.col("unit_price").cast("double"))
    )

    if "order_value" not in df_clean.columns:
        df_clean = df_clean.withColumn("order_value", F.col("quantity") * F.col("unit_price"))
    else:
        df_clean = df_clean.withColumn("order_value", F.coalesce(F.col("order_value").cast("double"), F.lit(0.0)))

    # Filter out null customer IDs
    df_valid = df_clean.filter(F.col("customer_id").isNotNull() & (F.trim(F.col("customer_id")) != ""))

    # Find maximum transaction date across entire dataset
    max_date_row = df_valid.select(F.max("invoice_date")).first()
    max_date = max_date_row[0] if max_date_row and max_date_row[0] is not None else None

    if max_date is None:
        raise ValueError("Cannot compute features: dataset contains no valid invoice dates.")

    # 1. Base aggregations for all transactions (including cancellations for total orders)
    df_with_flags = (
        df_valid
        .withColumn("is_positive_sale", F.when(F.col("quantity") > 0, 1).otherwise(0))
        .withColumn("is_cancellation", F.when(F.col("invoice_no").startswith("C") | (F.col("quantity") < 0), 1).otherwise(0))
        .withColumn("sale_value", F.when(F.col("quantity") > 0, F.col("order_value")).otherwise(0.0))
        .withColumn("hour", F.hour(F.col("invoice_date")))
        .withColumn("days_since_invoice", F.datediff(F.lit(max_date), F.col("invoice_date")))
    )

    # 2. Velocity windows (30 days and 90 days)
    df_with_velocity = (
        df_with_flags
        .withColumn("sale_value_30d", F.when((F.col("is_positive_sale") == 1) & (F.col("days_since_invoice") <= 30), F.col("order_value")).otherwise(0.0))
        .withColumn("sale_value_90d", F.when((F.col("is_positive_sale") == 1) & (F.col("days_since_invoice") <= 90), F.col("order_value")).otherwise(0.0))
    )

    # 3. Main Customer Aggregations
    # Calculate recency, frequency, monetary, velocity, cancellation counts
    customer_agg = (
        df_with_velocity
        .groupBy("customer_id")
        .agg(
            F.min("days_since_invoice").alias("recency"),
            F.countDistinct(F.when(F.col("is_positive_sale") == 1, F.col("invoice_no"))).alias("frequency"),
            F.sum("sale_value").alias("monetary"),
            F.sum("sale_value_30d").alias("monetary_30"),
            F.sum("sale_value_90d").alias("monetary_90"),
            F.countDistinct(F.when(F.col("is_cancellation") == 1, F.col("invoice_no"))).alias("cancelled_orders"),
            F.countDistinct("invoice_no").alias("total_orders")
        )
    )

    # 4. Compute Preferred Shopping Hour (Mode Hour per Customer)
    hour_counts = (
        df_with_flags
        .groupBy("customer_id", "hour")
        .count()
    )

    hour_window = Window.partitionBy("customer_id").orderBy(F.col("count").desc(), F.col("hour").asc())

    preferred_hour = (
        hour_counts
        .withColumn("rank", F.row_number().over(hour_window))
        .filter(F.col("rank") == 1)
        .select("customer_id", F.col("hour").alias("preferred_shopping_hour"))
    )

    # 5. Join and Finalize Feature Ratios
    rfm_final = (
        customer_agg
        .join(preferred_hour, on="customer_id", how="left")
        .withColumn("recency", F.coalesce(F.col("recency"), F.lit(0)).cast("int"))
        .withColumn("frequency", F.coalesce(F.col("frequency"), F.lit(0)).cast("int"))
        .withColumn("monetary", F.round(F.coalesce(F.col("monetary"), F.lit(0.0)), 2))
        .withColumn(
            "avg_order_value",
            F.round(
                F.when(F.col("frequency") > 0, F.col("monetary") / F.col("frequency")).otherwise(0.0),
                2
            )
        )
        .withColumn(
            "spending_velocity",
            F.round(
                F.when(
                    F.col("monetary_90") > 0,
                    F.col("monetary_30") / (F.col("monetary_90") / 3.0)
                ).otherwise(1.0),
                2
            )
        )
        .withColumn(
            "cancellation_rate",
            F.round(
                F.when(
                    F.col("total_orders") > 0,
                    F.col("cancelled_orders") / F.col("total_orders")
                ).otherwise(0.0),
                4
            )
        )
        .withColumn("preferred_shopping_hour", F.coalesce(F.col("preferred_shopping_hour"), F.lit(12)).cast("int"))
        .select(
            "customer_id",
            "recency",
            "frequency",
            "avg_order_value",
            "spending_velocity",
            "cancellation_rate",
            "preferred_shopping_hour"
        )
    )

    return rfm_final


def run_pyspark_pipeline(
    input_path: str,
    output_path: str,
    project_id: Optional[str] = None,
    write_to_bigquery: bool = False,
    bigquery_table: str = "retail_data.rfm_features"
):
    """
    Orchestrates the PySpark batch feature engineering pipeline on Dataproc.
    """
    project_id = project_id or os.getenv("GCP_PROJECT", "anna-ml-pipeline")
    logger.info("Initializing PySpark Feature Engineering on Dataproc for project %s...", project_id)
    
    spark = get_spark_session()
    
    try:
        if input_path.startswith("bq://") or input_path.startswith("bigquery://"):
            table_ref = input_path.replace("bq://", "").replace("bigquery://", "")
            logger.info("Reading input from BigQuery table %s via Spark-BigQuery connector...", table_ref)
            df = spark.read.format("bigquery").load(table_ref)
        elif input_path.endswith(".parquet"):
            logger.info("Reading input Parquet from %s...", input_path)
            df = spark.read.parquet(input_path)
        else:
            logger.info("Reading input CSV from %s...", input_path)
            df = spark.read.option("header", "true").option("inferSchema", "true").csv(input_path)
            
        logger.info("Computing distributed customer features...")
        rfm_features = compute_pyspark_rfm_features(df)
        
        logger.info("Writing feature dataset to %s in Parquet format...", output_path)
        rfm_features.write.mode("overwrite").parquet(output_path)
        
        if write_to_bigquery:
            logger.info("Writing feature table to BigQuery table %s...", bigquery_table)
            (
                rfm_features.write
                .format("bigquery")
                .option("table", f"{project_id}.{bigquery_table}")
                .mode("overwrite")
                .save()
            )
            
        logger.info("PySpark Feature Engineering completed successfully!")
        return rfm_features
        
    finally:
        spark.stop()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="PySpark Feature Engineering on GCP Dataproc")
    parser.add_argument("--input", default="data/retail_cleaned.csv", help="Input GCS/Local file path or BigQuery table")
    parser.add_argument("--output", default="data/rfm_customers_spark.parquet", help="Output GCS/Local Parquet path")
    parser.add_argument("--project", default=os.getenv("GCP_PROJECT", "anna-ml-pipeline"), help="GCP Project ID")
    parser.add_argument("--write-bq", action="store_true", help="Write directly to BigQuery feature table")

    args = parser.parse_args()
    run_pyspark_pipeline(
        input_path=args.input,
        output_path=args.output,
        project_id=args.project,
        write_to_bigquery=args.write_bq
    )
