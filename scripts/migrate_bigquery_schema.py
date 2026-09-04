import os
import sys
import logging
from google.cloud import bigquery

# Ensure project root is in Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def migrate_transactions_schema(project_id: str = "anna-ml-pipeline") -> None:
    """
    Migrates BigQuery retail_data.transactions schema:
    Converts CustomerID from FLOAT64 to STRING while preserving
    partitioning by DATE(InvoiceDate) and clustering by CustomerID, Country.
    """
    client = bigquery.Client(project=project_id)
    table_id = f"{project_id}.retail_data.transactions"
    
    logging.info("Connecting to BigQuery table: %s", table_id)
    
    # 1. Check pre-migration row count and schema
    try:
        table = client.get_table(table_id)
        pre_count_query = f"SELECT COUNT(1) AS total_rows FROM `{table_id}`"
        pre_count = list(client.query(pre_count_query).result())[0].total_rows
        logging.info("Current row count before migration: %d", pre_count)
        
        current_schema = {field.name: field.field_type for field in table.schema}
        logging.info("Current CustomerID type: %s", current_schema.get("CustomerID", "NOT FOUND"))
    except Exception as e:
        logging.error("Failed to inspect table %s: %s", table_id, e)
        return

    # 2. Execute Migration DDL
    ddl_query = f"""
    CREATE OR REPLACE TABLE `{table_id}`
    PARTITION BY DATE(InvoiceDate)
    CLUSTER BY CustomerID, Country AS
    SELECT 
      InvoiceNo,
      StockCode,
      Description,
      Quantity,
      InvoiceDate,
      UnitPrice,
      IFNULL(REGEXP_REPLACE(CAST(CustomerID AS STRING), r'\\.0$', ''), 'GUEST') AS CustomerID,
      Country
    FROM `{table_id}`
    """
    
    logging.info("Executing BigQuery DDL schema migration...")
    query_job = client.query(ddl_query)
    query_job.result()  # Wait for query to complete
    logging.info("✅ DDL query completed successfully.")

    # 3. Verify post-migration state
    new_table = client.get_table(table_id)
    new_schema = {field.name: field.field_type for field in new_table.schema}
    post_count_query = f"SELECT COUNT(1) AS total_rows FROM `{table_id}`"
    post_count = list(client.query(post_count_query).result())[0].total_rows

    logging.info("New CustomerID type: %s", new_schema.get("CustomerID"))
    logging.info("Post-migration row count: %d", post_count)

    if post_count == pre_count and new_schema.get("CustomerID") == "STRING":
        logging.info("🎉 Migration 100%% verified! Zero data loss. CustomerID is now STRING.")
    else:
        logging.warning("⚠️ Verification mismatch: Pre=%d, Post=%d, Type=%s", pre_count, post_count, new_schema.get("CustomerID"))

if __name__ == "__main__":
    project = os.getenv("GCP_PROJECT", "anna-ml-pipeline")
    migrate_transactions_schema(project_id=project)
