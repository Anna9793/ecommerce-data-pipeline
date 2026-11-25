"""Deploys BigQuery Medallion Views (Silver clean_transactions & Gold customer_features)."""
import os
import sys
import logging
from pathlib import Path
from google.cloud import bigquery

# Ensure project root is in path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.bq_queries import get_medallion_ddl_script
from config.paths import BIGQUERY_MEDALLION_SQL

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def deploy_medallion_views(project_id: str = None) -> None:
    """Deploys or updates the Silver and Gold Medallion views in BigQuery."""
    if project_id is None:
        project_id = os.getenv("GCP_PROJECT", "anna-ml-pipeline")

    logging.info("Connecting to BigQuery project: %s", project_id)
    client = bigquery.Client(project=project_id)

    # Use centralized Python DDL query definition, fallback to SQL file if available
    if BIGQUERY_MEDALLION_SQL.exists():
        sql_content = BIGQUERY_MEDALLION_SQL.read_text(encoding="utf-8")
    else:
        sql_content = get_medallion_ddl_script(project_id=project_id)

    logging.info("Executing Medallion DDL statements for project %s...", project_id)
    job_config = bigquery.QueryJobConfig(default_dataset=f"{project_id}.retail_data")
    query_job = client.query(sql_content, job_config=job_config)
    query_job.result()  # Wait for DDL completion

    logging.info("✅ Successfully deployed BigQuery Medallion views:")
    logging.info("   - Silver: %s.retail_data.clean_transactions", project_id)
    logging.info("   - Silver: %s.retail_data.raw_products", project_id)
    logging.info("   - Gold:   %s.retail_data.customer_features", project_id)


if __name__ == "__main__":
    project = os.getenv("GCP_PROJECT", "anna-ml-pipeline")
    deploy_medallion_views(project_id=project)
