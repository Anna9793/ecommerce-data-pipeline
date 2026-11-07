"""
Deploys BigQuery Medallion Views (Silver clean_transactions & Gold rfm_features).
Reads sql/bigquery_medallion.sql and executes DDL statements on GCP.
"""
import os
import sys
import logging
from pathlib import Path
from google.cloud import bigquery

# Ensure project root is in path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

SQL_FILE = ROOT_DIR / "sql" / "bigquery_medallion.sql"


def deploy_medallion_views(project_id: str = None) -> None:
    """
    Deploys or updates the Silver and Gold Medallion views in BigQuery.
    """
    if project_id is None:
        project_id = os.getenv("GCP_PROJECT", "anna-ml-pipeline")

    logging.info("Connecting to BigQuery project: %s", project_id)
    client = bigquery.Client(project=project_id)

    if not SQL_FILE.exists():
        raise FileNotFoundError(f"SQL file not found at: {SQL_FILE}")

    sql_content = SQL_FILE.read_text(encoding="utf-8")

    logging.info("Executing Medallion DDL statements from %s...", SQL_FILE.name)
    job_config = bigquery.QueryJobConfig(default_dataset=f"{project_id}.retail_data")
    query_job = client.query(sql_content, job_config=job_config)
    query_job.result()  # Wait for DDL completion

    logging.info("✅ Successfully deployed BigQuery Medallion views:")
    logging.info("   - Silver: %s.retail_data.clean_transactions", project_id)
    logging.info("   - Gold:   %s.retail_data.rfm_features", project_id)


if __name__ == "__main__":
    project = os.getenv("GCP_PROJECT", "anna-ml-pipeline")
    deploy_medallion_views(project_id=project)
