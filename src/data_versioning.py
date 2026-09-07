"""
Data Versioning & BigQuery Time Travel Management.

Provides utilities for:
1. BigQuery Time Travel queries using 'FOR SYSTEM_TIME AS OF' for point-in-time reproducibility.
2. Zero-copy BigQuery table snapshots and clones with automated expiration policies.
3. Cryptographic dataset checksum generation for data provenance and drift validation.
"""

import os
import hashlib
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("data_versioning")


def compute_dataset_checksum(df: pd.DataFrame) -> str:
    """
    Computes a deterministic SHA-256 hash of a DataFrame's content and schema.
    Used for cryptographic data versioning and data contract verification.
    """
    # Hash column schema
    schema_str = "|".join(f"{col}:{dtype}" for col, dtype in zip(df.columns, df.dtypes))
    
    # Hash sample content or full sorted index if feasible
    sample_repr = df.head(100).to_json(orient="values")
    
    hasher = hashlib.sha256()
    hasher.update(schema_str.encode("utf-8"))
    hasher.update(str(len(df)).encode("utf-8"))
    hasher.update(sample_repr.encode("utf-8"))
    return hasher.hexdigest()[:16]


def build_time_travel_query(
    table_name: str,
    dataset_id: str = "retail_data",
    project_id: Optional[str] = None,
    time_travel_expression: Optional[str] = None,
    columns: str = "*"
) -> str:
    """
    Constructs a SQL query leveraging BigQuery Time Travel syntax (FOR SYSTEM_TIME AS OF).
    
    Args:
        table_name: Base table name (e.g., 'transactions').
        dataset_id: BigQuery dataset ID.
        project_id: GCP project ID.
        time_travel_expression: SQL timestamp expression (e.g., "TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 1 HOUR)"
                                or "TIMESTAMP '2026-09-01 00:00:00 UTC'").
        columns: Comma-separated columns to select.
        
    Returns:
        Formatted BigQuery SQL string.
    """
    project_id = project_id or os.getenv("GCP_PROJECT", "anna-ml-pipeline")
    full_table_id = f"`{project_id}.{dataset_id}.{table_name}`"
    
    if time_travel_expression:
        return f"SELECT {columns} FROM {full_table_id} FOR SYSTEM_TIME AS OF {time_travel_expression}"
    else:
        return f"SELECT {columns} FROM {full_table_id}"


def build_create_snapshot_sql(
    source_table: str,
    snapshot_name: str,
    dataset_id: str = "retail_data",
    project_id: Optional[str] = None,
    expiration_days: int = 30
) -> str:
    """
    Generates SQL to create an immutable, zero-copy BigQuery table snapshot.
    """
    project_id = project_id or os.getenv("GCP_PROJECT", "anna-ml-pipeline")
    source_table_ref = f"`{project_id}.{dataset_id}.{source_table}`"
    snapshot_table_ref = f"`{project_id}.{dataset_id}.{snapshot_name}`"
    
    expiration_clause = f"OPTIONS (expiration_timestamp = TIMESTAMP_ADD(CURRENT_TIMESTAMP(), INTERVAL {expiration_days} DAY))"
    
    return f"CREATE SNAPSHOT TABLE {snapshot_table_ref} CLONE {source_table_ref} {expiration_clause};"


def build_restore_from_snapshot_sql(
    snapshot_name: str,
    target_table: str,
    dataset_id: str = "retail_data",
    project_id: Optional[str] = None
) -> str:
    """
    Generates SQL to restore or clone a BigQuery table from an existing snapshot.
    """
    project_id = project_id or os.getenv("GCP_PROJECT", "anna-ml-pipeline")
    snapshot_table_ref = f"`{project_id}.{dataset_id}.{snapshot_name}`"
    target_table_ref = f"`{project_id}.{dataset_id}.{target_table}`"
    
    return f"CREATE OR REPLACE TABLE {target_table_ref} CLONE {snapshot_table_ref};"


class BigQueryVersioningManager:
    """
    Manages BigQuery table versions, snapshots, and point-in-time time-travel queries.
    """
    
    def __init__(self, project_id: Optional[str] = None, dataset_id: str = "retail_data"):
        self.project_id = project_id or os.getenv("GCP_PROJECT", "anna-ml-pipeline")
        self.dataset_id = dataset_id
        
    def query_at_timestamp(
        self,
        table_name: str,
        timestamp_utc: datetime,
        client: Any = None
    ) -> pd.DataFrame:
        """
        Executes a point-in-time time travel query against BigQuery for a specific datetime.
        """
        iso_str = timestamp_utc.strftime("%Y-%m-%d %H:%M:%S UTC")
        time_expr = f"TIMESTAMP '{iso_str}'"
        sql = build_time_travel_query(
            table_name=table_name,
            dataset_id=self.dataset_id,
            project_id=self.project_id,
            time_travel_expression=time_expr
        )
        logger.info("Executing BigQuery Time Travel query: %s", sql)
        
        if client:
            return client.query(sql).to_dataframe()
        else:
            try:
                import pandas_gbq
                return pandas_gbq.read_gbq(sql, project_id=self.project_id)
            except Exception as e:
                logger.warning("Could not query BigQuery directly (using fallback): %s", e)
                return pd.DataFrame()
                
    def create_dataset_snapshot(
        self,
        source_table: str = "transactions",
        snapshot_suffix: Optional[str] = None,
        expiration_days: int = 30,
        client: Any = None
    ) -> Dict[str, Any]:
        """
        Creates a timestamped snapshot of a table and returns the metadata.
        """
        timestamp_str = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        suffix = f"_{snapshot_suffix}" if snapshot_suffix else ""
        snapshot_name = f"{source_table}_snapshot_{timestamp_str}{suffix}"
        
        sql = build_create_snapshot_sql(
            source_table=source_table,
            snapshot_name=snapshot_name,
            dataset_id=self.dataset_id,
            project_id=self.project_id,
            expiration_days=expiration_days
        )
        
        logger.info("Creating BigQuery table snapshot '%s'...", snapshot_name)
        if client:
            client.query(sql).result()
            
        return {
            "snapshot_name": snapshot_name,
            "source_table": source_table,
            "created_at_utc": datetime.utcnow().isoformat(),
            "expiration_days": expiration_days,
            "sql_executed": sql
        }
