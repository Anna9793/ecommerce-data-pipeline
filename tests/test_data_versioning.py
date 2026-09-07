import os
import json
import pytest
import pandas as pd
from datetime import datetime
from pathlib import Path
from src.data_versioning import (
    compute_dataset_checksum,
    build_time_travel_query,
    build_create_snapshot_sql,
    build_restore_from_snapshot_sql,
    BigQueryVersioningManager
)
from src.lineage import DataLineageTracker, get_current_git_commit


def test_dataset_checksum_deterministic():
    """Verify that dataset checksums are deterministic and sensitive to changes."""
    df1 = pd.DataFrame({
        "customer_id": ["123", "456"],
        "spend": [100.50, 200.00]
    })
    df2 = pd.DataFrame({
        "customer_id": ["123", "456"],
        "spend": [100.50, 200.00]
    })
    df_diff = pd.DataFrame({
        "customer_id": ["123", "456"],
        "spend": [100.50, 999.99]
    })
    
    hash1 = compute_dataset_checksum(df1)
    hash2 = compute_dataset_checksum(df2)
    hash_diff = compute_dataset_checksum(df_diff)
    
    assert hash1 == hash2
    assert hash1 != hash_diff
    assert len(hash1) == 16


def test_build_time_travel_query():
    """Verify BigQuery Time Travel SQL query construction."""
    query_sub = build_time_travel_query(
        table_name="transactions",
        dataset_id="retail_data",
        project_id="anna-ml-pipeline",
        time_travel_expression="TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 1 DAY)"
    )
    assert "SELECT * FROM `anna-ml-pipeline.retail_data.transactions`" in query_sub
    assert "FOR SYSTEM_TIME AS OF TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 1 DAY)" in query_sub

    query_standard = build_time_travel_query(
        table_name="transactions",
        dataset_id="retail_data",
        project_id="anna-ml-pipeline"
    )
    assert "FOR SYSTEM_TIME AS OF" not in query_standard


def test_build_create_snapshot_sql():
    """Verify snapshot table creation SQL with expiration clauses."""
    sql = build_create_snapshot_sql(
        source_table="transactions",
        snapshot_name="transactions_snapshot_20260907",
        dataset_id="retail_data",
        project_id="anna-ml-pipeline",
        expiration_days=14
    )
    assert "CREATE SNAPSHOT TABLE `anna-ml-pipeline.retail_data.transactions_snapshot_20260907`" in sql
    assert "CLONE `anna-ml-pipeline.retail_data.transactions`" in sql
    assert "INTERVAL 14 DAY" in sql


def test_build_restore_from_snapshot_sql():
    """Verify restore from snapshot SQL cloning."""
    sql = build_restore_from_snapshot_sql(
        snapshot_name="transactions_snapshot_20260907",
        target_table="transactions_restored",
        dataset_id="retail_data",
        project_id="anna-ml-pipeline"
    )
    assert "CREATE OR REPLACE TABLE `anna-ml-pipeline.retail_data.transactions_restored` CLONE `anna-ml-pipeline.retail_data.transactions_snapshot_20260907`" in sql


def test_bigquery_versioning_manager_metadata():
    """Verify snapshot generation metadata."""
    manager = BigQueryVersioningManager(project_id="anna-ml-pipeline", dataset_id="retail_data")
    meta = manager.create_dataset_snapshot(
        source_table="transactions",
        snapshot_suffix="v1",
        expiration_days=7
    )
    assert "transactions_snapshot_" in meta["snapshot_name"]
    assert meta["source_table"] == "transactions"
    assert meta["expiration_days"] == 7
    assert "CREATE SNAPSHOT TABLE" in meta["sql_executed"]


def test_data_lineage_tracker(tmp_path):
    """Verify data lineage stage recording, manifest serialization, and Mermaid diagram generation."""
    manifest_file = tmp_path / "test_lineage_manifest.json"
    tracker = DataLineageTracker(manifest_file=manifest_file)
    
    inputs = [{"uri": "gs://anna-ml-pipeline-bucket/raw/nordic_wear.csv", "rows": 1000}]
    outputs = [{"table": "retail_data.transactions", "rows": 1000}]
    
    record = tracker.record_stage_execution(
        stage_name="raw_streaming_ingestion",
        inputs=inputs,
        outputs=outputs,
        transformation_type="pubsub_stream"
    )
    
    assert record["stage_name"] == "raw_streaming_ingestion"
    assert record["git_commit"] is not None
    assert manifest_file.exists()
    
    with open(manifest_file, "r") as f:
        data = json.load(f)
        assert len(data) == 1
        assert data[0]["stage_name"] == "raw_streaming_ingestion"
        
    mermaid = tracker.get_latest_lineage_graph_mermaid()
    assert "graph LR" in mermaid
    assert "BigQuery" in mermaid
    assert "Dataproc" in mermaid
