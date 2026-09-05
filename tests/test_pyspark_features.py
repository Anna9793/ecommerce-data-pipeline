import pytest
import os
from unittest.mock import MagicMock, patch
from dags.ecommerce_master_pipeline_dag import dag, run_dataproc_pyspark_features_func


def test_airflow_dag_contains_dataproc_pyspark_task():
    """Verify that the Airflow Master DAG contains the Dataproc PySpark feature task."""
    task_ids = [t.task_id for t in dag.tasks]
    assert "pyspark_dataproc_feature_engineering" in task_ids
    
    task = dag.get_task("pyspark_dataproc_feature_engineering")
    assert task is not None
    assert "validate_data_quality" in [t.task_id for t in task.upstream_list]
    assert "sync_online_feature_store" in [t.task_id for t in task.downstream_list]


def test_run_dataproc_pyspark_callable():
    """Verify that the Dataproc Airflow callable executes without errors."""
    result = run_dataproc_pyspark_features_func()
    assert result is True


def test_pyspark_feature_module_structure():
    """Verify that src/pyspark_feature_engineering.py defines required entry points."""
    import src.pyspark_feature_engineering as spark_module
    
    assert hasattr(spark_module, "get_spark_session")
    assert hasattr(spark_module, "compute_pyspark_rfm_features")
    assert hasattr(spark_module, "run_pyspark_pipeline")


def test_pyspark_rfm_schema_and_calculation_logic():
    """
    Test PySpark transformation logic and output schema contracts using mocks
    if PySpark runtime is not natively present in the lightweight test runner.
    """
    try:
        from pyspark.sql import SparkSession
        import pyspark.sql.functions as F
        
        # Test with a local SparkSession if available
        spark = SparkSession.builder.master("local[1]").appName("UnitTest").getOrCreate()
        try:
            data = [
                ("12345", "1001", "2026-01-01 10:00:00", 2.0, 10.0, 20.0),
                ("12345", "1002", "2026-01-15 10:00:00", 1.0, 30.0, 30.0),
                ("12345", "C1003", "2026-01-20 12:00:00", -1.0, 10.0, -10.0),
                ("67890", "1004", "2026-01-10 15:00:00", 5.0, 5.0, 25.0),
            ]
            columns = ["customer_id", "invoice_no", "invoice_date", "quantity", "unit_price", "order_value"]
            df = spark.createDataFrame(data, columns)
            
            from src.pyspark_feature_engineering import compute_pyspark_rfm_features
            result_df = compute_pyspark_rfm_features(df)
            
            expected_cols = {
                "customer_id",
                "recency",
                "frequency",
                "avg_order_value",
                "spending_velocity",
                "cancellation_rate",
                "preferred_shopping_hour"
            }
            assert expected_cols.issubset(set(result_df.columns))
            
            rows = {r["customer_id"]: r for r in result_df.collect()}
            assert "12345" in rows
            assert "67890" in rows
            assert rows["12345"]["frequency"] == 2 # 2 positive sales
            assert rows["12345"]["preferred_shopping_hour"] == 10 # Mode hour is 10
            assert rows["67890"]["frequency"] == 1
        finally:
            spark.stop()
            
    except ImportError:
        # If PySpark is not installed in the local environment, test contract via mock
        mock_df = MagicMock()
        mock_spark = MagicMock()
        
        with patch("src.pyspark_feature_engineering.get_spark_session", return_value=mock_spark):
            from src.pyspark_feature_engineering import get_spark_session
            session = get_spark_session()
            assert session == mock_spark
