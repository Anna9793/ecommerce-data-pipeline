import pytest
from pathlib import Path
from src.dbt_runner import DbtProjectInspector


def test_dbt_project_configuration():
    """Verify that dbt_project.yml exists and contains expected configuration."""
    inspector = DbtProjectInspector()
    assert inspector.config is not None
    assert inspector.config.get("name") == "ecommerce_dbt_pipeline"
    assert "dbt_models" in inspector.config.get("model-paths", [])


def test_dbt_models_discovered():
    """Verify that staging, intermediate, and marts models are all discovered."""
    inspector = DbtProjectInspector()
    model_names = set(inspector.models.keys())
    
    expected_models = {
        "stg_transactions",
        "int_customer_orders_rollup",
        "fct_customer_rfm"
    }
    assert expected_models.issubset(model_names)


def test_dbt_model_lineage_and_dependencies():
    """Verify the Medallion dependency DAG: staging -> intermediate -> marts."""
    inspector = DbtProjectInspector()
    lineage = inspector.get_lineage_graph()
    
    # stg_transactions reads from source (no upstream refs)
    assert len(lineage["stg_transactions"]) == 0
    
    # int_customer_orders_rollup references stg_transactions
    assert "stg_transactions" in lineage["int_customer_orders_rollup"]
    
    # fct_customer_rfm references int_customer_orders_rollup
    assert "int_customer_orders_rollup" in lineage["fct_customer_rfm"]


def test_dbt_schema_tests_and_data_contracts():
    """Verify that schema tests (not_null, unique, accepted_values) are defined."""
    inspector = DbtProjectInspector()
    assert len(inspector.tests) > 0
    
    # Check for primary key constraints on fct_customer_rfm
    rfm_tests = [t for t in inspector.tests if t["model"] == "fct_customer_rfm"]
    customer_id_tests = {t["test_type"] for t in rfm_tests if t["column"] == "customer_id"}
    assert "not_null" in customer_id_tests
    assert "unique" in customer_id_tests
    
    # Check for accepted values test on rfm_segment_tier
    tier_tests = [t for t in rfm_tests if t["column"] == "rfm_segment_tier"]
    assert any(t["test_type"] == "accepted_values" for t in tier_tests)


def test_dbt_sql_compilation():
    """Verify that Jinja refs and configs compile into valid BigQuery SQL."""
    inspector = DbtProjectInspector()
    compiled_rfm = inspector.compile_model_sql("fct_customer_rfm")
    
    # Assert ref was replaced with BigQuery identifier
    assert "`anna-ml-pipeline.retail_data.int_customer_orders_rollup`" in compiled_rfm
    # Assert config block was stripped
    assert "config(" not in compiled_rfm
    # Assert SQL structure remains intact
    assert "SELECT" in compiled_rfm
    assert "customer_id" in compiled_rfm
    assert "rfm_segment_tier" in compiled_rfm
