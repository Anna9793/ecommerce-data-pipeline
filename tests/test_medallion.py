import os
import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch

from src.bronze import ingest_raw_source, run_bronze
from src.silver import clean_and_enrich_transactions, run_silver
from src.gold import compute_gold_features, load_gold, run_gold


def test_bronze_ingestion(tmp_path):
    """
    Verifies that Bronze layer correctly ingests raw Excel files and persists immutable Bronze CSV.
    """
    raw_source = tmp_path / "raw_source.xlsx"
    bronze_target = tmp_path / "bronze_target.csv"

    sample_df = pd.DataFrame({
        "InvoiceNo": ["536365"],
        "Quantity": [6],
        "UnitPrice": [2.55]
    })
    sample_df.to_excel(raw_source, index=False)

    # 1. Test ingest_raw_source
    ingested_df = ingest_raw_source(source_path=raw_source, target_path=bronze_target)
    assert bronze_target.exists()
    assert len(ingested_df) == 1

    # 2. Test direct CSV read and data integrity
    loaded_df = pd.read_csv(bronze_target)
    assert len(loaded_df) == 1
    assert loaded_df["InvoiceNo"].iloc[0] == 536365


def test_run_bronze():
    """
    Verifies that run_bronze returns a valid non-empty DataFrame.
    """
    df = run_bronze()
    assert isinstance(df, pd.DataFrame)
    assert not df.empty


@pytest.fixture
def sample_raw_data():
    return pd.DataFrame({
        "InvoiceNo":   ["536365", "536365", "536366", "C536367", "536368", "536369", "536370", "536371"],
        "StockCode":   ["85123A", "85123A", "71053",  "84406B",  "22752",  "21730",  "22633",  "22632"],
        "Description": ["HEART",  "HEART",  "LANTERN", "COAT",    "SHOES",  "DOLL",   "BAG",    "BOTTLE"],
        "Quantity":    [6,        6,        2,         -1,        10,       -3,       0,        5],
        "InvoiceDate": ["2010-12-01 08:26:00"] * 8,
        "UnitPrice":   [2.55,     2.55,     3.39,      10.0,      0.0,      2.0,      4.0,      5.0],
        "CustomerID":  [17850.0,  17850.0,  np.nan,    17850.0,   13047.0,  12583.0,  15100.0,  "GUEST"],
        "Country":     ["United Kingdom"] * 8
    })


def test_silver_cleaning_and_enrichment(sample_raw_data):
    """
    Verifies that Silver layer performs atomic cleaning, typing, and transactional enrichment.
    """
    silver = clean_and_enrich_transactions(sample_raw_data)

    # 1. Column standardization and types
    assert "invoice_no" in silver.columns
    assert "customer_id" in silver.columns
    assert "order_value" in silver.columns
    assert "invoice_year" in silver.columns
    assert "invoice_hour" in silver.columns
    assert pd.api.types.is_datetime64_any_dtype(silver["invoice_date"])

    # 2. Excluded rows:
    # - Duplicate (536365 duplicate dropped)
    # - Missing CustomerID (np.nan dropped)
    # - CustomerID == 'GUEST' dropped
    # - UnitPrice == 0.0 dropped
    # - Quantity == 0 dropped
    # - Negative quantity without 'C' prefix dropped (-3 on 536369)
    # Surviving rows:
    # - 536365 (purchase by 17850)
    # - C536367 (cancellation by 17850)
    assert len(silver) == 2
    assert "17850" in silver["customer_id"].values
    assert "GUEST" not in silver["customer_id"].values

    # 3. Enrichment calculations
    row_purchase = silver[silver["quantity"] > 0].iloc[0]
    assert row_purchase["order_value"] == round(6 * 2.55, 2)
    assert row_purchase["invoice_hour"] == 8


def test_gold_feature_computation():
    """
    Verifies that Gold layer produces customer features with exactly 0 NaNs
    and handles edge cases (like zero monetary in 90 days).
    """
    dates = pd.date_range("2011-01-01", periods=10, freq="10D")
    silver_df = pd.DataFrame({
        "invoice_no":   ["101", "102", "103", "C104", "105"],
        "customer_id":  ["CUST_A", "CUST_A", "CUST_B", "CUST_A", "CUST_C"],
        "quantity":     [2, 3, 5, -1, -2],
        "unit_price":   [10.0, 20.0, 15.0, 20.0, 5.0],
        "order_value":  [20.0, 60.0, 75.0, -20.0, -10.0],
        "invoice_date": [dates[-1], dates[-2], dates[-3], dates[-1], dates[-1]],
        "invoice_hour": [10, 10, 14, 10, 9]
    })

    gold = compute_gold_features(silver_df)

    # 1. Customer C only had cancellations (frequency 0) -> filtered out
    assert set(gold["customer_id"].values) == {"CUST_A", "CUST_B"}

    # 2. Zero NaNs guaranteed
    assert gold.isnull().sum().sum() == 0

    # 3. Customer A verification
    cust_a = gold[gold["customer_id"] == "CUST_A"].iloc[0]
    assert cust_a["frequency"] == 2
    assert cust_a["avg_order_value"] == 40.0  # (20 + 60) / 2
    assert cust_a["cancellation_rate"] == round(1 / 3, 4)  # 1 cancellation out of 3 total orders
    assert cust_a["preferred_shopping_hour"] == 10
    assert cust_a["spending_velocity"] >= 0.0


def test_data_mode_precedence(monkeypatch):
    """
    Verifies that explicit parameter use_bigquery=False forces local loading
    even when USE_BIGQUERY=true is set in the environment.
    """
    monkeypatch.setenv("USE_BIGQUERY", "true")

    # load_gold with explicit use_bigquery=False must read local file without querying BigQuery
    df = load_gold(use_bigquery=False)
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert df.isnull().sum().sum() == 0


@patch("pandas_gbq.read_gbq")
def test_cloud_mode_queries_bigquery_views(mock_read_gbq, monkeypatch):
    """
    Verifies that use_bigquery=True routes to the appropriate BigQuery views.
    """
    monkeypatch.setenv("USE_BIGQUERY", "true")
    monkeypatch.setenv("GCP_PROJECT", "test-project")
    
    mock_read_gbq.return_value = pd.DataFrame({
        "customer_id": ["101"],
        "recency": [10],
        "frequency": [2],
        "avg_order_value": [50.0],
        "spending_velocity": [1.0],
        "cancellation_rate": [0.0],
        "preferred_shopping_hour": [12]
    })

    df = load_gold(use_bigquery=True)
    assert mock_read_gbq.called
    query_called = mock_read_gbq.call_args[0][0]
    assert "retail_data.rfm_features" in query_called
    assert len(df) == 1
