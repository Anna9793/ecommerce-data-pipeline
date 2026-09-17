import pytest
import pandas as pd
from unittest.mock import patch, MagicMock
from src.bqml_enrichment import BigQueryMLEnrichmentService, STANDARD_CATEGORIES


def test_bqml_sql_query_generation():
    service = BigQueryMLEnrichmentService(project_id="test-project", dataset_id="test_dataset")
    
    # 1. Text Enrichment SQL
    text_sql = service.build_bqml_text_enrichment_query()
    assert "ML.GENERATE_TEXT" in text_sql
    assert "test-project.test_dataset.gemini_flash_remote" in text_sql
    assert "application/json" in text_sql
    assert "dim_products_enriched" in text_sql
    
    # 2. Vector Embedding SQL
    vector_sql = service.build_bqml_vector_embedding_query()
    assert "ML.GENERATE_EMBEDDING" in vector_sql
    assert "test-project.test_dataset.text_embedding_remote" in vector_sql
    assert "768 AS output_dimensionality" in vector_sql
    assert "product_catalog_vectors" in vector_sql


def test_bqml_enrich_products_batch_local():
    service = BigQueryMLEnrichmentService(project_id="test-project")
    service.use_bigquery = False

    df_raw = pd.DataFrame([
        {
            "stock_code": "85123A",
            "description": "WHITE HANGING HEART T-LIGHT HOLDER",
            "unit_price": 2.55,
            "tenant_id": "giftshop_uk"
        },
        {
            "stock_code": "SKU-TECH-001",
            "description": "Nordic Pro ANC Wireless Headphones",
            "unit_price": 149.00,
            "tenant_id": "nordic_tech"
        }
    ])

    enriched_df = service.enrich_products_batch(df_raw)
    
    assert len(enriched_df) == 2
    assert "category" in enriched_df.columns
    assert "tags" in enriched_df.columns
    assert "document_text" in enriched_df.columns
    
    # Check UK GiftShop category classification
    assert enriched_df.iloc[0]["category"] == "Home Decor & Lighting"
    assert "Product: WHITE HANGING HEART T-LIGHT HOLDER" in enriched_df.iloc[0]["document_text"]
    
    # Check Nordic Tech category classification
    assert enriched_df.iloc[1]["category"] == "Smart Audio & Acoustics"
    assert "Nordic Pro ANC Wireless Headphones" in enriched_df.iloc[1]["document_text"]


def test_bqml_generate_embeddings_batch_local():
    service = BigQueryMLEnrichmentService(project_id="test-project")
    service.use_bigquery = False

    df_raw = pd.DataFrame([
        {
            "stock_code": "85123A",
            "description": "WHITE HANGING HEART T-LIGHT HOLDER",
            "unit_price": 2.55,
            "tenant_id": "giftshop_uk"
        }
    ])

    enriched_df = service.enrich_products_batch(df_raw)
    vectorized_df = service.generate_embeddings_batch(enriched_df)
    
    assert "embedding" in vectorized_df.columns
    assert len(vectorized_df["embedding"].iloc[0]) == 768
    assert isinstance(vectorized_df["embedding"].iloc[0][0], float)


@patch("google.cloud.bigquery.Client")
def test_bqml_cloud_mode_execution(mock_bq_client_class):
    mock_client = mock_bq_client_class.return_value
    mock_query_job = MagicMock()
    mock_client.query.return_value = mock_query_job
    
    # Mock to_dataframe result
    mock_df_result = pd.DataFrame([
        {
            "stock_code": "85123A",
            "tenant_id": "giftshop_uk",
            "description": "WHITE HANGING HEART T-LIGHT HOLDER",
            "category": "Home Decor & Lighting",
            "tags": ["lighting", "heart"],
            "unit_price": 2.55,
            "document_text": "Product: WHITE HANGING HEART T-LIGHT HOLDER..."
        }
    ])
    mock_query_job.to_dataframe.return_value = mock_df_result

    service = BigQueryMLEnrichmentService(project_id="test-project")
    service.use_bigquery = True

    df_input = pd.DataFrame([{"stock_code": "85123A", "description": "WHITE HANGING HEART T-LIGHT HOLDER", "unit_price": 2.55, "tenant_id": "giftshop_uk"}])
    result = service.enrich_products_batch(df_input)

    assert len(result) == 1
    assert result.iloc[0]["category"] == "Home Decor & Lighting"
    assert mock_client.query.called
