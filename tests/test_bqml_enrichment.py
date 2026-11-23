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
    for cat in STANDARD_CATEGORIES:
        assert cat in text_sql
    
    # 2. Vector Embedding SQL
    vector_sql = service.build_bqml_vector_embedding_query()
    assert "ML.GENERATE_EMBEDDING" in vector_sql
    assert "test-project.test_dataset.text_embedding_remote" in vector_sql
    assert "768 AS output_dimensionality" in vector_sql
    assert "product_catalog_vectors" in vector_sql


def test_bqml_fail_fast_when_not_in_cloud_mode():
    service = BigQueryMLEnrichmentService(project_id="test-project")
    service.use_bigquery = False

    df_raw = pd.DataFrame([
        {
            "stock_code": "85123A",
            "description": "WHITE HANGING HEART T-LIGHT HOLDER",
            "unit_price": 2.55,
        }
    ])

    with pytest.raises(RuntimeError, match="requires cloud mode"):
        service.enrich_products_batch(df_raw)

    with pytest.raises(RuntimeError, match="requires cloud mode"):
        service.generate_embeddings_batch(df_raw)


@patch("google.cloud.bigquery.Client")
def test_bqml_cloud_enrichment_execution(mock_bq_client_class):
    mock_client = mock_bq_client_class.return_value
    mock_query_job = MagicMock()
    mock_client.query.return_value = mock_query_job
    
    # Mock to_dataframe result
    mock_df_result = pd.DataFrame([
        {
            "stock_code": "85123A",
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

    df_input = pd.DataFrame([{"stock_code": "85123A", "description": "WHITE HANGING HEART T-LIGHT HOLDER", "unit_price": 2.55}])
    result = service.enrich_products_batch(df_input)

    assert len(result) == 1
    assert result.iloc[0]["category"] == "Home Decor & Lighting"
    assert mock_client.query.called


@patch("google.cloud.bigquery.Client")
def test_bqml_cloud_enrichment_fail_fast(mock_bq_client_class):
    mock_client = mock_bq_client_class.return_value
    mock_client.query.side_effect = Exception("BigQuery connection timeout / quota exceeded")

    service = BigQueryMLEnrichmentService(project_id="test-project")
    service.use_bigquery = True

    df_input = pd.DataFrame([{"stock_code": "85123A", "description": "WHITE HANGING HEART T-LIGHT HOLDER", "unit_price": 2.55}])
    
    with pytest.raises(RuntimeError, match="BigQuery ML text enrichment failed"):
        service.enrich_products_batch(df_input)


@patch("vertexai.init")
@patch("vertexai.language_models.TextEmbeddingModel.from_pretrained")
def test_bqml_cloud_vector_embedding_execution(mock_model_class, mock_vertex_init):
    mock_model = mock_model_class.return_value
    mock_emb = MagicMock()
    mock_emb.values = [0.05] * 768
    mock_model.get_embeddings.return_value = [mock_emb]

    service = BigQueryMLEnrichmentService(project_id="test-project")
    service.use_bigquery = True

    df_input = pd.DataFrame([
        {
            "stock_code": "85123A",
            "document_text": "Product: WHITE HANGING HEART T-LIGHT HOLDER"
        }
    ])
    result = service.generate_embeddings_batch(df_input)

    assert "embedding" in result.columns
    assert len(result["embedding"].iloc[0]) == 768
    assert result["embedding"].iloc[0][0] == 0.05


@patch("vertexai.init")
@patch("vertexai.language_models.TextEmbeddingModel.from_pretrained")
def test_bqml_cloud_vector_embedding_fail_fast(mock_model_class, mock_vertex_init):
    mock_model = mock_model_class.return_value
    mock_model.get_embeddings.side_effect = Exception("Vertex AI quota exceeded")

    service = BigQueryMLEnrichmentService(project_id="test-project")
    service.use_bigquery = True

    df_input = pd.DataFrame([
        {
            "stock_code": "85123A",
            "document_text": "Product: WHITE HANGING HEART T-LIGHT HOLDER"
        }
    ])

    with pytest.raises(RuntimeError, match="Vertex AI embedding generation failed"):
        service.generate_embeddings_batch(df_input)
