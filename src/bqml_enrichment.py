import os
import json
import logging
import pandas as pd
from typing import List, Dict, Optional, Tuple

from src.bq_queries import (
    STANDARD_CATEGORIES,
    build_text_enrichment_query,
    build_vector_embedding_query,
)


class BigQueryMLEnrichmentService:
    """
    In-Database Machine Learning & GenAI Service using Google BigQuery ML (BQML).
    Executes ML.GENERATE_TEXT (Gemini 1.5 Flash) for automated taxonomy categorization
    and ML.GENERATE_EMBEDDING (text-embedding-004) for in-warehouse vectorization.
    """

    def __init__(self, project_id: Optional[str] = None, dataset_id: str = "retail_data"):
        self.project_id = project_id or os.getenv("GCP_PROJECT", "anna-ml-pipeline")
        self.dataset_id = dataset_id
        self.use_bigquery = os.getenv("USE_BIGQUERY", "false").lower() == "true"

    def build_bqml_text_enrichment_query(self, source_table: str = "raw_products", target_table: str = "dim_products_enriched") -> str:
        """Builds the BigQuery ML SQL query executing ML.GENERATE_TEXT with Gemini 1.5 Flash."""
        return build_text_enrichment_query(
            project_id=self.project_id,
            dataset_id=self.dataset_id,
            source_table=source_table,
            target_table=target_table,
            categories=STANDARD_CATEGORIES,
        )

    def build_bqml_vector_embedding_query(self, source_table: str = "dim_products_enriched", target_table: str = "product_catalog_vectors") -> str:
        """Builds the BigQuery ML SQL query executing ML.GENERATE_EMBEDDING with text-embedding-004."""
        return build_vector_embedding_query(
            project_id=self.project_id,
            dataset_id=self.dataset_id,
            source_table=source_table,
            target_table=target_table,
        )

    def enrich_products_batch(self, df_products: pd.DataFrame) -> pd.DataFrame:
        """
        Enriches product catalog with taxonomy category, semantic tags, and document_text
        using BigQuery ML Remote Model (Gemini 1.5 Flash).
        Requires USE_BIGQUERY=true.
        """
        if df_products.empty:
            return df_products

        if not self.use_bigquery:
            raise RuntimeError(
                "BigQueryMLEnrichmentService requires cloud mode (USE_BIGQUERY=true). "
                "Local execution without BigQuery ML is not supported."
            )

        try:
            from google.cloud import bigquery
            client = bigquery.Client(project=self.project_id)
            query = self.build_bqml_text_enrichment_query()
            logging.info("Executing BigQuery ML Gemini text enrichment job in project %s...", self.project_id)
            query_job = client.query(query)
            query_job.result()
            
            # Fetch back enriched data
            enriched_df = client.query(f"SELECT * FROM `{self.project_id}.{self.dataset_id}.dim_products_enriched`").to_dataframe()
            logging.info("Successfully fetched %d BQML enriched products from BigQuery.", len(enriched_df))
            return enriched_df
        except Exception as e:
            logging.error("BigQuery ML text enrichment failed: %s", e)
            raise RuntimeError(f"BigQuery ML text enrichment failed: {e}") from e

    def generate_embeddings_batch(self, df_products: pd.DataFrame) -> pd.DataFrame:
        """
        Generates 768-dimensional vector embeddings for enriched product catalog using Vertex AI text-embedding-004.
        Requires USE_BIGQUERY=true.
        """
        if df_products.empty:
            return df_products

        if not self.use_bigquery:
            raise RuntimeError(
                "BigQueryMLEnrichmentService requires cloud mode (USE_BIGQUERY=true). "
                "Local execution without Vertex AI / BigQuery is not supported."
            )

        df = df_products.copy()
        texts = df["document_text"].tolist()

        try:
            import vertexai
            from vertexai.language_models import TextEmbeddingModel, TextEmbeddingInput
            vertexai.init(project=self.project_id, location="us-central1")
            model = TextEmbeddingModel.from_pretrained("text-embedding-004")
            
            batch_size = 100
            all_embeddings = []
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i+batch_size]
                inputs = [TextEmbeddingInput(t, "RETRIEVAL_DOCUMENT") for t in batch]
                res = model.get_embeddings(inputs)
                all_embeddings.extend([emb.values for emb in res])
            df["embedding"] = all_embeddings
            return df
        except Exception as e:
            logging.error("Vertex AI embedding generation failed: %s", e)
            raise RuntimeError(f"Vertex AI embedding generation failed: {e}") from e

    def execute_text_enrichment_job(self, source_table: str = "raw_products", target_table: str = "dim_products_enriched") -> None:
        """
        Executes BigQuery ML Gemini 1.5 Flash text enrichment query in-database.
        """
        if not self.use_bigquery:
            raise RuntimeError("BigQueryMLEnrichmentService requires cloud mode (USE_BIGQUERY=true).")
        try:
            from google.cloud import bigquery
            client = bigquery.Client(project=self.project_id)
            query = self.build_bqml_text_enrichment_query(source_table=source_table, target_table=target_table)
            logging.info("Executing BigQuery ML Gemini text enrichment job in project %s...", self.project_id)
            query_job = client.query(query)
            query_job.result()
            logging.info("Successfully materialized %s.%s.%s in BigQuery.", self.project_id, self.dataset_id, target_table)
        except Exception as e:
            logging.error("BigQuery ML text enrichment job failed: %s", e)
            raise RuntimeError(f"BigQuery ML text enrichment job failed: {e}") from e

    def execute_vector_embedding_job(self, source_table: str = "dim_products_enriched", target_table: str = "product_catalog_vectors") -> None:
        """
        Executes BigQuery ML text-embedding-004 vector embedding query in-database.
        """
        if not self.use_bigquery:
            raise RuntimeError("BigQueryMLEnrichmentService requires cloud mode (USE_BIGQUERY=true).")
        try:
            from google.cloud import bigquery
            client = bigquery.Client(project=self.project_id)
            query = self.build_bqml_vector_embedding_query(source_table=source_table, target_table=target_table)
            logging.info("Executing BigQuery ML vector embedding job in project %s...", self.project_id)
            query_job = client.query(query)
            query_job.result()
            logging.info("Successfully materialized %s.%s.%s in BigQuery.", self.project_id, self.dataset_id, target_table)
        except Exception as e:
            logging.error("BigQuery ML vector embedding job failed: %s", e)
            raise RuntimeError(f"BigQuery ML vector embedding job failed: {e}") from e

    def execute_catalog_vectorization_pipeline(self) -> None:
        """
        Executes the complete in-database BQML pipeline:
        1. raw_products -> dim_products_enriched (Gemini)
        2. dim_products_enriched -> product_catalog_vectors (Embeddings)
        """
        self.execute_text_enrichment_job()
        self.execute_vector_embedding_job()

    def fetch_catalog_vectors(self) -> pd.DataFrame:
        """
        Fetches the precomputed product vectors from BigQuery table product_catalog_vectors.
        """
        if not self.use_bigquery:
            raise RuntimeError("BigQueryMLEnrichmentService requires cloud mode (USE_BIGQUERY=true).")
        try:
            from google.cloud import bigquery
            client = bigquery.Client(project=self.project_id)
            query = f"""
                SELECT stock_code, description, category, unit_price, document_text, embedding
                FROM `{self.project_id}.{self.dataset_id}.product_catalog_vectors`
            """
            return client.query(query).to_dataframe()
        except Exception as e:
            logging.error("Failed to fetch product catalog vectors from BigQuery: %s", e)
            raise RuntimeError(f"Failed to fetch product catalog vectors from BigQuery: {e}") from e
