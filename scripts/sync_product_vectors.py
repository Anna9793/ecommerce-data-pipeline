import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import logging
import pandas as pd
import psycopg2
from psycopg2.extras import execute_batch
from app.db_postgres import get_connection, release_connection, init_pgvector_table
from src.bqml_enrichment import BigQueryMLEnrichmentService

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def sync_product_vectors():
    """
    Synchronizes precomputed 768d vector embeddings from BigQuery ML table
    'retail_data.product_catalog_vectors' directly into PostgreSQL pgvector table.
    
    100% In-Database: BigQuery executes Gemini & Text-Embedding-004 transformations;
    this sync worker only replicates pre-materialized vectors to local/online OLTP storage.
    """
    logging.info("Starting Product Catalog Vector Sync (BigQuery ML -> PostgreSQL pgvector)...")
    
    use_bigquery = os.getenv("USE_BIGQUERY", "false").lower() == "true"
    if not use_bigquery:
        raise RuntimeError(
            "sync_product_vectors requires cloud mode (USE_BIGQUERY=true). "
            "BigQuery ML table 'retail_data.product_catalog_vectors' is required as source."
        )

    # 1. Fetch precomputed vectors from BigQuery
    bqml_service = BigQueryMLEnrichmentService()
    logging.info("Fetching precomputed product vectors from BigQuery table 'product_catalog_vectors'...")
    vectorized_catalog = bqml_service.fetch_catalog_vectors()
    logging.info("Retrieved %d vectorized products from BigQuery.", len(vectorized_catalog))
    
    if vectorized_catalog.empty:
        logging.warning("No product vectors found in BigQuery 'product_catalog_vectors'. Nothing to sync.")
        return

    # Validate schema contract
    required_cols = ["stock_code", "description", "category", "unit_price", "document_text", "embedding"]
    missing = set(required_cols) - set(vectorized_catalog.columns)
    if missing:
        raise ValueError(f"BigQuery product_catalog_vectors schema violation! Missing columns: {missing}")

    # 2. Upsert into PostgreSQL pgvector table
    logging.info("Connecting to PostgreSQL to populate pgvector table 'product_catalog_vectors'...")
    init_pgvector_table()
    
    conn = get_connection()
    if not conn:
        raise RuntimeError("Could not obtain PostgreSQL connection for vector sync.")
        
    try:
        cursor = conn.cursor()
        
        # Prepare records
        records = []
        for _, row in vectorized_catalog.iterrows():
            vec_list = list(row["embedding"])
            vec_str = "[" + ",".join(map(str, vec_list)) + "]"
            records.append((
                str(row["stock_code"]),
                str(row["description"]),
                str(row["category"]),
                float(row["unit_price"]),
                str(row["document_text"]),
                vec_str
            ))
            
        upsert_query = """
            INSERT INTO product_catalog_vectors (stock_code, description, category, unit_price, document_text, embedding)
            VALUES (%s, %s, %s, %s, %s, %s::vector)
            ON CONFLICT (stock_code) DO UPDATE SET
                description = EXCLUDED.description,
                category = EXCLUDED.category,
                unit_price = EXCLUDED.unit_price,
                document_text = EXCLUDED.document_text,
                embedding = EXCLUDED.embedding;
        """
        
        execute_batch(cursor, upsert_query, records, page_size=100)
        conn.commit()
        cursor.close()
        logging.info("✅ Successfully upserted %d product vector records into PostgreSQL pgvector table!", len(records))
    except Exception as e:
        logging.error("Failed to sync pgvector catalog to PostgreSQL: %s", e)
        raise RuntimeError(f"Failed to sync pgvector catalog to PostgreSQL: {e}") from e
    finally:
        release_connection(conn)

if __name__ == "__main__":
    sync_product_vectors()
