import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import logging
import pandas as pd
import numpy as np
import psycopg2
from psycopg2.extras import execute_batch
from app.db_postgres import get_connection, release_connection, init_pgvector_table
from src.bqml_enrichment import BigQueryMLEnrichmentService

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def sync_product_vectors():
    """
    Extracts product catalog items, enriches taxonomy using BigQuery ML (Gemini 1.5 Flash),
    generates 768d vector embeddings, and upserts into PostgreSQL pgvector table.
    """
    logging.info("Starting Product Catalog Vector Sync via BigQuery ML...")
    
    # 1. Load unique products (BigQuery in production vs CSV in local/CI)
    if os.getenv("USE_BIGQUERY", "false").lower() == "true":
        logging.info("Querying latest product transactions directly from BigQuery table...")
        from google.cloud import bigquery
        project_id = os.getenv("GCP_PROJECT", "anna-ml-pipeline")
        client = bigquery.Client(project=project_id)
        query = f"""
            SELECT 
                StockCode AS stock_code, 
                Description AS description, 
                UnitPrice AS unit_price, 
                InvoiceNo AS invoice_no
            FROM `{project_id}.retail_data.transactions`
            WHERE Description IS NOT NULL AND UnitPrice > 0
        """
        df = client.query(query).to_dataframe()
    else:
        logging.info("Loading product transactions from local CSV fallback...")
        csv_path = "data/processed/clean_retail.csv"
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
        else:
            csv_path = "data/raw/online_retail.csv"
            df = pd.read_csv(csv_path)
        
    df.columns = df.columns.str.lower()
    
    stock_col = "stock_code" if "stock_code" in df.columns else "stockcode"
    unit_col = "unit_price" if "unit_price" in df.columns else "unitprice"
    desc_col = "description"
    inv_col = "invoice_no" if "invoice_no" in df.columns else "invoiceno"
    
    df = df.dropna(subset=[desc_col, stock_col, unit_col])
    df = df[df[unit_col] > 0]
    
    catalog = df.groupby(desc_col).agg({
        stock_col: "first",
        unit_col: "median",
        inv_col: "count"
    }).reset_index()
    
    catalog.rename(columns={
        desc_col: "description",
        stock_col: "stock_code",
        unit_col: "unit_price",
        inv_col: "sales_count"
    }, inplace=True)
    
    catalog = catalog.sort_values(by="sales_count", ascending=False).head(500)
    logging.info("Selected top %d unique products from retail catalog.", len(catalog))
    
    # 2. BigQuery ML Automated Taxonomy Enrichment (Gemini 1.5 Flash)
    bqml_service = BigQueryMLEnrichmentService()
    logging.info("Enriching %d products via BigQuery ML service...", len(catalog))
    enriched_catalog = bqml_service.enrich_products_batch(catalog)
    
    # 3. Generate 768d Dense Vector Embeddings (text-embedding-004)
    logging.info("Generating 768-dimensional embeddings for product catalog...")
    vectorized_catalog = bqml_service.generate_embeddings_batch(enriched_catalog)
    
    # 4. Upsert into PostgreSQL pgvector table
    logging.info("Connecting to PostgreSQL to populate pgvector table 'product_catalog_vectors'...")
    init_pgvector_table()
    
    conn = get_connection()
    if not conn:
        logging.error("Could not obtain PostgreSQL connection for vector sync.")
        return
        
    try:
        cursor = conn.cursor()
        
        # Prepare records
        records = []
        for _, row in vectorized_catalog.iterrows():
            vec_str = "[" + ",".join(map(str, row["embedding"])) + "]"
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
    finally:
        release_connection(conn)

if __name__ == "__main__":
    sync_product_vectors()
