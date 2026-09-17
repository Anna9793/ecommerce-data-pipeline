import os
import logging
import pandas as pd
import numpy as np
import psycopg2
from psycopg2.extras import execute_batch
from app.db_postgres import get_connection, release_connection
from src.bqml_enrichment import BigQueryMLEnrichmentService

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def sync_product_vectors():
    """
    Extracts multi-tenant catalog items, enriches taxonomy using BigQuery ML (Gemini 1.5 Flash),
    generates 768d vector embeddings, and upserts into PostgreSQL pgvector table.
    """
    logging.info("Starting Multi-Tenant Product Catalog Vector Sync via BigQuery ML...")
    
    # 1. Load unique products from local dataset (GiftShop UK)
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
    catalog["tenant_id"] = "giftshop_uk"
    logging.info("Selected top %d unique products for 'giftshop_uk'.", len(catalog))
    
    # 2. Add Nordic Tech Catalog Products
    from app.rag_service import STORE_PROFILES
    nordic_items = STORE_PROFILES.get("nordic_tech", {}).get("fallback_catalog", [])
    nordic_rows = []
    for item in nordic_items:
        nordic_rows.append({
            "stock_code": item["stock_code"],
            "description": item["description"],
            "unit_price": float(item["unit_price"]),
            "tenant_id": "nordic_tech"
        })
    df_nordic = pd.DataFrame(nordic_rows)
    all_catalog = pd.concat([catalog, df_nordic], ignore_index=True)
    
    # 3. BigQuery ML Automated Taxonomy Enrichment (Gemini 1.5 Flash)
    bqml_service = BigQueryMLEnrichmentService()
    logging.info("Enriching %d multi-tenant products via BigQuery ML service...", len(all_catalog))
    enriched_catalog = bqml_service.enrich_products_batch(all_catalog)
    
    # 4. Generate 768d Dense Vector Embeddings (text-embedding-004)
    logging.info("Generating 768-dimensional embeddings for multi-tenant catalog...")
    vectorized_catalog = bqml_service.generate_embeddings_batch(enriched_catalog)
    
    # 5. Upsert into PostgreSQL pgvector table
    logging.info("Connecting to PostgreSQL to populate pgvector table 'product_catalog_vectors'...")
    conn = get_connection()
    if not conn:
        logging.error("Could not obtain PostgreSQL connection for vector sync.")
        return
        
    try:
        cursor = conn.cursor()
        
        # Ensure pgvector extension and table exist
        cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS product_catalog_vectors (
                stock_code VARCHAR(50) NOT NULL,
                tenant_id VARCHAR(50) NOT NULL DEFAULT 'giftshop_uk',
                description TEXT NOT NULL,
                category VARCHAR(100),
                unit_price DOUBLE PRECISION NOT NULL,
                document_text TEXT NOT NULL,
                embedding vector(768),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (stock_code, tenant_id)
            );
            CREATE INDEX IF NOT EXISTS product_vector_idx 
            ON product_catalog_vectors 
            USING hnsw (embedding vector_cosine_ops);
            CREATE INDEX IF NOT EXISTS idx_product_catalog_tenant 
            ON product_catalog_vectors (tenant_id);
        """)
        conn.commit()
        
        # Prepare records
        records = []
        for _, row in vectorized_catalog.iterrows():
            vec_str = "[" + ",".join(map(str, row["embedding"])) + "]"
            records.append((
                str(row["stock_code"]),
                str(row["tenant_id"]),
                str(row["description"]),
                str(row["category"]),
                float(row["unit_price"]),
                str(row["document_text"]),
                vec_str
            ))
            
        upsert_query = """
            INSERT INTO product_catalog_vectors (stock_code, tenant_id, description, category, unit_price, document_text, embedding)
            VALUES (%s, %s, %s, %s, %s, %s, %s::vector)
            ON CONFLICT (stock_code, tenant_id) DO UPDATE SET
                description = EXCLUDED.description,
                category = EXCLUDED.category,
                unit_price = EXCLUDED.unit_price,
                document_text = EXCLUDED.document_text,
                embedding = EXCLUDED.embedding;
        """
        
        execute_batch(cursor, upsert_query, records, page_size=100)
        conn.commit()
        cursor.close()
        logging.info("✅ Successfully upserted %d multi-tenant product vector records into PostgreSQL pgvector table!", len(records))
    except Exception as e:
        logging.error("Failed to sync pgvector catalog to PostgreSQL: %s", e)
    finally:
        release_connection(conn)

if __name__ == "__main__":
    sync_product_vectors()
