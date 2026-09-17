import os
import logging
import pandas as pd
import numpy as np
import psycopg2
from psycopg2.extras import execute_batch

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def categorize_product(desc: str) -> tuple:
    """Assigns category and contextual semantic tags based on product keywords."""
    d = str(desc).upper()
    
    if any(k in d for k in ["TEA", "MUG", "CUP", "PLATE", "BOWL", "CAKE", "BAKING", "CUTLERY", "JAR", "BOTTLE", "DISH", "FORK", "SPOON"]):
        category = "Kitchen & Dining"
        tags = "kitchenware, cookware, cozy tea time, dining, tableware, host gifts"
    elif any(k in d for k in ["HEART", "LIGHT", "CANDLE", "CLOCK", "FRAME", "MIRROR", "CUSHION", "HANGING", "SIGN", "LANTERN", "VASE"]):
        category = "Home Decor & Lighting"
        tags = "home accents, cozy lighting, warm interior styling, romantic decor, ambient lighting"
    elif any(k in d for k in ["CHRISTMAS", "TREE", "STAR", "SNOW", "SANTA", "WINTER", "ADVENT", "BELL", "REINDEER", "HOLIDAY"]):
        category = "Holiday & Seasonal"
        tags = "winter festive gifts, cozy christmas celebration, holiday decoration, seasonal cheer"
    elif any(k in d for k in ["BAG", "TOTE", "BOX", "TIN", "CASE", "PURSE", "BASKET", "STORAGE", "DRAWER", "LUNCH"]):
        category = "Storage & Accessories"
        tags = "daily accessories, travel storage, organizer essentials, stylish bags, portable gifts"
    elif any(k in d for k in ["PARTY", "BUNTING", "BALLOON", "GARLAND", "WRAP", "PAPER", "CARD", "RIBBON", "STICKER"]):
        category = "Party & Celebration"
        tags = "celebrations, party supplies, gift wrap, crafting, festive gatherings"
    elif any(k in d for k in ["TOY", "DOLL", "GAME", "PUZZLE", "CHILD", "PLUSH", "PENCIL"]):
        category = "Kids & Toys"
        tags = "playful gifts, family games, creative toys, children favorites"
    else:
        category = "Gifts & Living"
        tags = "lifestyle items, thoughtful novelty gifts, general merchandise"
        
    return category, tags

from app.db_postgres import get_connection, release_connection

def generate_embeddings_batch(texts: list) -> list:
    """Generates 768-dimensional embeddings using Vertex AI or deterministic fallback."""
    project_id = os.getenv("GCP_PROJECT", "anna-ml-pipeline")
    try:
        import vertexai
        from vertexai.language_models import TextEmbeddingModel, TextEmbeddingInput
        vertexai.init(project=project_id, location="us-central1")
        model = TextEmbeddingModel.from_pretrained("text-embedding-004")
        
        batch_size = 100
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            inputs = [TextEmbeddingInput(t, "RETRIEVAL_DOCUMENT") for t in batch]
            res = model.get_embeddings(inputs)
            all_embeddings.extend([emb.values for emb in res])
        return all_embeddings
    except Exception as e:
        logging.warning("Vertex AI embedding generation unavailable (%s). Generating deterministic normalized embeddings.", e)
        embeddings = []
        for text in texts:
            # Deterministic pseudo-embedding based on hash seed
            seed = sum(ord(c) for c in text) % 10000
            rng = np.random.RandomState(seed)
            v = rng.randn(768).astype(np.float32)
            v /= np.linalg.norm(v)
            embeddings.append(v.tolist())
        return embeddings

def sync_product_vectors():
    """Extracts catalog items, builds rich contextual representations, and upserts into pgvector."""
    logging.info("Starting Product Catalog Vector Sync...")
    
    # 1. Load unique products from local dataset
    csv_path = "data/processed/clean_retail.csv"
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
    else:
        # Fallback to online_retail.csv
        csv_path = "data/raw/online_retail.csv"
        df = pd.read_csv(csv_path)
        
    df.columns = df.columns.str.lower()
    
    # Map possible column name variations
    stock_col = "stock_code" if "stock_code" in df.columns else "stockcode"
    unit_col = "unit_price" if "unit_price" in df.columns else "unitprice"
    desc_col = "description"
    inv_col = "invoice_no" if "invoice_no" in df.columns else "invoiceno"
    
    df = df.dropna(subset=[desc_col, stock_col, unit_col])
    df = df[df[unit_col] > 0]
    
    # Aggregate to top unique products
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
    logging.info("Selected top %d unique products from historical transactions.", len(catalog))
    
    # 2. Enrich with categories & rich document text
    categories = []
    document_texts = []
    
    for _, row in catalog.iterrows():
        desc = str(row["description"]).strip()
        price = float(row["unit_price"])
        cat, tags = categorize_product(desc)
        
        doc_text = f"Product: {desc} | Category: {cat} | Price: ${price:.2f} | Tags: {tags}"
        categories.append(cat)
        document_texts.append(doc_text)
        
    catalog["category"] = categories
    catalog["document_text"] = document_texts
    catalog["tenant_id"] = "giftshop_uk"
    
    # 3. Add Nordic Tech Catalog Products
    from app.rag_service import STORE_PROFILES
    nordic_items = STORE_PROFILES.get("nordic_tech", {}).get("fallback_catalog", [])
    nordic_rows = []
    for item in nordic_items:
        desc = item["description"]
        cat = item["category"]
        price = float(item["unit_price"])
        doc_text = f"Product: {desc} | Category: {cat} | Price: ${price:.2f} | Store: NordicWear & Tech | Tags: Scandinavian acoustics, performance activewear, workspace gear"
        nordic_rows.append({
            "stock_code": item["stock_code"],
            "description": desc,
            "category": cat,
            "unit_price": price,
            "document_text": doc_text,
            "tenant_id": "nordic_tech"
        })
    df_nordic = pd.DataFrame(nordic_rows)
    all_catalog = pd.concat([catalog, df_nordic], ignore_index=True)
    
    # 4. Generate 768d Vector Embeddings
    logging.info("Computing 768-dimensional contextual vector embeddings for %d multi-tenant products...", len(all_catalog))
    embeddings = generate_embeddings_batch(all_catalog["document_text"].tolist())
    all_catalog["embedding"] = embeddings
    
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
        for _, row in all_catalog.iterrows():
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
