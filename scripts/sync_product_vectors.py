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

def get_db_connection():
    db_host = os.getenv("POSTGRES_HOST", "localhost")
    db_port = int(os.getenv("POSTGRES_PORT", 5433))
    db_name = os.getenv("POSTGRES_DB", "ml_pipeline")
    db_user = os.getenv("POSTGRES_USER", "postgres")
    db_password = os.getenv("POSTGRES_PASSWORD", "passione")
    return psycopg2.connect(
        host=db_host,
        port=db_port,
        dbname=db_name,
        user=db_user,
        password=db_password
    )

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
        
    df = df.dropna(subset=["Description", "StockCode", "UnitPrice"])
    df = df[df["UnitPrice"] > 0]
    
    # Aggregate to top unique products
    catalog = df.groupby("Description").agg({
        "StockCode": "first",
        "UnitPrice": "median",
        "InvoiceNo": "count"
    }).reset_index()
    
    catalog.rename(columns={"InvoiceNo": "sales_count"}, inplace=True)
    catalog = catalog.sort_values(by="sales_count", ascending=False).head(500)
    logging.info("Selected top %d unique products from historical transactions.", len(catalog))
    
    # 2. Enrich with categories & rich document text
    categories = []
    document_texts = []
    
    for _, row in catalog.iterrows():
        desc = str(row["Description"]).strip()
        price = float(row["UnitPrice"])
        cat, tags = categorize_product(desc)
        
        doc_text = f"Product: {desc} | Category: {cat} | Price: ${price:.2f} | Tags: {tags}"
        categories.append(cat)
        document_texts.append(doc_text)
        
    catalog["category"] = categories
    catalog["document_text"] = document_texts
    
    # 3. Generate 768d Vector Embeddings
    logging.info("Computing 768-dimensional contextual vector embeddings...")
    embeddings = generate_embeddings_batch(document_texts)
    catalog["embedding"] = embeddings
    
    # 4. Upsert into PostgreSQL pgvector table
    logging.info("Connecting to PostgreSQL to populate pgvector table 'product_catalog_vectors'...")
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Ensure pgvector extension and table exist
        cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS product_catalog_vectors (
                stock_code VARCHAR(50) PRIMARY KEY,
                description TEXT NOT NULL,
                category VARCHAR(100),
                unit_price DOUBLE PRECISION NOT NULL,
                document_text TEXT NOT NULL,
                embedding vector(768),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            CREATE INDEX IF NOT EXISTS product_vector_idx 
            ON product_catalog_vectors 
            USING hnsw (embedding vector_cosine_ops);
        """)
        conn.commit()
        
        # Prepare records
        records = []
        for _, row in catalog.iterrows():
            vec_str = "[" + ",".join(map(str, row["embedding"])) + "]"
            records.append((
                str(row["StockCode"]),
                str(row["Description"]),
                str(row["category"]),
                float(row["UnitPrice"]),
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
        logging.info("✅ Successfully upserted %d product vector records into PostgreSQL pgvector table!", len(records))
        
        cursor.close()
        conn.close()
    except Exception as e:
        logging.error("Failed to sync pgvector catalog to PostgreSQL: %s", e)

if __name__ == "__main__":
    sync_product_vectors()
