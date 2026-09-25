import os
import json
import logging
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple

logging.basicConfig(level=logging.INFO)

# Standard Category Taxonomy for E-Commerce Platform
STANDARD_CATEGORIES = [
    "Kitchen & Dining",
    "Home Decor & Lighting",
    "Holiday & Seasonal",
    "Smart Audio & Acoustics",
    "Performance Activewear",
    "Ergonomic Workspace",
    "Storage & Accessories",
    "Party & Celebration",
    "Kids & Toys",
    "Gifts & Living"
]


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
        """
        Builds the BigQuery ML SQL query executing ML.GENERATE_TEXT with Gemini 1.5 Flash.
        """
        return f"""
        CREATE OR REPLACE TABLE `{self.project_id}.{self.dataset_id}.{target_table}` AS
        WITH raw_prompts AS (
          SELECT 
            stock_code,
            description,
            unit_price,
            tenant_id,
            CONCAT(
              'You are an expert e-commerce catalog taxonomist. Analyze this product description: "', 
              description, 
              '". Return a JSON object with "category" (e.g. Kitchen & Dining, Home Decor & Lighting, Holiday & Seasonal, Smart Audio & Acoustics, Performance Activewear, Storage & Accessories) and "tags" (array of 3-5 search keywords). JSON only.'
            ) AS prompt
          FROM `{self.project_id}.{self.dataset_id}.{source_table}`
          WHERE description IS NOT NULL AND TRIM(description) != ''
        ),
        llm_responses AS (
          SELECT 
            stock_code,
            description,
            unit_price,
            tenant_id,
            ml_generate_text_result
          FROM ML.GENERATE_TEXT(
            MODEL `{self.project_id}.{self.dataset_id}.gemini_flash_remote`,
            TABLE raw_prompts,
            STRUCT(
              0.1 AS temperature,
              128 AS max_output_tokens,
              'application/json' AS response_mime_type,
              TRUE AS flatten_json_output
            )
          )
        )
        SELECT
          stock_code,
          tenant_id,
          description,
          COALESCE(JSON_EXTRACT_SCALAR(ml_generate_text_result, '$.category'), 'General Merchandise') AS category,
          COALESCE(JSON_EXTRACT_STRING_ARRAY(ml_generate_text_result, '$.tags'), ARRAY<STRING>['lifestyle', 'general']) AS tags,
          unit_price,
          CONCAT(
            'Product: ', description, 
            ' | Category: ', COALESCE(JSON_EXTRACT_SCALAR(ml_generate_text_result, '$.category'), 'General Merchandise'), 
            ' | Price: $', CAST(ROUND(unit_price, 2) AS STRING),
            ' | Tags: ', ARRAY_TO_STRING(COALESCE(JSON_EXTRACT_STRING_ARRAY(ml_generate_text_result, '$.tags'), ARRAY<STRING>['gift', 'lifestyle']), ', ')
          ) AS document_text,
          CURRENT_TIMESTAMP() AS enriched_at
        FROM llm_responses;
        """.strip()

    def build_bqml_vector_embedding_query(self, source_table: str = "dim_products_enriched", target_table: str = "product_catalog_vectors") -> str:
        """
        Builds the BigQuery ML SQL query executing ML.GENERATE_EMBEDDING with text-embedding-004.
        """
        return f"""
        CREATE OR REPLACE TABLE `{self.project_id}.{self.dataset_id}.{target_table}` AS
        SELECT
          stock_code,
          tenant_id,
          description,
          category,
          unit_price,
          document_text,
          ml_generate_embedding_result AS embedding,
          CURRENT_TIMESTAMP() AS vectorized_at
        FROM ML.GENERATE_EMBEDDING(
          MODEL `{self.project_id}.{self.dataset_id}.text_embedding_remote`,
          (
            SELECT 
              stock_code,
              tenant_id,
              description,
              category,
              unit_price,
              document_text,
              document_text AS content
            FROM `{self.project_id}.{self.dataset_id}.{source_table}`
          ),
          STRUCT(
            'RETRIEVAL_DOCUMENT' AS task_type,
            768 AS output_dimensionality
          )
        );
        """.strip()

    def enrich_products_batch(self, df_products: pd.DataFrame) -> pd.DataFrame:
        """
        Enriches product catalog with taxonomy category, semantic tags, and document_text.
        In Cloud Mode (USE_BIGQUERY=true): Executes BigQuery ML Remote Model transformation.
        In Local/CI Mode (USE_BIGQUERY=false): Executes semantic zero-shot categorization fallback.
        """
        if df_products.empty:
            return df_products

        df = df_products.copy()

        if self.use_bigquery:
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
                logging.warning("BigQuery ML execution failed or unavailable (%s). Using deterministic zero-shot fallback.", e)

        # Deterministic / Zero-shot semantic categorization fallback
        categories = []
        tags_list = []
        doc_texts = []

        for _, row in df.iterrows():
            desc = str(row.get("description", "")).strip()
            price = float(row.get("unit_price", 0.0))
            tenant = str(row.get("tenant_id", "giftshop_uk"))
            
            cat, tags = self._classify_product_zero_shot(desc, tenant)
            doc_text = f"Product: {desc} | Category: {cat} | Price: ${price:.2f} | Store: {tenant} | Tags: {tags}"
            
            categories.append(cat)
            tags_list.append(tags.split(", "))
            doc_texts.append(doc_text)

        df["category"] = categories
        df["tags"] = tags_list
        df["document_text"] = doc_texts
        return df

    def generate_embeddings_batch(self, df_products: pd.DataFrame) -> pd.DataFrame:
        """
        Generates 768-dimensional vector embeddings for enriched product catalog.
        In Cloud Mode: Executes ML.GENERATE_EMBEDDING.
        In Local/CI Mode: Uses Vertex AI TextEmbeddingModel or deterministic fallback.
        """
        if df_products.empty:
            return df_products

        df = df_products.copy()
        texts = df["document_text"].tolist()

        if self.use_bigquery:
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
                logging.warning("Vertex AI / BQML embedding service in local fallback mode: %s", e)

        # Deterministic normalized vector fallback (768 dimensions)
        embeddings = []
        for text in texts:
            seed = sum(ord(c) for c in text) % 10000
            rng = np.random.RandomState(seed)
            v = rng.randn(768).astype(np.float32)
            v /= np.linalg.norm(v)
            embeddings.append(v.tolist())
        df["embedding"] = embeddings
        return df

    def _classify_product_zero_shot(self, desc: str, tenant_id: str) -> Tuple[str, str]:
        """
        Semantic rule engine emulating Gemini 1.5 Flash taxonomy resolution for local execution.
        """
        d = desc.upper()
        if tenant_id in ("nordic_tech", "shopify"):
            if any(k in d for k in ["HEADPHONE", "EARBUD", "AUDIO", "SPEAKER", "SOUND", "ANC"]):
                return "Smart Audio & Acoustics", "wireless audio, ANC acoustic, studio sound, bluetooth"
            elif any(k in d for k in ["KEYBOARD", "MOUSE", "STAND", "RGB", "DESK", "GAMING", "ERGONOMIC"]):
                return "Ergonomic Workspace", "mechanical keyboard, ergonomic setup, workspace gear, desk accessories"
            elif any(k in d for k in ["HOODIE", "PARKA", "SOCK", "THERMAL", "RUNNING", "MERINO", "GORE-TEX"]):
                return "Performance Activewear", "scandinavian activewear, thermal performance, outdoor outerwear"
            elif any(k in d for k in ["WATCH", "BOTTLE", "GADGET", "TITANIUM"]):
                return "Smart Wearables", "titanium smartwatch, smart lifestyle gadgets, fitness gear"

        # General / GiftShop UK Taxonomy
        if any(k in d for k in ["TEA", "MUG", "CUP", "PLATE", "BOWL", "CAKE", "BAKING", "CUTLERY", "JAR", "BOTTLE", "DISH", "FORK", "SPOON"]):
            return "Kitchen & Dining", "kitchenware, cookware, cozy tea time, dining, tableware, host gifts"
        elif any(k in d for k in ["HEART", "LIGHT", "CANDLE", "CLOCK", "FRAME", "MIRROR", "CUSHION", "HANGING", "SIGN", "LANTERN", "VASE"]):
            return "Home Decor & Lighting", "home accents, cozy lighting, warm interior styling, romantic decor, ambient lighting"
        elif any(k in d for k in ["CHRISTMAS", "TREE", "STAR", "SNOW", "SANTA", "WINTER", "ADVENT", "BELL", "REINDEER", "HOLIDAY"]):
            return "Holiday & Seasonal", "winter festive gifts, cozy christmas celebration, holiday decoration, seasonal cheer"
        elif any(k in d for k in ["BAG", "TOTE", "BOX", "TIN", "CASE", "PURSE", "BASKET", "STORAGE", "DRAWER", "LUNCH"]):
            return "Storage & Accessories", "daily accessories, travel storage, organizer essentials, stylish bags, portable gifts"
        elif any(k in d for k in ["PARTY", "BUNTING", "BALLOON", "GARLAND", "WRAP", "PAPER", "CARD", "RIBBON", "STICKER"]):
            return "Party & Celebration", "celebrations, party supplies, gift wrap, crafting, festive gatherings"
        elif any(k in d for k in ["TOY", "DOLL", "GAME", "PUZZLE", "CHILD", "PLUSH", "PENCIL"]):
            return "Kids & Toys", "playful gifts, family games, creative toys, children favorites"
        
        return "Gifts & Living", "lifestyle items, thoughtful novelty gifts, general merchandise"
