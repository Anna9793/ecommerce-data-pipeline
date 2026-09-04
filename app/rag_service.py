import os
import json
import logging
import numpy as np
from typing import Optional, List
from pydantic import BaseModel, Field
import vertexai
from vertexai.language_models import TextEmbeddingModel, TextEmbeddingInput
from vertexai.generative_models import GenerativeModel, GenerationConfig
from app.db_postgres import search_product_catalog_pgvector

logging.basicConfig(level=logging.INFO)

# ==========================================
# 1. Pydantic Structured Output Schemas
# ==========================================

def _get_pydantic_schema(model_cls):
    """Safely extracts dictionary JSON schema from Pydantic model for Vertex AI SDK compatibility."""
    if hasattr(model_cls, "model_json_schema"):
        return model_cls.model_json_schema()
    elif hasattr(model_cls, "schema"):
        return model_cls.schema()
    return model_cls

class RecommendedProduct(BaseModel):
    stock_code: str = Field(description="Product SKU code")
    description: str = Field(description="Product title")
    category: str = Field(description="Product category")
    unit_price: float = Field(description="Unit price in USD")
    similarity: float = Field(description="Vector match similarity score between 0.0 and 1.0")
    why_recommended: str = Field(description="1-2 sentences explaining why this matches the user's request")

class ProductAdvisorResponse(BaseModel):
    user_query: str = Field(description="Original user search request")
    budget_applied: float = Field(default=0.0, description="Max budget constraint if applied, or 0.0 if not specified")
    intro_message: str = Field(description="Warm, helpful 1-2 sentence assistant opening")
    recommendations: List[RecommendedProduct] = Field(description="List of top matching products with justifications")
    shopping_tip: str = Field(description="A helpful styling, gifting, or shopping tip")

# ==========================================
# 2. Product Advisor RAG Service
# ==========================================

STORE_PROFILES = {
    "giftshop_uk": {
        "store_name": "GiftShop UK Boutique",
        "description": "specialized boutique gift, vintage tea sets, home decor, ambient lighting, and holiday lifestyle shop",
        "departments": """
        - Home Decor & Lighting (lanterns, candles, wall signs, clocks, vases)
        - Kitchen & Dining (vintage tea sets, mugs, cake stands, tableware)
        - Holiday & Seasonal (Christmas ornaments, festive decor, seasonal gifts)
        - Storage & Accessories (tote bags, trinket boxes, cases)
        - Party & Celebration (bunting, banners, party supplies)
        - Kids & Toys (playful novelty items, plush toys, puzzles)
        """,
        "fallback_catalog": [
            {"stock_code": "85123A", "description": "WHITE HANGING HEART T-LIGHT HOLDER", "category": "Home Decor & Lighting", "unit_price": 2.55, "similarity": 0.92},
            {"stock_code": "22423", "description": "REGENCY CAKESTAND 3 TIER", "category": "Kitchen & Dining", "unit_price": 12.75, "similarity": 0.88},
            {"stock_code": "47566", "description": "PARTY BUNTING", "category": "Party & Celebration", "unit_price": 4.95, "similarity": 0.82},
            {"stock_code": "22086", "description": "PAPER CHAIN KIT 50'S CHRISTMAS", "category": "Holiday & Seasonal", "unit_price": 2.95, "similarity": 0.78},
            {"stock_code": "84879", "description": "ASSORTED COLOUR BIRD ORNAMENT", "category": "Home Decor & Lighting", "unit_price": 1.69, "similarity": 0.75}
        ]
    },
    "nordic_tech": {
        "store_name": "NordicWear & Tech (Shopify)",
        "description": "modern Scandinavian e-commerce brand specializing in smart audio acoustics, performance activewear, and ergonomic tech workspace gear",
        "departments": """
        - Smart Audio & Acoustics (ANC wireless headphones, studio monitor earbuds, sound gear)
        - Performance Activewear (Merino wool thermal hoodies, trail running caps, compression socks)
        - All-Weather Mountain Apparel (Gore-Tex all-weather parkas, technical outerwear)
        - Smart Wearables & Gadgets (ultralight titanium smartwatches, smart temperature bottles)
        - Ergonomic Workspace & Gaming (RGB hot-swap mechanical keyboards, wireless ergonomic gaming mice, aluminum laptop stands)
        """,
        "fallback_catalog": [
            {"stock_code": "SKU-TECH-001", "description": "Nordic Pro ANC Wireless Headphones", "category": "Smart Audio & Acoustics", "unit_price": 149.00, "similarity": 0.94},
            {"stock_code": "SKU-FASH-002", "description": "Merino Wool Thermal Performance Hoodie", "category": "Performance Activewear", "unit_price": 89.50, "similarity": 0.90},
            {"stock_code": "SKU-TECH-003", "description": "UltraLight Titanium Smartwatch v3", "category": "Smart Wearables", "unit_price": 199.99, "similarity": 0.86},
            {"stock_code": "SKU-TECH-004", "description": "Ergonomic Wireless Gaming Mouse", "category": "Ergonomic Workspace", "unit_price": 45.00, "similarity": 0.83},
            {"stock_code": "SKU-FASH-008", "description": "Gore-Tex All-Weather Mountain Parka", "category": "All-Weather Apparel", "unit_price": 249.00, "similarity": 0.81},
            {"stock_code": "SKU-TECH-006", "description": "Mechanical RGB Hot-Swap Keyboard", "category": "Ergonomic Workspace", "unit_price": 120.00, "similarity": 0.79},
            {"stock_code": "SKU-TECH-010", "description": "Minimalist Aluminum Laptop Stand", "category": "Ergonomic Workspace", "unit_price": 79.00, "similarity": 0.77},
            {"stock_code": "SKU-TECH-011", "description": "Studio Monitor Wireless Earbuds", "category": "Smart Audio & Acoustics", "unit_price": 110.00, "similarity": 0.75}
        ]
    }
}

# ==========================================
# 2. Product Advisor RAG Service
# ==========================================

class ProductAdvisorService:
    def __init__(self):
        self.project_id = os.getenv("GCP_PROJECT", "anna-ml-pipeline")
        
        try:
            vertexai.init(project=self.project_id, location="us-central1")
            self.embedding_model = TextEmbeddingModel.from_pretrained("text-embedding-004")
            self.gemini_model = GenerativeModel("gemini-1.5-flash")
        except Exception as e:
            logging.warning("Vertex AI initialization skipped in local test mode: %s", e)
            self.embedding_model = None
            self.gemini_model = None

    def get_query_embedding(self, query_text: str) -> list:
        """Generates a 768-dimensional embedding for user query."""
        if self.embedding_model:
            try:
                inputs = [TextEmbeddingInput(query_text, "RETRIEVAL_QUERY")]
                embeddings = self.embedding_model.get_embeddings(inputs)
                return embeddings[0].values
            except Exception as e:
                logging.warning("Vertex query embedding failed: %s. Using deterministic fallback.", e)
                
        # Deterministic fallback embedding
        seed = sum(ord(c) for c in query_text) % 10000
        rng = np.random.RandomState(seed)
        v = rng.randn(768).astype(np.float32)
        v /= np.linalg.norm(v)
        return v.tolist()

    def search_products(self, query_text: str, budget_max: Optional[float] = None, top_k: int = 4, tenant_id: str = "giftshop_uk") -> list:
        """Retrieves top products filtered by tenant/store catalog and budget."""
        tenant_key = "nordic_tech" if tenant_id in ("nordic_tech", "shopify") else "giftshop_uk"
        store_profile = STORE_PROFILES.get(tenant_key, STORE_PROFILES["giftshop_uk"])
        
        query_vector = self.get_query_embedding(query_text)
        results = search_product_catalog_pgvector(query_vector, budget_max=budget_max, top_k=top_k)
        
        # If pgvector returned results from default UK catalog but active store is Nordic, or if pgvector is offline
        if tenant_key == "nordic_tech" or not results:
            logging.info("Using multi-tenant catalog for '%s'...", tenant_key)
            catalog = store_profile["fallback_catalog"]
            if budget_max:
                catalog = [p for p in catalog if p["unit_price"] <= budget_max]
            
            # Rank products based on simple query relevance matching
            q_lower = query_text.lower()
            scored = []
            for p in catalog:
                desc = p["description"].lower()
                cat = p["category"].lower()
                score = 0.70
                if any(w in desc or w in cat for w in q_lower.split() if len(w) > 3):
                    score = 0.93
                scored.append({**p, "similarity": score})
            
            scored.sort(key=lambda x: x["similarity"], reverse=True)
            results = scored[:top_k] if scored else catalog[:top_k]
            
        return results

    def advise(self, query_text: str, budget_max: Optional[float] = None, top_k: int = 4, tenant_id: str = "giftshop_uk") -> dict:
        """
        Executes full RAG workflow for the active store tenant:
        1. Retrieval: Vector search & catalog filtering for the store
        2. Reasoning: Gemini generates personalized justifications per product based on store identity
        """
        tenant_key = "nordic_tech" if tenant_id in ("nordic_tech", "shopify") else "giftshop_uk"
        store_profile = STORE_PROFILES.get(tenant_key, STORE_PROFILES["giftshop_uk"])
        
        retrieved_products = self.search_products(query_text, budget_max=budget_max, top_k=top_k, tenant_id=tenant_key)
        
        products_context = ""
        for idx, p in enumerate(retrieved_products, 1):
            products_context += f"{idx}. Code: {p['stock_code']} | '{p['description']}' | Category: {p['category']} | Price: ${p['unit_price']:.2f} | Match Score: {p['similarity']*100:.1f}%\n"

        prompt = f"""
        You are an expert E-Commerce Personal Shopper & Product Advisor Agent for '{store_profile['store_name']}', a {store_profile['description']}.
        
        STORE SPECIALTY & DEPARTMENTS:
        {store_profile['departments']}
        
        CUSTOMER SEARCH QUERY: "{query_text}"
        APPLIED BUDGET LIMIT: {f"${budget_max:.2f}" if budget_max else "None specified"}
        STORE TENANT: {tenant_key}
        
        RETRIEVED CATALOG CANDIDATES (Retrieved via store catalog semantic search):
        {products_context}
        
        INSTRUCTIONS:
        1. Write a friendly, engaging intro message tailored to '{store_profile['store_name']}'. If the customer asked for items our store does not sell, politely clarify our store's specialty and recommend the closest high-quality match from our catalog.
        2. For each retrieved product, provide a specific justification ('why_recommended') explaining why it perfectly matches their search.
        3. Provide an insightful 'shopping_tip' aligned with '{store_profile['store_name']}' products (e.g. cold-weather layering, acoustic tuning, desk setup, or gift wrapping).
        """
        
        try:
            if not self.gemini_model:
                raise ValueError("Gemini model not initialized.")
                
            config = GenerationConfig(
                response_mime_type="application/json",
                response_schema=_get_pydantic_schema(ProductAdvisorResponse)
            )
            response = self.gemini_model.generate_content(prompt, generation_config=config)
            data = json.loads(response.text)
            
            # Merge vector similarity scores into final structure
            recs = []
            for item in data.get("recommendations", []):
                recs.append({
                    "stock_code": item.get("stock_code", ""),
                    "description": item.get("description", ""),
                    "category": item.get("category", "General"),
                    "unit_price": float(item.get("unit_price", 0.0)),
                    "similarity": float(item.get("similarity", 0.85)),
                    "why_recommended": item.get("why_recommended", "Matches your search criteria.")
                })
                
            return {
                "tenant_id": tenant_key,
                "store_name": store_profile["store_name"],
                "user_query": query_text,
                "budget_applied": budget_max,
                "intro_message": data.get("intro_message", f"Welcome to {store_profile['store_name']}! Here are our top handpicked recommendations:"),
                "recommendations": recs if recs else retrieved_products,
                "shopping_tip": data.get("shopping_tip", "Pair with coordinating accessories for maximum performance!")
            }
        except Exception as e:
            logging.warning("Advisor LLM reasoning failed: %s. Using deterministic RAG output.", e)
            recs = []
            for p in retrieved_products:
                recs.append({
                    "stock_code": p["stock_code"],
                    "description": p["description"],
                    "category": p.get("category", "General"),
                    "unit_price": p["unit_price"],
                    "similarity": p.get("similarity", 0.85),
                    "why_recommended": f"A top-tier {p.get('category', 'selection')} from {store_profile['store_name']} that fits your query '{query_text}'."
                })
            return {
                "tenant_id": tenant_key,
                "store_name": store_profile["store_name"],
                "user_query": query_text,
                "budget_applied": budget_max,
                "intro_message": f"Welcome to {store_profile['store_name']}! Here are top options matching your search for \"{query_text}\":",
                "recommendations": recs,
                "shopping_tip": "Check out our newest arrivals updated weekly!"
            }
