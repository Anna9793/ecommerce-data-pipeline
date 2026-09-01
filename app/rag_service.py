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

    def search_products(self, query_text: str, budget_max: Optional[float] = None, top_k: int = 4) -> list:
        """Retrieves top products from PostgreSQL pgvector table with optional budget filtering."""
        query_vector = self.get_query_embedding(query_text)
        results = search_product_catalog_pgvector(query_vector, budget_max=budget_max, top_k=top_k)
        
        if not results:
            logging.info("pgvector returned empty or offline. Using catalog fallback.")
            # Safe default fallback products
            fallback_catalog = [
                {"stock_code": "85123A", "description": "WHITE HANGING HEART T-LIGHT HOLDER", "category": "Home Decor & Lighting", "unit_price": 2.55, "similarity": 0.88},
                {"stock_code": "22423", "description": "REGENCY CAKESTAND 3 TIER", "category": "Kitchen & Dining", "unit_price": 12.75, "similarity": 0.82},
                {"stock_code": "47566", "description": "PARTY BUNTING", "category": "Party & Celebration", "unit_price": 4.95, "similarity": 0.76},
                {"stock_code": "22086", "description": "PAPER CHAIN KIT 50'S CHRISTMAS", "category": "Holiday & Seasonal", "unit_price": 2.95, "similarity": 0.74}
            ]
            if budget_max:
                fallback_catalog = [p for p in fallback_catalog if p["unit_price"] <= budget_max]
            results = fallback_catalog[:top_k]
            
        return results

    def advise(self, query_text: str, budget_max: Optional[float] = None, top_k: int = 4) -> dict:
        """
        Executes full RAG workflow:
        1. Retrieval: Vector search in pgvector + budget filter
        2. Reasoning: Gemini generates personalized justifications per product
        """
        retrieved_products = self.search_products(query_text, budget_max=budget_max, top_k=top_k)
        
        products_context = ""
        for idx, p in enumerate(retrieved_products, 1):
            products_context += f"{idx}. Code: {p['stock_code']} | '{p['description']}' | Category: {p['category']} | Price: ${p['unit_price']:.2f} | Match Score: {p['similarity']*100:.1f}%\n"

        prompt = f"""
        You are an expert E-Commerce Personal Shopper & Product Advisor Agent for a specialized boutique gift, home decor, and lifestyle shop.
        
        STORE SPECIALTY & DEPARTMENTS:
        - Home Decor & Lighting (lanterns, candles, wall signs, clocks, vases)
        - Kitchen & Dining (vintage tea sets, mugs, cake stands, tableware)
        - Holiday & Seasonal (Christmas ornaments, festive decor, seasonal gifts)
        - Storage & Accessories (tote bags, trinket boxes, cases)
        - Party & Celebration (bunting, banners, party supplies)
        - Kids & Toys (playful novelty items, plush toys, puzzles)
        
        CUSTOMER SEARCH QUERY: "{query_text}"
        APPLIED BUDGET LIMIT: {f"${budget_max:.2f}" if budget_max else "None specified"}
        
        RETRIEVED CATALOG CANDIDATES (Retrieved via pgvector cosine similarity):
        {products_context}
        
        INSTRUCTIONS:
        1. Write a friendly, engaging intro message. If the customer asked for items our store does not sell (e.g. sports equipment, electronics, power tools, car parts), politely clarify our store's specialty and mention that you found the closest related novelty or lifestyle alternatives.
        2. For each retrieved product, provide a specific justification ('why_recommended') explaining why it is a charming choice or alternative.
        3. Provide an insightful 'shopping_tip' (e.g. gift presentation, pairing suggestions, or seasonal styling).
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
                "user_query": query_text,
                "budget_applied": budget_max,
                "intro_message": data.get("intro_message", "Here are the top handpicked items that match your style:"),
                "recommendations": recs if recs else retrieved_products,
                "shopping_tip": data.get("shopping_tip", "Pair with coordinating accessories for a cohesive look!")
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
                    "similarity": p.get("similarity", 0.80),
                    "why_recommended": f"A wonderful {p.get('category', 'item')} selection that matches your search for '{query_text}' perfectly."
                })
            return {
                "user_query": query_text,
                "budget_applied": budget_max,
                "intro_message": f"I found these fantastic options matching your search for \"{query_text}\":",
                "recommendations": recs,
                "shopping_tip": "Check back often as new seasonal additions arrive weekly!"
            }
