from pydantic import BaseModel, Field
from typing import Optional, List

class PredictionRequest(BaseModel):
    customer_id: Optional[str] = None
    recency: Optional[float] = None
    frequency: Optional[float] = None
    avg_order_value: Optional[float] = None

class ChurnPredictionRequest(BaseModel):
    customer_id: Optional[str] = None
    recency: Optional[float] = None
    frequency: Optional[float] = None
    avg_order_value: Optional[float] = None
    spending_velocity: Optional[float] = None
    cancellation_rate: Optional[float] = None
    preferred_shopping_hour: Optional[int] = None

class ChurnPredictionResponse(BaseModel):
    customer_id: Optional[str] = None
    churn_probability: float
    is_churn: int

class ProductAdvisorRequest(BaseModel):
    query: str = Field(..., description="Customer natural language search query")
    budget_max: Optional[float] = Field(None, description="Optional maximum price filter")
    top_k: Optional[int] = Field(4, description="Number of products to retrieve")
    tenant_id: Optional[str] = Field("giftshop_uk", description="Tenant / Store identifier (giftshop_uk, nordic_tech, olist)")

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

class TwoTowerRecommendationRequest(BaseModel):
    customer_id: Optional[str] = Field(None, description="Customer ID for feature lookup")
    recency: Optional[float] = None
    frequency: Optional[float] = None
    avg_order_value: Optional[float] = None
    spending_velocity: Optional[float] = None
    cancellation_rate: Optional[float] = None
    preferred_shopping_hour: Optional[int] = None
    top_k: Optional[int] = Field(4, description="Number of recommendations to return")

