from pydantic import BaseModel, Field
from typing import Optional

class PredictionRequest(BaseModel):
    customer_id: Optional[int] = None
    recency: float = Field(..., ge=0)
    frequency: float = Field(..., ge=0)
    avg_order_value: float = Field(..., ge=0)

class ChurnPredictionRequest(BaseModel):
    customer_id: Optional[str] = None
    recency: float = Field(..., ge=0)
    frequency: float = Field(..., ge=0)
    avg_order_value: float = Field(..., ge=0)
    spending_velocity: float = Field(..., ge=0)
    cancellation_rate: float = Field(..., ge=0)
    preferred_shopping_hour: int = Field(..., ge=0)

class ChurnPredictionResponse(BaseModel):
    customer_id: Optional[str] = None
    churn_probability: float
    is_churn: int
