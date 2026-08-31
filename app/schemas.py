from pydantic import BaseModel, Field
from typing import Optional

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
