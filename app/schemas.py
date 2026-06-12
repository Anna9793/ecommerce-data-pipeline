from pydantic import BaseModel, Field
from typing import Optional


class PredictionRequest(BaseModel):
    customer_id: Optional[int] = None
    recency: float = Field(...,ge=0)
    frequency: float = Field(...,ge=0)
    avg_order_value: float = Field(...,ge=0)

