from pydantic import BaseModel
from typing import Dict

class PredictionRequest(BaseModel):
    customer_id: int
    features: Dict[str, float]

class PredictionResponse(BaseModel):
    customer_id: int
    cluster: int