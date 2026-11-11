from pydantic import BaseModel, Field
from typing import List


class ClusteringConfig(BaseModel):
    features: List[str] = ["recency", "frequency", "avg_order_value"]
    cluster_range: List[int] = [4]
    random_state: int = 42


class ChurnConfig(BaseModel):
    cutoff_days: int = 90
    test_size: float = 0.2
    random_state: int = 42
    features: List[str] = [
        "recency",
        "frequency",
        "avg_order_value",
        "spending_velocity",
        "cancellation_rate",
        "preferred_shopping_hour",
    ]
    experiment_name: str = "customer_churn"


class TrackingConfig(BaseModel):
    experiment_name: str = "customer_segmentation"


class ExperimentConfig(BaseModel):
    clustering: ClusteringConfig = Field(default_factory=ClusteringConfig)
    tracking: TrackingConfig = Field(default_factory=TrackingConfig)
    churn: ChurnConfig = Field(default_factory=ChurnConfig)


