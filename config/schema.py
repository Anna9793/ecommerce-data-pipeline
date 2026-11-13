from pydantic import BaseModel, Field
from typing import List


class ClusteringConfig(BaseModel):
    experiment_name: str = "customer_segmentation"
    features: List[str] = ["recency", "frequency", "avg_order_value"]
    cluster_range: List[int] = [4]
    random_state: int = 42
    test_size: float = 0.2


class ChurnConfig(BaseModel):
    experiment_name: str = "customer_churn"
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


class ExperimentConfig(BaseModel):
    clustering: ClusteringConfig = Field(default_factory=ClusteringConfig)
    churn: ChurnConfig = Field(default_factory=ChurnConfig)


