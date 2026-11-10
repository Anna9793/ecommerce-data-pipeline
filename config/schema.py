from pydantic import BaseModel
from typing import List, Optional


class ClusteringConfig(BaseModel):
    features: List[str]
    cluster_range: List[int]
    random_state: int


class TrackingConfig(BaseModel):
    experiment_name: str


# MLflowConfig alias for backwards compatibility
MLflowConfig = TrackingConfig


class ExperimentConfig(BaseModel):
    clustering: ClusteringConfig
    tracking: Optional[TrackingConfig] = None
    mlflow: Optional[MLflowConfig] = None

    def get_tracking(self) -> TrackingConfig:
        return self.tracking or self.mlflow or TrackingConfig(experiment_name="customer_segmentation")
