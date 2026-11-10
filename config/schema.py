from pydantic import BaseModel, Field
from typing import List


class ClusteringConfig(BaseModel):
    features: List[str]
    cluster_range: List[int]
    random_state: int


class TrackingConfig(BaseModel):
    experiment_name: str = "customer_segmentation"


class ExperimentConfig(BaseModel):
    clustering: ClusteringConfig
    tracking: TrackingConfig = Field(default_factory=TrackingConfig)

