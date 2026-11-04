from pydantic import BaseModel
from typing import List


class ClusteringConfig(BaseModel):
    features: List[str]
    cluster_range: List[int]
    random_state: int


class MLflowConfig(BaseModel):
    experiment_name: str


class ExperimentConfig(BaseModel):
    clustering: ClusteringConfig
    mlflow: MLflowConfig
