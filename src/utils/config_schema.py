from pydantic import BaseModel
from typing import List

class ClusteringConfig(BaseModel):
    cluster_range: List[int]
    random_state: int

class MLflowConfig(BaseModel):
    experiment_name: str

class FeatureConfig(BaseModel):
    columns: List[str]

class ExperimentConfig(BaseModel):
    clustering: ClusteringConfig
    mlflow: MLflowConfig
    features: FeatureConfig