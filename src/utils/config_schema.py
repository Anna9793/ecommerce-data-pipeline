from pydantic import BaseModel
from typing import List, Dict, Optional

class ClusteringConfig(BaseModel):
    cluster_range: List[int]
    random_state: int

class MLflowConfig(BaseModel):
    experiment_name: str

class SingleExperimentConfig(BaseModel):
    features: List[str]
    scaler: Optional[str] = None

class ExperimentConfig(BaseModel):
    clustering: ClusteringConfig
    mlflow: MLflowConfig
    experiments: Dict[str, SingleExperimentConfig]

