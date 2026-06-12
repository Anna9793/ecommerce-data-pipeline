import pandas as pd
import mlflow
import mlflow.sklearn
from mlflow import MlflowClient
from config.paths import CLUSTER_PROFILE

mlflow.set_tracking_uri("file:./mlruns")

PROD_MODEL_URI = "models:/customer_segmentation_model@production"

prod_pipeline = mlflow.sklearn.load_model(PROD_MODEL_URI)

client = MlflowClient()

version_info = client.get_model_version_by_alias(
    "customer_segmentation_model",
    "production"
)

MODEL_VERSION = version_info.version

profile_df = pd.read_csv(CLUSTER_PROFILE)

cluster_labels = dict(
    zip(
        profile_df["cluster"],
        profile_df["segment"]
    )
)

def predict_cluster(features: dict):

    df = pd.DataFrame([features])

    cluster = int(prod_pipeline.predict(df)[0])
    label = cluster_labels.get(cluster, "Unknown")
    
    return cluster, label
