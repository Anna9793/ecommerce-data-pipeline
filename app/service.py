import os
import pandas as pd
import mlflow
import mlflow.sklearn
from mlflow import MlflowClient
from config.paths import CLUSTER_PROFILE

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns"))

PROD_MODEL_URI = "models:/customer_segmentation_model@production"
PROD_CHURN_MODEL_URI = "models:/customer_churn_model@production"

def load_model_from_gcs(model_name):
    from google.cloud import storage
    import joblib
    
    project_id = os.getenv("GCP_PROJECT", "anna-ml-pipeline")
    bucket_name = os.getenv("GCS_BUCKET", "anna-ml-pipeline-bucket")
    local_path = f"/tmp/{model_name}"
    
    if not os.path.exists(local_path):
        logging.info("Downloading %s from GCS bucket %s", model_name, bucket_name)
        storage_client = storage.Client(project=project_id)
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(f"models/{model_name}")
        blob.download_to_filename(local_path)
        
    return joblib.load(local_path)

if os.getenv("USE_BIGQUERY", "false").lower() == "true":
    logging.info("Cloud mode active: loading models directly from GCS")
    prod_pipeline = load_model_from_gcs("customer_segmentation_model.pkl")
    MODEL_VERSION = "gcs_v50"
    prod_churn_pipeline = load_model_from_gcs("customer_churn_model.pkl")
    CHURN_MODEL_VERSION = "gcs_v4"
else:
    # Local mode: load from MLflow
    prod_pipeline = mlflow.sklearn.load_model(PROD_MODEL_URI)
    client = MlflowClient()
    version_info = client.get_model_version_by_alias(
        "customer_segmentation_model",
        "production"
    )
    MODEL_VERSION = version_info.version

    prod_churn_pipeline = mlflow.sklearn.load_model(PROD_CHURN_MODEL_URI)
    churn_version_info = client.get_model_version_by_alias(
        "customer_churn_model",
        "production"
    )
    CHURN_MODEL_VERSION = churn_version_info.version


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

def predict_churn_service(features: dict):

    df = pd.DataFrame([features])

    is_churn = int(prod_churn_pipeline.predict(df)[0])
    churn_probability = float(prod_churn_pipeline.predict_proba(df)[0][1])

    return is_churn, churn_probability

