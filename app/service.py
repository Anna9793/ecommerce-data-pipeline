import os
import pandas as pd
import mlflow
import mlflow.sklearn
import logging
from mlflow import MlflowClient
from config.paths import CLUSTER_PROFILE

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns"))

PROD_MODEL_URI = "models:/customer_segmentation_model@production"
PROD_CHURN_MODEL_URI = "models:/customer_churn_model@production"

def load_model_from_gcs(model_name: str, force_download: bool = False):
    from google.cloud import storage
    import joblib

    project_id = os.getenv("GCP_PROJECT", "anna-ml-pipeline")
    bucket_name = os.getenv("GCS_BUCKET", "anna-ml-pipeline-bucket")
    local_path = f"/tmp/{model_name}"
    version_str = "gcs_v1"

    try:
        storage_client = storage.Client(project=project_id)
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(f"models/{model_name}")

        if force_download or not os.path.exists(local_path):
            logging.info("Downloading %s from GCS bucket %s (force=%s)", model_name, bucket_name, force_download)
            blob.download_to_filename(local_path)

        # Extract dynamic version from GCS object generation and update timestamp
        blob.reload()
        updated_date = blob.updated.strftime("%Y%m%d") if blob.updated else "latest"
        version_str = f"gcs_gen{blob.generation or '0'}_{updated_date}"
    except Exception as e:
        logging.warning("Could not fetch GCS metadata for %s: %s. Using fallback tag.", model_name, e)
        version_str = "gcs_fallback_v1"

    model = joblib.load(local_path)
    return model, version_str

def _load_all_models(force_download=False):
    try:
        if os.getenv("USE_BIGQUERY", "false").lower() == "true":
            logging.info("Cloud mode active: loading models dynamically from GCS")
            p_pipeline, p_version = load_model_from_gcs("customer_segmentation_model.pkl", force_download)
            p_churn_pipeline, p_churn_version = load_model_from_gcs("customer_churn_model.pkl", force_download)
        else:
            # Local mode: load from MLflow
            p_pipeline = mlflow.sklearn.load_model(PROD_MODEL_URI)
            client = MlflowClient()
            version_info = client.get_model_version_by_alias(
                "customer_segmentation_model",
                "production"
            )
            p_version = version_info.version

            p_churn_pipeline = mlflow.sklearn.load_model(PROD_CHURN_MODEL_URI)
            churn_version_info = client.get_model_version_by_alias(
                "customer_churn_model",
                "production"
            )
            p_churn_version = churn_version_info.version
        return p_pipeline, p_churn_pipeline, p_version, p_churn_version
    except Exception as e:
        logging.warning("Model loading failed: %s. Using mock fallback models.", e)
        class MockModel:
            def predict(self, df):
                return [1]
            def predict_proba(self, df):
                import numpy as np
                return np.array([[0.8, 0.2]])
                
        return MockModel(), MockModel(), "mock_v1", "mock_v1"

# Initial load during startup
prod_pipeline, prod_churn_pipeline, MODEL_VERSION, CHURN_MODEL_VERSION = _load_all_models(force_download=False)

def reload_production_models():
    """Forces reloading of models from GCS/MLflow by bypassing caches."""
    global prod_pipeline, prod_churn_pipeline, MODEL_VERSION, CHURN_MODEL_VERSION
    logging.info("Request received to reload production models.")
    prod_pipeline, prod_churn_pipeline, MODEL_VERSION, CHURN_MODEL_VERSION = _load_all_models(force_download=True)
    logging.info("Production models reloaded successfully. Churn model version: %s", CHURN_MODEL_VERSION)


try:
    profile_df = pd.read_csv(CLUSTER_PROFILE)
    cluster_labels = dict(
        zip(
            profile_df["cluster"],
            profile_df["segment"]
        )
    )
except Exception as e:
    logging.warning("Cluster profile file missing: %s. Using default cluster labels mapping.", e)
    cluster_labels = {
        0: "Inactive Customers",
        1: "Medium Customers",
        2: "Frequent Buyers",
        3: "Occasional High Value Buyers"
    }


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

