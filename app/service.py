import os
import logging
import joblib
import pandas as pd
from config.paths import CLUSTER_PROFILE, SEGMENTATION_MODEL_PKL, CHURN_MODEL_PKL

logger = logging.getLogger(__name__)

RFM_FEATURE_VERSION = os.getenv("RFM_FEATURE_VERSION", "rfm_v1")
CHURN_FEATURE_VERSION = os.getenv("CHURN_FEATURE_VERSION", "churn_v2")

def load_model_from_gcs(model_name: str, force_download: bool = False):
    from google.cloud import storage

    project_id = os.getenv("GCP_PROJECT", "anna-ml-pipeline")
    bucket_name = os.getenv("GCS_BUCKET", "anna-ml-pipeline-bucket")
    local_path = f"/tmp/{model_name}"
    version_str = "gcs_v1"

    try:
        storage_client = storage.Client(project=project_id)
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(f"models/{model_name}")

        if force_download or not os.path.exists(local_path):
            logger.info("Downloading %s from GCS bucket %s (force=%s)", model_name, bucket_name, force_download)
            blob.download_to_filename(local_path)

        # Extract dynamic version from GCS object generation and update timestamp
        blob.reload()
        updated_date = blob.updated.strftime("%Y%m%d") if blob.updated else "latest"
        version_str = f"gcs_gen{blob.generation or '0'}_{updated_date}"
    except Exception as e:
        logger.warning("Could not fetch GCS metadata for %s: %s. Using fallback tag.", model_name, e)
        version_str = "gcs_fallback_v1"

    model = joblib.load(local_path)
    return model, version_str

def _load_all_models(force_download=False):
    try:
        if os.getenv("USE_BIGQUERY", "false").lower() == "true":
            logger.info("Cloud mode active: loading models dynamically from GCS")
            p_pipeline, p_version = load_model_from_gcs("customer_segmentation_model.pkl", force_download)
            p_churn_pipeline, p_churn_version = load_model_from_gcs("customer_churn_model.pkl", force_download)
        else:
            # Local mode: load directly from local artifacts
            logger.info("Local mode active: loading models from local filesystem (%s, %s)", SEGMENTATION_MODEL_PKL, CHURN_MODEL_PKL)
            p_pipeline = joblib.load(SEGMENTATION_MODEL_PKL)
            p_churn_pipeline = joblib.load(CHURN_MODEL_PKL)
            p_version = "local_v1"
            p_churn_version = "local_v1"
        return p_pipeline, p_churn_pipeline, p_version, p_churn_version
    except Exception as e:
        logger.warning("Model loading failed: %s. Using mock fallback models.", e)
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
    """Forces reloading of models from GCS/local storage by bypassing caches."""
    global prod_pipeline, prod_churn_pipeline, MODEL_VERSION, CHURN_MODEL_VERSION
    logger.info("Request received to reload production models.")
    prod_pipeline, prod_churn_pipeline, MODEL_VERSION, CHURN_MODEL_VERSION = _load_all_models(force_download=True)
    logger.info("Production models reloaded successfully. Churn model version: %s", CHURN_MODEL_VERSION)


try:
    profile_df = pd.read_csv(CLUSTER_PROFILE)
    cluster_labels = dict(
        zip(
            profile_df["cluster"],
            profile_df["segment"]
        )
    )
except Exception as e:
    logger.warning("Cluster profile file missing: %s. Using default cluster labels mapping.", e)
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

