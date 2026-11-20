import os
import logging
from datetime import datetime
from typing import Optional, Union, Any
from pydantic import BaseModel
import joblib
import pandas as pd
from config.paths import (
    SEGMENTATION_MODEL_PKL,
    CHURN_MODEL_PKL,
    EXPERIMENT_CONFIG_PATH,
)
from config.loader import load_config

logger = logging.getLogger(__name__)

config = load_config(EXPERIMENT_CONFIG_PATH)

CLUSTERING_FEATURE_VERSION = os.getenv("CLUSTERING_FEATURE_VERSION", os.getenv("RFM_FEATURE_VERSION", config.clustering.feature_version))
RFM_FEATURE_VERSION = CLUSTERING_FEATURE_VERSION
CHURN_FEATURE_VERSION = os.getenv("CHURN_FEATURE_VERSION", config.churn.feature_version)
CHURN_DECISION_THRESHOLD = float(os.getenv("CHURN_DECISION_THRESHOLD", config.churn.classification_threshold))

def _load_model_from_gcs(model_name: str, force_download: bool = False):
    from google.cloud import storage

    project_id = os.getenv("GCP_PROJECT", "anna-ml-pipeline")
    bucket_name = os.getenv("GCS_BUCKET", "anna-ml-pipeline-bucket")
    local_path = f"/tmp/{model_name}"

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

def _get_local_model_version(path) -> str:
    """Computes a dynamic local model version based on file modification timestamp."""
    try:
        mtime = datetime.fromtimestamp(os.path.getmtime(path)).strftime("%Y%m%d")
        return f"local_{mtime}"
    except Exception:
        return "local_v1"


def _load_all_models(force_download=False):
    if os.getenv("USE_BIGQUERY", "false").lower() == "true":
        logger.info("Cloud mode active: loading models dynamically from GCS")
        seg_pipeline, seg_version = _load_model_from_gcs("customer_segmentation_model.pkl", force_download)
        churn_pipeline, churn_version = _load_model_from_gcs("customer_churn_model.pkl", force_download)
    else:
        # Local mode: load directly from local artifacts with dynamic file timestamp version
        logger.info("Local mode active: loading models from local filesystem (%s, %s)", SEGMENTATION_MODEL_PKL, CHURN_MODEL_PKL)
        seg_pipeline = joblib.load(SEGMENTATION_MODEL_PKL)
        churn_pipeline = joblib.load(CHURN_MODEL_PKL)
        seg_version = _get_local_model_version(SEGMENTATION_MODEL_PKL)
        churn_version = _get_local_model_version(CHURN_MODEL_PKL)

    # Enforce strict model contract
    if not hasattr(seg_pipeline, "cluster_labels_") or not isinstance(seg_pipeline.cluster_labels_, dict):
        raise ValueError(
            "Model Contract Violation: El modelo de segmentación cargado no contiene "
            "el atributo obligatorio 'cluster_labels_'."
        )

    return seg_pipeline, churn_pipeline, seg_version, churn_version

# Initial load during startup
prod_clustering_pipeline, prod_churn_pipeline, CLUSTERING_MODEL_VERSION, CHURN_MODEL_VERSION = _load_all_models(force_download=False)

# Backward compatibility aliases
prod_pipeline = prod_clustering_pipeline
MODEL_VERSION = CLUSTERING_MODEL_VERSION

def reload_production_models():
    """Forces reloading of models from GCS/local storage by bypassing caches."""
    global prod_clustering_pipeline, prod_churn_pipeline, CLUSTERING_MODEL_VERSION, CHURN_MODEL_VERSION, prod_pipeline, MODEL_VERSION
    logger.info("Request received to reload production models.")
    prod_clustering_pipeline, prod_churn_pipeline, CLUSTERING_MODEL_VERSION, CHURN_MODEL_VERSION = _load_all_models(force_download=True)
    prod_pipeline = prod_clustering_pipeline
    MODEL_VERSION = CLUSTERING_MODEL_VERSION
    logger.info("Production models reloaded successfully. Churn model version: %s", CHURN_MODEL_VERSION)


def predict_clustering_service(features: Any):
    features_dict = features.model_dump() if isinstance(features, BaseModel) else features
    df = pd.DataFrame([features_dict])
    cluster = int(prod_clustering_pipeline.predict(df)[0])
    label = prod_clustering_pipeline.cluster_labels_.get(cluster, "Unknown")
    return cluster, label

# Backward compatibility alias
predict_cluster = predict_clustering_service


def predict_churn_service(features: Any, threshold: Optional[float] = None):
    applied_threshold = threshold if threshold is not None else CHURN_DECISION_THRESHOLD
    features_dict = features.model_dump() if isinstance(features, BaseModel) else features
    df = pd.DataFrame([features_dict])

    churn_probability = float(prod_churn_pipeline.predict_proba(df)[0][1])
    is_churn = int(churn_probability >= applied_threshold)

    return is_churn, churn_probability, applied_threshold

