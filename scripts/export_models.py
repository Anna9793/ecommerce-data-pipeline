"""
Uploads trained model artifacts from models/ to Google Cloud Storage (GCS).
"""
import os
import sys
import logging
from pathlib import Path
from google.cloud import storage

# Ensure project root is in Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config.paths import MODELS_DIR, SEGMENTATION_MODEL_PKL, CHURN_MODEL_PKL

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def export_and_upload():
    """Uploads local model .pkl artifacts to the project's GCS bucket."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    models_to_upload = [
        ("customer_segmentation_model.pkl", SEGMENTATION_MODEL_PKL),
        ("customer_churn_model.pkl", CHURN_MODEL_PKL)
    ]
    
    uploaded_files = []
    for model_name, path in models_to_upload:
        if path.exists():
            uploaded_files.append((model_name, str(path)))
            logging.info("Found local model artifact: %s", path)
        else:
            logging.warning("Model file not found at %s. Skipping.", path)

    if uploaded_files:
        bucket_name = os.getenv("GCS_BUCKET", "anna-ml-pipeline-bucket")
        project_id = os.getenv("GCP_PROJECT", "anna-ml-pipeline")
        logging.info("Connecting to GCS bucket: gs://%s (project=%s)", bucket_name, project_id)
        storage_client = storage.Client(project=project_id)
        bucket = storage_client.bucket(bucket_name)
        
        for model_name, local_path in uploaded_files:
            blob = bucket.blob(f"models/{model_name}")
            logging.info("Uploading %s to gs://%s/models/%s...", local_path, bucket_name, model_name)
            blob.upload_from_filename(local_path)
            
        logging.info("✅ All models successfully uploaded to Google Cloud Storage!")
    else:
        logging.warning("No model files found to upload in %s", MODELS_DIR)


if __name__ == "__main__":
    export_and_upload()
