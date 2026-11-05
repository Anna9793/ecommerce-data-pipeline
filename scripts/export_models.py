import os
import sys
from pathlib import Path
import mlflow.sklearn
from mlflow import MlflowClient
from google.cloud import storage
import joblib

# Ensure project root is in Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config.paths import MODELS_DIR, SEGMENTATION_MODEL_PKL, CHURN_MODEL_PKL

# Configure local MLflow
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns"))


def export_and_upload():
    # Ensure local models folder exists
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    client = MlflowClient()
    uploaded_files = []
    
    # 1. Export segmentation model
    print("Loading segmentation model...")
    try:
        seg_model = mlflow.sklearn.load_model("models:/customer_segmentation_model@production")
        joblib.dump(seg_model, SEGMENTATION_MODEL_PKL)
        print(f"Saved segmentation model to {SEGMENTATION_MODEL_PKL}")
        uploaded_files.append((SEGMENTATION_MODEL_PKL.name, str(SEGMENTATION_MODEL_PKL)))
    except Exception as e:
        print(f"Could not load customer_segmentation_model from registry: {e}. Skipping segmentation model export.")
    
    # 2. Export churn model
    print("Loading churn model...")
    try:
        churn_model = mlflow.sklearn.load_model("models:/customer_churn_model@production")
        joblib.dump(churn_model, CHURN_MODEL_PKL)
        print(f"Saved churn model to {CHURN_MODEL_PKL}")
        uploaded_files.append((CHURN_MODEL_PKL.name, str(CHURN_MODEL_PKL)))
    except Exception as e:
        print(f"Could not load customer_churn_model from registry: {e}. Cannot export churn model.")
    
    # 3. Upload to GCS
    if uploaded_files:
        bucket_name = os.getenv("GCS_BUCKET", "anna-ml-pipeline-bucket")
        project_id = os.getenv("GCP_PROJECT", "anna-ml-pipeline")
        storage_client = storage.Client(project=project_id)
        bucket = storage_client.bucket(bucket_name)
        
        for model_name, local_path in uploaded_files:
            blob = bucket.blob(f"models/{model_name}")
            print(f"Uploading {local_path} to gs://{bucket_name}/models/{model_name}...")
            blob.upload_from_filename(local_path)
            
        print("Model export and upload complete!")
    else:
        print("No models were exported. Nothing to upload.")


if __name__ == "__main__":
    export_and_upload()
