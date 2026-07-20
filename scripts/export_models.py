import os
import mlflow.sklearn
from mlflow import MlflowClient
from google.cloud import storage
import joblib

# Configure local MLflow
mlflow.set_tracking_uri("file:./mlruns")

def export_and_upload():
    # Create a local models folder if not exists
    os.makedirs("models", exist_ok=True)
    
    client = MlflowClient()
    uploaded_files = []
    
    # 1. Export segmentation model
    print("Loading segmentation model...")
    try:
        seg_model = mlflow.sklearn.load_model("models:/customer_segmentation_model@production")
        seg_path = "models/customer_segmentation_model.pkl"
        joblib.dump(seg_model, seg_path)
        print(f"Saved segmentation model to {seg_path}")
        uploaded_files.append(("customer_segmentation_model.pkl", seg_path))
    except Exception as e:
        print(f"Could not load customer_segmentation_model from registry: {e}. Skipping segmentation model export.")
    
    # 2. Export churn model
    print("Loading churn model...")
    try:
        churn_model = mlflow.sklearn.load_model("models:/customer_churn_model@production")
        churn_path = "models/customer_churn_model.pkl"
        joblib.dump(churn_model, churn_path)
        print(f"Saved churn model to {churn_path}")
        uploaded_files.append(("customer_churn_model.pkl", churn_path))
    except Exception as e:
        print(f"Could not load customer_churn_model from registry: {e}. Cannot export churn model.")
    
    # 3. Upload to GCS
    if uploaded_files:
        bucket_name = "anna-ml-pipeline-bucket"
        storage_client = storage.Client(project="anna-ml-pipeline")
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
