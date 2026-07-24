import os
from kfp import dsl
from kfp import compiler
from kfp.dsl import component, Output, Input, Dataset, Model, Metrics

@component(base_image="us-central1-docker.pkg.dev/anna-ml-pipeline/ecommerce-ml-pipeline/pipeline-image:latest")
def extract_data_comp(
    project_id: str,
    dataset_output: Output[Dataset]
):
    import os
    import pandas as pd
    import logging
    from src.transformation import load_clean_data, transform_data
    
    logging.basicConfig(level=logging.INFO)
    os.environ["USE_BIGQUERY"] = "true"
    os.environ["GCP_PROJECT"] = project_id
    
    logging.info("Extracting and transforming dataset from BigQuery...")
    df = load_clean_data("data/processed/clean_retail.csv")
    df = transform_data(df)
    
    # Write to KFP outputs
    df.to_csv(dataset_output.path, index=False)
    logging.info("Data extraction successfully complete.")

@component(base_image="us-central1-docker.pkg.dev/anna-ml-pipeline/ecommerce-ml-pipeline/pipeline-image:latest")
def train_churn_comp(
    dataset_input: Input[Dataset],
    model_output: Output[Model],
    metrics_output: Output[Metrics]
):
    import os
    import pandas as pd
    import logging
    import mlflow.sklearn
    import joblib
    from src.churn_training import run_churn_training
    
    logging.basicConfig(level=logging.INFO)
    os.environ["MLFLOW_TRACKING_URI"] = "file:./mlruns"
    
    logging.info("Loading KFP Dataset artifact...")
    df = pd.read_csv(dataset_input.path)
    
    logging.info("Running model retraining...")
    results = run_churn_training(df=df)
    
    # Track metrics in KFP dashboard UI
    metrics_output.log_metric("f1_score", results["f1_score"])
    metrics_output.log_metric("model_version", results["model_version"])
    
    # Save model artifact to KFP output path
    logging.info("Saving trained model to KFP artifact path...")
    candidate_model = mlflow.sklearn.load_model(f"runs:/{results['run_id']}/model")
    joblib.dump(candidate_model, model_output.path)
    
    # Pass metadata parameters down the DAG line
    model_output.metadata["run_id"] = results["run_id"]
    model_output.metadata["f1_score"] = results["f1_score"]
    model_output.metadata["model_version"] = results["model_version"]
    logging.info("Model training successfully complete.")

@component(base_image="us-central1-docker.pkg.dev/anna-ml-pipeline/ecommerce-ml-pipeline/pipeline-image:latest")
def train_segmentation_comp(
    dataset_input: Input[Dataset],
    model_output: Output[Model]
):
    import os
    import pandas as pd
    import logging
    import mlflow.sklearn
    import joblib
    from src.rfm_features import compute_rfm
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from src.features.transformers import LogTransformer
    from src.features.selection import ColumnSelector
    from src.features.to_numpy import ToNumpy
    
    logging.basicConfig(level=logging.INFO)
    os.environ["MLFLOW_TRACKING_URI"] = "file:./mlruns"
    
    logging.info("Loading KFP Dataset artifact...")
    df = pd.read_csv(dataset_input.path)
    
    logging.info("Computing RFM features...")
    rfm_df = compute_rfm(df)
    
    logging.info("Training KMeans customer segmentation model...")
    feature_cols = ["recency", "frequency", "avg_order_value"]
    
    pipeline = Pipeline([
        ("selector", ColumnSelector(columns=feature_cols)),
        ("log", LogTransformer(columns=["frequency", "avg_order_value"])),
        ("to_numpy", ToNumpy()),
        ("scaler", StandardScaler()),
        ("model", KMeans(n_clusters=4, random_state=42, n_init=10))
    ])
    
    pipeline.fit(rfm_df)
    
    import mlflow
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("customer_segmentation")
    with mlflow.start_run() as run:
        mlflow.log_params({"n_clusters": 4, "random_state": 42})
        mlflow.sklearn.log_model(sk_model=pipeline, artifact_path="model")
        
        logging.info("Saving trained segmentation model to KFP artifact path...")
        joblib.dump(pipeline, model_output.path)
        model_output.metadata["run_id"] = run.info.run_id
        logging.info("Segmentation model training complete.")

@component(base_image="us-central1-docker.pkg.dev/anna-ml-pipeline/ecommerce-ml-pipeline/pipeline-image:latest")
def evaluate_deploy_comp(
    churn_model_input: Input[Model],
    seg_model_input: Input[Model],
    bucket_name: str,
    project_id: str,
    api_url: str
):
    import os
    import json
    import logging
    from google.cloud import storage
    
    logging.basicConfig(level=logging.INFO)
    run_id = churn_model_input.metadata["run_id"]
    candidate_f1 = churn_model_input.metadata["f1_score"]
    model_version = churn_model_input.metadata["model_version"]
    
    logging.info("Evaluating candidate churn model: Version %s, F1-Score %s", model_version, candidate_f1)
    
    # Download current production model details from GCS
    storage_client = storage.Client(project=project_id)
    bucket = storage_client.bucket(bucket_name)
    metadata_blob = bucket.blob("models/model_metadata.json")
    
    active_f1 = 0.0
    if metadata_blob.exists():
        try:
            metadata_str = metadata_blob.download_as_string()
            metadata = json.loads(metadata_str)
            active_f1 = metadata.get("f1_score", 0.0)
            logging.info("Active production model F1-Score: %s", active_f1)
        except Exception as e:
            logging.warning("Error reading active model metadata: %s. Defaulting to 0.0.", e)
            
    # Evaluation gate
    if candidate_f1 >= active_f1:
        logging.info("Candidate model is better or equal. Deploying to GCS production path...")
        
        # Upload candidate models direct from KFP download paths
        model_blob = bucket.blob("models/customer_churn_model.pkl")
        model_blob.upload_from_filename(churn_model_input.path)
        
        seg_model_blob = bucket.blob("models/customer_segmentation_model.pkl")
        seg_model_blob.upload_from_filename(seg_model_input.path)
        
        # Write new metadata
        new_metadata = {
            "f1_score": candidate_f1,
            "model_version": model_version,
            "run_id": run_id
        }
        metadata_blob.upload_from_string(json.dumps(new_metadata, indent=2))
        logging.info("Deployment successful.")
        
        # Trigger dynamic reload of models on the live API service
        if api_url and api_url.startswith("http"):
            import requests
            try:
                logging.info("Triggering dynamic model reload on live API: %s", api_url)
                reload_url = f"{api_url.rstrip('/')}/reload-models"
                response = requests.post(reload_url, timeout=15)
                logging.info("Reload response: %s - %s", response.status_code, response.text)
            except Exception as e:
                logging.warning("Failed to trigger model reload on live API service: %s", e)
    else:
        logging.info("Candidate performance was lower than active model. Deployment skipped.")

@dsl.pipeline(
    name="churn-prediction-retraining-pipeline",
    description="An end-to-end KFP pipeline for feature extraction, model retraining, and gated deployment."
)
def churn_retraining_pipeline(
    project_id: str = "anna-ml-pipeline",
    bucket_name: str = "anna-ml-pipeline-bucket",
    api_url: str = "http://localhost:8000"
):
    extract_task = extract_data_comp(project_id=project_id)
    
    train_task = train_churn_comp(dataset_input=extract_task.outputs["dataset_output"])
    train_seg_task = train_segmentation_comp(dataset_input=extract_task.outputs["dataset_output"])
    
    evaluate_deploy_comp(
        churn_model_input=train_task.outputs["model_output"],
        seg_model_input=train_seg_task.outputs["model_output"],
        bucket_name=bucket_name,
        project_id=project_id,
        api_url=api_url
    )

if __name__ == "__main__":
    compiler.Compiler().compile(
        pipeline_func=churn_retraining_pipeline,
        package_path="churn_pipeline.yaml"
    )
