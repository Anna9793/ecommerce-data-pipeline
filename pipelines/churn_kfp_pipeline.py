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
def evaluate_deploy_comp(
    model_input: Input[Model],
    bucket_name: str,
    project_id: str
):
    import os
    import json
    import logging
    from google.cloud import storage
    
    logging.basicConfig(level=logging.INFO)
    run_id = model_input.metadata["run_id"]
    candidate_f1 = model_input.metadata["f1_score"]
    model_version = model_input.metadata["model_version"]
    
    logging.info("Evaluating candidate model: Version %s, F1-Score %s", model_version, candidate_f1)
    
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
        
        # Upload candidate model direct from KFP download path
        model_blob = bucket.blob("models/customer_churn_model.pkl")
        model_blob.upload_from_filename(model_input.path)
        
        # Write new metadata
        new_metadata = {
            "f1_score": candidate_f1,
            "model_version": model_version,
            "run_id": run_id
        }
        metadata_blob.upload_from_string(json.dumps(new_metadata, indent=2))
        logging.info("Deployment successful.")
    else:
        logging.info("Candidate performance was lower than active model. Deployment skipped.")

@dsl.pipeline(
    name="churn-prediction-retraining-pipeline",
    description="An end-to-end KFP pipeline for feature extraction, model retraining, and gated deployment."
)
def churn_retraining_pipeline(
    project_id: str = "anna-ml-pipeline",
    bucket_name: str = "anna-ml-pipeline-bucket"
):
    extract_task = extract_data_comp(project_id=project_id)
    
    train_task = train_churn_comp(dataset_input=extract_task.outputs["dataset_output"])
    
    evaluate_deploy_comp(
        model_input=train_task.outputs["model_output"],
        bucket_name=bucket_name,
        project_id=project_id
    )

if __name__ == "__main__":
    compiler.Compiler().compile(
        pipeline_func=churn_retraining_pipeline,
        package_path="churn_pipeline.yaml"
    )
