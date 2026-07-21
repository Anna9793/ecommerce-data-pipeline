import os
import logging
from google.cloud import aiplatform

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def submit_vertex_training_job():
    project_id = os.getenv("GCP_PROJECT", "anna-ml-pipeline")
    location = os.getenv("GCP_LOCATION", "us-central1")
    bucket_name = os.getenv("GCS_BUCKET", "anna-ml-pipeline-bucket")
    
    # 1. Compile the pipeline
    logging.info("Compiling Kubeflow pipeline...")
    from pipelines.churn_kfp_pipeline import churn_retraining_pipeline
    from kfp import compiler
    
    pipeline_yaml_path = "churn_pipeline.yaml"
    compiler.Compiler().compile(
        pipeline_func=churn_retraining_pipeline,
        package_path=pipeline_yaml_path
    )
    logging.info("Pipeline compiled successfully to %s", pipeline_yaml_path)
    
    # 2. Initialize Vertex AI
    logging.info("Initializing Vertex AI SDK for project: %s in %s", project_id, location)
    aiplatform.init(project=project_id, location=location, staging_bucket=f"gs://{bucket_name}")
    
    # 3. Create and Run PipelineJob
    logging.info("Submitting PipelineJob to Vertex AI Pipelines...")
    pipeline_job = aiplatform.PipelineJob(
        display_name="churn-prediction-retraining-pipeline",
        template_path=pipeline_yaml_path,
        pipeline_root=f"gs://{bucket_name}/pipeline_root",
        parameter_values={
            "project_id": project_id,
            "bucket_name": bucket_name
        }
    )
    
    pipeline_job.submit()
    logging.info("Vertex AI Pipeline Job submitted successfully! Monitor it in the Google Cloud Console under Vertex AI Pipelines.")
    return pipeline_job.name

if __name__ == "__main__":
    submit_vertex_training_job()

