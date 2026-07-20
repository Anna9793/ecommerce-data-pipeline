import os
import logging
from google.cloud import aiplatform

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def submit_vertex_training_job():
    project_id = os.getenv("GCP_PROJECT", "anna-ml-pipeline")
    location = os.getenv("GCP_LOCATION", "us-central1")
    bucket_name = os.getenv("GCS_BUCKET", "anna-ml-pipeline-bucket")
    
    image_uri = f"{location}-docker.pkg.dev/{project_id}/ecommerce-ml-pipeline/pipeline-image:latest"
    
    logging.info("Initializing Vertex AI SDK for project: %s in %s", project_id, location)
    aiplatform.init(project=project_id, location=location, staging_bucket=f"gs://{bucket_name}")
    
    logging.info("Defining custom container training job using image: %s", image_uri)
    
    # We run training and then immediately run the model exporter to GCS
    job = aiplatform.CustomContainerTrainingJob(
        display_name="churn-prediction-retraining",
        container_uri=image_uri,
        command=["sh", "-c", "export PYTHONPATH=. && python src/churn_training.py && python scripts/export_models.py"],
    )
    
    environment_variables = {
        "GCP_PROJECT": project_id,
        "GCS_BUCKET": bucket_name,
        "MLFLOW_TRACKING_URI": os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns"),
        "USE_BIGQUERY": "true",
    }
    
    logging.info("Submitting custom job to Vertex AI...")
    
    # Submit asynchronously to avoid blocking API requests
    job.run(
        replica_count=1,
        machine_type="n1-standard-4",
        environment_variables=environment_variables,
        sync=False,
    )
    
    logging.info("Job submitted successfully! You can monitor it in the Google Cloud Console under Vertex AI Custom Jobs.")

if __name__ == "__main__":
    submit_vertex_training_job()
