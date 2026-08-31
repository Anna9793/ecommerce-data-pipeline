import subprocess
import os
import logging
import sys

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def run_cmd(args):
    try:
        result = subprocess.run(args, capture_output=True, text=True, check=True)
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        logging.error("Command failed: %s\nError output: %s", " ".join(args), e.stderr)
        return None

def setup_scheduler():
    project_id = os.getenv("GCP_PROJECT")
    if not project_id:
        # Fallback to gcloud default project
        project_id = run_cmd(["gcloud", "config", "get-value", "project"])
        
    if not project_id or project_id == "(unset)":
        logging.error("GCP project ID could not be determined. Please set the GCP_PROJECT environment variable or run 'gcloud config set project'.")
        sys.exit(1)
        
    location = os.getenv("GCP_LOCATION", "us-central1")
    
    logging.info("Resolving Cloud Run api-service URL in project %s (%s)...", project_id, location)
    api_url = run_cmd([
        "gcloud", "run", "services", "describe", "api-service",
        "--platform", "managed",
        "--region", location,
        "--format", "value(status.url)",
        "--project", project_id
    ])
    
    if not api_url:
        logging.error("Could not resolve Cloud Run api-service URL. Is the service deployed?")
        sys.exit(1)
        
    logging.info("Resolved API Service URL: %s", api_url)
    webhook_url = f"{api_url}/monitoring/check-and-retrain"
    
    # Resolve GCP project number to construct default compute service account
    logging.info("Resolving project number for project: %s...", project_id)
    project_number = run_cmd([
        "gcloud", "projects", "describe", project_id,
        "--format", "value(projectNumber)"
    ])
    
    if not project_number:
        logging.error("Could not resolve project number.")
        sys.exit(1)
        
    # Default Compute Engine service account has the necessary permissions to invoke Cloud Run
    service_account = f"{project_number}-compute@developer.gserviceaccount.com"
    logging.info("Using service account for authentication: %s", service_account)
    
    job_name = "automated-drift-retrain-job"
    schedule = "0 0 * * 0"  # Every Sunday at midnight
    
    logging.info("Checking if Cloud Scheduler job %s already exists...", job_name)
    # Check if job exists
    existing_job = run_cmd([
        "gcloud", "scheduler", "jobs", "describe", job_name,
        "--location", location,
        "--project", project_id
    ])
    
    if existing_job:
        logging.info("Job already exists. Updating existing Cloud Scheduler job...")
        cmd = [
            "gcloud", "scheduler", "jobs", "update", "http", job_name,
            "--schedule", schedule,
            "--uri", webhook_url,
            "--http-method", "POST",
            "--oidc-service-account-email", service_account,
            "--location", location,
            "--project", project_id
        ]
    else:
        logging.info("Creating new Cloud Scheduler job...")
        cmd = [
            "gcloud", "scheduler", "jobs", "create", "http", job_name,
            "--schedule", schedule,
            "--uri", webhook_url,
            "--http-method", "POST",
            "--oidc-service-account-email", service_account,
            "--location", location,
            "--project", project_id
        ]
        
    result = run_cmd(cmd)
    if result is not None:
        logging.info("Cloud Scheduler job set up successfully!\nSchedule: %s\nURI: %s", schedule, webhook_url)
    else:
        logging.error("Failed to configure Cloud Scheduler job.")
        sys.exit(1)

if __name__ == "__main__":
    setup_scheduler()
