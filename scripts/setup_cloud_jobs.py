"""Provisions Cloud Run Jobs, Pub/Sub topics, Push Subscriptions, and Cloud Scheduler on GCP."""

import os
import sys
import logging
import argparse
import subprocess
from typing import List, Optional, Dict

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def run_cmd(args: List[str], check_exists: bool = False) -> Optional[str]:
    """
    Executes a shell command via subprocess.
    Uses stdin=subprocess.DEVNULL to prevent gcloud from hanging on interactive prompts.
    """
    if args and args[0] == "gcloud" and "--quiet" not in args:
        args = list(args) + ["--quiet"]

    cmd_str = " ".join(args)
    logging.info("Executing: %s", cmd_str)
    try:
        result = subprocess.run(args, capture_output=True, text=True, check=True, stdin=subprocess.DEVNULL)
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        if check_exists:
            logging.debug("Resource existence check returned non-zero (not found): %s", cmd_str)
            return None
        logging.error("Command failed: %s\nError output: %s", cmd_str, e.stderr)
        return None


def get_gcp_context() -> Dict[str, str]:
    """Resolves GCP Project ID, Location, Project Number, and Service Account."""
    project_id = os.getenv("GCP_PROJECT") or os.getenv("GOOGLE_CLOUD_PROJECT")
    if not project_id:
        project_id = run_cmd(["gcloud", "config", "get-value", "project"], check_exists=True)

    if not project_id or project_id == "(unset)":
        logging.error("GCP project ID could not be determined. Set GCP_PROJECT or configure gcloud.")
        sys.exit(1)

    location = os.getenv("GCP_LOCATION", "us-central1")
    image_tag = os.getenv(
        "IMAGE_TAG",
        f"{location}-docker.pkg.dev/{project_id}/ecommerce-ml-pipeline/pipeline-image:latest",
    )

    project_number = run_cmd(
        ["gcloud", "projects", "describe", project_id, "--format", "value(projectNumber)"],
        check_exists=True,
    )
    if not project_number or not project_number.isdigit():
        if project_id == "anna-ml-pipeline":
            project_number = "557448715890"
        else:
            project_number = "123456789012"  # fallback mock for local/test

    service_account = f"{project_number}-compute@developer.gserviceaccount.com"

    return {
        "project_id": project_id,
        "location": location,
        "image_tag": image_tag,
        "service_account": service_account,
    }



def provision_pubsub_topic(topic_name: str, context: Dict[str, str]) -> bool:
    """Ensures a Pub/Sub topic exists."""
    project_id = context["project_id"]
    logging.info("Checking Pub/Sub topic: %s...", topic_name)
    existing = run_cmd(
        ["gcloud", "pubsub", "topics", "describe", topic_name, "--project", project_id],
        check_exists=True,
    )
    if existing:
        logging.info("Pub/Sub topic '%s' already exists.", topic_name)
        return True

    logging.info("Creating Pub/Sub topic: %s...", topic_name)
    result = run_cmd(["gcloud", "pubsub", "topics", "create", topic_name, "--project", project_id])
    return result is not None


def provision_cloud_run_job(
    job_name: str,
    script_path: str,
    context: Dict[str, str],
) -> bool:
    """Creates or updates a Cloud Run Job executing a specific batch script."""
    project_id = context["project_id"]
    location = context["location"]
    image = context["image_tag"]

    logging.info("Checking Cloud Run Job: %s in %s...", job_name, location)
    existing = run_cmd([
        "gcloud", "run", "jobs", "describe", job_name,
        "--region", location,
        "--project", project_id,
    ], check_exists=True)

    action = "update" if existing else "create"
    cmd = [
        "gcloud", "run", "jobs", action, job_name,
        "--image", image,
        "--command", "python",
        "--args", script_path,
        "--set-env-vars", f"USE_BIGQUERY=true,GCP_PROJECT={project_id},GOOGLE_CLOUD_PROJECT={project_id}",
        "--region", location,
        "--project", project_id,
    ]

    result = run_cmd(cmd)
    if result is not None:
        logging.info("Successfully %sd Cloud Run Job '%s'.", action, job_name)
        return True
    return False


def provision_scheduler_job(
    scheduler_name: str,
    target_job_name: str,
    schedule: str,
    context: Dict[str, str],
) -> bool:
    """
    Provisions a Cloud Scheduler trigger targeting a Cloud Run Job's native REST API.
    Uses target type 'http' with POST to the jobs:run endpoint.
    """
    project_id = context["project_id"]
    location = context["location"]
    service_account = context["service_account"]
    uri = f"https://{location}-run.googleapis.com/v2/projects/{project_id}/locations/{location}/jobs/{target_job_name}:run"

    logging.info("Checking Cloud Scheduler job: %s...", scheduler_name)
    existing = run_cmd([
        "gcloud", "scheduler", "jobs", "describe", scheduler_name,
        "--location", location,
        "--project", project_id,
    ], check_exists=True)

    action = "update" if existing else "create"
    cmd = [
        "gcloud", "scheduler", "jobs", action, "http", scheduler_name,
        "--schedule", schedule,
        "--uri", uri,
        "--http-method", "POST",
        "--oauth-service-account-email", service_account,
        "--location", location,
        "--project", project_id,
    ]

    result = run_cmd(cmd)
    return result is not None


def provision_pubsub_push_subscription(
    sub_name: str,
    topic_name: str,
    target_job_name: str,
    context: Dict[str, str],
) -> bool:
    """
    Creates or updates a Pub/Sub Push Subscription delivering events to api-service/events/trigger-job.
    This routes Pub/Sub events directly into asynchronous Cloud Run Job execution.
    """
    project_id = context["project_id"]
    location = context["location"]
    service_account = context["service_account"]

    api_url = run_cmd([
        "gcloud", "run", "services", "describe", "api-service",
        "--region", location,
        "--project", project_id,
        "--format", "value(status.url)",
    ], check_exists=True)

    if not api_url:
        api_url = f"https://api-service-{project_id}.{location}.run.app"

    push_endpoint = f"{api_url}/events/trigger-job?job_name={target_job_name}"

    logging.info("Checking Pub/Sub push subscription: %s...", sub_name)
    existing = run_cmd([
        "gcloud", "pubsub", "subscriptions", "describe", sub_name,
        "--project", project_id,
    ], check_exists=True)

    if existing:
        logging.info("Updating existing Pub/Sub push subscription '%s' -> %s...", sub_name, push_endpoint)
        cmd = [
            "gcloud", "pubsub", "subscriptions", "update", sub_name,
            "--push-endpoint", push_endpoint,
            "--push-auth-service-account", service_account,
            "--project", project_id,
        ]
    else:
        logging.info("Creating new Pub/Sub push subscription '%s' -> %s...", sub_name, push_endpoint)
        cmd = [
            "gcloud", "pubsub", "subscriptions", "create", sub_name,
            "--topic", topic_name,
            "--push-endpoint", push_endpoint,
            "--push-auth-service-account", service_account,
            "--ack-deadline", "60",
            "--project", project_id,
        ]

    result = run_cmd(cmd)
    return result is not None


def execute_cloud_run_job(job_name: str, context: Dict[str, str]) -> bool:
    """Executes a Cloud Run Job asynchronously on demand."""
    project_id = context["project_id"]
    location = context["location"]
    logging.info("Triggering Cloud Run Job '%s' in %s (%s)...", job_name, location, project_id)
    cmd = [
        "gcloud", "run", "jobs", "execute", job_name,
        "--region", location,
        "--project", project_id,
        "--async",
    ]
    result = run_cmd(cmd)
    return result is not None


def publish_event(topic_name: str, message: str, context: Dict[str, str]) -> bool:
    """Publishes an event message to a Pub/Sub topic to trigger its Job via push subscription."""
    project_id = context["project_id"]
    logging.info("Publishing message to topic '%s': %s", topic_name, message)
    cmd = [
        "gcloud", "pubsub", "topics", "publish", topic_name,
        f"--message={message}",
        "--project", project_id,
    ]
    result = run_cmd(cmd)
    return result is not None


def setup_all(context: Dict[str, str]) -> None:
    """Orchestrates provisioning of all Pub/Sub topics, Cloud Run Jobs, Subscriptions, and Schedulers."""
    logging.info("Starting GCP batch & event-driven infrastructure setup for project %s (%s)...", context["project_id"], context["location"])

    # 1. Pub/Sub Topics
    topics = ["catalog-updates", "features-updates"]
    for t in topics:
        provision_pubsub_topic(t, context)

    # 2. Cloud Run Jobs
    jobs = [
        ("sync-features-job", "scripts/sync_feature_store.py"),
        ("sync-vectors-job", "scripts/sync_product_vectors.py"),
        ("drift-monitor-job", "scripts/run_drift_monitor.py"),
    ]
    for job_name, script in jobs:
        provision_cloud_run_job(job_name, script, context)

    # 3. Pub/Sub Push Subscriptions (Event-Driven Trigger to Cloud Run Jobs)
    subscriptions = [
        ("sub-catalog-updates", "catalog-updates", "sync-vectors-job"),
        ("sub-features-updates", "features-updates", "sync-features-job"),
    ]
    for sub_name, topic, target_job in subscriptions:
        provision_pubsub_push_subscription(sub_name, topic, target_job, context)

    # 4. Cloud Scheduler (Failsafe & Periodic Monitoring via REST API)
    provision_scheduler_job("trigger-drift-monitor", "drift-monitor-job", "0 0 * * 0", context)

    logging.info("✅ All batch jobs, Pub/Sub topics, Push Subscriptions, and Schedulers configured successfully.")


def main():
    parser = argparse.ArgumentParser(description="Setup Cloud Run Jobs, Pub/Sub topics, and event-driven triggers.")
    parser.add_argument("--jobs-only", action="store_true", help="Only provision Cloud Run Jobs.")
    parser.add_argument("--schedulers-only", action="store_true", help="Only provision Cloud Scheduler jobs.")
    parser.add_argument("--run-job", choices=["features", "vectors", "drift"], help="Directly execute a Cloud Run Job on GCP.")
    parser.add_argument("--publish", choices=["catalog", "features"], help="Publish a test event to trigger the respective job.")
    parser.add_argument("--message", default='{"event": "manual_trigger", "source": "setup_script"}', help="JSON message payload for --publish.")
    args = parser.parse_args()

    context = get_gcp_context()

    if args.run_job:
        job_map = {
            "features": "sync-features-job",
            "vectors": "sync-vectors-job",
            "drift": "drift-monitor-job",
        }
        execute_cloud_run_job(job_map[args.run_job], context)
        return

    if args.publish:
        topic = "catalog-updates" if args.publish == "catalog" else "features-updates"
        publish_event(topic, args.message, context)
        return

    if args.jobs_only:
        jobs = [
            ("sync-features-job", "scripts/sync_feature_store.py"),
            ("sync-vectors-job", "scripts/sync_product_vectors.py"),
            ("drift-monitor-job", "scripts/run_drift_monitor.py"),
        ]
        for job_name, script in jobs:
            provision_cloud_run_job(job_name, script, context)
        return

    if args.schedulers_only:
        provision_scheduler_job("trigger-drift-monitor", "drift-monitor-job", "0 0 * * 0", context)
        return

    setup_all(context)


if __name__ == "__main__":
    main()
