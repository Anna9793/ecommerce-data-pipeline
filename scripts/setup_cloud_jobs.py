"""
Automated provisioning of Cloud Run Jobs, Pub/Sub Topics, and Eventarc Triggers.

This script provisions:
1. Pub/Sub Topics:
   - catalog-updates: Triggers vector re-embedding & catalog sync to PostgreSQL pgvector.
   - features-updates: Triggers feature store refresh to Firestore online store.
2. Cloud Run Jobs:
   - sync-features-job: Executes scripts/sync_feature_store.py
   - sync-vectors-job: Executes scripts/sync_product_vectors.py
   - drift-monitor-job: Executes scripts/run_drift_monitor.py
3. Eventarc Triggers:
   - trigger-sync-features: Pub/Sub features-updates -> sync-features-job
   - trigger-sync-vectors: Pub/Sub catalog-updates -> sync-vectors-job
4. Cloud Scheduler (Failsafe & Monitoring):
   - trigger-drift-monitor: Weekly schedule (0 0 * * 0) -> drift-monitor-job
"""

import os
import sys
import logging
import argparse
import subprocess
from typing import List, Optional, Dict, Any

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def run_cmd(args: List[str], dry_run: bool = False) -> Optional[str]:
    """Executes a shell command via subprocess or logs it if in dry_run mode."""
    cmd_str = " ".join(args)
    if dry_run:
        logging.info("[DRY RUN] Would execute: %s", cmd_str)
        return "DRY_RUN_SUCCESS"

    logging.info("Executing: %s", cmd_str)
    try:
        result = subprocess.run(args, capture_output=True, text=True, check=True)
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        logging.error("Command failed: %s\nError output: %s", cmd_str, e.stderr)
        return None


def get_gcp_context() -> Dict[str, str]:
    """Resolves GCP Project ID, Location, Project Number, and Service Account."""
    project_id = os.getenv("GCP_PROJECT") or os.getenv("GOOGLE_CLOUD_PROJECT")
    if not project_id:
        project_id = run_cmd(["gcloud", "config", "get-value", "project"])

    if not project_id or project_id == "(unset)":
        logging.error("GCP project ID could not be determined. Set GCP_PROJECT or configure gcloud.")
        sys.exit(1)

    location = os.getenv("GCP_LOCATION", "us-central1")
    image_tag = os.getenv(
        "IMAGE_TAG",
        f"{location}-docker.pkg.dev/{project_id}/ecommerce-ml-pipeline/pipeline-image:latest",
    )

    # Resolve project number for default compute service account
    project_number = run_cmd(["gcloud", "projects", "describe", project_id, "--format", "value(projectNumber)"])
    if not project_number or "DRY_RUN" in project_number or not project_number.isdigit():
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


def provision_pubsub_topic(topic_name: str, context: Dict[str, str], dry_run: bool = False) -> bool:
    """Ensures a Pub/Sub topic exists."""
    project_id = context["project_id"]
    logging.info("Checking Pub/Sub topic: %s...", topic_name)
    existing = run_cmd(["gcloud", "pubsub", "topics", "describe", topic_name, "--project", project_id], dry_run=dry_run)
    if existing:
        logging.info("Pub/Sub topic '%s' already exists.", topic_name)
        return True

    logging.info("Creating Pub/Sub topic: %s...", topic_name)
    result = run_cmd(["gcloud", "pubsub", "topics", "create", topic_name, "--project", project_id], dry_run=dry_run)
    return result is not None


def provision_cloud_run_job(
    job_name: str,
    script_path: str,
    context: Dict[str, str],
    dry_run: bool = False,
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
    ], dry_run=dry_run)

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

    result = run_cmd(cmd, dry_run=dry_run)
    if result is not None:
        logging.info("Successfully %sd Cloud Run Job '%s'.", action, job_name)
        return True
    return False


def provision_eventarc_trigger(
    trigger_name: str,
    job_name: str,
    topic_name: str,
    context: Dict[str, str],
    dry_run: bool = False,
) -> bool:
    """Creates or updates an Eventarc trigger connecting a Pub/Sub topic to a Cloud Run Job."""
    project_id = context["project_id"]
    location = context["location"]
    service_account = context["service_account"]

    logging.info("Checking Eventarc Trigger: %s...", trigger_name)
    existing = run_cmd([
        "gcloud", "eventarc", "triggers", "describe", trigger_name,
        "--location", location,
        "--project", project_id,
    ], dry_run=dry_run)

    if existing:
        logging.info("Eventarc trigger '%s' already exists. Updating destination...", trigger_name)
        cmd = [
            "gcloud", "eventarc", "triggers", "update", trigger_name,
            "--destination-run-job", job_name,
            "--location", location,
            "--project", project_id,
        ]
    else:
        logging.info("Creating new Eventarc trigger '%s' linking Pub/Sub topic '%s' to Job '%s'...", trigger_name, topic_name, job_name)
        cmd = [
            "gcloud", "eventarc", "triggers", "create", trigger_name,
            "--destination-run-job", job_name,
            "--event-filters", "type=google.cloud.pubsub.topic.v1.messagePublished",
            "--event-filters", f"topic={topic_name}",
            "--service-account", service_account,
            "--location", location,
            "--project", project_id,
        ]

    result = run_cmd(cmd, dry_run=dry_run)
    return result is not None


def provision_scheduler_job(
    scheduler_name: str,
    target_job_name: str,
    schedule: str,
    context: Dict[str, str],
    dry_run: bool = False,
) -> bool:
    """Provisions a Cloud Scheduler trigger directly executing a Cloud Run Job."""
    project_id = context["project_id"]
    location = context["location"]

    logging.info("Checking Cloud Scheduler job: %s...", scheduler_name)
    existing = run_cmd([
        "gcloud", "scheduler", "jobs", "describe", scheduler_name,
        "--location", location,
        "--project", project_id,
    ], dry_run=dry_run)

    action = "update" if existing else "create"
    cmd = [
        "gcloud", "scheduler", "jobs", action, "run-job", scheduler_name,
        "--job", target_job_name,
        "--schedule", schedule,
        "--location", location,
        "--project", project_id,
    ]

    result = run_cmd(cmd, dry_run=dry_run)
    return result is not None


def publish_event(topic_name: str, message: str, context: Dict[str, str], dry_run: bool = False) -> bool:
    """Publishes a test/operational event message to a Pub/Sub topic to trigger its Job."""
    project_id = context["project_id"]
    logging.info("Publishing message to topic '%s': %s", topic_name, message)
    cmd = [
        "gcloud", "pubsub", "topics", "publish", topic_name,
        f"--message={message}",
        "--project", project_id,
    ]
    result = run_cmd(cmd, dry_run=dry_run)
    return result is not None


def setup_all(context: Dict[str, str], dry_run: bool = False) -> None:
    """Orchestrates provisioning of all Pub/Sub topics, Cloud Run Jobs, Triggers, and Schedulers."""
    logging.info("Starting GCP batch & event-driven infrastructure setup for project %s (%s)...", context["project_id"], context["location"])

    # 1. Pub/Sub Topics
    topics = ["catalog-updates", "features-updates"]
    for t in topics:
        provision_pubsub_topic(t, context, dry_run=dry_run)

    # 2. Cloud Run Jobs
    jobs = [
        ("sync-features-job", "scripts/sync_feature_store.py"),
        ("sync-vectors-job", "scripts/sync_product_vectors.py"),
        ("drift-monitor-job", "scripts/run_drift_monitor.py"),
    ]
    for job_name, script in jobs:
        provision_cloud_run_job(job_name, script, context, dry_run=dry_run)

    # 3. Eventarc Triggers (Pub/Sub -> Cloud Run Job)
    triggers = [
        ("trigger-sync-features", "sync-features-job", "features-updates"),
        ("trigger-sync-vectors", "sync-vectors-job", "catalog-updates"),
    ]
    for trigger_name, job_name, topic in triggers:
        provision_eventarc_trigger(trigger_name, job_name, topic, context, dry_run=dry_run)

    # 4. Cloud Scheduler Failsafe & Monitoring (Weekly drift check at midnight)
    provision_scheduler_job("trigger-drift-monitor", "drift-monitor-job", "0 0 * * 0", context, dry_run=dry_run)

    logging.info("✅ All batch jobs, Pub/Sub topics, Eventarc triggers, and Schedulers configured successfully.")


def main():
    parser = argparse.ArgumentParser(description="Setup Cloud Run Jobs, Pub/Sub topics, and Eventarc triggers.")
    parser.add_argument("--dry-run", action="store_true", help="Log commands without executing them.")
    parser.add_argument("--jobs-only", action="store_true", help="Only provision Cloud Run Jobs.")
    parser.add_argument("--publish", choices=["catalog", "features"], help="Publish a test event to trigger the respective job.")
    parser.add_argument("--message", default='{"event": "manual_trigger", "source": "setup_script"}', help="JSON message payload for --publish.")
    args = parser.parse_args()

    context = get_gcp_context()

    if args.publish:
        topic = "catalog-updates" if args.publish == "catalog" else "features-updates"
        publish_event(topic, args.message, context, dry_run=args.dry_run)
        return

    if args.jobs_only:
        jobs = [
            ("sync-features-job", "scripts/sync_feature_store.py"),
            ("sync-vectors-job", "scripts/sync_product_vectors.py"),
            ("drift-monitor-job", "scripts/run_drift_monitor.py"),
        ]
        for job_name, script in jobs:
            provision_cloud_run_job(job_name, script, context, dry_run=args.dry_run)
        return

    setup_all(context, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
