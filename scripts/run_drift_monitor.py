"""
Standalone Drift Monitoring and Automated Retraining Worker (Batch Job).
Evaluates Kolmogorov-Smirnov / PSI feature drift between baseline and production distributions.
If significant drift is detected, triggers a Vertex AI retraining pipeline.
"""

import os
import sys
import logging
from typing import Dict, Any

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("drift_monitor")


def evaluate_drift_and_retrain() -> Dict[str, Any]:
    """
    Core business logic for MLOps drift evaluation and retraining trigger.
    Reusable across CLI batch executions (Cloud Run Jobs) and API endpoints.
    """
    from src.monitoring import calculate_feature_drift
    from scripts.train_on_vertex import submit_vertex_training_job

    logger.info("Starting automated feature drift evaluation...")
    report = calculate_feature_drift()
    
    drift_detected = report.get("drift_detected", False)
    
    if drift_detected:
        logger.warning("🚨 Data drift detected! Launching automated retraining pipeline on Vertex AI...")
        job_name = submit_vertex_training_job()
        project_id = os.getenv("GCP_PROJECT", "anna-ml-pipeline")
        location = os.getenv("GCP_LOCATION", "us-central1")
        console_url = f"https://console.cloud.google.com/vertex-ai/pipelines/locations/{location}/runs/{job_name}?project={project_id}"
        
        logger.info("✅ Retraining job submitted successfully: %s", job_name)
        return {
            "status": "drift_detected",
            "message": "Data drift detected! Retraining pipeline submitted successfully.",
            "drift_details": report,
            "job_name": job_name,
            "console_url": console_url
        }
    else:
        logger.info("✅ Production features are healthy. No retraining triggered.")
        return {
            "status": "healthy",
            "message": "Features are healthy. Retraining is not required.",
            "drift_details": report
        }


def main():
    try:
        result = evaluate_drift_and_retrain()
        logger.info("Drift monitor completed: status=%s", result["status"])
        if result["status"] == "drift_detected":
            logger.info("Vertex AI Console URL: %s", result["console_url"])
    except Exception as e:
        logger.error("Drift monitoring batch job failed: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
