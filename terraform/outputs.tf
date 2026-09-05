# ============================================================
# Terraform Outputs
# ============================================================

output "api_service_url" {
  description = "Public URL of the FastAPI Backend Cloud Run service"
  value       = google_cloud_run_v2_service.api_service.uri
}

output "dashboard_service_url" {
  description = "Public URL of the Streamlit Dashboard Cloud Run service"
  value       = google_cloud_run_v2_service.dashboard_service.uri
}

output "gcs_bucket_name" {
  description = "Google Cloud Storage bucket for ML artifacts"
  value       = google_storage_bucket.mlops_artifacts.name
}

output "bigquery_dataset_id" {
  description = "BigQuery dataset ID"
  value       = google_bigquery_dataset.retail_data.dataset_id
}

output "cloud_scheduler_job_name" {
  description = "Name of the automated retraining Cloud Scheduler job"
  value       = google_cloud_scheduler_job.weekly_retraining_check.name
}

output "pubsub_topic_name" {
  description = "Google Cloud Pub/Sub topic for streaming transactions"
  value       = google_pubsub_topic.transactions_topic.name
}

output "pubsub_subscription_name" {
  description = "Google Cloud Pub/Sub worker subscription name"
  value       = google_pubsub_subscription.worker_subscription.name
}

output "pubsub_dead_letter_topic_name" {
  description = "Google Cloud Pub/Sub Dead Letter Queue topic name"
  value       = google_pubsub_topic.dead_letter_topic.name
}

output "api_gateway_hostname" {
  description = "Google Cloud API Gateway default hostname"
  value       = google_api_gateway_gateway.gateway.default_hostname
}

output "api_gateway_url" {
  description = "Google Cloud API Gateway base HTTPS invocation URL"
  value       = "https://${google_api_gateway_gateway.gateway.default_hostname}"
}

output "dataproc_cluster_name" {
  description = "Name of the Google Cloud Dataproc PySpark cluster"
  value       = google_dataproc_cluster.pyspark_cluster.name
}

output "dataproc_cluster_region" {
  description = "Region of the Dataproc PySpark cluster"
  value       = google_dataproc_cluster.pyspark_cluster.region
}

