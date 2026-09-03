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
