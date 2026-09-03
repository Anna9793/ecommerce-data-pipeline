variable "project_id" {
  description = "Google Cloud Project ID"
  type        = string
  default     = "anna-ml-pipeline"
}

variable "region" {
  description = "Default Google Cloud Region"
  type        = string
  default     = "us-central1"
}

variable "environment" {
  description = "Deployment environment (e.g. production, staging, dev)"
  type        = string
  default     = "production"
}

variable "gcs_bucket_name" {
  description = "Name of the Google Cloud Storage bucket for ML models and artifacts"
  type        = string
  default     = "anna-ml-pipeline-bucket"
}

variable "container_image" {
  description = "Docker image URI in Google Artifact Registry"
  type        = string
  default     = "us-central1-docker.pkg.dev/anna-ml-pipeline/ecommerce-ml-pipeline/pipeline-image:latest"
}

variable "scheduler_cron" {
  description = "Cron expression for the weekly automated retraining check"
  type        = string
  default     = "0 0 * * 0"
}
