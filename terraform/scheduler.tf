# ============================================================
# Google Cloud Scheduler Cron Job (Automated Retraining Check)
# ============================================================

resource "google_cloud_scheduler_job" "weekly_retraining_check" {
  name             = "weekly-model-retraining-check"
  description      = "Triggers weekly K-S drift check and automated model retraining"
  schedule         = var.scheduler_cron
  time_zone        = "UTC"
  attempt_deadline = "320s"

  http_target {
    http_method = "POST"
    uri         = "${google_cloud_run_v2_service.api_service.uri}/monitoring/check-and-retrain"

    oidc_token {
      service_account_email = "${var.project_id}@appspot.gserviceaccount.com"
      audience              = google_cloud_run_v2_service.api_service.uri
    }
  }

  depends_on = [google_cloud_run_v2_service.api_service]
}
