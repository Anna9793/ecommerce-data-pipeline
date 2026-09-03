# ============================================================
# Google Cloud Pub/Sub Messaging & Streaming Layer
# ============================================================

# 1. Central Transaction Streaming Topic
resource "google_pubsub_topic" "transactions_topic" {
  name = "retail-transactions-topic"

  labels = {
    environment = var.environment
    managed_by  = "terraform"
    service     = "streaming-ingestion"
  }
}

# 2. Dead Letter Queue Topic & Subscription (Infrastructure Protection)
resource "google_pubsub_topic" "dead_letter_topic" {
  name = "retail-transactions-dead-letter-topic"

  labels = {
    environment = var.environment
    managed_by  = "terraform"
    service     = "dead-letter-queue"
  }
}

resource "google_pubsub_subscription" "dead_letter_subscription" {
  name  = "retail-transactions-dead-letter-sub"
  topic = google_pubsub_topic.dead_letter_topic.name

  message_retention_duration = "604800s" # 7 days retention for debugging

  labels = {
    environment = var.environment
    managed_by  = "terraform"
  }
}

# 3. Worker Pull Subscription for Online Feature Store & Real-time Services
resource "google_pubsub_subscription" "worker_subscription" {
  name  = "retail-transactions-sub"
  topic = google_pubsub_topic.transactions_topic.name

  # Acknowledge deadline in seconds
  ack_deadline_seconds = 20

  # Retain unacknowledged messages for 24 hours
  message_retention_duration = "86400s"

  # Retain acknowledged messages for 4 hours (replayability)
  retain_acked_messages = true

  expiration_policy {
    ttl = "" # Never expire due to inactivity
  }

  retry_policy {
    minimum_backoff = "10s"
    maximum_backoff = "600s"
  }

  # Forward poisoned messages to DLQ after 5 failed delivery attempts
  dead_letter_policy {
    dead_letter_topic     = google_pubsub_topic.dead_letter_topic.id
    max_delivery_attempts = 5
  }

  labels = {
    environment = var.environment
    managed_by  = "terraform"
  }
}

# 3. Direct BigQuery Subscription (Zero-code Warehouse Ingestion)
resource "google_pubsub_subscription" "bigquery_subscription" {
  name  = "retail-transactions-bigquery-sub"
  topic = google_pubsub_topic.transactions_topic.name

  bigquery_config {
    table               = "${var.project_id}:${google_bigquery_dataset.retail_data.dataset_id}.${google_bigquery_table.transactions.table_id}"
    use_topic_schema    = false
    write_metadata      = false
    drop_unknown_fields = true
  }

  depends_on = [
    google_bigquery_table.transactions,
    google_pubsub_topic.transactions_topic
  ]

  labels = {
    environment = var.environment
    managed_by  = "terraform"
  }
}
