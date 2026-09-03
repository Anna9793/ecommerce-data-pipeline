# ============================================================
# Google Cloud Dataflow Streaming Pipeline Infrastructure
# ============================================================

# BigQuery Table for Real-Time Windowed Customer Aggregates
resource "google_bigquery_table" "streaming_customer_aggregates" {
  dataset_id          = google_bigquery_dataset.retail_data.dataset_id
  table_id            = "streaming_customer_aggregates"
  deletion_protection = false

  time_partitioning {
    type  = "DAY"
    field = "window_end"
  }

  clustering = ["customer_id"]

  schema = jsonencode([
    {
      name = "customer_id"
      type = "STRING"
      mode = "REQUIRED"
    },
    {
      name = "window_start"
      type = "TIMESTAMP"
      mode = "REQUIRED"
    },
    {
      name = "window_end"
      type = "TIMESTAMP"
      mode = "REQUIRED"
    },
    {
      name = "total_spend"
      type = "FLOAT"
      mode = "REQUIRED"
    },
    {
      name = "order_count"
      type = "INTEGER"
      mode = "REQUIRED"
    },
    {
      name = "item_count"
      type = "INTEGER"
      mode = "REQUIRED"
    },
    {
      name = "cancellation_count"
      type = "INTEGER"
      mode = "REQUIRED"
    },
    {
      name = "cancellation_ratio"
      type = "FLOAT"
      mode = "REQUIRED"
    },
    {
      name = "spending_velocity"
      type = "FLOAT"
      mode = "REQUIRED"
    }
  ])

  labels = {
    pipeline    = "apache-beam-dataflow"
    environment = var.environment
    managed_by  = "terraform"
  }
}
