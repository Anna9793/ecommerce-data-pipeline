# ============================================================
# BigQuery Data Warehouse Layer
# ============================================================

resource "google_bigquery_dataset" "retail_data" {
  dataset_id                  = "retail_data"
  friendly_name               = "E-commerce Retail Data"
  description                 = "Data warehouse storing transactions, product catalog, and analytical RFM views"
  location                    = var.region
  default_table_expiration_ms = null

  labels = {
    environment = var.environment
    managed_by  = "terraform"
  }
}

# 1. Transactions Table
resource "google_bigquery_table" "transactions" {
  dataset_id          = google_bigquery_dataset.retail_data.dataset_id
  table_id            = "transactions"
  deletion_protection = false

  time_partitioning {
    type  = "DAY"
    field = "InvoiceDate"
  }

  clustering = ["CustomerID", "Country"]

  schema = jsonencode([
    {
      name = "InvoiceNo"
      type = "STRING"
      mode = "NULLABLE"
    },
    {
      name = "StockCode"
      type = "STRING"
      mode = "NULLABLE"
    },
    {
      name = "Description"
      type = "STRING"
      mode = "NULLABLE"
    },
    {
      name = "Quantity"
      type = "INTEGER"
      mode = "NULLABLE"
    },
    {
      name = "InvoiceDate"
      type = "TIMESTAMP"
      mode = "NULLABLE"
    },
    {
      name = "UnitPrice"
      type = "FLOAT"
      mode = "NULLABLE"
    },
    {
      name = "CustomerID"
      type = "STRING"
      mode = "NULLABLE"
    },
    {
      name = "Country"
      type = "STRING"
      mode = "NULLABLE"
    }
  ])
}

# 2. Product Catalog Table with Embeddings
resource "google_bigquery_table" "product_catalog" {
  dataset_id          = google_bigquery_dataset.retail_data.dataset_id
  table_id            = "product_catalog"
  deletion_protection = false

  schema = jsonencode([
    {
      name = "stock_code"
      type = "STRING"
      mode = "REQUIRED"
    },
    {
      name = "description"
      type = "STRING"
      mode = "NULLABLE"
    },
    {
      name = "unit_price"
      type = "FLOAT"
      mode = "NULLABLE"
    },
    {
      name = "embedding"
      type = "FLOAT"
      mode = "REPEATED"
    }
  ])
}

# 3. RFM Features Analytical View
resource "google_bigquery_table" "rfm_features_view" {
  dataset_id          = google_bigquery_dataset.retail_data.dataset_id
  table_id            = "rfm_features"
  deletion_protection = false

  view {
    query          = <<-SQL
      WITH customer_metrics AS (
        SELECT 
          CustomerID AS customer_id,
          DATE_DIFF(DATE('2011-12-09'), DATE(MAX(InvoiceDate)), DAY) AS recency,
          COUNT(DISTINCT InvoiceNo) AS frequency,
          AVG(Quantity * UnitPrice) AS avg_order_value,
          COALESCE(
            COUNTIF(STARTS_WITH(InvoiceNo, 'C')) / NULLIF(COUNT(DISTINCT InvoiceNo), 0),
            0.0
          ) AS cancellation_rate,
          EXTRACT(HOUR FROM MAX(InvoiceDate)) AS preferred_shopping_hour
        FROM `${var.project_id}.retail_data.transactions`
        WHERE CustomerID IS NOT NULL
        GROUP BY CustomerID
      )
      SELECT 
        customer_id,
        recency,
        frequency,
        avg_order_value,
        cancellation_rate,
        preferred_shopping_hour,
        1.0 AS spending_velocity
      FROM customer_metrics
    SQL
    use_legacy_sql = false
  }

  depends_on = [google_bigquery_table.transactions]
}
