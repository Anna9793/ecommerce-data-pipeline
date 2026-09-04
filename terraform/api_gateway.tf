# ============================================================
# Google Cloud API Gateway Ingress Infrastructure
# ============================================================

# 1. API Resource
resource "google_api_gateway_api" "api_gateway" {
  provider     = google
  api_id       = "ecommerce-api-gateway"
  display_name = "E-Commerce Intelligence API Gateway"

  labels = {
    environment = var.environment
    managed_by  = "terraform"
  }
}

# 2. API Gateway Config (Attaches OpenAPI Specification)
resource "google_api_gateway_api_config" "api_cfg" {
  provider      = google
  api           = google_api_gateway_api.api_gateway.api_id
  api_config_id = "v1-config"
  display_name  = "V1 Production OpenAPI Config"

  openapi_documents {
    document {
      path     = "openapi.yaml"
      contents = filebase64("${path.module}/../api_gateway/openapi.yaml")
    }
  }

  lifecycle {
    create_before_destroy = true
  }
}

# 3. Regional API Gateway Instance
resource "google_api_gateway_gateway" "gateway" {
  provider   = google
  gateway_id = "ecommerce-gateway"
  api_config = google_api_gateway_api_config.api_cfg.id
  region     = var.region

  display_name = "E-Commerce Regional API Gateway (${var.region})"

  labels = {
    environment = var.environment
    managed_by  = "terraform"
  }
}
