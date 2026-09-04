import os
import yaml
import pytest

OPENAPI_PATH = os.path.join(os.path.dirname(__file__), "..", "api_gateway", "openapi.yaml")

@pytest.fixture
def openapi_spec():
    """Loads and parses the OpenAPI YAML specification."""
    assert os.path.exists(OPENAPI_PATH), f"OpenAPI contract missing at {OPENAPI_PATH}"
    with open(OPENAPI_PATH, "r", encoding="utf-8") as f:
        spec = yaml.safe_load(f)
    return spec

def test_openapi_spec_structure(openapi_spec):
    """Verifies standard metadata and top-level OpenAPI structure."""
    assert "swagger" in openapi_spec or "openapi" in openapi_spec
    assert "info" in openapi_spec
    assert openapi_spec["info"]["title"] == "Enterprise E-Commerce Intelligence API"
    assert "paths" in openapi_spec
    assert "definitions" in openapi_spec

def test_openapi_security_definitions(openapi_spec):
    """Verifies API Key security definitions in header and query params."""
    sec_defs = openapi_spec.get("securityDefinitions", {})
    assert "api_key_header" in sec_defs
    assert sec_defs["api_key_header"]["type"] == "apiKey"
    assert sec_defs["api_key_header"]["name"] == "x-api-key"
    assert sec_defs["api_key_header"]["in"] == "header"

def test_core_endpoints_presence(openapi_spec):
    """Verifies all 6 mission-critical endpoints are declared in the contract."""
    paths = openapi_spec["paths"]
    
    expected_paths = {
        "/v1/health",
        "/v1/predict/churn",
        "/v1/predict/campaign",
        "/v1/predict/campaign-graph/{customer_id}",
        "/v1/rag/advisor",
        "/v1/monitoring/check-and-retrain"
    }
    
    for path in expected_paths:
        assert path in paths, f"Missing path {path} in OpenAPI spec"

def test_churn_predict_endpoint_contract(openapi_spec):
    """Verifies churn endpoint configuration, security, and backend extension."""
    churn_path = openapi_spec["paths"]["/v1/predict/churn"]
    assert "post" in churn_path
    op = churn_path["post"]
    
    assert "x-google-backend" in op
    assert "address" in op["x-google-backend"]
    assert "security" in op
    assert any("api_key_header" in s for s in op["security"])

def test_langgraph_campaign_endpoint_contract(openapi_spec):
    """Verifies LangGraph dynamic campaign endpoint and path parameter."""
    graph_path = openapi_spec["paths"]["/v1/predict/campaign-graph/{customer_id}"]
    assert "get" in graph_path
    op = graph_path["get"]
    
    assert "parameters" in op
    params = op["parameters"]
    assert any(p["name"] == "customer_id" and p["in"] == "path" for p in params)
    assert "x-google-backend" in op

def test_rag_advisor_definitions(openapi_spec):
    """Verifies RAG definitions and response schema."""
    defs = openapi_spec.get("definitions", {})
    assert "RagAdvisorRequest" in defs
    assert "RagAdvisorResponse" in defs
    assert "query" in defs["RagAdvisorRequest"]["properties"]
