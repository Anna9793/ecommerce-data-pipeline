import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

@patch("app.rag_service.ProductAdvisorService")
def test_rag_advisor_endpoint(mock_service_class):
    mock_instance = mock_service_class.return_value
    mock_instance.advise.return_value = {
        "user_query": "Cozy winter gift under $20",
        "budget_applied": 20.0,
        "intro_message": "Here are wonderful cozy winter gifts matching your budget!",
        "recommendations": [
            {
                "stock_code": "85123A",
                "description": "WHITE HANGING HEART T-LIGHT HOLDER",
                "category": "Home Decor & Lighting",
                "unit_price": 2.55,
                "similarity": 0.91,
                "why_recommended": "Creates a warm, cozy ambient light perfect for winter evenings."
            }
        ],
        "shopping_tip": "Pair with scented tea lights for an extra special holiday gift."
    }

    response = client.post("/rag/advisor", json={
        "query": "Cozy winter gift under $20",
        "budget_max": 20.0,
        "top_k": 3
    })
    
    assert response.status_code == 200
    data = response.json()
    assert data["user_query"] == "Cozy winter gift under $20"
    assert data["budget_applied"] == 20.0
    assert len(data["recommendations"]) == 1
    assert data["recommendations"][0]["stock_code"] == "85123A"
    assert "why_recommended" in data["recommendations"][0]
    assert "shopping_tip" in data

@patch("app.db_postgres.get_connection")
def test_search_product_catalog_pgvector(mock_get_conn):
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_get_conn.return_value = mock_conn
    mock_conn.cursor.return_value = mock_cursor
    
    mock_cursor.fetchall.return_value = [
        ("85123A", "WHITE HANGING HEART T-LIGHT HOLDER", "Home Decor", 2.55, "Product: ...", 0.9123),
        ("22423", "REGENCY CAKESTAND 3 TIER", "Kitchen", 12.75, "Product: ...", 0.8245)
    ]
    
    from app.db_postgres import search_product_catalog_pgvector
    fake_vector = [0.1] * 768
    results = search_product_catalog_pgvector(fake_vector, budget_max=15.0, top_k=2)
    
    assert len(results) == 2
    assert results[0]["stock_code"] == "85123A"
    assert results[0]["unit_price"] == 2.55
    assert results[0]["similarity"] == 0.9123
    assert mock_cursor.execute.called

@patch("app.rag_service.vertexai.init")
@patch("app.rag_service.TextEmbeddingModel.from_pretrained")
@patch("app.rag_service.GenerativeModel")
@patch("app.rag_service.search_product_catalog_pgvector")
def test_product_advisor_service_advise(mock_search, mock_gen_model, mock_embed_model, mock_vertex):
    mock_search.return_value = [
        {
            "stock_code": "47566",
            "description": "PARTY BUNTING",
            "category": "Party & Celebration",
            "unit_price": 4.95,
            "document_text": "Product: ...",
            "similarity": 0.88
        }
    ]
    
    mock_model_instance = mock_gen_model.return_value
    mock_response = mock_model_instance.generate_content.return_value
    mock_response.text = '{"user_query": "party decor", "intro_message": "Here are festive decor picks!", "recommendations": [{"stock_code": "47566", "description": "PARTY BUNTING", "category": "Party", "unit_price": 4.95, "similarity": 0.88, "why_recommended": "Vibrant party bunting adds instant festive energy."}], "shopping_tip": "Hang above the dessert table!"}'
    
    from app.rag_service import ProductAdvisorService
    service = ProductAdvisorService()
    advice = service.advise("party decor", budget_max=10.0, top_k=1, tenant_id="giftshop_uk")
    
    assert advice["user_query"] == "party decor"
    assert len(advice["recommendations"]) == 1
    assert advice["recommendations"][0]["why_recommended"] == "Vibrant party bunting adds instant festive energy."
    assert "shopping_tip" in advice

@patch("app.rag_service.vertexai.init")
@patch("app.rag_service.TextEmbeddingModel.from_pretrained")
@patch("app.rag_service.GenerativeModel")
def test_product_advisor_service_nordic_tenant(mock_gen_model, mock_embed_model, mock_vertex):
    from app.rag_service import ProductAdvisorService
    service = ProductAdvisorService()
    
    # Test local/deterministic fallback search for NordicWear & Tech
    results = service.search_products("noise cancelling headphones", budget_max=160.0, top_k=2, tenant_id="nordic_tech")
    
    assert len(results) > 0
    assert any("Headphones" in p["description"] or "Audio" in p["category"] or "SKU-TECH" in p["stock_code"] for p in results)
    assert all(p["unit_price"] <= 160.0 for p in results)

    # Test full advise workflow for Nordic tenant
    mock_model_instance = mock_gen_model.return_value
    mock_response = mock_model_instance.generate_content.return_value
    mock_response.text = '{"user_query": "headphones", "intro_message": "Welcome to NordicWear & Tech! Here are our ANC headphones:", "recommendations": [{"stock_code": "SKU-TECH-001", "description": "Nordic Pro ANC Wireless Headphones", "category": "Smart Audio", "unit_price": 149.00, "similarity": 0.94, "why_recommended": "Industry-leading active noise cancellation designed in Scandinavia."}], "shopping_tip": "Enable spatial audio in settings."}'

    advice = service.advise("headphones", budget_max=150.0, top_k=1, tenant_id="nordic_tech")
    assert advice["tenant_id"] == "nordic_tech"
    assert "NordicWear" in advice["store_name"]
    assert len(advice["recommendations"]) == 1
    assert advice["recommendations"][0]["stock_code"] == "SKU-TECH-001"
