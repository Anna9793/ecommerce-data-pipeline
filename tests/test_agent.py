from fastapi.testclient import TestClient
from unittest.mock import patch
from app.main import app

client = TestClient(app)

@patch("app.agent_service.MarketingAgentService")
def test_generate_campaign_endpoint(mock_service_class):
    # Mock the instance returned by instantiating the class
    mock_instance = mock_service_class.return_value
    mock_instance.generate_marketing_campaign.return_value = {
        "customer_id": "17850",
        "profile": {
            "recency": 30,
            "frequency": 5,
            "avg_order_value": 100.0,
            "segment": "Medium Customers",
            "last_purchased": "RED RETROSPOT WRAP",
            "churn_probability": 0.15,
            "is_churn": 0
        },
        "recommendations": [
            {"stock_code": "85123A", "description": "WHITE HANGING HEART T-LIGHT HOLDER", "unit_price": 2.55, "similarity": 0.85}
        ],
        "campaign_draft": "Mock email campaign draft copy."
    }

    response = client.get("/predict/campaign/17850")
    assert response.status_code == 200
    data = response.json()
    assert data["customer_id"] == "17850"
    assert "profile" in data
    assert "recommendations" in data
    assert "campaign_draft" in data
    assert len(data["recommendations"]) > 0
