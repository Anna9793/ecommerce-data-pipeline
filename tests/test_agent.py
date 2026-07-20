from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_generate_campaign_endpoint():
    response = client.get("/predict/campaign/17850")
    assert response.status_code == 200
    data = response.json()
    assert data["customer_id"] == "17850"
    assert "profile" in data
    assert "recommendations" in data
    assert "campaign_draft" in data
    assert len(data["recommendations"]) > 0
