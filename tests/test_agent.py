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

@patch("app.agent_service.vertexai.init")
@patch("app.agent_service.TextEmbeddingModel.from_pretrained")
@patch("app.agent_service.GenerativeModel")
@patch("app.agent_service.bigquery.Client")
def test_multi_agent_internal_pipeline(mock_bq, mock_gen_model, mock_embed_model, mock_vertex):
    from app.agent_service import MarketingAgentService
    
    # Mock Gemini generate_content returning JSON
    mock_model_instance = mock_gen_model.return_value
    mock_response = mock_model_instance.generate_content.return_value
    mock_response.text = '{"review_notes": "Quality check approved.", "final_subject": "Exclusive styles for you!", "final_body": "Hi there, we handpicked these items for you.", "theme": "VIP Retention", "incentive_code": "LOYALTYVIP", "action_plan": "Highlight tea accessories.", "subject": "Exclusive styles for you!", "body": "Hi there!"}'

    service = MarketingAgentService()
    profile = {
        "customer_id": "12345",
        "recency": 45,
        "frequency": 8,
        "avg_order_value": 220.0,
        "spending_velocity": 1.2,
        "cancellation_rate": 0.02,
        "preferred_shopping_hour": 14,
        "segment": "Champions",
        "last_purchased": "SET 3 RETROSPOT TEA TINS",
        "churn_probability": 0.05,
        "is_churn": 0
    }
    recs = [
        {"stock_code": "22423", "description": "REGENCY CAKESTAND 3 TIER", "unit_price": 12.75, "similarity": 0.88}
    ]
    
    diagnosis = service._run_analyst_agent(profile)
    assert isinstance(diagnosis, str)
    
    strategy = service._run_strategist_agent(diagnosis, profile, recs)
    assert isinstance(strategy, str)
    
    draft = service._run_copywriter_agent(strategy, profile, recs)
    assert isinstance(draft, dict)
    assert "subject" in draft
    assert "body" in draft
    
    critic = service._run_critic_agent(draft, profile)
    assert "review_notes" in critic
    assert "final_subject" in critic
    assert "final_body" in critic
