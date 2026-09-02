import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

@patch("app.agent_graph.MarketingGraphOrchestrator")
def test_campaign_graph_endpoint(mock_orch_class):
    mock_orch = mock_orch_class.return_value
    mock_orch.run.return_value = {
        "customer_id": "17850",
        "segment": "High-Value Loyal",
        "churn_risk": "8.5%",
        "subject": "Handcrafted picks for you",
        "body": "Discover new styles curated for your home.",
        "delivery_meta": "Schedule Delivery for 14:00",
        "recommended_products": ["WHITE HANGING HEART T-LIGHT HOLDER"],
        "iterations_required": 1,
        "graph_engine": "LangGraph (StateGraph with Cyclic Feedback)",
        "agent_traces": [
            {"node": "analyst", "title": "Agent 1: Behavioral Analyst", "content": "Healthy buyer."}
        ]
    }

    response = client.get("/predict/campaign-graph/17850")
    assert response.status_code == 200
    data = response.json()
    assert data["customer_id"] == "17850"
    assert data["graph_engine"] == "LangGraph (StateGraph with Cyclic Feedback)"
    assert data["iterations_required"] == 1
    assert "agent_traces" in data

@patch("app.agent_service.vertexai.init")
@patch("app.agent_service.TextEmbeddingModel.from_pretrained")
@patch("app.agent_service.GenerativeModel")
@patch("app.agent_service.bigquery.Client")
def test_langgraph_execution_success(mock_bq, mock_gen_model, mock_embed, mock_vertex):
    from app.agent_graph import MarketingGraphOrchestrator
    
    orch = MarketingGraphOrchestrator()
    orch.agent_service.get_customer_profile = MagicMock(return_value={
        "customer_id": "17850",
        "recency": 10.0,
        "frequency": 8,
        "avg_order_value": 85.0,
        "spending_velocity": 1.2,
        "cancellation_rate": 0.0,
        "preferred_shopping_hour": 14,
        "cluster": 0,
        "label": "High-Value Loyal",
        "churn_probability": 0.05,
        "is_churn": 0
    })
    orch.agent_service.find_similar_products = MagicMock(return_value=[
        {"stock_code": "85123A", "description": "WHITE HANGING HEART T-LIGHT HOLDER", "unit_price": 2.55}
    ])
    
    # Mock LLM generation responses for Strategist, Copywriter, Critic
    mock_model_inst = mock_gen_model.return_value
    
    def mock_generate_content(prompt, generation_config=None):
        mock_resp = MagicMock()
        p = str(prompt)
        if "Strategist" in p or "Chief Commercial" in p:
            mock_resp.text = '{"theme": "VIP Appreciation", "incentive_code": "LOYALTYVIP", "action_plan": "Feature premium candleware"}'
        elif "Copywriter" in p:
            mock_resp.text = '{"subject": "Exclusive VIP picks for your home", "body": "Thank you for being our loyal customer. Enjoy 10% off!"}'
        elif "Critic" in p:
            mock_resp.text = '{"is_approved": true, "review_notes": "Perfect compliance and warm tone.", "critique_feedback": "Approved", "final_subject": "Exclusive VIP picks for your home", "final_body": "Thank you for being our loyal customer. Enjoy 10% off!"}'
        else:
            mock_resp.text = '{"theme": "Loyalty"}'
        return mock_resp

    orch.agent_service.gemini_model.generate_content.side_effect = mock_generate_content

    result = orch.run("17850")
    assert result["customer_id"] == "17850"
    assert result["segment"] == "High-Value Loyal"
    assert result["iterations_required"] == 1
    assert len(result["agent_traces"]) == 4  # analyst, strategist, copywriter, critic

@patch("app.agent_service.vertexai.init")
@patch("app.agent_service.TextEmbeddingModel.from_pretrained")
@patch("app.agent_service.GenerativeModel")
@patch("app.agent_service.bigquery.Client")
def test_langgraph_cyclic_revision_loop(mock_bq, mock_gen_model, mock_embed, mock_vertex):
    from app.agent_graph import MarketingGraphOrchestrator
    
    orch = MarketingGraphOrchestrator()
    orch.agent_service.get_customer_profile = MagicMock(return_value={
        "customer_id": "12347",
        "recency": 40.0,
        "frequency": 3,
        "avg_order_value": 40.0,
        "spending_velocity": 0.8,
        "cancellation_rate": 0.0,
        "preferred_shopping_hour": 12,
        "cluster": 1,
        "label": "Moderate Spenders",
        "churn_probability": 0.65,
        "is_churn": 1
    })
    orch.agent_service.find_similar_products = MagicMock(return_value=[
        {"stock_code": "22423", "description": "REGENCY CAKESTAND 3 TIER", "unit_price": 12.75}
    ])
    
    mock_model_inst = mock_gen_model.return_value
    call_counts = {"critic": 0}
    
    def mock_generate_content_with_rejection(prompt, generation_config=None):
        mock_resp = MagicMock()
        p = str(prompt)
        if "Strategist" in p or "Chief Commercial" in p:
            mock_resp.text = '{"theme": "Win-back Offer", "incentive_code": "WINBACK20", "action_plan": "Feature discount coupon"}'
        elif "Copywriter" in p:
            mock_resp.text = '{"subject": "We miss you! 20% off inside", "body": "Come back and save 20% on our cakestand collection!"}'
        elif "Critic" in p:
            call_counts["critic"] += 1
            if call_counts["critic"] == 1:
                # Cycle 1: Reject due to aggressive tone
                mock_resp.text = '{"is_approved": false, "review_notes": "Tone too urgent and pushy.", "critique_feedback": "Please soften the tone and focus on product quality rather than urgency.", "final_subject": "", "final_body": ""}'
            else:
                # Cycle 2: Approve revised copy
                mock_resp.text = '{"is_approved": true, "review_notes": "Polished, respectful win-back email.", "critique_feedback": "Approved", "final_subject": "A special welcome back treat for you", "final_body": "We thought of you! Enjoy a 20% discount on your next visit."}'
        else:
            mock_resp.text = '{"theme": "Retention"}'
        return mock_resp

    orch.agent_service.gemini_model.generate_content.side_effect = mock_generate_content_with_rejection

    result = orch.run("12347")
    assert result["customer_id"] == "12347"
    assert result["iterations_required"] == 2  # Proves cyclic feedback loop ran!
    assert len(result["agent_traces"]) == 6  # analyst, strategist, copywriter1, critic1, copywriter2, critic2
    assert "special welcome back treat" in result["subject"]
