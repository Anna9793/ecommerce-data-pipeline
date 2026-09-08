"""
Unit Tests for Phase 27: Enterprise MCP Server, Agent-to-Agent (A2A) Protocol & Google Workspace Bridge.
"""

import json
from unittest.mock import MagicMock, patch
import pytest

from src.mcp_server import EnterpriseMCPServer
from src.a2a_orchestrator import (
    A2AMessageEnvelope,
    A2ADispatcher,
    CustomerSupportAgent,
    MarketingRetentionAgent,
    RiskOperationsAgent
)
from app.iap_middleware import IAPSecurityValidator, IAPUserContext


# =============================================================================
# 1. MCP Server Protocol Tests
# =============================================================================

def test_mcp_tools_list():
    """Validates that MCP tools/list returns all required tool definitions and schemas."""
    server = EnterpriseMCPServer(project_id="test-project")
    request = {
        "jsonrpc": "2.0",
        "id": 100,
        "method": "tools/list",
        "params": {}
    }
    response = server.handle_mcp_request(request)

    assert response["jsonrpc"] == "2.0"
    assert response["id"] == 100
    tools = response["result"]["tools"]
    tool_names = [t["name"] for t in tools]

    assert "lookup_customer_rfm" in tool_names
    assert "score_churn_risk" in tool_names
    assert "generate_retention_campaign" in tool_names
    assert "semantic_product_search" in tool_names
    assert "escalate_to_jira" in tool_names


def test_mcp_tools_call_churn_scoring():
    """Validates execution of score_churn_risk tool via MCP."""
    server = EnterpriseMCPServer(project_id="test-project")
    request = {
        "jsonrpc": "2.0",
        "id": 101,
        "method": "tools/call",
        "params": {
            "name": "score_churn_risk",
            "arguments": {"customer_id": "15311"}
        }
    }
    response = server.handle_mcp_request(request)

    assert response["jsonrpc"] == "2.0"
    assert response["result"]["isError"] is False
    content_text = response["result"]["content"][0]["text"]
    data = json.loads(content_text)

    assert data["customer_id"] == "15311"
    assert "churn_probability" in data
    assert data["churn_risk_tier"] == "At Risk"


def test_mcp_tools_call_jira_escalation():
    """Validates execution of Jira escalation tool via MCP."""
    server = EnterpriseMCPServer(project_id="test-project")
    request = {
        "jsonrpc": "2.0",
        "id": 102,
        "method": "tools/call",
        "params": {
            "name": "escalate_to_jira",
            "arguments": {
                "customer_id": "17850",
                "summary": "VIP Customer requested urgent refund",
                "priority": "CRITICAL"
            }
        }
    }
    response = server.handle_mcp_request(request)

    assert response["result"]["isError"] is False
    data = json.loads(response["result"]["content"][0]["text"])
    assert data["status"] == "created"
    assert "ticket_id" in data
    assert data["priority"] == "CRITICAL"
    assert "atlassian.net" in data["jira_url"]


def test_mcp_invalid_tool():
    """Validates error response on unknown tool call."""
    server = EnterpriseMCPServer(project_id="test-project")
    request = {
        "jsonrpc": "2.0",
        "id": 103,
        "method": "tools/call",
        "params": {"name": "non_existent_tool", "arguments": {}}
    }
    response = server.handle_mcp_request(request)
    assert "error" in response
    assert response["error"]["code"] == -32601


# =============================================================================
# 2. Agent-to-Agent (A2A) Bus Tests
# =============================================================================

def test_a2a_dispatcher_support_delegation():
    """Validates multi-agent triage: SupportAgent delegates to Marketing and Ops."""
    dispatcher = A2ADispatcher()

    envelope = A2AMessageEnvelope(
        sender_agent="workspace_sheets_addon",
        recipient_agent="support_agent",
        intent="process_complaint",
        payload={
            "customer_id": "15311",
            "text": "I want to cancel my account immediately.",
            "is_complaint": True,
            "spend": 450.00,
            "priority": "CRITICAL"
        }
    )

    response = dispatcher.dispatch(envelope)

    assert response.sender_agent == "support_agent"
    assert response.payload["status"] == "resolved"
    actions = response.payload["actions"]
    assert len(actions) == 2  # Retention offer generated + Jira ticket created

    action_types = [a["action"] for a in actions]
    assert "retention_offer_requested" in action_types
    assert "jira_ticket_escalated" in action_types


# =============================================================================
# 3. Google IAP Security Middleware Tests
# =============================================================================

def test_iap_security_authenticated_user():
    """Validates parsing and role extraction from Google IAP headers."""
    validator = IAPSecurityValidator()
    headers = {
        "x-goog-authenticated-user-email": "accounts.google.com:anna@paradigma.digital",
        "x-goog-authenticated-user-id": "accounts.google.com:987654321",
        "x-goog-iap-jwt-assertion": "mock-jwt-signature-xyz"
    }

    user = validator.extract_user_from_headers(headers)
    assert user is not None
    assert user.email == "anna@paradigma.digital"
    assert user.role == "admin"
    assert user.is_authenticated is True


def test_iap_security_missing_credentials_rejection():
    """Validates rejection when IAP auth is strictly required and headers are missing."""
    with patch.dict("os.environ", {"REQUIRE_IAP_AUTH": "true"}):
        validator = IAPSecurityValidator()
        mock_request = MagicMock()
        mock_request.headers = {}

        with pytest.raises(Exception) as exc_info:
            validator.validate_request(mock_request)
        assert "401" in str(exc_info.value) or "Unauthorized" in str(exc_info.value)




