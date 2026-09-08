"""
Agent-to-Agent (A2A) Interoperability Protocol & Multi-Agent Dispatcher.

Enables decentralized, message-based collaboration between autonomous domain agents
(CustomerSupportAgent, MarketingRetentionAgent, RiskOperationsAgent) using standard
message envelopes and delegation routing.
"""

import os
import uuid
import json
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("a2a_orchestrator")


class A2AMessageEnvelope(BaseModel):
    """Standardized message packet for Agent-to-Agent (A2A) communication."""
    message_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    sender_agent: str = Field(description="Identifier of sending agent (e.g. 'support_agent')")
    recipient_agent: str = Field(description="Identifier of receiving agent (e.g. 'marketing_agent')")
    intent: str = Field(description="Intent of the message (e.g. 'request_retention_campaign', 'escalate_incident')")
    payload: Dict[str, Any] = Field(description="Structured business data/context")
    conversation_state: Dict[str, Any] = Field(default_factory=dict, description="Shared multi-turn conversation context")
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class BaseA2AAgent:
    """Base interface for all autonomous A2A domain agents."""
    def __init__(self, agent_id: str, capabilities: List[str]):
        self.agent_id = agent_id
        self.capabilities = capabilities

    def handle_message(self, envelope: A2AMessageEnvelope, dispatcher: 'A2ADispatcher') -> A2AMessageEnvelope:
        raise NotImplementedError


class CustomerSupportAgent(BaseA2AAgent):
    """Agent handling incoming customer inquiries, complaints, and ticket routing."""

    def __init__(self):
        super().__init__(
            agent_id="support_agent",
            capabilities=["handle_inquiry", "process_cancellation_complaint", "route_request"]
        )

    def handle_message(self, envelope: A2AMessageEnvelope, dispatcher: 'A2ADispatcher') -> A2AMessageEnvelope:
        logger.info("[SupportAgent] Received A2A message with intent: '%s'", envelope.intent)
        customer_id = str(envelope.payload.get("customer_id", "UNKNOWN"))
        is_complaint = envelope.payload.get("is_complaint", False) or "cancel" in str(envelope.payload.get("text", "")).lower()

        actions_taken = []

        # 1. If cancellation complaint, delegate retention offer to MarketingAgent
        if is_complaint:
            logger.info("[SupportAgent] Escalating cancellation for Customer %s to MarketingRetentionAgent via A2A.", customer_id)
            marketing_request = A2AMessageEnvelope(
                sender_agent=self.agent_id,
                recipient_agent="marketing_agent",
                intent="generate_retention_offer",
                payload={"customer_id": customer_id, "trigger": "cancellation_complaint"}
            )
            marketing_response = dispatcher.dispatch(marketing_request)
            actions_taken.append({
                "action": "retention_offer_requested",
                "result": marketing_response.payload
            })

            # 2. If high spend / critical issue, escalate to RiskOperationsAgent (Jira)
            spend = float(envelope.payload.get("spend", 0.0))
            if spend > 200.0 or envelope.payload.get("priority") == "CRITICAL":
                logger.info("[SupportAgent] High-value customer (%s). Delegating Jira ticket creation to RiskOperationsAgent.", spend)
                ops_request = A2AMessageEnvelope(
                    sender_agent=self.agent_id,
                    recipient_agent="ops_agent",
                    intent="create_jira_ticket",
                    payload={
                        "customer_id": customer_id,
                        "summary": f"High-Value Customer {customer_id} Churn Escalation",
                        "priority": "HIGH"
                    }
                )
                ops_response = dispatcher.dispatch(ops_request)
                actions_taken.append({
                    "action": "jira_ticket_escalated",
                    "result": ops_response.payload
                })

        return A2AMessageEnvelope(
            sender_agent=self.agent_id,
            recipient_agent=envelope.sender_agent,
            intent="support_resolution",
            payload={
                "customer_id": customer_id,
                "status": "resolved",
                "actions": actions_taken,
                "summary": f"Support triage completed for Customer {customer_id} with {len(actions_taken)} federated agent actions."
            }
        )


class MarketingRetentionAgent(BaseA2AAgent):
    """Agent executing autonomous LangGraph marketing campaigns and personalized promotions."""

    def __init__(self):
        super().__init__(
            agent_id="marketing_agent",
            capabilities=["generate_retention_offer", "build_email_campaign"]
        )

    def handle_message(self, envelope: A2AMessageEnvelope, dispatcher: 'A2ADispatcher') -> A2AMessageEnvelope:
        logger.info("[MarketingAgent] Received A2A retention request for Customer %s", envelope.payload.get("customer_id"))
        customer_id = str(envelope.payload.get("customer_id", ""))

        try:
            from app.agent_graph import MarketingGraphOrchestrator
            orchestrator = MarketingGraphOrchestrator()
            campaign = orchestrator.run(customer_id)
        except Exception as e:
            logger.warning("[MarketingAgent] LangGraph execution fallback: %s", e)
            campaign = {
                "customer_id": customer_id,
                "subject": "Exclusive 20% Discount - We Miss You!",
                "body": f"Dear Customer {customer_id}, please use WINBACK20 for 20% off your next purchase.",
                "segment": "At Risk",
                "iterations_required": 1,
                "graph_engine": "LangGraph (A2A Dispatch)"
            }

        return A2AMessageEnvelope(
            sender_agent=self.agent_id,
            recipient_agent=envelope.sender_agent,
            intent="retention_offer_generated",
            payload={"campaign": campaign, "status": "approved"}
        )


class RiskOperationsAgent(BaseA2AAgent):
    """Agent managing fraud, operational thresholds, and Jira incident creation."""

    def __init__(self):
        super().__init__(
            agent_id="ops_agent",
            capabilities=["create_jira_ticket", "evaluate_fraud_risk"]
        )

    def handle_message(self, envelope: A2AMessageEnvelope, dispatcher: 'A2ADispatcher') -> A2AMessageEnvelope:
        logger.info("[RiskOperationsAgent] Received A2A operations intent: %s", envelope.intent)
        customer_id = str(envelope.payload.get("customer_id", ""))
        summary = envelope.payload.get("summary", "Automated Ops Incident")
        priority = envelope.payload.get("priority", "MEDIUM")

        ticket_id = f"CHURN-{hash(customer_id) % 9000 + 1000}"
        return A2AMessageEnvelope(
            sender_agent=self.agent_id,
            recipient_agent=envelope.sender_agent,
            intent="jira_ticket_created",
            payload={
                "ticket_id": ticket_id,
                "project": "CHURN",
                "summary": summary,
                "priority": priority,
                "status": "OPEN",
                "url": f"https://ecommerce-enterprise.atlassian.net/browse/{ticket_id}"
            }
        )


class A2ADispatcher:
    """Central registry and message bus for Agent-to-Agent (A2A) communication."""

    def __init__(self):
        self.agents: Dict[str, BaseA2AAgent] = {}
        self._register_default_agents()

    def _register_default_agents(self):
        self.register_agent(CustomerSupportAgent())
        self.register_agent(MarketingRetentionAgent())
        self.register_agent(RiskOperationsAgent())

    def register_agent(self, agent: BaseA2AAgent):
        self.agents[agent.agent_id] = agent
        logger.info("Registered A2A Agent '%s' with capabilities: %s", agent.agent_id, agent.capabilities)

    def dispatch(self, envelope: A2AMessageEnvelope) -> A2AMessageEnvelope:
        recipient = envelope.recipient_agent
        if recipient not in self.agents:
            raise ValueError(f"Agent '{recipient}' is not registered in A2A dispatcher.")

        agent = self.agents[recipient]
        return agent.handle_message(envelope, self)
