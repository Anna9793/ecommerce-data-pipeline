"""
Event-Driven Agentic AI Retention Trigger & Background Worker.

Consumes streaming transaction and churn trigger events from Google Cloud Pub/Sub,
evaluates churn risk / cancellation status in real time, executes autonomous
LangGraph multi-agent retention workflows (Analyst -> Strategist -> Copywriter <-> Critic),
and persists generated campaigns directly into the Online Feature Store (Firestore / PostgreSQL).
"""

import os
import json
import logging
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List, Tuple, Callable
from google.cloud import pubsub_v1

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("retention_worker")


class RetentionEventClassifier:
    """Classifies streaming transaction and churn events to determine retention triggers."""

    @staticmethod
    def is_retention_event(event: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Evaluates event payload. Returns (should_trigger_retention, reason).
        """
        # 1. Check explicit cancellation flags
        is_canc = event.get("is_cancellation", False)
        invoice_no = str(event.get("InvoiceNo", event.get("invoice_no", "")))
        quantity = float(event.get("Quantity", event.get("quantity", 1)))
        unit_price = float(event.get("UnitPrice", event.get("unit_price", 0)))
        amount = float(event.get("line_item_amount", quantity * unit_price))

        if is_canc or invoice_no.upper().startswith("C") or quantity < 0 or amount < 0:
            return True, "order_cancellation"

        # 2. Check explicit event types
        event_type = str(event.get("event_type", event.get("type", ""))).lower()
        if event_type in ["churn_alert", "cancellation", "retention_trigger", "high_churn_risk"]:
            return True, f"explicit_event_{event_type}"

        # 3. Check ML churn risk thresholds
        churn_prob = float(event.get("churn_probability", event.get("churn_prob", 0.0)))
        if churn_prob >= 0.60:
            return True, f"high_churn_probability_{churn_prob:.2f}"

        churn_tier = str(event.get("churn_risk_tier", event.get("segment", ""))).lower()
        if churn_tier in ["at risk", "churn risk", "high risk", "lost"]:
            return True, f"high_risk_tier_{churn_tier}"

        return False, "regular_transaction"


class RetentionOfferStore:
    """Manages persistence of generated retention campaigns in Firestore and PostgreSQL."""

    def __init__(self, project_id: Optional[str] = None):
        self.project_id = project_id or os.getenv("GCP_PROJECT", "anna-ml-pipeline")
        self.use_bigquery = os.getenv("USE_BIGQUERY", "false").lower() == "true"

    def save_offer(self, customer_id: str, campaign_payload: Dict[str, Any], trigger_reason: str) -> bool:
        """Saves generated retention campaign to Firestore and PostgreSQL."""
        record = {
            "customer_id": str(customer_id),
            "trigger_reason": trigger_reason,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "status": "pending_dispatch",
            "campaign": campaign_payload,
            "subject": campaign_payload.get("subject", "Special Offer For You"),
            "body": campaign_payload.get("body", ""),
            "segment": campaign_payload.get("segment", "Valued Customer"),
            "iterations_required": campaign_payload.get("iterations_required", 1),
            "graph_engine": campaign_payload.get("graph_engine", "LangGraph")
        }

        # 1. Firestore Cloud Persistence
        use_bigquery = self.use_bigquery or os.getenv("USE_BIGQUERY", "false").lower() == "true"
        if use_bigquery:
            try:
                from google.cloud import firestore
                db = firestore.Client(project=self.project_id)
                doc_ref = db.collection("retention_offers").document(str(customer_id))
                doc_ref.set(record, merge=True)
                logger.info("Saved retention offer for customer %s to Firestore.", customer_id)
            except Exception as e:
                logger.warning("Firestore save_offer failed: %s", e)

        # 2. PostgreSQL Local Persistence
        try:
            from app.db_postgres import get_connection
            conn = get_connection()
            if conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        CREATE TABLE IF NOT EXISTS retention_offers (
                            customer_id VARCHAR(64) PRIMARY KEY,
                            trigger_reason VARCHAR(128),
                            subject VARCHAR(256),
                            body TEXT,
                            segment VARCHAR(64),
                            iterations_required INT,
                            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                            payload JSONB
                        );
                    """)
                    cur.execute("""
                        INSERT INTO retention_offers (customer_id, trigger_reason, subject, body, segment, iterations_required, payload)
                        VALUES (%s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (customer_id) DO UPDATE SET
                            trigger_reason = EXCLUDED.trigger_reason,
                            subject = EXCLUDED.subject,
                            body = EXCLUDED.body,
                            segment = EXCLUDED.segment,
                            iterations_required = EXCLUDED.iterations_required,
                            payload = EXCLUDED.payload,
                            created_at = CURRENT_TIMESTAMP;
                    """, (
                        str(customer_id),
                        trigger_reason,
                        record["subject"],
                        record["body"],
                        record["segment"],
                        record["iterations_required"],
                        json.dumps(record)
                    ))
                    conn.commit()
                conn.close()
                logger.info("Saved retention offer for customer %s to PostgreSQL.", customer_id)
        except Exception as e:
            logger.warning("PostgreSQL save_offer failed or skipped: %s", e)

        return True


class RetentionWorker:
    """
    Event-driven subscriber worker that consumes transactions/events from Pub/Sub,
    triggers LangGraph agent workflows on churn/cancellation, and saves offers.
    """

    def __init__(
        self,
        project_id: Optional[str] = None,
        subscription_name: str = "retail-transactions-sub",
        outbound_topic_name: str = "retail-retention-offers-topic"
    ):
        self.project_id = project_id or os.getenv("GCP_PROJECT", "anna-ml-pipeline")
        self.subscription_name = subscription_name
        self.outbound_topic_name = outbound_topic_name
        self.subscription_path = f"projects/{self.project_id}/subscriptions/{self.subscription_name}"
        self.outbound_topic_path = f"projects/{self.project_id}/topics/{self.outbound_topic_name}"
        self.classifier = RetentionEventClassifier()
        self.store = RetentionOfferStore(project_id=self.project_id)
        self.subscriber = pubsub_v1.SubscriberClient()
        self.publisher = pubsub_v1.PublisherClient()

    def process_message_payload(self, message_data: bytes) -> Dict[str, Any]:
        """Decodes and parses message bytes into dictionary."""
        payload_str = message_data.decode("utf-8")
        return json.loads(payload_str)

    def process_event(self, event: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Evaluates event, runs LangGraph on retention trigger, saves offer, and publishes outbound notification.
        """
        customer_id = event.get("CustomerID") or event.get("customer_id") or event.get("Customer ID")
        if not customer_id:
            logger.info("Skipping event with no customer ID.")
            return None

        should_trigger, reason = self.classifier.is_retention_event(event)
        if not should_trigger:
            logger.info("Event for customer %s is a standard transaction. No retention trigger needed.", customer_id)
            return None

        logger.info(
            "⚡ RETENTION TRIGGER ACTIVATED for Customer %s! Reason: %s",
            customer_id,
            reason
        )

        # Execute LangGraph Multi-Agent Orchestrator
        try:
            from app.agent_graph import MarketingGraphOrchestrator
            orchestrator = MarketingGraphOrchestrator()
            campaign = orchestrator.run(str(customer_id))
        except Exception as e:
            logger.warning("LangGraph orchestrator execution error (%s). Generating fallback retention offer.", e)
            campaign = {
                "customer_id": str(customer_id),
                "segment": "At Risk",
                "churn_risk": "High",
                "subject": "We Miss You! Here is an Exclusive 20% Discount",
                "body": f"Dear Customer {customer_id},\nWe noticed your recent cancellation. Use promo code WINBACK20 for 20% off your next order!",
                "delivery_meta": "Immediate Retention Dispatch",
                "recommended_products": ["Premium Gift Set", "Customer Favorite Bundle"],
                "iterations_required": 1,
                "graph_engine": "LangGraph (Fallback Self-Healing Mode)",
                "agent_traces": []
            }

        # Persist generated retention campaign
        self.store.save_offer(str(customer_id), campaign, reason)

        # Dispatch outbound Pub/Sub notification
        self._publish_outbound_offer(str(customer_id), campaign, reason)

        return campaign

    def _publish_outbound_offer(self, customer_id: str, campaign: Dict[str, Any], trigger_reason: str):
        """Publishes the finalized retention offer to outbound Pub/Sub topic for CRM dispatch."""
        try:
            outbound_payload = {
                "customer_id": customer_id,
                "trigger_reason": trigger_reason,
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "campaign": campaign
            }
            data_bytes = json.dumps(outbound_payload).encode("utf-8")
            self.publisher.publish(self.outbound_topic_path, data=data_bytes)
            logger.info("Published outbound retention offer event to %s", self.outbound_topic_path)
        except Exception as e:
            logger.warning("Outbound Pub/Sub publish skipped or failed: %s", e)

    def _subscriber_callback(self, message: pubsub_v1.subscriber.message.Message):
        """Streaming pull callback."""
        try:
            event = self.process_message_payload(message.data)
            self.process_event(event)
            message.ack()
        except Exception as e:
            logger.error("Failed to process streaming retention message: %s", e)
            message.nack()

    def start_listening(self, callback: Optional[Callable] = None):
        """Starts background streaming pull subscription listener."""
        handler = callback or self._subscriber_callback
        future = self.subscriber.subscribe(self.subscription_path, callback=handler)
        logger.info("RetentionWorker listening on %s...", self.subscription_path)
        return future

    def pull_and_process(self, max_messages: int = 10, timeout: float = 5.0) -> List[Dict[str, Any]]:
        """Synchronously pulls and processes a batch of messages from Pub/Sub."""
        processed_campaigns = []
        try:
            response = self.subscriber.pull(
                request={"subscription": self.subscription_path, "max_messages": max_messages},
                timeout=timeout
            )
            if not response.received_messages:
                logger.info("No unread messages in retention subscription.")
                return []

            ack_ids = []
            for msg in response.received_messages:
                event = self.process_message_payload(msg.message.data)
                result = self.process_event(event)
                if result:
                    processed_campaigns.append(result)
                ack_ids.append(msg.ack_id)

            self.subscriber.acknowledge(
                request={"subscription": self.subscription_path, "ack_ids": ack_ids}
            )
            logger.info("Successfully processed and acknowledged %d messages.", len(ack_ids))
        except Exception as e:
            if "DeadlineExceeded" not in str(type(e).__name__) and "timed out" not in str(e).lower():
                logger.error("Error during retention worker pull: %s", e)

        return processed_campaigns
