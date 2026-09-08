"""
Unit Tests for Phase 26: Event-Driven Agentic AI Retention Worker.
"""

import json
from unittest.mock import MagicMock, patch
import pytest

from src.retention_worker import (
    RetentionEventClassifier,
    RetentionOfferStore,
    RetentionWorker
)


def test_classifier_cancellation_detection():
    """Validates that order cancellations trigger retention evaluation."""
    classifier = RetentionEventClassifier()

    # Case A: Invoice starting with 'C'
    event_canc_invoice = {
        "InvoiceNo": "C536379",
        "CustomerID": "12345",
        "Quantity": 1,
        "UnitPrice": 10.0
    }
    should_trigger, reason = classifier.is_retention_event(event_canc_invoice)
    assert should_trigger is True
    assert reason == "order_cancellation"

    # Case B: Negative Quantity
    event_neg_qty = {
        "InvoiceNo": "536380",
        "CustomerID": "12345",
        "Quantity": -2,
        "UnitPrice": 15.0
    }
    should_trigger, reason = classifier.is_retention_event(event_neg_qty)
    assert should_trigger is True
    assert reason == "order_cancellation"

    # Case C: Explicit cancellation flag
    event_flag = {
        "CustomerID": "12345",
        "is_cancellation": True
    }
    should_trigger, reason = classifier.is_retention_event(event_flag)
    assert should_trigger is True
    assert reason == "order_cancellation"


def test_classifier_churn_risk_thresholds():
    """Validates that high churn probability and at-risk tiers trigger retention."""
    classifier = RetentionEventClassifier()

    # Case A: High churn probability >= 0.60
    event_high_churn = {
        "CustomerID": "17850",
        "Quantity": 1,
        "UnitPrice": 20.0,
        "churn_probability": 0.85
    }
    should_trigger, reason = classifier.is_retention_event(event_high_churn)
    assert should_trigger is True
    assert "high_churn_probability" in reason

    # Case B: At-risk customer segment
    event_at_risk = {
        "CustomerID": "17850",
        "Quantity": 1,
        "UnitPrice": 20.0,
        "churn_risk_tier": "At Risk"
    }
    should_trigger, reason = classifier.is_retention_event(event_at_risk)
    assert should_trigger is True
    assert "high_risk_tier" in reason


def test_classifier_regular_transaction_ignored():
    """Validates that regular positive purchases do not trigger retention intervention."""
    classifier = RetentionEventClassifier()
    event_normal = {
        "InvoiceNo": "536365",
        "CustomerID": "17850",
        "Quantity": 6,
        "UnitPrice": 2.55,
        "churn_probability": 0.12,
        "churn_risk_tier": "Champions"
    }
    should_trigger, reason = classifier.is_retention_event(event_normal)
    assert should_trigger is False
    assert reason == "regular_transaction"


def test_retention_offer_store():
    """Validates that retention offers are formatted and saved properly."""
    store = RetentionOfferStore(project_id="test-project")
    campaign_payload = {
        "customer_id": "17850",
        "segment": "At Risk",
        "subject": "Exclusive 20% Discount",
        "body": "We want to offer you 20% off your next purchase.",
        "iterations_required": 1,
        "graph_engine": "LangGraph"
    }

    with patch.dict("os.environ", {"USE_BIGQUERY": "true"}):
        with patch("google.cloud.firestore.Client") as mock_firestore:
            mock_db = MagicMock()
            mock_doc = MagicMock()
            mock_db.collection.return_value.document.return_value = mock_doc
            mock_firestore.return_value = mock_db

            success = store.save_offer("17850", campaign_payload, "order_cancellation")
            assert success is True
            mock_doc.set.assert_called_once()


@patch("src.retention_worker.pubsub_v1.PublisherClient")
@patch("src.retention_worker.pubsub_v1.SubscriberClient")
def test_retention_worker_process_event(mock_sub, mock_pub):
    """Validates event-driven retention workflow when cancellation occurs."""
    worker = RetentionWorker(project_id="test-project")

    cancellation_event = {
        "InvoiceNo": "C536379",
        "CustomerID": "17850",
        "Quantity": -1,
        "UnitPrice": 25.0
    }

    with patch.object(worker.store, "save_offer", return_value=True) as mock_save:
        with patch.object(worker, "_publish_outbound_offer") as mock_outbound:
            result = worker.process_event(cancellation_event)

            assert result is not None
            assert result["customer_id"] == "17850"
            assert "subject" in result
            assert "body" in result
            mock_save.assert_called_once()
            mock_outbound.assert_called_once()


@patch("src.retention_worker.pubsub_v1.PublisherClient")
@patch("src.retention_worker.pubsub_v1.SubscriberClient")
def test_retention_worker_pull_and_process(mock_sub, mock_pub):
    """Validates batch pull and acknowledgement of retention events."""
    mock_subscriber_instance = MagicMock()
    mock_sub.return_value = mock_subscriber_instance

    # Mock Pub/Sub pulled message
    mock_msg = MagicMock()
    mock_msg.message.data = json.dumps({
        "InvoiceNo": "C999999",
        "CustomerID": "14444",
        "Quantity": -2,
        "UnitPrice": 50.0
    }).encode("utf-8")
    mock_msg.ack_id = "ack-12345"

    mock_response = MagicMock()
    mock_response.received_messages = [mock_msg]
    mock_subscriber_instance.pull.return_value = mock_response

    worker = RetentionWorker(project_id="test-project")
    
    with patch.object(worker, "process_event", return_value={"status": "offer_generated"}):
        results = worker.pull_and_process(max_messages=5)
        assert len(results) == 1
        mock_subscriber_instance.acknowledge.assert_called_once()
