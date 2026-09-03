import json
import pytest
from unittest.mock import patch, MagicMock
from src.pubsub_publisher import TransactionPublisher
from src.pubsub_consumer import TransactionConsumer

@patch("src.pubsub_publisher.pubsub_v1.PublisherClient")
def test_publisher_single_transaction(mock_pub_client_cls):
    mock_pub_client = mock_pub_client_cls.return_value
    mock_future = MagicMock()
    mock_future.result.return_value = "msg-12345"
    mock_pub_client.publish.return_value = mock_future

    publisher = TransactionPublisher(project_id="test-project", topic_name="test-topic")
    tx = {
        "InvoiceNo": "581492",
        "StockCode": "85123A",
        "Description": "WHITE HANGING HEART T-LIGHT HOLDER",
        "Quantity": 2,
        "UnitPrice": 2.55,
        "CustomerID": "17850"
    }

    msg_id = publisher.publish_transaction(tx)
    assert msg_id == "msg-12345"
    assert mock_pub_client.publish.called
    topic_arg, data_arg = mock_pub_client.publish.call_args[0]
    assert topic_arg == "projects/test-project/topics/test-topic"
    assert b"17850" in data_arg

@patch("src.pubsub_publisher.pubsub_v1.PublisherClient")
def test_publisher_batch_transactions(mock_pub_client_cls):
    mock_pub_client = mock_pub_client_cls.return_value
    mock_future1 = MagicMock()
    mock_future1.result.return_value = "msg-1"
    mock_future2 = MagicMock()
    mock_future2.result.return_value = "msg-2"
    mock_pub_client.publish.side_effect = [mock_future1, mock_future2]

    publisher = TransactionPublisher(project_id="test-project", topic_name="test-topic")
    txs = [
        {"InvoiceNo": "1", "CustomerID": "101", "UnitPrice": 10.0},
        {"InvoiceNo": "2", "CustomerID": "102", "UnitPrice": 20.0}
    ]

    msg_ids = publisher.publish_batch(txs)
    assert msg_ids == ["msg-1", "msg-2"]
    assert mock_pub_client.publish.call_count == 2

@patch("src.pubsub_consumer.pubsub_v1.SubscriberClient")
def test_consumer_message_payload_parsing(mock_sub_client_cls):
    consumer = TransactionConsumer(project_id="test-project", subscription_name="test-sub")
    raw_payload = json.dumps({
        "InvoiceNo": "581492",
        "CustomerID": "17850",
        "UnitPrice": 2.55,
        "Quantity": 3
    }).encode("utf-8")

    parsed = consumer.process_message_payload(raw_payload)
    assert parsed["CustomerID"] == "17850"
    assert parsed["UnitPrice"] == 2.55
    assert parsed["Quantity"] == 3

@patch("src.pubsub_consumer.pubsub_v1.SubscriberClient")
def test_consumer_callback_acknowledgment(mock_sub_client_cls):
    consumer = TransactionConsumer(project_id="test-project", subscription_name="test-sub")
    consumer._update_online_feature_store = MagicMock()

    mock_msg = MagicMock()
    mock_msg.data = json.dumps({
        "InvoiceNo": "581492",
        "CustomerID": "17850",
        "UnitPrice": 2.55,
        "Quantity": 3
    }).encode("utf-8")

    consumer._default_callback(mock_msg)
    assert mock_msg.ack.called
    assert not mock_msg.nack.called
    consumer._update_online_feature_store.assert_called_once_with(
        "17850",
        {"InvoiceNo": "581492", "CustomerID": "17850", "UnitPrice": 2.55, "Quantity": 3}
    )
