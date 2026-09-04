import os
import json
import logging
from typing import Dict, Any, List, Optional
from google.cloud import pubsub_v1

logging.basicConfig(level=logging.INFO)

class TransactionPublisher:
    """Asynchronous Publisher for Google Cloud Pub/Sub Streaming Ingestion."""

    def __init__(self, project_id: Optional[str] = None, topic_name: str = "retail-transactions-topic"):
        self.project_id = project_id or os.getenv("GCP_PROJECT", "anna-ml-pipeline")
        self.topic_name = topic_name
        self.topic_path = f"projects/{self.project_id}/topics/{self.topic_name}"

        # Configure batch settings for high-throughput streaming
        batch_settings = pubsub_v1.types.BatchSettings(
            max_messages=100,
            max_bytes=1024 * 1024,  # 1 MB
            max_latency=0.05       # 50 ms
        )

        self.publisher = pubsub_v1.PublisherClient(batch_settings=batch_settings)
        logging.info("Initialized Pub/Sub Publisher for topic: %s", self.topic_path)

    def _ensure_topic_exists(self):
        """Creates topic if it does not exist in the GCP project."""
        try:
            self.publisher.get_topic(request={"topic": self.topic_path})
        except Exception:
            try:
                logging.info("Topic %s does not exist. Auto-creating...", self.topic_path)
                self.publisher.create_topic(request={"name": self.topic_path})
                logging.info("✅ Created Pub/Sub topic: %s", self.topic_path)
            except Exception as e:
                logging.warning("Could not auto-create topic %s: %s", self.topic_path, e)

    def publish_transaction(self, transaction: Dict[str, Any], attributes: Optional[Dict[str, str]] = None) -> str:
        """Publishes a single JSON transaction event to Pub/Sub."""
        payload_bytes = json.dumps(transaction, default=str).encode("utf-8")
        attrs = attributes or {}
        
        try:
            future = self.publisher.publish(self.topic_path, payload_bytes, **attrs)
            message_id = future.result(timeout=10)
            return message_id
        except Exception as e:
            if "404" in str(e) or "NotFound" in str(type(e).__name__):
                self._ensure_topic_exists()
                future = self.publisher.publish(self.topic_path, payload_bytes, **attrs)
                return future.result(timeout=10)
            raise

    def publish_batch(self, transactions: List[Dict[str, Any]]) -> List[str]:
        """Publishes a batch of transactions concurrently with self-healing topic creation."""
        self._ensure_topic_exists()
        futures = []
        for tx in transactions:
            payload_bytes = json.dumps(tx, default=str).encode("utf-8")
            futures.append(self.publisher.publish(self.topic_path, payload_bytes))

        message_ids = []
        for future in futures:
            try:
                msg_id = future.result(timeout=15)
                message_ids.append(msg_id)
            except Exception as e:
                logging.error("Failed to publish transaction to Pub/Sub: %s", e)
                
        logging.info("Successfully published %d/%d transactions to %s", len(message_ids), len(transactions), self.topic_name)
        return message_ids
