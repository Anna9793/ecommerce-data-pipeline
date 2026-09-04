import os
import json
import logging
from typing import Optional, Callable
from google.cloud import pubsub_v1

logging.basicConfig(level=logging.INFO)

class TransactionConsumer:
    """Streaming Subscriber Consumer for Google Cloud Pub/Sub transactions."""

    def __init__(
        self,
        project_id: Optional[str] = None,
        subscription_name: str = "retail-transactions-sub"
    ):
        self.project_id = project_id or os.getenv("GCP_PROJECT", "anna-ml-pipeline")
        self.subscription_name = subscription_name
        self.subscription_path = f"projects/{self.project_id}/subscriptions/{self.subscription_name}"
        self.subscriber = pubsub_v1.SubscriberClient()
        self.topic_path = f"projects/{self.project_id}/topics/retail-transactions-topic"
        logging.info("Initialized Pub/Sub Consumer for subscription: %s", self.subscription_path)
        self._ensure_subscription_exists()

    def _ensure_subscription_exists(self):
        """Creates subscription if it does not exist in the GCP project."""
        try:
            self.subscriber.get_subscription(request={"subscription": self.subscription_path})
        except Exception:
            try:
                logging.info("Subscription %s does not exist. Auto-creating...", self.subscription_path)
                self.subscriber.create_subscription(
                    request={"name": self.subscription_path, "topic": self.topic_path}
                )
                logging.info("✅ Created Pub/Sub subscription: %s", self.subscription_path)
            except Exception as e:
                logging.warning("Could not auto-create subscription %s: %s", self.subscription_path, e)

    def process_message_payload(self, message_data: bytes) -> dict:
        """Parses and validates incoming message bytes."""
        payload_str = message_data.decode("utf-8")
        transaction = json.loads(payload_str)
        return transaction

    def _default_callback(self, message: pubsub_v1.subscriber.message.Message):
        """Processes transaction event and updates the Online Feature Store."""
        try:
            transaction = self.process_message_payload(message.data)
            customer_id = transaction.get("CustomerID") or transaction.get("customer_id")
            
            if customer_id:
                logging.info(
                    "Consumed transaction event for Customer %s | Invoice %s | Amount: $%.2f",
                    customer_id,
                    transaction.get("InvoiceNo", "N/A"),
                    float(transaction.get("UnitPrice", 0)) * float(transaction.get("Quantity", 1))
                )
                
                # Update Online Feature Store if applicable
                self._update_online_feature_store(customer_id, transaction)

            message.ack()
        except Exception as e:
            logging.error("Error processing Pub/Sub message: %s", e)
            message.nack()

    def _update_online_feature_store(self, customer_id: str, transaction: dict):
        """Updates low-latency customer features in Firestore/PostgreSQL."""
        use_bigquery = os.getenv("USE_BIGQUERY", "false").lower() == "true"
        
        if use_bigquery:
            try:
                from google.cloud import firestore
                db = firestore.Client(project=self.project_id)
                doc_ref = db.collection("online_customer_features").document(str(customer_id))
                doc_ref.set({
                    "customer_id": str(customer_id),
                    "last_transaction": transaction,
                    "last_updated": firestore.SERVER_TIMESTAMP
                }, merge=True)
            except Exception as e:
                logging.warning("Firestore feature store update failed: %s", e)
        else:
            try:
                from app.db_postgres import get_db_connection
                conn = get_db_connection()
                if conn:
                    with conn.cursor() as cur:
                        cur.execute("""
                            UPDATE online_customer_features
                            SET recency = 0.0,
                                frequency = frequency + 1
                            WHERE customer_id = %s;
                        """, (str(customer_id),))
                        conn.commit()
                    conn.close()
            except Exception as e:
                logging.warning("PostgreSQL feature store update failed: %s", e)

    def start_listening(self, callback: Optional[Callable] = None, max_messages: Optional[int] = None):
        """Starts the streaming pull worker."""
        handler = callback or self._default_callback
        streaming_pull_future = self.subscriber.subscribe(self.subscription_path, callback=handler)
        logging.info("Listening for messages on %s...", self.subscription_path)
        return streaming_pull_future
