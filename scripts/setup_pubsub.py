import os
import sys
import logging
from google.cloud import pubsub_v1

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def setup_pubsub_infrastructure(project_id: str = "anna-ml-pipeline"):
    """Provisions Pub/Sub topics, subscriptions, and Dead-Letter Queues."""
    publisher = pubsub_v1.PublisherClient()
    subscriber = pubsub_v1.SubscriberClient()

    main_topic = f"projects/{project_id}/topics/retail-transactions-topic"
    dlq_topic = f"projects/{project_id}/topics/retail-transactions-dead-letter-topic"
    main_sub = f"projects/{project_id}/subscriptions/retail-transactions-sub"
    dlq_sub = f"projects/{project_id}/subscriptions/retail-transactions-dead-letter-sub"

    # 1. Create Dead-Letter Topic
    try:
        publisher.get_topic(topic=dlq_topic)
        logging.info("DLQ Topic already exists: %s", dlq_topic)
    except Exception:
        publisher.create_topic(name=dlq_topic)
        logging.info("✅ Created DLQ Topic: %s", dlq_topic)

    # 2. Create DLQ Subscription
    try:
        subscriber.get_subscription(subscription=dlq_sub)
        logging.info("DLQ Subscription already exists: %s", dlq_sub)
    except Exception:
        subscriber.create_subscription(name=dlq_sub, topic=dlq_topic)
        logging.info("✅ Created DLQ Subscription: %s", dlq_sub)

    # 3. Create Main Topic
    try:
        publisher.get_topic(topic=main_topic)
        logging.info("Main Topic already exists: %s", main_topic)
    except Exception:
        publisher.create_topic(name=main_topic)
        logging.info("✅ Created Main Topic: %s", main_topic)

    # 4. Create Main Worker Subscription with DLQ Policy
    try:
        subscriber.get_subscription(subscription=main_sub)
        logging.info("Main Subscription already exists: %s", main_sub)
    except Exception:
        dead_letter_policy = pubsub_v1.types.DeadLetterPolicy(
            dead_letter_topic=dlq_topic,
            max_delivery_attempts=5
        )
        subscriber.create_subscription(
            name=main_sub,
            topic=main_topic,
            dead_letter_policy=dead_letter_policy,
            ack_deadline_seconds=20
        )
        logging.info("✅ Created Main Subscription with DLQ policy: %s", main_sub)

if __name__ == "__main__":
    project = os.getenv("GCP_PROJECT", "anna-ml-pipeline")
    setup_pubsub_infrastructure(project_id=project)
