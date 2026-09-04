import os
import sys
import json
import logging
import argparse

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.pubsub_consumer import TransactionConsumer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def consume_stream(subscription_name: str = "retail-transactions-sub", limit: int = 10, timeout: float = 10.0):
    project_id = os.getenv("GCP_PROJECT", "anna-ml-pipeline")
    consumer = TransactionConsumer(project_id=project_id, subscription_name=subscription_name)
    
    logging.info("Pulling up to %d messages from Pub/Sub subscription '%s'...", limit, subscription_name)
    messages = consumer.pull_batch(max_messages=limit, timeout=timeout)
    
    if not messages:
        print("\n📭 No hay mensajes pendientes en la bandeja de Pub/Sub.")
        print("💡 Consejo: Primero publica eventos con:")
        print("   python scripts/simulate_stream.py --tenant nordic_tech --limit 5 --use-pubsub\n")
        return
        
    print(f"\n✨ [CONSUMED {len(messages)} TRANSACTIONS FROM PUB/SUB] ✨")
    for i, msg in enumerate(messages, 1):
        print(f"\n--- Transacción #{i} ---")
        print(json.dumps(msg, indent=2))
    print("---------------------------------------------------\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Consume streaming transactions from Google Cloud Pub/Sub.")
    parser.add_argument("--subscription", type=str, default="retail-transactions-sub")
    parser.add_argument("--limit", "--records", "-n", dest="limit", type=int, default=10)
    parser.add_argument("--timeout", type=float, default=10.0)
    args = parser.parse_args()
    
    consume_stream(subscription_name=args.subscription, limit=args.limit, timeout=args.timeout)
