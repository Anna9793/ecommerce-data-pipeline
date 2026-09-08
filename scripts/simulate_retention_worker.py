"""
Interactive Simulation Script for Phase 26: Event-Driven Agentic AI Retention Worker.

Demonstrates how streaming events from Pub/Sub are classified in real-time,
triggering LangGraph multi-agent retention campaigns upon order cancellations
or high churn probability, and persisting results to Firestore / PostgreSQL.
"""

import os
import json
import logging
from datetime import datetime, timezone
from src.retention_worker import RetentionWorker, RetentionEventClassifier

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("simulation")


def run_retention_simulation():
    print("\n" + "=" * 80)
    print("🚀 PHASE 26: STREAMING AGENTIC AI RETENTION WORKER SIMULATION")
    print("=" * 80)

    # Initialize Retention Worker
    worker = RetentionWorker(project_id=os.getenv("GCP_PROJECT", "anna-ml-pipeline"))
    use_bq = os.getenv("USE_BIGQUERY", "false").lower() == "true"
    storage_engine = "Google Cloud Firestore" if use_bq else "PostgreSQL / In-Memory Mock"
    print(f"📡 Worker initialized targeting: {storage_engine}\n")

    # Sample Stream of Events
    sample_events = [
        {
            "description": "Event 1: Standard Checkout (Should be ignored by Retention Worker)",
            "payload": {
                "InvoiceNo": "581475",
                "CustomerID": "17850",
                "Quantity": 3,
                "UnitPrice": 15.00,
                "line_item_amount": 45.00,
                "churn_probability": 0.12,
                "churn_risk_tier": "Champions"
            }
        },
        {
            "description": "Event 2: High Churn Risk Alert (Should TRIGGER Retention Worker)",
            "payload": {
                "InvoiceNo": "581480",
                "CustomerID": "13047",
                "Quantity": 1,
                "UnitPrice": 22.50,
                "line_item_amount": 22.50,
                "churn_probability": 0.85,
                "churn_risk_tier": "At Risk",
                "event_type": "high_churn_risk"
            }
        },
        {
            "description": "Event 3: Sudden Order Cancellation (Should TRIGGER Retention Worker)",
            "payload": {
                "InvoiceNo": "C581484",
                "CustomerID": "15311",
                "Quantity": -5,
                "UnitPrice": 19.99,
                "line_item_amount": -99.95,
                "is_cancellation": True
            }
        }
    ]

    for idx, item in enumerate(sample_events, 1):
        print("-" * 80)
        print(f"📥 [{idx}/3] Processing: {item['description']}")
        print(f"   Payload: {json.dumps(item['payload'])}")

        should_trigger, reason = RetentionEventClassifier.is_retention_event(item["payload"])
        
        if not should_trigger:
            print(f"   ⏩ Decision: Filtered out ({reason}). No expensive LLM call required.")
        else:
            print(f"   ⚡ Decision: TRIGGER ACTIVATED! Reason: '{reason}'")
            print("   🤖 Running LangGraph Multi-Agent Orchestrator (Analyst -> Strategist -> Copywriter <-> Critic)...")
            
            result = worker.process_event(item["payload"])
            
            if result:
                print("\n   ✅ Generated Retention Campaign Result:")
                print(f"      • Customer ID : {result.get('customer_id')}")
                print(f"      • Segment     : {result.get('segment')}")
                print(f"      • Churn Risk  : {result.get('churn_risk')}")
                print(f"      • Subject     : {result.get('subject')}")
                print(f"      • Body Copy   : {result.get('body')}")
                print(f"      • Engine      : {result.get('graph_engine')}")
                print(f"      • Iterations  : {result.get('iterations_required', 1)}")
                print(f"   💾 Persisted offer to: {storage_engine} (Collection/Table: retention_offers)")
                print(f"   📬 Dispatched notification to Pub/Sub: 'retail-retention-offers-topic'")

        print()

    print("=" * 80)
    print("🎉 SIMULATION COMPLETED SUCCESSFULLY")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    run_retention_simulation()
