import os
import sys
import random
import logging
from datetime import datetime

# Ensure project root is in Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from google.cloud import bigquery

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def generate_mock_transactions(mode: str = "standard", num_records: int = 50) -> list:
    """
    Generates mock transactions matching the schema of retail_data.transactions.
    Supports standard, drift_cancellations, and drift_velocity modes.
    """
    # Use real Customer IDs from the dataset to update existing profiles
    customer_ids = ["17850", "13047", "12583", "13748", "15100", "15291", "14688", "17809", "15311", "16098"]
    stock_codes = ["85123A", "71053", "84406B", "22752", "21730", "22633", "22632", "22386", "84029G"]
    descriptions = [
        "WHITE HANGING HEART T-LIGHT HOLDER",
        "WHITE METAL LANTERN",
        "CREAM CUPID HEARTS COAT HANGER",
        "JAM MAKING SET WITH JARS",
        "RED WOOLLY HATTIE SHOES",
        "FELTCRAFT PRINCESS CHARLOTTE DOLL",
        "HAND WARMER RED POLKA DOT",
        "JUMBO BAG PINK POLKA DOT",
        "KNITTED UNION FLAG HOT WATER BOTTLE"
    ]
    countries = ["United Kingdom", "France", "Germany", "Spain", "Netherlands"]
    
    rows = []
    for _ in range(num_records):
        cust_id = random.choice(customer_ids)
        stock_idx = random.randint(0, len(stock_codes) - 1)
        stock = stock_codes[stock_idx]
        desc = descriptions[stock_idx]
        country = random.choice(countries)
        
        # Configure is_cancel depending on simulation mode
        if mode == "drift_cancellations":
            is_cancel = random.random() < 0.40  # 40% cancellations!
        else:
            is_cancel = random.random() < 0.05  # 5% baseline cancellations
            
        if is_cancel:
            qty = -random.randint(1, 5)
            inv_no = f"C{random.randint(536365, 581587)}"
        else:
            if mode == "drift_velocity":
                qty = random.randint(20, 100)  # Heavy volume!
            else:
                qty = random.randint(1, 10)
            inv_no = str(random.randint(536365, 581587))
            
        if mode == "drift_velocity":
            price = round(random.uniform(15.0, 75.0), 2)  # Higher prices
        else:
            price = round(random.uniform(0.5, 12.0), 2)
            
        # Use 2011-12-09 (UCI dataset epoch) to keep the recency max date calculations stable
        inv_date = f"2011-12-09 {random.randint(13, 23)}:{random.randint(10, 59)}:{random.randint(10, 59)}"
        
        rows.append({
            "InvoiceNo": inv_no,
            "StockCode": stock,
            "Description": desc,
            "Quantity": qty,
            "InvoiceDate": inv_date,
            "UnitPrice": price,
            "CustomerID": cust_id,
            "Country": country
        })
        
    return rows

def insert_transactions_to_bq(rows: list, project_id: str = "anna-ml-pipeline") -> int:
    """Streams rows to the BigQuery transactions table."""
    client = bigquery.Client(project=project_id)
    table_id = f"{project_id}.retail_data.transactions"
    
    logging.info("Streaming %s records to BigQuery table: %s", len(rows), table_id)
    errors = client.insert_rows_json(table_id, rows)
    if errors:
        raise RuntimeError(f"BigQuery streaming failed: {errors}")
        
    logging.info("Streaming successfully completed.")
    return len(rows)

def publish_transactions_to_pubsub(rows: list, project_id: str = "anna-ml-pipeline", topic: str = "retail-transactions-topic") -> int:
    """Publishes transaction events to Google Cloud Pub/Sub topic."""
    from src.pubsub_publisher import TransactionPublisher
    publisher = TransactionPublisher(project_id=project_id, topic_name=topic)
    logging.info("Publishing %d transactions via Google Cloud Pub/Sub topic: %s", len(rows), topic)
    message_ids = publisher.publish_batch(rows)
    logging.info("Pub/Sub ingestion complete. %d messages acknowledged.", len(message_ids))
    return len(message_ids)

def generate_shopify_transactions(num_records: int = 10) -> list:
    """Loads realistic Shopify transactions from data/raw/shopify_nordic_store.csv."""
    import pandas as pd
    csv_path = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "shopify_nordic_store.csv")
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        sample = df.sample(min(num_records, len(df)), replace=True)
        return sample.to_dict(orient="records")
    return generate_mock_transactions(num_records=num_records)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Simulate real-time transactions ingestion stream.")
    parser.add_argument("--mode", type=str, default="standard", choices=["standard", "drift_cancellations", "drift_velocity", "pubsub"])
    parser.add_argument("--tenant", type=str, default="giftshop_uk", choices=["giftshop_uk", "nordic_tech", "shopify"])
    parser.add_argument("--records", "--limit", "-n", "--count", "--num-records", dest="records", type=int, default=10, help="Number of records to simulate (alias: --limit, -n, --count)")
    parser.add_argument("--use-pubsub", action="store_true", help="Publish stream to Google Cloud Pub/Sub instead of direct BigQuery insert")
    parser.add_argument("--dry-run", action="store_true", help="Normalize and print sample records locally without attempting GCP network calls")
    args = parser.parse_args()
    
    project = os.getenv("GCP_PROJECT", "anna-ml-pipeline")
    sim_mode = "standard" if args.mode == "pubsub" else args.mode
    
    if args.tenant in ("nordic_tech", "shopify"):
        logging.info("Simulating Shopify transactions for tenant: NordicWear & Tech...")
        raw_rows = generate_shopify_transactions(num_records=args.records)
    else:
        logging.info("Simulating UCI transactions for tenant: GiftShop UK...")
        raw_rows = generate_mock_transactions(mode=sim_mode, num_records=args.records)
    
    from src.schema_adapters import SchemaAdapterFactory
    canonical_rows = [SchemaAdapterFactory.normalize(r).to_dict() for r in raw_rows]
    logging.info("✅ Normalized %d transactions via SchemaAdapterFactory.", len(canonical_rows))

    if args.dry_run:
        import json
        print("\n--- [DRY RUN: Normalized Canonical Transactions Preview] ---")
        for i, row in enumerate(canonical_rows[:5], 1):
            print(f"[{i}] {json.dumps(row, indent=2)}")
        if len(canonical_rows) > 5:
            print(f"... and {len(canonical_rows) - 5} more records.")
        print("------------------------------------------------------------\n")
    elif args.use_pubsub or args.mode == "pubsub":
        publish_transactions_to_pubsub(raw_rows, project_id=project)
    else:
        try:
            insert_transactions_to_bq(canonical_rows, project_id=project)
        except Exception as e:
            logging.info("Note: BigQuery streaming skipped in offline/local mode (%s)", e)

