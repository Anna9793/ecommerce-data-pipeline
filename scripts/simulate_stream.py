import os
import random
import logging
from datetime import datetime
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
            
        inv_date = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        
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

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Simulate real-time transactions ingestion stream.")
    parser.add_argument("--mode", type=str, default="standard", choices=["standard", "drift_cancellations", "drift_velocity"])
    parser.add_argument("--records", type=int, default=10)
    args = parser.parse_args()
    
    project = os.getenv("GCP_PROJECT", "anna-ml-pipeline")
    mock_rows = generate_mock_transactions(mode=args.mode, num_records=args.records)
    insert_transactions_to_bq(mock_rows, project_id=project)
