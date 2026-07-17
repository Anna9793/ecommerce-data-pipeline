import os
import pandas as pd
import numpy as np
import vertexai
from vertexai.language_models import TextEmbeddingModel, TextEmbeddingInput
from google.cloud import bigquery

# Configure GCP project
project_id = os.getenv("GCP_PROJECT", "anna-ml-pipeline")

def generate_product_catalog():
    print(f"Initializing BigQuery client for project: {project_id}...")
    bq_client = bigquery.Client(project=project_id)
    
    # 1. Fetch top 1000 unique products by sales volume
    query = """
    SELECT 
        Description as description, 
        MAX(StockCode) as stock_code, 
        MAX(UnitPrice) as unit_price, 
        COUNT(*) as sales_volume
    FROM `retail_data.transactions`
    WHERE Description IS NOT NULL
    GROUP BY Description
    ORDER BY sales_volume DESC
    LIMIT 1000
    """
    
    print("Querying transaction history from BigQuery...")
    df = bq_client.query(query).to_dataframe()
    print(f"Retrieved {len(df)} unique products.")
    
    # 2. Initialize Vertex AI Text Embeddings
    print("Initializing Vertex AI...")
    vertexai.init(project=project_id, location="us-central1")
    model = TextEmbeddingModel.from_pretrained("text-embedding-004")
    
    # 3. Generate embeddings in batches of 100
    batch_size = 100
    embeddings = []
    
    print("Generating embeddings for product descriptions...")
    for i in range(0, len(df), batch_size):
        batch = df["description"].iloc[i:i+batch_size].tolist()
        print(f"Processing batch {i // batch_size + 1} / {len(df) // batch_size + 1} (size: {len(batch)})...")
        
        # Wrap inputs
        inputs = [TextEmbeddingInput(text, "RETRIEVAL_DOCUMENT") for text in batch]
        
        # Get embeddings
        batch_embeddings = model.get_embeddings(inputs)
        embeddings.extend([emb.values for emb in batch_embeddings])
        
    df["embedding"] = embeddings
    
    # Clean up dataframe columns
    catalog_df = df[["stock_code", "description", "unit_price", "embedding"]]
    
    # 4. Save to BigQuery table retail_data.product_catalog
    table_id = f"{project_id}.retail_data.product_catalog"
    print(f"Uploading catalog to BigQuery table {table_id}...")
    
    # Configure schema to specify REPEATED FLOAT for embedding
    job_config = bigquery.LoadJobConfig(
        schema=[
            bigquery.SchemaField("stock_code", "STRING"),
            bigquery.SchemaField("description", "STRING"),
            bigquery.SchemaField("unit_price", "FLOAT64"),
            bigquery.SchemaField("embedding", "FLOAT64", mode="REPEATED"),
        ],
        write_disposition="WRITE_TRUNCATE"
    )
    
    job = bq_client.load_table_from_dataframe(catalog_df, table_id, job_config=job_config)
    job.result() # Wait for job to complete
    
    print("Successfully populated retail_data.product_catalog table!")

if __name__ == "__main__":
    generate_product_catalog()
