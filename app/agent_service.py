import os
import logging
import numpy as np
import pandas as pd
import vertexai
from vertexai.language_models import TextEmbeddingModel, TextEmbeddingInput
from vertexai.generative_models import GenerativeModel
from google.cloud import bigquery

# Configure logging
logging.basicConfig(level=logging.INFO)

class MarketingAgentService:
    def __init__(self):
        self.project_id = os.getenv("GCP_PROJECT", "anna-ml-pipeline")
        self.bucket_name = os.getenv("GCS_BUCKET", "anna-ml-pipeline-bucket")
        self.bq_client = bigquery.Client(project=self.project_id)
        
        # Initialize Vertex AI
        vertexai.init(project=self.project_id, location="us-central1")
        self.embedding_model = TextEmbeddingModel.from_pretrained("text-embedding-004")
        self.gemini_model = GenerativeModel("gemini-1.5-flash")

    def get_customer_profile(self, customer_id: str) -> dict:
        """
        Fetches RFM values, cluster segment, last purchased product,
        and recent churn predictions for a customer from BigQuery.
        """
        # Query RFM features from view
        rfm_query = f"""
        SELECT 
            recency, 
            frequency, 
            avg_order_value
        FROM `{self.project_id}.retail_data.rfm_features`
        WHERE CAST(customer_id AS STRING) = @customer_id
        LIMIT 1
        """
        
        job_config = bigquery.QueryJobConfig(
            query_parameters=[bigquery.ScalarQueryParameter("customer_id", "STRING", str(customer_id))]
        )
        
        rfm_df = self.bq_client.query(rfm_query, job_config=job_config).to_dataframe()
        
        if rfm_df.empty:
            logging.warning("Customer ID %s not found in BigQuery rfm_features.", customer_id)
            return None
            
        profile = rfm_df.iloc[0].to_dict()
        
        # Predict segment dynamically using the production segmentation model
        from app.service import predict_cluster
        features_dict = {
            "recency": float(profile["recency"]),
            "frequency": int(profile["frequency"]),
            "avg_order_value": float(profile["avg_order_value"])
        }
        prediction = predict_cluster(features_dict)
        profile["segment"] = prediction[1] if prediction else "Medium Customers"


        
        # Query last purchased product description
        last_product_query = f"""
        SELECT Description as description
        FROM `{self.project_id}.retail_data.transactions`
        WHERE CAST(CustomerID AS STRING) = @customer_id AND Description IS NOT NULL
        ORDER BY InvoiceDate DESC
        LIMIT 1
        """
        
        last_prod_df = self.bq_client.query(last_product_query, job_config=job_config).to_dataframe()
        profile["last_purchased"] = last_prod_df.iloc[0]["description"] if not last_prod_df.empty else "Unknown Product"
        
        # Query churn prediction log
        churn_query = f"""
        SELECT churn_probability, is_churn
        FROM `{self.project_id}.retail_data.churn_predictions_log`
        WHERE customer_id = @customer_id
        ORDER BY created_at DESC
        LIMIT 1
        """
        
        churn_df = self.bq_client.query(churn_query, job_config=job_config).to_dataframe()
        if not churn_df.empty:
            profile["churn_probability"] = float(churn_df.iloc[0]["churn_probability"])
            profile["is_churn"] = int(churn_df.iloc[0]["is_churn"])
        else:
            profile["churn_probability"] = 0.5 if profile["recency"] > 90 else 0.1
            profile["is_churn"] = 1 if profile["recency"] > 90 else 0
            
        return profile

    def find_similar_products(self, query_text: str, limit: int = 3) -> list:
        """
        Generates text embedding of the query, fetches the product catalog from BigQuery,
        and computes Cosine Similarity in Python. Fallbacks to list matches if Vertex fails.
        """
        try:
            # Generate query embedding
            inputs = [TextEmbeddingInput(query_text, "RETRIEVAL_QUERY")]
            query_embedding = self.embedding_model.get_embeddings(inputs)[0].values
            query_vector = np.array(query_embedding)
            
            # Fetch catalog from BigQuery
            catalog_query = f"SELECT stock_code, description, unit_price, embedding FROM `{self.project_id}.retail_data.product_catalog`"
            catalog_df = self.bq_client.query(catalog_query).to_dataframe()
            
            if catalog_df.empty:
                logging.error("Product catalog is empty. Run generate_embeddings.py first.")
                return []
                
            # Compute Cosine Similarity using numpy
            embeddings_matrix = np.vstack(catalog_df["embedding"].values)
            dot_products = np.dot(embeddings_matrix, query_vector)
            norms_matrix = np.linalg.norm(embeddings_matrix, axis=1)
            norm_query = np.linalg.norm(query_vector)
            
            similarities = dot_products / (norms_matrix * norm_query)
            catalog_df["similarity"] = similarities
            
            # Exclude exact match of query description if present, then sort
            catalog_df = catalog_df[catalog_df["description"].str.lower() != query_text.lower()]
            top_matches = catalog_df.sort_values(by="similarity", ascending=False).head(limit)
            
            return top_matches[["stock_code", "description", "unit_price", "similarity"]].to_dict(orient="records")
        except Exception as e:
            logging.warning("Vertex AI vector search failed: %s. Using default recommendations.", e)
            # Safe static fallback recommendations
            return [
                {"stock_code": "85123A", "description": "WHITE HANGING HEART T-LIGHT HOLDER", "unit_price": 2.55, "similarity": 0.85},
                {"stock_code": "22423", "description": "REGENCY CAKESTAND 3 TIER", "unit_price": 12.75, "similarity": 0.72},
                {"stock_code": "47566", "description": "PARTY BUNTING", "unit_price": 4.95, "similarity": 0.68}
            ][:limit]

    def generate_marketing_campaign(self, customer_id: str) -> dict:
        """
        Orchestrates customer context and vector search to draft a highly
        personalized marketing copy using Gemini. Fallbacks to default template if Gemini fails.
        """
        # 1. Fetch customer context
        profile = self.get_customer_profile(customer_id)
        if not profile:
            # Fallback profile for unknown/dummy customer
            profile = {
                "recency": 30,
                "frequency": 5,
                "avg_order_value": 100.0,
                "segment": "Medium Customers",
                "last_purchased": "RED RETROSPOT WRAP",
                "churn_probability": 0.15,
                "is_churn": 0
            }
            
        # 2. Get vector search recommendations
        recommendations = self.find_similar_products(profile["last_purchased"], limit=3)
        
        # 3. Create Gemini Prompt
        rec_list_str = ""
        for idx, rec in enumerate(recommendations, 1):
            rec_list_str += f"{idx}. \"{rec['description']}\" (Price: ${rec['unit_price']:.2f}, Similarity: {rec['similarity']*100:.1f}%)\n"
            
        prompt = f"""
        You are an expert e-commerce marketing agent. Based on the customer profile below, draft a personalized email campaign to re-engage the customer.
        
        CUSTOMER CONTEXT:
        - Customer ID: {customer_id}
        - Segment Profile: {profile['segment']}
        - Churn Probability: {profile['churn_probability']*100:.1f}% (Predicted Churn Status: {"At Risk" if profile['is_churn'] == 1 else "Loyal"})
        - Last Purchased Product: "{profile['last_purchased']}"
        
        RECOMMENDED PRODUCTS (Identified via vector search based on preferences):
        {rec_list_str}
        
        CAMPAIGN INSTRUCTIONS:
        1. Draft a catchy, subject line tailored to their segment.
        2. Write a short, warm, and highly personalized email body.
        3. Address their segment characteristics:
           - If they are predicted "At Risk" of churn, offer a special "We Miss You" 20% discount.
           - If they are a loyal segment, thank them for their loyalty and offer early access to new items.
        4. Integrate the recommended products naturally into the email body, highlighting how they match their style.
        5. Return the response in clean, professional Markdown.
        """
        
        # 4. Generate email with Gemini
        try:
            response = self.gemini_model.generate_content(prompt)
            email_draft = response.text
        except Exception as e:
            logging.warning("Gemini generation failed: %s. Using default mock campaign copy.", e)
            discount_offer = "20% off with code MISSYOU20" if profile["is_churn"] == 1 else "early access with code LOYALTYVIP"
            email_draft = f"""# Subject: Special Offer: Inspired by your recent purchase!

Dear Customer {customer_id},

We hope you are having a wonderful day! We noticed you recently purchased the **"{profile['last_purchased']}"** and loved it. 

Based on your taste, we thought you might enjoy these top recommendations from our collection:
1.  **{recommendations[0]['description']}** — ${recommendations[0]['unit_price']:.2f}
2.  **{recommendations[1]['description']}** — ${recommendations[1]['unit_price']:.2f}
3.  **{recommendations[2]['description']}** — ${recommendations[2]['unit_price']:.2f}

Because you are a valued customer in our **{profile['segment']}** segment, we'd like to offer you a special benefit: **{discount_offer}**!

Enter the code at checkout. We hope to see you again soon!

Best regards,  
*Your E-Commerce Marketing Team*
"""
        
        return {
            "customer_id": customer_id,
            "profile": profile,
            "recommendations": recommendations,
            "campaign_draft": email_draft
        }

