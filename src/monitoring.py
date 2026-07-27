import os
import logging
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from config.paths import RFM_CUSTOMERS

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def calculate_feature_drift(project_id: str = "anna-ml-pipeline") -> dict:
    """
    Computes Kolmogorov-Smirnov test to detect data drift between the training baseline
    dataset and the live streaming transactions in BigQuery.
    """
    use_bigquery = os.getenv("USE_BIGQUERY", "false").lower() == "true"
    
    # 1. Load Baseline (Training) Dataset
    if os.path.exists(RFM_CUSTOMERS):
        baseline_df = pd.read_csv(RFM_CUSTOMERS)
    else:
        logging.warning("Baseline RFM file not found at %s. Creating a mock baseline for safety.", RFM_CUSTOMERS)
        # Fallback mock baseline matching original distributions
        baseline_df = pd.DataFrame({
            "recency": np.random.exponential(scale=50, size=500),
            "frequency": np.random.geometric(p=0.2, size=500),
            "avg_order_value": np.random.lognormal(mean=3.5, sigma=0.8, size=500),
            "spending_velocity": np.random.normal(loc=1.0, scale=0.2, size=500),
            "cancellation_rate": np.random.beta(a=0.5, b=5, size=500),
            "preferred_shopping_hour": np.random.randint(0, 24, size=500)
        })

    # 2. Load Target (Live) Dataset
    if use_bigquery:
        try:
            logging.info("Querying live RFM features from BigQuery view...")
            query = f"SELECT * FROM `{project_id}.retail_data.rfm_features`"
            from google.cloud import bigquery
            client = bigquery.Client(project=project_id)
            target_df = client.query(query).to_dataframe()
        except Exception as e:
            logging.error("Failed to query live RFM data from BigQuery: %s", e)
            target_df = pd.DataFrame()
    else:
        # Local test mode: Use baseline or shift it depending on environment
        logging.info("Local mode: generating target dataset from baseline.")
        target_df = baseline_df.copy()
        
        # Inject artificial drift locally if test environment is flagged
        if os.getenv("TEST_DRIFT_ACTIVE", "false").lower() == "true":
            if "cancellation_rate" in target_df.columns:
                target_df["cancellation_rate"] = target_df["cancellation_rate"] + 0.35
            target_df["avg_order_value"] = target_df["avg_order_value"] * 2.5

    if target_df.empty:
        logging.warning("Target dataset is empty. Cannot perform drift check.")
        return {
            "status": "No Target Data Available",
            "drift_detected": False,
            "features": {}
        }

    # 3. Perform Kolmogorov-Smirnov Statistical Tests
    features_to_check = [
        "recency",
        "frequency",
        "avg_order_value",
        "spending_velocity",
        "cancellation_rate",
        "preferred_shopping_hour"
    ]
    
    results = {}
    drift_detected = False
    
    for feature in features_to_check:
        if feature not in baseline_df.columns or feature not in target_df.columns:
            logging.warning("Feature %s missing from baseline or target. Skipping.", feature)
            continue
            
        baseline_vals = baseline_df[feature].dropna().values
        target_vals = target_df[feature].dropna().values
        
        if len(baseline_vals) < 5 or len(target_vals) < 5:
            logging.warning("Insufficient samples for feature %s. Skipping.", feature)
            continue
            
        # Run two-sample K-S test
        stat, p_val = ks_2samp(baseline_vals, target_vals)
        
        # p-value < 0.05 rejects the null hypothesis that distributions are identical
        has_drifted = bool(p_val < 0.05)
        
        if has_drifted:
            drift_detected = True
            
        results[feature] = {
            "ks_statistic": float(stat),
            "p_value": float(p_val),
            "drifted": has_drifted,
            "baseline_mean": float(np.mean(baseline_vals)),
            "target_mean": float(np.mean(target_vals)),
            "baseline_values": list(map(float, baseline_vals)),
            "target_values": list(map(float, target_vals))
        }
        
    return {
        "status": "Drift Detected" if drift_detected else "Healthy",
        "drift_detected": drift_detected,
        "features": results
    }

if __name__ == "__main__":
    import json
    drift_report = calculate_feature_drift()
    print(json.dumps(drift_report, indent=2))
