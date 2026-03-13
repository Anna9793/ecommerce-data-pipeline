from pathlib import Path

PROJECT_ROOT = Path(".")

DATA_DIR = PROJECT_ROOT/ "data"
REPORTS_DIR = PROJECT_ROOT / "reports"
MODELS_DIR = PROJECT_ROOT / "models"

RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
PREDICTIONS_DIR = DATA_DIR/ "predictions"

DIRECTORIES = [
    DATA_DIR,
    RAW_DIR,
    PREDICTIONS_DIR,
    PROCESSED_DIR,
    REPORTS_DIR,
    MODELS_DIR
]

for directory in DIRECTORIES:
    directory.mkdir(exist_ok=True)


ONLINE_RETAIL_XLSX = RAW_DIR / "Online_Retail.xlsx"
ONLINE_RETAIL_CSV = RAW_DIR / "online_retail.csv"

CLEAN_RETAIL = PROCESSED_DIR / "clean_retail.csv"
FEATURE_RETAIL = PROCESSED_DIR / "feature_retail.csv"

RFM_CUSTOMERS = PROCESSED_DIR / "rfm_customers.csv"

TRAIN_CLUSTERS = PROCESSED_DIR / "rfm_train_clusters.csv"
PREDICTED_CLUSTERS = PROCESSED_DIR / "rfm_predicted_clusters.csv"
CLUSTER_PROFILE = PROCESSED_DIR / "cluster_profile.csv"

CUSTOMER_CLUSTERS = PREDICTIONS_DIR / "customer_clusters.csv"

CLUSTER_PLOT = REPORTS_DIR / "cluster.visualization.png"

BEST_MODEL_PATH = MODELS_DIR / "best_kmeans_pipeline.pkl"