from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR/ "data"
REPORTS_DIR = BASE_DIR / "reports"
MODELS_DIR = BASE_DIR / "models"
CONFIG_DIR = BASE_DIR/ "config"

RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
PREDICTIONS_DIR = DATA_DIR/ "predictions"

DIRECTORIES = [
    DATA_DIR,
    RAW_DIR,
    PREDICTIONS_DIR,
    PROCESSED_DIR,
    REPORTS_DIR,
    MODELS_DIR,
    CONFIG_DIR
]

for directory in DIRECTORIES:
    directory.mkdir(exist_ok=True)


ONLINE_RETAIL_XLSX = RAW_DIR / "Online_Retail.xlsx"
ONLINE_RETAIL_CSV = RAW_DIR / "online_retail.csv"

CLEAN_RETAIL = PROCESSED_DIR / "clean_retail.csv"
FEATURE_RETAIL = PROCESSED_DIR / "feature_retail.csv"

RFM_CUSTOMERS = PROCESSED_DIR / "rfm_customers.csv"

TRAIN_CLUSTERS = PROCESSED_DIR / "rfm_train_clusters.csv"
CLUSTER_PROFILE = PROCESSED_DIR / "cluster_profile.csv"

CUSTOMER_CLUSTERS = PREDICTIONS_DIR / "customer_clusters.csv"
CUSTOMER_CLUSTERS_V2 = PREDICTIONS_DIR / "customer_clusters_v2.csv"
CUSTOMER_CLUSTERS_LABELED = PREDICTIONS_DIR / "customer_clusters_labeled.csv"
CUSTOMER_CLUSTERS_DB = PREDICTIONS_DIR / "customer_clusters.db"

EXPERIMENT_CONFIG_PATH = CONFIG_DIR / "experiment.yaml"