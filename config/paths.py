"""
Rutas del proyecto (FileSystem Paths).
Gestiona las rutas absolutas y relativas a la carpeta raíz del proyecto.
"""

from pathlib import Path

# 1. Carpeta raíz del proyecto
BASE_DIR = Path(__file__).resolve().parent.parent

# 2. Directorios principales
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
PREDICTIONS_DIR = DATA_DIR / "predictions"
MODELS_DIR = BASE_DIR / "models"
CONFIG_DIR = BASE_DIR / "config"
SQL_DIR = BASE_DIR / "sql"

DIRECTORIES = [
    DATA_DIR,
    RAW_DIR,
    PROCESSED_DIR,
    PREDICTIONS_DIR,
    MODELS_DIR,
    CONFIG_DIR,
    SQL_DIR
]

# Crear directorios si no existen
for directory in DIRECTORIES:
    directory.mkdir(exist_ok=True)

# 3. Datasets del Pipeline (Arquitectura Medallion)
RAW_EXCEL = RAW_DIR / "Online_Retail.xlsx"
BRONZE_RETAIL = RAW_DIR / "online_retail.csv"
SILVER_RETAIL = PROCESSED_DIR / "silver_retail.csv"
GOLD_RFM = PROCESSED_DIR / "gold_rfm.csv"

# 4. Modelos, Perfiles y Configuración
SEGMENTATION_MODEL_PKL = MODELS_DIR / "customer_segmentation_model.pkl"
CHURN_MODEL_PKL = MODELS_DIR / "customer_churn_model.pkl"
CLUSTER_PROFILE = PROCESSED_DIR / "cluster_profile.csv"
CUSTOMER_CLUSTERS = PREDICTIONS_DIR / "customer_clusters.csv"
EXPERIMENT_CONFIG_PATH = CONFIG_DIR / "experiment.yaml"

# 5. Scripts SQL de Infraestructura
BIGQUERY_MEDALLION_SQL = SQL_DIR / "bigquery_medallion.sql"
