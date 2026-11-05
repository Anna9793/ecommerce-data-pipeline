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
REPORTS_DIR = BASE_DIR / "reports"
MODELS_DIR = BASE_DIR / "models"
CONFIG_DIR = BASE_DIR / "config"

DIRECTORIES = [
    DATA_DIR,
    RAW_DIR,
    PROCESSED_DIR,
    PREDICTIONS_DIR,
    REPORTS_DIR,
    MODELS_DIR,
    CONFIG_DIR
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
CLUSTER_PROFILE = PROCESSED_DIR / "cluster_profile.csv"
CUSTOMER_CLUSTERS = PREDICTIONS_DIR / "customer_clusters.csv"
EXPERIMENT_CONFIG_PATH = CONFIG_DIR / "experiment.yaml"
