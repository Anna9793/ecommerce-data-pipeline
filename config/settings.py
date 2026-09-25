import os
from functools import lru_cache
from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Centralized Application Configuration implementing The Twelve-Factor App (Factor III: Config in Environment).
    Loads environment variables from OS environment and optional .env file with strict validation and safe defaults.
    """
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False
    )

    # --------------------------------------------------------------------------
    # 1. Google Cloud Platform (GCP) Settings
    # --------------------------------------------------------------------------
    GCP_PROJECT: str = "anna-ml-pipeline"
    GCP_LOCATION: str = "us-central1"
    GCS_BUCKET: str = "anna-ml-pipeline-bucket"
    USE_BIGQUERY: bool = False
    GOOGLE_APPLICATION_CREDENTIALS: Optional[str] = None

    # --------------------------------------------------------------------------
    # 2. Database & Feature Store (PostgreSQL / pgvector / Firestore)
    # --------------------------------------------------------------------------
    POSTGRES_DB: str = "postgres"
    POSTGRES_USER: str = "postgres"
    POSTGRES_PASSWORD: str = "postgres"
    POSTGRES_HOST: str = "localhost"
    POSTGRES_PORT: str = "5432"

    # --------------------------------------------------------------------------
    # 3. Machine Learning & Feature Lineage Tracking
    # --------------------------------------------------------------------------
    MLFLOW_TRACKING_URI: str = "file:./mlruns"
    RFM_FEATURE_VERSION: str = "rfm_v1"
    CHURN_FEATURE_VERSION: str = "churn_v2"
    TWO_TOWER_VERSION: str = "two_tower_v1"

    # --------------------------------------------------------------------------
    # 4. API & Service Configuration
    # --------------------------------------------------------------------------
    API_PORT: int = 8000
    API_HOST: str = "0.0.0.0"
    TEST_DRIFT_ACTIVE: bool = False

    # --------------------------------------------------------------------------
    # 5. Filesystem Directory Layout (Derived from project root)
    # --------------------------------------------------------------------------
    @property
    def BASE_DIR(self) -> Path:
        return Path(__file__).resolve().parent.parent

    @property
    def DATA_DIR(self) -> Path:
        d = self.BASE_DIR / "data"
        d.mkdir(exist_ok=True)
        return d

    @property
    def RAW_DIR(self) -> Path:
        d = self.DATA_DIR / "raw"
        d.mkdir(exist_ok=True)
        return d

    @property
    def PROCESSED_DIR(self) -> Path:
        d = self.DATA_DIR / "processed"
        d.mkdir(exist_ok=True)
        return d

    @property
    def PREDICTIONS_DIR(self) -> Path:
        d = self.DATA_DIR / "predictions"
        d.mkdir(exist_ok=True)
        return d

    @property
    def REPORTS_DIR(self) -> Path:
        d = self.BASE_DIR / "reports"
        d.mkdir(exist_ok=True)
        return d

    @property
    def MODELS_DIR(self) -> Path:
        d = self.BASE_DIR / "models"
        d.mkdir(exist_ok=True)
        return d

    @property
    def CONFIG_DIR(self) -> Path:
        d = self.BASE_DIR / "config"
        d.mkdir(exist_ok=True)
        return d

    # --------------------------------------------------------------------------
    # 6. Canonical Dataset File Paths
    # --------------------------------------------------------------------------
    @property
    def ONLINE_RETAIL_XLSX(self) -> Path:
        return self.RAW_DIR / "Online_Retail.xlsx"

    @property
    def ONLINE_RETAIL_CSV(self) -> Path:
        return self.RAW_DIR / "online_retail.csv"

    @property
    def CLEAN_RETAIL(self) -> Path:
        return self.PROCESSED_DIR / "clean_retail.csv"

    @property
    def FEATURE_RETAIL(self) -> Path:
        return self.PROCESSED_DIR / "feature_retail.csv"

    @property
    def RFM_CUSTOMERS(self) -> Path:
        return self.PROCESSED_DIR / "rfm_customers.csv"

    @property
    def TRAIN_CLUSTERS(self) -> Path:
        return self.PROCESSED_DIR / "rfm_train_clusters.csv"

    @property
    def CLUSTER_PROFILE(self) -> Path:
        return self.PROCESSED_DIR / "cluster_profile.csv"

    @property
    def CUSTOMER_CLUSTERS(self) -> Path:
        return self.PREDICTIONS_DIR / "customer_clusters.csv"

    @property
    def CUSTOMER_CLUSTERS_V2(self) -> Path:
        return self.PREDICTIONS_DIR / "customer_clusters_v2.csv"

    @property
    def CUSTOMER_CLUSTERS_LABELED(self) -> Path:
        return self.PREDICTIONS_DIR / "customer_clusters_labeled.csv"

    @property
    def CUSTOMER_CLUSTERS_DB(self) -> Path:
        return self.PREDICTIONS_DIR / "customer_clusters.db"

    @property
    def EXPERIMENT_CONFIG_PATH(self) -> Path:
        return self.CONFIG_DIR / "experiment.yaml"


@lru_cache()
def get_settings() -> Settings:
    """Cached singleton provider for application settings."""
    return Settings()
