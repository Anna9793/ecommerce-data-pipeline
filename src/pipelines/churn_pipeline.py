"""
Orchestration pipeline for Customer Churn Prediction.
Coordinates data preparation, candidate model training, evaluation, and model registration.
"""
import logging
from pathlib import Path
from typing import Optional, Dict, Any
import pandas as pd

from config.paths import SILVER_RETAIL, CHURN_MODEL_PKL, EXPERIMENT_CONFIG_PATH
from config.loader import load_config
from config.schema import ChurnConfig
from src.silver import run_silver
from src.gold import compute_gold_features
from src.churn_models import train_and_evaluate_churn_models
from src.utils.tracker import VertexExperimentTracker
from sklearn.model_selection import train_test_split


def create_churn_target(feature_df: pd.DataFrame, future_df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes binary churn target labels based on presence in the future evaluation window.
    Churn = 1 if the customer made 0 purchases in the future window, 0 otherwise.
    """
    active_customers = set(future_df["customer_id"].unique())
    customers = feature_df["customer_id"].drop_duplicates()
    target_df = pd.DataFrame({"customer_id": customers})
    target_df["churn"] = (~target_df["customer_id"].isin(active_customers)).astype(int)
    return target_df


def prepare_churn_data(
    df: Optional[pd.DataFrame] = None,
    cutoff_days: int = 90,
    feature_cols: Optional[list] = None
):
    """
    Loads curated transactions and constructs the churn feature matrix X and target y
    using temporal cutoff splitting to prevent data leakage.
    """
    if df is None:
        logging.info("Loading curated transactions from Silver Layer (%s)", SILVER_RETAIL)
        if not SILVER_RETAIL.exists():
            run_silver()
        df = pd.read_csv(SILVER_RETAIL, parse_dates=["invoice_date"])
    else:
        logging.info("Using provided DataFrame for training")
        if "invoice_date" in df.columns:
            df["invoice_date"] = pd.to_datetime(df["invoice_date"])

    max_date = df["invoice_date"].max()
    cutoff_date = max_date - pd.Timedelta(days=cutoff_days)
    logging.info("Cutoff date for churn split: %s", cutoff_date)

    feature_df = df[df["invoice_date"] < cutoff_date]
    future_df = df[df["invoice_date"] >= cutoff_date]
    logging.info("Historical feature rows: %d | Future target rows: %d", len(feature_df), len(future_df))

    rfm_df = compute_gold_features(feature_df)
    target_df = create_churn_target(feature_df, future_df)
    training_df = rfm_df.merge(target_df, on="customer_id", how="inner")

    cols = feature_cols or [
        "recency", "frequency", "avg_order_value",
        "spending_velocity", "cancellation_rate", "preferred_shopping_hour"
    ]
    X = training_df[cols]
    y = training_df["churn"]
    return X, y, training_df


def run_churn_pipeline(
    config=None,
    df: Optional[pd.DataFrame] = None,
    cutoff_days: Optional[int] = None,
    test_size: Optional[float] = None,
    random_state: Optional[int] = None,
    features: Optional[list] = None
) -> Dict[str, Any]:
    """
    Executes the end-to-end customer churn training pipeline:
    1. Loads configuration from Pydantic schema (with default fallback).
    2. Prepares temporal feature store (Silver -> Gold RFM + Churn Target).
    3. Trains and benchmarks candidate models (Logistic Regression, Random Forest, Gradient Boosting).
    4. Logs metrics and hyperparameters to Vertex AI Experiments.
    5. Persists the winning model artifact locally and registers it in GCS.
    """
    logging.info("Starting Customer Churn training pipeline")

    if config is None:
        try:
            full_config = load_config(EXPERIMENT_CONFIG_PATH)
            churn_cfg = full_config.churn
        except Exception as e:
            logging.warning("Could not load experiment config (%s), falling back to ChurnConfig defaults.", e)
            churn_cfg = ChurnConfig()
    elif hasattr(config, "churn"):
        churn_cfg = config.churn
    elif isinstance(config, ChurnConfig):
        churn_cfg = config
    else:
        churn_cfg = ChurnConfig()

    effective_cutoff = cutoff_days if cutoff_days is not None else churn_cfg.cutoff_days
    effective_test_size = test_size if test_size is not None else churn_cfg.test_size
    effective_random_state = random_state if random_state is not None else churn_cfg.random_state
    effective_features = features if features is not None else churn_cfg.features
    experiment_name = churn_cfg.experiment_name

    tracker = VertexExperimentTracker(experiment_name=experiment_name)

    # 1. Prepare data
    X, y, training_df = prepare_churn_data(
        df=df,
        cutoff_days=effective_cutoff,
        feature_cols=effective_features
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=effective_test_size,
        random_state=effective_random_state,
        stratify=y
    )

    # 2. Train and benchmark candidate models
    best_pipeline, best_f1, best_model_name = train_and_evaluate_churn_models(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        feature_cols=effective_features,
        tracker=tracker,
        cutoff_days=effective_cutoff,
        test_size=effective_test_size,
        random_state=effective_random_state,
        churn_rate=float(training_df["churn"].mean())
    )

    # 3. Persist and register the winning model artifact
    if best_pipeline is not None:
        logging.info(
            "Registering %s (F1-score: %.4f) as production model",
            best_model_name,
            best_f1
        )
        tracker.save_and_register_model(
            model=best_pipeline,
            model_name="customer_churn_model",
            local_path=CHURN_MODEL_PKL
        )

        return {
            "f1_score": float(best_f1),
            "model_name": "customer_churn_model",
            "status": "registered"
        }

    return {"status": "failed", "f1_score": 0.0}
