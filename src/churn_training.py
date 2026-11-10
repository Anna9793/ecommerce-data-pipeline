import os
import sys
import logging
from pathlib import Path
import pandas as pd

# Ensure project root is in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    classification_report
)
from config.paths import SILVER_RETAIL, CHURN_MODEL_PKL
from src.silver import run_silver
from src.gold import compute_gold_features
from src.churn_target import create_churn_target
from src.utils.tracker import VertexExperimentTracker

def run_churn_training(
    cutoff_days=90,
    test_size=0.2,
    random_state=42,
    df=None
):

    tracker = VertexExperimentTracker(experiment_name="customer_churn")

    if df is None:
        logging.info("Loading curated transactions from Silver Layer (%s)", SILVER_RETAIL)
        if not SILVER_RETAIL.exists():
            run_silver()
        df = pd.read_csv(SILVER_RETAIL, parse_dates=["invoice_date"])
    else:
        logging.info("Using provided DataFrame for training")
        if "invoice_date" in df.columns:
            df["invoice_date"] = pd.to_datetime(df["invoice_date"])

    # Creating cutoff date
    max_date = df["invoice_date"].max()
    cutoff_date = max_date - pd.Timedelta(days=cutoff_days)
    logging.info("Cutoff date: %s", cutoff_date)

    # Split past and future datasets
    feature_df = df[df["invoice_date"] < cutoff_date]
    logging.info("Feature rows: %s", len(feature_df))

    future_df = df[df["invoice_date"] >= cutoff_date]
    logging.info("Future rows: %s", len(future_df))

    # Creating RFM features
    rfm_df = compute_gold_features(feature_df)

    # Creating the target
    target_df = create_churn_target(feature_df, future_df)
    logging.info("Target rows: %s", len(target_df))

    # Merge features and target
    training_df = rfm_df.merge(target_df, on="customer_id", how="inner")

    # Separating predictor features (past) and target variable (future)
    X = training_df[["recency", "frequency", "avg_order_value", "spending_velocity", "cancellation_rate", "preferred_shopping_hour"]]
    y = training_df["churn"]

    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    candidate_models = {
        "LogisticRegression": LogisticRegression(random_state=random_state),
        "RandomForest": RandomForestClassifier(random_state=random_state, n_estimators=100),
        "GradientBoosting": GradientBoostingClassifier(random_state=random_state)
    }

    lr_f1 = -1.0
    lr_pipeline = None

    for model_name, model_instance in candidate_models.items():
        logging.info("Training candidate model: %s", model_name)
        
        with tracker.start_run(run_name=model_name):
            # Build training pipeline (scaler + classifier)
            pipeline = Pipeline([
                ("scaler", StandardScaler()),
                ("model", model_instance)
            ])

            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)

            # Evaluation
            report = classification_report(y_test, y_pred, output_dict=True)
            f1 = report["1"]["f1-score"]

            tracker.log_params({
                "model_type": model_name,
                "cutoff_days": cutoff_days,
                "random_state": random_state,
                "test_size": test_size,
                "scaler": "StandardScaler"
            })

            metrics_to_log = {
                "accuracy": float(report["accuracy"]),
                "churn_precision": float(report["1"]["precision"]),
                "churn_recall": float(report["1"]["recall"]),
                "churn_f1": float(f1),
                "churn_rate": float(training_df["churn"].mean())
            }

            # Feature coefficients or importance logging
            fitted_model = pipeline.named_steps["model"]
            if hasattr(fitted_model, "coef_"):
                for feature, coef in zip(X.columns, fitted_model.coef_[0]):
                    metrics_to_log[f"coef_{feature}"] = float(coef)
            elif hasattr(fitted_model, "feature_importances_"):
                for feature, importance in zip(X.columns, fitted_model.feature_importances_):
                    metrics_to_log[f"importance_{feature}"] = float(importance)

            tracker.log_metrics(metrics_to_log)

            # Specifically track Logistic Regression for registry promotion
            if model_name == "LogisticRegression":
                lr_f1 = f1
                lr_pipeline = pipeline

    # Persist and register the Logistic Regression model
    if lr_pipeline is not None:
        logging.info("Registering Logistic Regression model (F1-score: %.4f) as production model", lr_f1)
        
        tracker.save_and_register_model(
            model=lr_pipeline,
            model_name="customer_churn_model",
            local_path=CHURN_MODEL_PKL
        )
            
        logging.info("Logistic Regression model registered under 'customer_churn_model'")
        return {
            "f1_score": float(lr_f1),
            "model_name": "customer_churn_model",
            "status": "registered"
        }

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    run_churn_training()