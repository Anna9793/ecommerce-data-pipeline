import os
import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    classification_report
)
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature

from src.transformation import (
    load_clean_data,
    transform_data
)
from src.rfm_features import compute_rfm
from src.churn_target import create_churn_target
from config.paths import CLEAN_RETAIL

def run_churn_training(
    cutoff_days=90,
    test_size=0.2,
    random_state=42,
    df=None
):

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns"))
    mlflow.set_experiment("customer_churn")

    if df is None:
        logging.info("Loading and transforming data")
        df = load_clean_data(CLEAN_RETAIL)
        df = transform_data(df)
    else:
        logging.info("Using provided DataFrame for training")

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
    rfm_df = compute_rfm(feature_df)

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
    lr_run_id = None

    for model_name, model_instance in candidate_models.items():
        logging.info("Training candidate model: %s", model_name)
        
        with mlflow.start_run(run_name=model_name) as run:
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

            mlflow.log_params({
                "model_type": model_name,
                "cutoff_days": cutoff_days,
                "random_state": random_state,
                "test_size": test_size,
                "scaler": "StandardScaler"
            })

            mlflow.log_metrics({
                "accuracy": report["accuracy"],
                "churn_precision": report["1"]["precision"],
                "churn_recall": report["1"]["recall"],
                "churn_f1": f1,
                "churn_rate": training_df["churn"].mean()
            })

            # Feature coefficients or importance logging
            fitted_model = pipeline.named_steps["model"]
            if hasattr(fitted_model, "coef_"):
                for feature, coef in zip(X.columns, fitted_model.coef_[0]):
                    mlflow.log_metric(f"coef_{feature}", float(coef))
            elif hasattr(fitted_model, "feature_importances_"):
                for feature, importance in zip(X.columns, fitted_model.feature_importances_):
                    mlflow.log_metric(f"importance_{feature}", float(importance))

            # Specifically track Logistic Regression for registry promotion
            if model_name == "LogisticRegression":
                lr_f1 = f1
                lr_pipeline = pipeline
                lr_run_id = run.info.run_id

    # Log and register the Logistic Regression model
    if lr_pipeline is not None:
        logging.info("Registering Logistic Regression model (F1-score: %.4f) as production model", lr_f1)
        
        with mlflow.start_run(run_id=lr_run_id):
            input_example = X_train.head(5)
            predictions = lr_pipeline.predict(input_example)
            signature = infer_signature(input_example, predictions)

            mlflow.sklearn.log_model(
                sk_model=lr_pipeline,
                artifact_path="model",
                input_example=input_example,
                signature=signature
            )

            model_version_info = mlflow.register_model(
                model_uri=f"runs:/{lr_run_id}/model",
                name="customer_churn_model"
            )

            client = mlflow.tracking.MlflowClient()
            client.set_registered_model_alias(
                "customer_churn_model",
                "production",
                model_version_info.version
            )
            
        logging.info("Logistic Regression model registered to MLflow registry under 'customer_churn_model' and tagged 'production'")
        return {
            "f1_score": float(lr_f1),
            "model_version": int(model_version_info.version),
            "run_id": str(lr_run_id)
        }

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    run_churn_training()