import logging
from typing import Optional, Dict, Any, Tuple
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

def train_and_evaluate_churn_models(
    X_train,
    X_test,
    y_train,
    y_test,
    feature_cols,
    tracker,
    cutoff_days,
    test_size,
    random_state,
    churn_rate
):
    """
    Benchmarks multiple candidate classification algorithms (LogisticRegression,
    RandomForest, GradientBoosting) and logs parameters, metrics, and coefficients/importances
    to Vertex AI Experiments.
    """
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
            pipeline = Pipeline([
                ("scaler", StandardScaler()),
                ("model", model_instance)
            ])

            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)

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
                "churn_rate": float(churn_rate)
            }

            fitted_model = pipeline.named_steps["model"]
            if hasattr(fitted_model, "coef_"):
                for feature, coef in zip(feature_cols, fitted_model.coef_[0]):
                    metrics_to_log[f"coef_{feature}"] = float(coef)
            elif hasattr(fitted_model, "feature_importances_"):
                for feature, importance in zip(feature_cols, fitted_model.feature_importances_):
                    metrics_to_log[f"importance_{feature}"] = float(importance)

            tracker.log_metrics(metrics_to_log)

            if model_name == "LogisticRegression":
                lr_f1 = f1
                lr_pipeline = pipeline

    return lr_pipeline, lr_f1, "customer_churn_model"


def run_churn_training(
    cutoff_days=None,
    test_size=None,
    random_state=None,
    features=None,
    df=None,
    config=None
):
    """
    Backward-compatible entry point for churn model training.
    Delegates orchestration to src.pipelines.churn_pipeline.run_churn_pipeline.
    """
    from src.pipelines.churn_pipeline import run_churn_pipeline
    return run_churn_pipeline(
        config=config,
        df=df,
        cutoff_days=cutoff_days,
        test_size=test_size,
        random_state=random_state,
        features=features
    )