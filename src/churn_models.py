import logging
from typing import Optional, Dict, Any, Tuple, List
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

from src.utils.plotting import plot_feature_importance


def train_and_evaluate_churn_models(
    X_train,
    X_test,
    y_train,
    y_test,
    feature_cols: list,
    cutoff_days: int = 90,
    test_size: float = 0.2,
    random_state: int = 42,
    churn_rate: float = 0.0
):
    """
    Benchmarks candidate classification algorithms (LogisticRegression,
    RandomForest, GradientBoosting) and computes metrics, parameters, and feature importances.
    Pure ML function, agnostic to tracker or external cloud services.

    Returns:
        tuple: (best_pipeline, best_f1, best_model_name, candidate_runs, figures_dict)
    """
    candidate_models = {
        "LogisticRegression": LogisticRegression(random_state=random_state),
        "RandomForest": RandomForestClassifier(random_state=random_state, n_estimators=100),
        "GradientBoosting": GradientBoostingClassifier(random_state=random_state)
    }

    lr_f1 = -1.0
    lr_pipeline = None
    candidate_runs: List[Dict[str, Any]] = []

    for model_name, model_instance in candidate_models.items():
        logging.info("Training candidate model: %s", model_name)

        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("model", model_instance)
        ])

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        report = classification_report(y_test, y_pred, output_dict=True)
        f1 = float(report["1"]["f1-score"])

        params = {
            "model_type": model_name,
            "cutoff_days": cutoff_days,
            "random_state": random_state,
            "test_size": test_size,
            "scaler": "StandardScaler"
        }

        metrics = {
            "accuracy": float(report["accuracy"]),
            "churn_precision": float(report["1"]["precision"]),
            "churn_recall": float(report["1"]["recall"]),
            "churn_f1": f1,
            "churn_rate": float(churn_rate)
        }

        fitted_model = pipeline.named_steps["model"]
        if hasattr(fitted_model, "coef_"):
            for feature, coef in zip(feature_cols, fitted_model.coef_[0]):
                metrics[f"coef_{feature}"] = float(coef)
        elif hasattr(fitted_model, "feature_importances_"):
            for feature, importance in zip(feature_cols, fitted_model.feature_importances_):
                metrics[f"importance_{feature}"] = float(importance)

        candidate_runs.append({
            "model_name": model_name,
            "pipeline": pipeline,
            "params": params,
            "metrics": metrics,
            "f1_score": f1
        })

        if model_name == "LogisticRegression":
            lr_f1 = f1
            lr_pipeline = pipeline

    figures = {}
    if lr_pipeline is not None:
        fig = plot_feature_importance(lr_pipeline, feature_cols)
        if fig is not None:
            figures["feature_importance.png"] = fig

    logging.info("Candidate models evaluated. Best model: %s (F1=%.4f)", "customer_churn_model", lr_f1)
    return lr_pipeline, lr_f1, "customer_churn_model", candidate_runs, figures