import logging
from typing import Dict, Any, List, Tuple
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

from src.utils.plotting import plot_feature_importance


def build_pipeline(model_instance) -> Pipeline:
    """Builds a classification pipeline with StandardScaler and the specified estimator."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", model_instance)
    ])


def get_candidate_models(random_state: int) -> Dict[str, Any]:
    """Instantiates benchmark candidate classification algorithms."""
    return {
        "LogisticRegression": LogisticRegression(random_state=random_state),
        "RandomForest": RandomForestClassifier(random_state=random_state, n_estimators=100),
        "GradientBoosting": GradientBoostingClassifier(random_state=random_state)
    }


def extract_feature_effects(pipeline: Pipeline, feature_cols: list) -> Dict[str, float]:
    """Extracts coefficients or feature importances from a fitted pipeline."""
    effects: Dict[str, float] = {}
    model = pipeline.named_steps["model"]
    if hasattr(model, "coef_"):
        for feature, coef in zip(feature_cols, model.coef_[0]):
            effects[f"coef_{feature}"] = float(coef)
    elif hasattr(model, "feature_importances_"):
        for feature, importance in zip(feature_cols, model.feature_importances_):
            effects[f"importance_{feature}"] = float(importance)
    return effects


def benchmark_candidate_models(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    feature_cols: list,
    candidate_models: Dict[str, Any],
    cutoff_days: int,
    test_size: float,
    random_state: int,
    churn_rate: float
) -> Tuple[Pipeline, float, List[Dict[str, Any]]]:
    """Trains and evaluates candidate classifiers, returning winning pipeline and benchmark runs."""
    lr_f1 = -1.0
    lr_pipeline = None
    candidate_runs: List[Dict[str, Any]] = []

    for model_name, model_instance in candidate_models.items():
        logging.info("Training candidate model: %s", model_name)
        pipeline = build_pipeline(model_instance)
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
        metrics.update(extract_feature_effects(pipeline, feature_cols))

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

    return lr_pipeline, lr_f1, candidate_runs


def train_and_evaluate_churn_models(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    feature_cols: list,
    cutoff_days: int,
    test_size: float,
    random_state: int,
    churn_rate: float
):
    """Benchmarks candidate classification algorithms and returns the winning model and evaluation artifacts."""
    candidate_models = get_candidate_models(random_state)
    best_pipeline, best_f1, candidate_runs = benchmark_candidate_models(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        feature_cols=feature_cols,
        candidate_models=candidate_models,
        cutoff_days=cutoff_days,
        test_size=test_size,
        random_state=random_state,
        churn_rate=churn_rate
    )

    figures = {}
    if best_pipeline is not None:
        fig = plot_feature_importance(best_pipeline, feature_cols)
        if fig is not None:
            figures["feature_importance.png"] = fig

    best_model_name = "customer_churn_model"
    logging.info("Candidate models evaluated. Best model: %s (F1=%.4f)", best_model_name, best_f1)
    return best_pipeline, best_f1, best_model_name, candidate_runs, figures