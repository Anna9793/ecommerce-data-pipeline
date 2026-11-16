import logging
from typing import Dict, Any, List, Tuple
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

from src.utils.plotting import (
    plot_feature_importance,
    plot_confusion_matrix,
    plot_roc_curve
)


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


def find_optimal_threshold(
    y_true: pd.Series,
    y_prob: Any,
    min_threshold: float = 0.10,
    max_threshold: float = 0.90,
    steps: int = 81
) -> Tuple[float, Dict[str, float]]:
    """
    Scans candidate decision thresholds to find the threshold maximizing the F1-score.
    Returns (optimal_threshold, metrics_dict).
    """
    import numpy as np
    from sklearn.metrics import precision_score, recall_score, f1_score

    best_thresh = 0.50
    best_f1 = -1.0
    best_metrics = {
        "threshold": 0.50,
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0
    }

    thresholds = np.linspace(min_threshold, max_threshold, steps)
    for thresh in thresholds:
        preds = (y_prob >= thresh).astype(int)
        p = float(precision_score(y_true, preds, zero_division=0))
        r = float(recall_score(y_true, preds, zero_division=0))
        f = float(f1_score(y_true, preds, zero_division=0))

        if f > best_f1:
            best_f1 = f
            best_thresh = float(thresh)
            best_metrics = {
                "threshold": round(best_thresh, 4),
                "precision": p,
                "recall": r,
                "f1": f
            }

    return best_thresh, best_metrics


def train_and_evaluate_churn_models(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    feature_cols: list,
    cutoff_days: int,
    test_size: float,
    random_state: int,
    churn_rate: float,
    classification_threshold: float,
    auto_optimize_threshold: bool
) -> Tuple[Pipeline, float, str, List[Dict[str, Any]], Dict[str, Any], float]:
    """Trains candidate classifiers, evaluates benchmarks with threshold tuning, and returns winning model and artifacts."""
    candidate_models = get_candidate_models(random_state)
    best_f1 = -1.0
    best_pipeline = None
    best_model_name = ""
    best_threshold = classification_threshold
    candidate_runs: List[Dict[str, Any]] = []

    for model_name, model_instance in candidate_models.items():
        logging.info("Training candidate model: %s", model_name)
        pipeline = build_pipeline(model_instance)
        pipeline.fit(X_train, y_train)
        y_prob = pipeline.predict_proba(X_test)[:, 1]

        if auto_optimize_threshold:
            model_threshold, _ = find_optimal_threshold(y_test, y_prob)
        else:
            model_threshold = classification_threshold

        y_pred_chosen = (y_prob >= model_threshold).astype(int)
        y_pred_default = (y_prob >= 0.50).astype(int)

        report_chosen = classification_report(y_test, y_pred_chosen, output_dict=True, zero_division=0)
        report_default = classification_report(y_test, y_pred_default, output_dict=True, zero_division=0)
        f1 = float(report_chosen["1"]["f1-score"])

        params = {
            "model_type": model_name,
            "cutoff_days": cutoff_days,
            "random_state": random_state,
            "test_size": test_size,
            "scaler": "StandardScaler",
            "decision_threshold": round(model_threshold, 4),
            "auto_optimize_threshold": auto_optimize_threshold
        }

        metrics = {
            "accuracy": float(report_chosen["accuracy"]),
            "churn_precision": float(report_chosen["1"]["precision"]),
            "churn_recall": float(report_chosen["1"]["recall"]),
            "churn_f1": f1,
            "churn_rate": float(churn_rate),
            "decision_threshold": round(model_threshold, 4),
            "default_f1": float(report_default["1"]["f1-score"]),
            "default_recall": float(report_default["1"]["recall"]),
            "recall_gain": float(report_chosen["1"]["recall"] - report_default["1"]["recall"]),
            **extract_feature_effects(pipeline, feature_cols)
        }

        candidate_runs.append({
            "model_name": model_name,
            "pipeline": pipeline,
            "params": params,
            "metrics": metrics,
            "f1_score": f1
        })

        if f1 > best_f1:
            best_f1 = f1
            best_pipeline = pipeline
            best_model_name = model_name
            best_threshold = model_threshold

    # Generate evaluation and explainability visualizations for winning model using tuned threshold
    y_test_prob = best_pipeline.predict_proba(X_test)[:, 1]
    y_test_pred = (y_test_prob >= best_threshold).astype(int)

    raw_figures = {
        "feature_importance.png": plot_feature_importance(best_pipeline, feature_cols),
        "confusion_matrix.png": plot_confusion_matrix(y_test, y_test_pred),
        "roc_curve.png": plot_roc_curve(y_test, y_test_prob),
    }
    figures = {name: fig for name, fig in raw_figures.items() if fig is not None}

    logging.info(
        "Candidate models evaluated. Best model: %s (F1=%.4f, threshold=%.4f)",
        best_model_name, best_f1, best_threshold
    )
    return best_pipeline, best_f1, best_model_name, candidate_runs, figures, best_threshold