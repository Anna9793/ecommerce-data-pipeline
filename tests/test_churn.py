from fastapi.testclient import TestClient
from unittest.mock import patch

from app.main import app

client = TestClient(app)

@patch("app.main.insert_churn_prediction")
@patch("app.main.predict_churn_service")
def test_predict_churn_endpoint_returns_prediction(
    mock_predict,
    mock_insert):
    
    mock_predict.return_value = (1, 0.85, 0.31)

    response = client.post(
        "/predict/churn",
        json={
            "recency": 90,
            "frequency": 2,
            "avg_order_value": 50,
            "spending_velocity": 1.0,
            "cancellation_rate": 0.0,
            "preferred_shopping_hour": 12
        }
    )

    assert response.status_code == 200

    data = response.json()

    assert data["is_churn"] == 1
    assert data["churn_probability"] == 0.85
    assert data["threshold_applied"] == 0.31
    assert mock_insert.called

def test_predict_churn_reject_invalid_input():

    response = client.post(
        "/predict/churn",
        json={
            "recency": "not-a-number",
            "frequency": 2,
            "avg_order_value": 50
        }
    )

    assert response.status_code == 422

@patch("scripts.train_on_vertex.submit_vertex_training_job")
def test_trigger_churn_retraining_endpoint(mock_submit):
    response = client.post("/train/churn")
    assert response.status_code == 200
    assert response.json()["status"] == "success"


@patch("src.pipelines.churn_pipeline.save_churn_outputs")
@patch("src.pipelines.churn_pipeline.VertexExperimentTracker")
@patch("src.pipelines.churn_pipeline.prepare_churn_data")
@patch("src.pipelines.churn_pipeline.train_and_evaluate_churn_models")
def test_run_churn_pipeline_respects_custom_config(
    mock_train_models,
    mock_prepare_data,
    mock_tracker_cls,
    mock_save_churn
):
    import pandas as pd
    import numpy as np
    from unittest.mock import MagicMock
    from config.schema import ChurnConfig
    from src.pipelines.churn_pipeline import run_churn_pipeline

    mock_tracker = MagicMock()
    mock_tracker_cls.return_value = mock_tracker

    custom_cfg = ChurnConfig(
        cutoff_days=45,
        test_size=0.3,
        random_state=123,
        experiment_name="custom-churn-test"
    )

    X_dummy = pd.DataFrame({"recency": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]})
    y_dummy = pd.Series([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    training_dummy = pd.DataFrame({"churn": y_dummy})
    mock_prepare_data.return_value = (X_dummy, y_dummy, training_dummy)

    mock_pipeline = MagicMock()
    mock_pipeline.predict_proba.return_value = np.zeros((len(X_dummy), 2))
    candidate_mock = [{"model_name": "LogisticRegression", "params": {}, "metrics": {}}]
    mock_train_models.return_value = (mock_pipeline, 0.85, "LogisticRegression", candidate_mock, {}, 0.40)

    result = run_churn_pipeline(config=custom_cfg)

    assert result["status"] == "registered"
    assert result["model_name"] == "customer_churn_model"
    assert result["f1_score"] == 0.85
    mock_tracker_cls.assert_called_with(experiment_name="custom-churn-test")
    assert mock_tracker.save_and_register_model.called
    assert mock_tracker.log_classification_metrics.called
    assert mock_save_churn.called


@patch("src.pipelines.churn_pipeline.save_churn_outputs")
@patch("src.pipelines.churn_pipeline.VertexExperimentTracker")
@patch("src.pipelines.churn_pipeline.prepare_churn_data")
@patch("src.pipelines.churn_pipeline.train_and_evaluate_churn_models")
def test_run_churn_pipeline_direct(
    mock_train_models,
    mock_prepare_data,
    mock_tracker_cls,
    mock_save_churn
):
    import pandas as pd
    import numpy as np
    from unittest.mock import MagicMock
    from config.schema import ChurnConfig
    from src.pipelines.churn_pipeline import run_churn_pipeline

    mock_tracker = MagicMock()
    mock_tracker_cls.return_value = mock_tracker

    X_dummy = pd.DataFrame({"recency": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]})
    y_dummy = pd.Series([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    training_dummy = pd.DataFrame({"churn": y_dummy})
    mock_prepare_data.return_value = (X_dummy, y_dummy, training_dummy)

    mock_pipeline = MagicMock()
    mock_pipeline.predict_proba.return_value = np.zeros((len(X_dummy), 2))
    candidate_mock = [{"model_name": "LogisticRegression", "params": {}, "metrics": {}}]
    mock_train_models.return_value = (mock_pipeline, 0.90, "LogisticRegression", candidate_mock, {}, 0.40)

    cfg = ChurnConfig()
    result = run_churn_pipeline(cfg)

    assert result["status"] == "registered"
    assert result["f1_score"] == 0.90
    assert mock_tracker.save_and_register_model.called
    assert mock_tracker.log_classification_metrics.called
    assert mock_save_churn.called


def test_train_and_evaluate_churn_models_direct():
    import pandas as pd
    import numpy as np
    from src.churn_models import train_and_evaluate_churn_models

    np.random.seed(42)
    n = 20
    X_train = pd.DataFrame({
        "recency": np.random.uniform(5, 100, n),
        "frequency": np.random.uniform(1, 20, n),
        "avg_order_value": np.random.uniform(10, 500, n)
    })
    y_train = pd.Series([0, 1] * (n // 2))

    X_test = pd.DataFrame({
        "recency": np.random.uniform(5, 100, 10),
        "frequency": np.random.uniform(1, 20, 10),
        "avg_order_value": np.random.uniform(10, 500, 10)
    })
    y_test = pd.Series([0, 1] * 5)

    features = ["recency", "frequency", "avg_order_value"]
    pipeline, best_f1, best_name, candidate_runs, figures, best_threshold = train_and_evaluate_churn_models(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        feature_cols=features,
        cutoff_days=90,
        test_size=0.2,
        random_state=42,
        churn_rate=0.5,
        classification_threshold=0.40,
        auto_optimize_threshold=True
    )

    assert pipeline is not None
    assert isinstance(best_f1, float)
    assert best_name in ["LogisticRegression", "RandomForest", "GradientBoosting"]
    assert len(candidate_runs) == 3
    assert "feature_importance.png" in figures
    assert "confusion_matrix.png" in figures
    assert "roc_curve.png" in figures
    assert isinstance(best_threshold, float)
    assert 0.0 < best_threshold < 1.0


def test_build_churn_pipeline():
    from sklearn.linear_model import LogisticRegression
    from src.churn_models import build_pipeline

    pipe = build_pipeline(LogisticRegression())
    assert "scaler" in pipe.named_steps
    assert "model" in pipe.named_steps


def test_get_candidate_models():
    from src.churn_models import get_candidate_models

    models = get_candidate_models(42)
    assert "LogisticRegression" in models
    assert "RandomForest" in models
    assert "GradientBoosting" in models


def test_extract_feature_effects():
    import pandas as pd
    from sklearn.linear_model import LogisticRegression
    from src.churn_models import build_pipeline, extract_feature_effects

    X = pd.DataFrame({"feat1": [1, 2, 3, 4], "feat2": [4, 3, 2, 1]})
    y = pd.Series([0, 0, 1, 1])
    pipe = build_pipeline(LogisticRegression())
    pipe.fit(X, y)

    effects = extract_feature_effects(pipe, ["feat1", "feat2"])
    assert "coef_feat1" in effects
    assert "coef_feat2" in effects


def test_prepare_churn_data_direct():
    import pandas as pd
    from src.pipelines.churn_pipeline import prepare_churn_data

    base_date = pd.Timestamp("2026-01-01")
    df = pd.DataFrame({
        "customer_id": ["C1", "C1", "C2", "C2"],
        "invoice_date": [
            base_date,
            base_date + pd.Timedelta(days=10),
            base_date + pd.Timedelta(days=5),
            base_date + pd.Timedelta(days=100)
        ],
        "quantity": [2, 1, 3, 1],
        "unit_price": [10.0, 20.0, 15.0, 50.0],
        "order_value": [20.0, 20.0, 45.0, 50.0],
        "invoice_no": ["INV1", "INV2", "INV3", "INV4"]
    })

    feature_cols = ["recency", "frequency", "avg_order_value"]
    X, y, training_df = prepare_churn_data(
        cutoff_days=30,
        feature_cols=feature_cols,
        df=df
    )

    assert list(X.columns) == feature_cols
    assert len(X) == len(y)
    assert "churn" in training_df.columns
    # C2 bought at Day 100 (future) -> churn = 0
    assert training_df.loc[training_df["customer_id"] == "C2", "churn"].values[0] == 0
    # C1 only bought in past -> churn = 1
    assert training_df.loc[training_df["customer_id"] == "C1", "churn"].values[0] == 1


def test_plot_confusion_matrix():
    import numpy as np
    from src.utils.plotting import plot_confusion_matrix

    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 1, 1, 0])
    fig = plot_confusion_matrix(y_true, y_pred)
    assert fig is not None


def test_plot_roc_curve():
    import numpy as np
    from src.utils.plotting import plot_roc_curve

    y_true = np.array([0, 1, 0, 1])
    y_prob = np.array([0.1, 0.8, 0.4, 0.9])
    fig = plot_roc_curve(y_true, y_prob)
    assert fig is not None


def test_plot_roc_curve_single_class():
    import numpy as np
    from src.utils.plotting import plot_roc_curve

    y_true = np.array([0, 0, 0, 0])
    y_prob = np.array([0.1, 0.2, 0.3, 0.4])
    fig = plot_roc_curve(y_true, y_prob)
    assert fig is None


def test_find_optimal_threshold():
    import numpy as np
    import pandas as pd
    from src.churn_models import find_optimal_threshold

    y_true = pd.Series([0, 0, 1, 1, 1])
    y_prob = np.array([0.1, 0.2, 0.45, 0.7, 0.9])

    thresh, metrics = find_optimal_threshold(y_true, y_prob)
    assert isinstance(thresh, float)
    assert 0.1 <= thresh <= 0.9
    assert metrics["f1"] > 0.0


@patch("app.main.insert_churn_prediction")
@patch("app.main.predict_churn_service")
def test_predict_churn_endpoint_custom_threshold(mock_predict, mock_insert):
    mock_predict.return_value = (1, 0.45, 0.40)
    response = client.post(
        "/predict/churn",
        json={
            "recency": 90,
            "frequency": 2,
            "avg_order_value": 50,
            "spending_velocity": 1.0,
            "cancellation_rate": 0.0,
            "preferred_shopping_hour": 12,
            "threshold": 0.40
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert data["is_churn"] == 1
    assert data["churn_probability"] == 0.45
    assert data["threshold_applied"] == 0.40


def test_predict_churn_service_threshold_logic():
    from app.service import predict_churn_service
    features = {
        "recency": 90.0,
        "frequency": 2.0,
        "avg_order_value": 50.0,
        "spending_velocity": 1.0,
        "cancellation_rate": 0.0,
        "preferred_shopping_hour": 12
    }
    is_churn_high, prob_high, applied_high = predict_churn_service(features, threshold=0.999)
    assert applied_high == 0.999
    assert is_churn_high == 0

    is_churn_low, prob_low, applied_low = predict_churn_service(features, threshold=0.001)
    assert applied_low == 0.001
    assert is_churn_low == 1

