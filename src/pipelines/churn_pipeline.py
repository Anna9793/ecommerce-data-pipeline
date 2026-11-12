import os
import logging
from pathlib import Path
from typing import Optional, Dict, Any
import pandas as pd

from config.paths import SILVER_RETAIL, CHURN_MODEL_PKL, CHURN_PREDICTIONS, EXPERIMENT_CONFIG_PATH
from config.loader import load_config
from config.schema import ChurnConfig
from src.silver import run_silver
from src.gold import compute_gold_features
from src.churn_models import train_and_evaluate_churn_models
from src.utils.tracker import VertexExperimentTracker
from sklearn.model_selection import train_test_split


def save_churn_outputs(scored_df: pd.DataFrame):
    """Persists batch customer churn predictions to local CSV and BigQuery."""
    logging.info("Saving batch customer churn predictions to %s", CHURN_PREDICTIONS)
    scored_df.to_csv(CHURN_PREDICTIONS, index=False)

    if os.getenv("USE_BIGQUERY", "false").lower() == "true":
        project_id = os.getenv("GCP_PROJECT", "anna-ml-pipeline")
        destination_table = "retail_data.customer_churn_predictions"
        logging.info("Exporting %d churn-scored customers to BigQuery: %s", len(scored_df), destination_table)
        try:
            import pandas_gbq
            bq_df = scored_df.copy()
            if "customer_id" in bq_df.columns:
                bq_df["customer_id"] = bq_df["customer_id"].astype(str)
            pandas_gbq.to_gbq(
                bq_df,
                destination_table=destination_table,
                project_id=project_id,
                if_exists="replace"
            )
            logging.info("Successfully exported churn predictions to BigQuery table %s", destination_table)
        except Exception as e:
            logging.error("Failed to export customer churn predictions to BigQuery: %s", e)
            raise


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
    """Executes the end-to-end customer churn training, benchmarking, and batch prediction pipeline."""
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

    # 2. Train and benchmark candidate models (Pure ML, tracker-agnostic)
    best_pipeline, best_f1, best_model_name, candidate_runs, figures = train_and_evaluate_churn_models(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        feature_cols=effective_features,
        cutoff_days=effective_cutoff,
        test_size=effective_test_size,
        random_state=effective_random_state,
        churn_rate=float(training_df["churn"].mean())
    )

    # 3. Log candidate benchmark runs to Vertex AI Experiments
    for run_info in candidate_runs:
        model_name = run_info["model_name"]
        with tracker.start_run(run_name=model_name):
            tracker.log_params(run_info["params"])
            tracker.log_metrics(run_info["metrics"])

    # 4. Register winning model and figures inside winning model run context
    if best_pipeline is not None:
        with tracker.start_run(run_name="customer_churn_production"):
            tracker.log_params({
                "model_type": "LogisticRegression",
                "cutoff_days": effective_cutoff,
                "random_state": effective_random_state,
                "test_size": effective_test_size
            })
            tracker.log_metrics({
                "churn_f1": float(best_f1)
            })
            for fig_name, fig in figures.items():
                tracker.log_figure(fig, fig_name)

            tracker.save_and_register_model(
                model=best_pipeline,
                model_name="customer_churn_model",
                local_path=CHURN_MODEL_PKL
            )

        # 5. Batch Customer Scoring & Persistence (Local CSV + BigQuery)
        scored_churn = training_df.copy()
        scored_churn["churn_probability"] = best_pipeline.predict_proba(X)[:, 1]
        scored_churn["churn_prediction"] = best_pipeline.predict(X)
        save_churn_outputs(scored_churn)

        logging.info("Customer churn pipeline completed successfully")
        return {
            "status": "registered",
            "model_name": "customer_churn_model",
            "f1_score": float(best_f1)
        }

    return {"status": "failed", "f1_score": 0.0}
