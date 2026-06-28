import pandas as pd 
import logging
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report
)
import mlflow

from src.transformation import(
    load_clean_data,
    transform_data
)

from src.rfm_features import compute_rfm

from src.churn_target import create_churn_target

from config.paths import CLEAN_RETAIL


def run_churn_training(
    cutoff_days=180,
    test_size=0.2,
    random_state=42
):

    mlflow.set_experiment(
        "customer_churn"
    )

    logging.info("Loading and transforming data")

    df = load_clean_data(CLEAN_RETAIL)

    df = transform_data(df)

#Creating cutoff date

    max_date = df["invoice_date"].max()

    cutoff_date = max_date - pd.Timedelta(days=cutoff_days)

    logging.info(
        "Cutoff date: %s",
        cutoff_date
    )

#Split past and future datasets
    feature_df = df[
    df["invoice_date"] < cutoff_date
]
    logging.info(
        "Feature rows: %s",
        len(feature_df)
    )

    future_df = df[
    df["invoice_date"] >= cutoff_date
]
    logging.info(
        "Future rows: %s",
        len(future_df)
    )
#Creating RFM features
    rfm_df = compute_rfm(feature_df)

#Creating the target

    target_df = create_churn_target(
    feature_df,
    future_df
)
    logging.info(
        "Target rows: %s",
        len(target_df)
    )
#Merge

    training_df = rfm_df.merge(
    target_df,
    on="customer_id",
    how="inner"
)

#Separating predictor features (past) and target variable (future).
    X = training_df[
    ["recency", "frequency", "avg_order_value"]
]

    y = training_df["churn"]

#Train/Test Split

    X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
#Scaling and mlflow start
    with mlflow.start_run():

        mlflow.log_params(
            {"model_type": "LogisticRegression",
            "cutoff_days": cutoff_days,
            "random_state":random_state,
            "test_size":test_size,
            "scaler":"StandardScaler"
        })

        scaler = StandardScaler()

        X_train_scaled = scaler.fit_transform(X_train)

        X_test_scaled = scaler.transform(X_test)

    #Training
        model = LogisticRegression(
        random_state=42
    )
        model.fit(
        X_train_scaled,
        y_train
    )

    #Prediction
        y_pred = model.predict(
        X_test_scaled
    )

    #Evaluation
        accuracy = accuracy_score(
            y_test,
            y_pred
    )

        report = classification_report(
            y_test,
            y_pred,
            output_dict=True
        )

        mlflow.log_metrics(
            {"accuracy": report["accuracy"],
            "churn_precision": report["1"]["precision"],
            "churn_recall": report["1"]["recall"],
            "churn_f1": report["1"]["f1-score"]}
        )

        coef_df = pd.DataFrame({
            "feature": X.columns,
            "coefficient": model.coef_[0]
        })

        for feature, coef in zip(
            X.columns,
            model.coef_[0]
        ):
            mlflow.log_metric(
                f"coef_{feature}",
                float(coef)
            )

        mlflow.log_metric(
            "churn_rate",
            training_df["churn"].mean()
        )


if __name__ == "__main__":
    run_churn_training()