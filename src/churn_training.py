import pandas as pd 
import logging
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report
)

from src.transformation import(
    load_clean_data,
    transform_data
)

from src.rfm_features import compute_rfm

from src.churn_target import create_churn_target

from config.paths import CLEAN_RETAIL


def run_churn_training():

    logging.info("Loading and transforming data")

    df = load_clean_data(CLEAN_RETAIL)

    df = transform_data(df)

#Creating cutoff date

    max_date = df["invoice_date"].max()

    cutoff_date = max_date - pd.Timedelta(days=90)

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
#Scaling   

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

    print(
        f"Accuracy: {accuracy:.3f}"
)

    print(
    classification_report(
        y_test,
        y_pred
    )
)

    coef_df = pd.DataFrame({
        "feature": X.columns,
        "coefficient": model.coef_[0]
    })

    print(coef_df)

if __name__ == "__main__":
    run_churn_training()