import pandas as pd 

def create_churn_target(feature_df, future_df):

    active_customers = set(
        future_df["customer_id"].unique()
    )

    customers = (
        feature_df["customer_id"]
        .drop_duplicates()
    )

    target_df = pd.DataFrame({
        "customer_id": customers
    })

    target_df["churn"] = (
        ~target_df["customer_id"]
        .isin(active_customers)
    ).astype(int)

    return target_df