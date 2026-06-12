import pandas as pd 

def create_churn_target(df):

    max_date = df["invoice_date"].max()

    cutoff_date = max_date - pd.Timedelta(days=90)

    last_purchase = (
        df.groupby("customer_id")["invoice_date"]
        .max()
        .reset_index()
    )

    last_purchase["churn"] = (
        last_purchase["invoice_date"] < cutoff_date
    ).astype(int)

    return last_purchase[
        ["customer_id", "churn"]
    ]