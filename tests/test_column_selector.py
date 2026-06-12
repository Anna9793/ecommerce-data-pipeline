import pandas as pd 
from src.features.selection import ColumnSelector

def test_column_selector_returns_requested_columns():

    df = pd.DataFrame({
        "recency": [10, 20],
        "frequency": [5, 3],
        "avg_order_value": [100, 50]
    })

    selector = ColumnSelector(
        columns=["recency", "frequency"]
    )

    result = selector.transform(df)

    assert list(result.columns) == [
        "recency",
        "frequency"
    ]