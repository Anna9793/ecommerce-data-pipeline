import pandas as pd
import numpy as np
import pytest
from src.cleaning import (
    remove_duplicates,
    standardize_column_names,
    remove_missing_customer_ids,
    filter_valid_transactions,
    clean_data,
)

def test_remove_duplicates():
    df = pd.DataFrame({
        "invoice_no": ["536365", "536365", "536366"],
        "customer_id": ["17850", "17850", "13047"]
    })
    cleaned = remove_duplicates(df)
    assert len(cleaned) == 2

def test_standardize_column_names():
    df = pd.DataFrame({"InvoiceNo": [1], "CustomerID": [2], "UnitPrice": [3.0]})
    standardized = standardize_column_names(df)
    assert list(standardized.columns) == ["invoice_no", "customer_id", "unit_price"]

def test_remove_missing_customer_ids():
    df = pd.DataFrame({
        "customer_id": ["12345", np.nan, "67890"],
        "unit_price": [10.0, 5.0, 15.0]
    })
    cleaned = remove_missing_customer_ids(df)
    assert len(cleaned) == 2
    assert "12345" in cleaned["customer_id"].values
    assert "67890" in cleaned["customer_id"].values

def test_filter_valid_transactions():
    df = pd.DataFrame({
        "quantity": [5, -1, 10, 2],
        "unit_price": [2.5, 3.0, -1.0, 0.0]
    })
    cleaned = filter_valid_transactions(df)
    assert len(cleaned) == 1
    assert cleaned.iloc[0]["quantity"] == 5
    assert cleaned.iloc[0]["unit_price"] == 2.5

def test_clean_data_end_to_end():
    df = pd.DataFrame({
        "InvoiceNo": ["1", "1", "2", "3", "4", "5"],
        "CustomerID": ["101", "101", np.nan, "102", "103", "104"],
        "Quantity": [5, 5, 2, -1, 3, 4],
        "UnitPrice": [2.0, 2.0, 5.0, 10.0, 0.0, 1.5]
    })
    cleaned = clean_data(df)
    assert len(cleaned) == 2
    assert list(cleaned["customer_id"].values) == ["101", "104"]
