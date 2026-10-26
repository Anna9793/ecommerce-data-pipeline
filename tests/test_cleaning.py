import pandas as pd
import numpy as np
from src.cleaning import clean_data


def test_clean_data_end_to_end():
    """
    Verifica el pipeline completo de limpieza con un mini-dataset sintético:
    - Normaliza nombres CamelCase a snake_case.
    - Elimina duplicados exactos.
    - Elimina registros con CustomerID nulo.
    - Filtra transacciones con cantidad negativa (cancelaciones) o precio <= 0.
    """
    raw_df = pd.DataFrame({
        "InvoiceNo":   ["1001", "1001", "1002", "1003", "1004", "1005"],
        "CustomerID":  ["101",  "101",  np.nan, "102",  "103",  "104"],
        "Quantity":    [5,      5,      2,      -1,     3,      4],
        "UnitPrice":   [2.5,    2.5,    5.0,    10.0,   0.0,    1.5]
    })

    cleaned = clean_data(raw_df)

    # 1. Columnas estandarizadas a snake_case
    assert "invoice_no" in cleaned.columns
    assert "customer_id" in cleaned.columns

    # 2. Solo sobreviven las filas válidas (101 deduplicada y 104)
    assert len(cleaned) == 2
    assert list(cleaned["customer_id"].values) == ["101", "104"]
    assert all(cleaned["quantity"] > 0)
    assert all(cleaned["unit_price"] > 0)
