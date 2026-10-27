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
        "CustomerID":  [101.0,  101.0,  np.nan, 102.0,  103.0,  104.0],
        "Quantity":    [5,      5,      2,      -1,     3,      4],
        "UnitPrice":   [2.5,    2.5,    5.0,    10.0,   0.0,    1.5],
        "InvoiceDate": ["2010-12-01 08:26:00"] * 6
    })

    cleaned = clean_data(raw_df)

    # 1. Columnas estandarizadas a snake_case y tipadas explícitamente
    assert "invoice_no" in cleaned.columns
    assert "customer_id" in cleaned.columns
    assert "invoice_date" in cleaned.columns
    assert pd.api.types.is_datetime64_any_dtype(cleaned["invoice_date"])

    # 2. Solo sobreviven las filas válidas (101 deduplicada y 104) con IDs casteados a string limpio
    assert len(cleaned) == 2
    assert list(cleaned["customer_id"].values) == ["101", "104"]
    assert all(cleaned["quantity"] > 0)
    assert all(cleaned["unit_price"] > 0)
