import pandas as pd
import numpy as np
from src.cleaning import clean_data


def test_clean_data_end_to_end():
    """
    Verifica el pipeline completo de limpieza con un mini-dataset sintético:
    - Normaliza nombres CamelCase a snake_case.
    - Elimina duplicados exactos.
    - Elimina registros con CustomerID nulo.
    - Preserva compras válidas y cancelaciones legítimas ('C').
    - Filtra transacciones con precio <= 0 o cantidades negativas anómalas (sin 'C').
    """
    raw_df = pd.DataFrame({
        "InvoiceNo":   ["1001", "1001", "1002", "C1003", "1004", "1005", "1006"],
        "CustomerID":  [101.0,  101.0,  np.nan, 102.0,   103.0,  104.0,  105.0],
        "Quantity":    [5,      5,      2,      -1,      3,      4,      -2],
        "UnitPrice":   [2.5,    2.5,    5.0,    10.0,    0.0,    1.5,    5.0],
        "InvoiceDate": ["2010-12-01 08:26:00"] * 7
    })

    cleaned = clean_data(raw_df)

    # 1. Columnas estandarizadas a snake_case y tipadas explícitamente
    assert "invoice_no" in cleaned.columns
    assert "customer_id" in cleaned.columns
    assert "invoice_date" in cleaned.columns
    assert pd.api.types.is_datetime64_any_dtype(cleaned["invoice_date"])

    # 2. Sobreviven las filas válidas: 101 (compra), 102 (cancelación 'C'), 104 (compra)
    assert len(cleaned) == 3
    assert list(cleaned["customer_id"].values) == ["101", "102", "104"]
    assert "C1003" in cleaned["invoice_no"].values
    assert all(cleaned["unit_price"] > 0)
