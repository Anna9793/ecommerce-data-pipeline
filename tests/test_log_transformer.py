import pandas as pd 
import numpy as np 

from src.features.transformers import LogTransformer

def test_log_transformer_applies_log1p():

    df = pd.DataFrame({
        "frequency": [9],
        "avg_order_value": [99]
    })

    transformer = LogTransformer(
        columns = ["frequency", "avg_order_value"]
    )

    result = transformer.transform(df)

    assert np.isclose(
        result["frequency"].iloc[0],
        np.log1p(9)
    )

    assert np.isclose(
        result["avg_order_value"].iloc[0],
        np.log1p(99)
    )