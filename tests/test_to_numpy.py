import pandas as pd
import numpy as np 

from src.features.to_numpy import ToNumpy

def test_to_numpy_returns_numpy_array():

    df = pd.DataFrame({
        "a": [1, 2],
        "b": [3, 4]
    })

    transformer = ToNumpy()

    result = transformer.transform(df)

    assert isinstance(result, np.ndarray)

    assert result.shape == (2,2)