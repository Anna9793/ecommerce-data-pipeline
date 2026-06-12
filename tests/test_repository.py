import pandas as pd

from app.repository import get_total_predictions
from app.repository import get_latest_predictions
from app.repository import get_segment_distribution

def test_get_total_predictions_returns_integer():

    count = get_total_predictions()

    assert isinstance(count, int)

    assert count >= 0

def test_get_latest_predictions_returns_dataframe():

    df = get_latest_predictions()

    assert isinstance(df, pd.DataFrame)

    assert "cluster" in df.columns

    assert "label" in df.columns

def test_get_segment_distribution_returns_dataframe():

    df = get_segment_distribution()

    assert isinstance(df, pd.DataFrame)

    assert "label" in df.columns
    
    assert "count" in df.columns