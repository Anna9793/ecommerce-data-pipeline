import os
# Force local mode for repository tests to bypass remote BigQuery queries
os.environ["USE_BIGQUERY"] = "false"

from unittest.mock import patch, MagicMock
import pandas as pd

from app.repository import get_total_predictions
from app.repository import get_latest_predictions
from app.repository import get_segment_distribution

@patch("app.repository.get_connection")
def test_get_total_predictions_returns_integer(mock_conn):
    # Set up mock cursor returning a single count
    mock_cursor = MagicMock()
    mock_cursor.fetchone.return_value = (10,)
    mock_conn.return_value.cursor.return_value = mock_cursor

    count = get_total_predictions()

    assert isinstance(count, int)
    assert count == 10

@patch("app.repository.get_connection")
def test_get_latest_predictions_returns_dataframe(mock_conn):
    # Set up mock pd.read_sql return value
    with patch("pandas.read_sql") as mock_read_sql:
        mock_read_sql.return_value = pd.DataFrame([
            {"cluster": 1, "label": "Test Segment"}
        ])

        df = get_latest_predictions()

        assert isinstance(df, pd.DataFrame)
        assert "cluster" in df.columns
        assert "label" in df.columns

@patch("app.repository.get_connection")
def test_get_segment_distribution_returns_dataframe(mock_conn):
    # Set up mock pd.read_sql return value
    with patch("pandas.read_sql") as mock_read_sql:
        mock_read_sql.return_value = pd.DataFrame([
            {"label": "Test Segment", "count": 5}
        ])

        df = get_segment_distribution()

        assert isinstance(df, pd.DataFrame)
        assert "label" in df.columns
        assert "count" in df.columns