import pytest
from unittest.mock import patch, MagicMock
import app.db_postgres as db_postgres

def test_get_postgres_conn_params_success(monkeypatch):
    monkeypatch.setenv("POSTGRES_DB", "test_db")
    monkeypatch.setenv("POSTGRES_USER", "test_user")
    monkeypatch.setenv("POSTGRES_PASSWORD", "test_pass")
    monkeypatch.setenv("POSTGRES_HOST", "db.example.com")
    monkeypatch.setenv("POSTGRES_PORT", "5433")

    params = db_postgres._get_postgres_conn_params()
    assert params == {
        "dbname": "test_db",
        "user": "test_user",
        "password": "test_pass",
        "host": "db.example.com",
        "port": "5433",
    }

def test_get_postgres_conn_params_missing_credentials(monkeypatch):
    monkeypatch.delenv("POSTGRES_DB", raising=False)
    monkeypatch.delenv("POSTGRES_USER", raising=False)
    monkeypatch.delenv("POSTGRES_PASSWORD", raising=False)

    with pytest.raises(ValueError, match="Missing required PostgreSQL environment variable"):
        db_postgres._get_postgres_conn_params()

def test_get_pool_returns_none_when_config_missing(monkeypatch):
    monkeypatch.delenv("POSTGRES_DB", raising=False)
    monkeypatch.delenv("POSTGRES_USER", raising=False)
    monkeypatch.delenv("POSTGRES_PASSWORD", raising=False)

    # Reset any existing pool state
    db_postgres.close_pool()

    pool = db_postgres.get_pool()
    assert pool is None

def test_get_connection_raises_value_error_when_config_missing(monkeypatch):
    monkeypatch.delenv("POSTGRES_DB", raising=False)
    monkeypatch.delenv("POSTGRES_USER", raising=False)
    monkeypatch.delenv("POSTGRES_PASSWORD", raising=False)

    # Reset any existing pool state
    db_postgres.close_pool()

    with pytest.raises(ValueError, match="Missing required PostgreSQL environment variable"):
        db_postgres.get_connection()

@patch("app.db_postgres.psycopg2.connect")
def test_get_connection_direct_connect_success(mock_connect, monkeypatch):
    monkeypatch.setenv("POSTGRES_DB", "test_db")
    monkeypatch.setenv("POSTGRES_USER", "test_user")
    monkeypatch.setenv("POSTGRES_PASSWORD", "test_pass")

    db_postgres.close_pool()

    with patch("app.db_postgres.get_pool", return_value=None):
        conn = db_postgres.get_connection()
        mock_connect.assert_called_once_with(
            dbname="test_db",
            user="test_user",
            password="test_pass",
            host="localhost",
            port="5432",
        )
