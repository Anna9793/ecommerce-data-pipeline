import json
import logging
from config.logging_config import StructuredJSONFormatter, setup_logging


def test_structured_json_formatter():
    """Verify that StructuredJSONFormatter emits valid single-line JSON adhering to Cloud Logging format."""
    formatter = StructuredJSONFormatter()
    record = logging.LogRecord(
        name="test_logger",
        level=logging.INFO,
        pathname="test.py",
        lineno=42,
        msg="Prediction executed successfully",
        args=(),
        exc_info=None,
        func="test_func"
    )
    # Inject extra metadata
    record.tenant_id = "giftshop_uk"
    record.latency_ms = 14.5

    output = formatter.format(record)
    data = json.loads(output)

    assert data["severity"] == "INFO"
    assert data["message"] == "Prediction executed successfully"
    assert data["logger"] == "test_logger"
    assert data["tenant_id"] == "giftshop_uk"
    assert data["latency_ms"] == 14.5
    assert "timestamp" in data


def test_setup_logging_text_and_json(monkeypatch):
    """Verify setup_logging dynamically configures text vs JSON handlers."""
    # 1. Text mode
    logger_text = setup_logging(json_format=False)
    assert len(logger_text.handlers) > 0
    assert not isinstance(logger_text.handlers[0].formatter, StructuredJSONFormatter)

    # 2. JSON mode
    logger_json = setup_logging(json_format=True)
    assert len(logger_json.handlers) > 0
    assert isinstance(logger_json.handlers[0].formatter, StructuredJSONFormatter)
