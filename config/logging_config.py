import json
import logging
import os
import sys
from datetime import datetime, timezone
from typing import Optional


class StructuredJSONFormatter(logging.Formatter):
    """
    Structured JSON Formatter conforming to 12-Factor App Factor XI (Logs as Event Streams)
    and Google Cloud Logging format specifications.
    Emits single-line JSON logs parseable by Cloud Logging, BigQuery Log Sinks, and Datadog.
    """

    def format(self, record: logging.LogRecord) -> str:
        # Map Python logging levels to Google Cloud Logging severity levels
        severity_map = {
            "DEBUG": "DEBUG",
            "INFO": "INFO",
            "WARNING": "WARNING",
            "ERROR": "ERROR",
            "CRITICAL": "CRITICAL"
        }

        # Base structured payload
        log_entry = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "severity": severity_map.get(record.levelname, record.levelname),
            "message": record.getMessage(),
            "logger": record.name,
            "module": record.module,
            "function": record.funcName,
            "lineno": record.lineno,
        }

        # Include exception traceback if present
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)

        # Include custom extra metadata passed via logger (e.g. logger.info("msg", extra={"tenant_id": ...}))
        reserved_keys = {
            "name", "msg", "args", "levelname", "levelno", "pathname", "filename",
            "module", "exc_info", "exc_text", "stack_info", "lineno", "funcName",
            "created", "msecs", "relativeCreated", "thread", "threadName",
            "processName", "process", "message"
        }
        for key, val in record.__dict__.items():
            if key not in reserved_keys and not key.startswith("_"):
                log_entry[key] = val

        return json.dumps(log_entry)


def setup_logging(
    log_level: Optional[str] = None,
    json_format: Optional[bool] = None
) -> logging.Logger:
    """
    Configures application-wide logging stream to stdout (12-Factor Factor XI).
    Automatically enables structured JSON in production (or when LOG_FORMAT=json).
    """
    level_name = (log_level or os.getenv("LOG_LEVEL", "INFO")).upper()
    level = getattr(logging, level_name, logging.INFO)

    # Determine if JSON format should be used
    if json_format is None:
        env_format = os.getenv("LOG_FORMAT", "").lower()
        use_json = env_format == "json" or os.getenv("USE_BIGQUERY", "false").lower() == "true"
    else:
        use_json = json_format

    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Remove existing handlers to prevent duplicates
    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)

    # Stream to stdout (Factor XI)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(level)

    if use_json:
        stream_handler.setFormatter(StructuredJSONFormatter())
    else:
        text_formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        stream_handler.setFormatter(text_formatter)

    root_logger.addHandler(stream_handler)
    return root_logger
