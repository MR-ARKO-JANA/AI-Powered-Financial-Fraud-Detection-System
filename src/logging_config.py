"""
logging_config.py — Centralized Structured Logging for FinGuard AI.

Provides JSON-formatted structured logging with request ID correlation,
replacing all module-level print() calls and duplicate basicConfig setups.
"""

import logging
import json
import uuid
import sys
from datetime import datetime, timezone
from typing import Any, Optional
from contextvars import ContextVar

# Context variable for request-scoped correlation ID
_request_id_var: ContextVar[str] = ContextVar("request_id", default="system")


def get_request_id() -> str:
    """Get current request correlation ID."""
    return _request_id_var.get()


def set_request_id(request_id: Optional[str] = None) -> str:
    """Set a new request correlation ID. Returns the ID."""
    rid = request_id or str(uuid.uuid4())[:12]
    _request_id_var.set(rid)
    return rid


class StructuredFormatter(logging.Formatter):
    """JSON-structured log formatter with request ID correlation."""

    def format(self, record: logging.LogRecord) -> str:
        log_entry: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "request_id": get_request_id(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        if record.exc_info:
            exc_type, exc_value, exc_tb = record.exc_info
            if exc_type is not None and exc_value is not None:
                log_entry["exception"] = {
                    "type": exc_type.__name__,
                    "message": str(exc_value),
                    "traceback": self.formatException(record.exc_info),
                }

        return json.dumps(log_entry, default=str)


def setup_logging(level: int = logging.INFO) -> None:
    """
    Configure centralized structured logging for the entire application.
    Call once at application startup. Subsequent calls are no-ops.
    """
    root_logger = logging.getLogger()

    # Avoid duplicate handler setup
    if any(isinstance(h, logging.StreamHandler) and isinstance(h.formatter, StructuredFormatter)
           for h in root_logger.handlers):
        return

    # Remove any existing handlers to prevent duplicate output
    root_logger.handlers.clear()

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(StructuredFormatter())
    root_logger.addHandler(handler)
    root_logger.setLevel(level)

    # Suppress noisy third-party loggers
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Get a named logger with structured formatting."""
    setup_logging()
    return logging.getLogger(name)
