# SPDX-License-Identifier: MIT
"""Structured logging configuration for MuJoCo Models.

Provides a JSON formatter and a helper to configure the root logger so that
all log output is emitted as structured JSON records suitable for ingestion
by log aggregators.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from typing import Any


class JSONFormatter(logging.Formatter):
    """Emit log records as newline-delimited JSON."""

    def format(self, record: logging.LogRecord) -> str:
        """Return a JSON string for *record*."""
        msg = super().format(record)
        payload: dict[str, Any] = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created))
            + f".{int(record.msecs):03d}Z",
            "level": record.levelname,
            "logger": record.name,
            "message": msg,
        }
        if hasattr(record, "event"):
            payload["event"] = record.event
        if hasattr(record, "duration_ms"):
            payload["duration_ms"] = record.duration_ms
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def configure_logging(level: int = logging.INFO, stream: Any = sys.stderr) -> None:
    """Configure the root logger with :class:`JSONFormatter`.

    Parameters
    ----------
    level:
        Minimum log level (default ``logging.INFO``).
    stream:
        Output stream (default ``sys.stderr``).
    """
    handler = logging.StreamHandler(stream)
    handler.setFormatter(JSONFormatter())
    root = logging.getLogger()
    root.handlers = []
    root.addHandler(handler)
    root.setLevel(level)