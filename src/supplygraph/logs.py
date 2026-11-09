"""Logging setup: human-readable by default, one JSON object per line for log collectors.

`SUPPLYGRAPH_LOG_FORMAT=json` (set in the container image) makes every record machine-parseable,
so Loki / CloudWatch / Elastic can filter on `level`, `logger` and the extra fields.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import UTC, datetime

# Attributes every LogRecord has; anything else was passed via `extra=` and is kept.
_STANDARD = set(vars(logging.LogRecord("", 0, "", 0, "", None, None))) | {"message", "asctime"}


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        out = {
            "ts": datetime.fromtimestamp(record.created, UTC).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        out |= {k: v for k, v in vars(record).items() if k not in _STANDARD}
        if record.exc_info:
            out["exception"] = self.formatException(record.exc_info)
        return json.dumps(out, ensure_ascii=False, default=str)


def configure(verbose: bool = False, fmt: str | None = None) -> None:
    fmt = (fmt or os.getenv("SUPPLYGRAPH_LOG_FORMAT") or "text").lower()
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter() if fmt == "json" else logging.Formatter(
        "%(asctime)s %(levelname)-7s %(name)s: %(message)s"))
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(logging.DEBUG if verbose else logging.INFO)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("neo4j").setLevel(logging.WARNING)  # "constraint already exists" noise
