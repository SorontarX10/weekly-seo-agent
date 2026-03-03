from __future__ import annotations

from datetime import datetime, timezone
import json
import logging

LOGGER_NAME = "manager_document_agent"

logger = logging.getLogger(LOGGER_NAME)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
logger.setLevel(logging.INFO)
logger.propagate = False


def log_event(event: str, **fields) -> None:
    payload = {
        "event": event,
        "ts": datetime.now(timezone.utc).isoformat(),
    }
    payload.update(fields)
    logger.info(json.dumps(payload, ensure_ascii=True))
