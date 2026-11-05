"""Structured JSON logs with request ids, and nothing in them that identifies a visitor.

Every record is one JSON object per line on standard output. What is logged is decided here, not by whoever writes a
log call:

* request bodies, messages and answers are never logged, only their sizes;
* a client address is never logged, only a short salted hash of it (enough to see that one address is misbehaving);
* every string that does get logged passes the redactor (e-mail addresses, phone and card numbers, bearer tokens and
  the credential shapes of this service), including the text of an exception;
* only plain values are logged: a record carrying an object adds nothing but its type name.
"""

from __future__ import annotations

import contextvars
import hashlib
import json
import logging
import sys
from datetime import UTC, datetime
from typing import Any

from rag_chatbot.domain.redaction import ALL_KINDS, redact

request_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default="-")

_STANDARD = frozenset(vars(logging.makeLogRecord({}))) | {"message", "asctime", "taskName"}
_MAX_VALUE_CHARS = 500


def ip_hash(address: str, salt: str) -> str:
    return hashlib.sha256(f"{salt}|{address}".encode()).hexdigest()[:10]


class JsonFormatter(logging.Formatter):
    def __init__(self, *, redaction_kinds: frozenset[str] = ALL_KINDS) -> None:
        super().__init__()
        # secrets are always redacted, whatever the operator chose for personal data
        self._kinds = frozenset(redaction_kinds) | {"secrets"}

    def _clean(self, value: object) -> object:
        if isinstance(value, bool) or value is None or isinstance(value, (int, float)):
            return value
        if isinstance(value, str):
            return redact(value[:_MAX_VALUE_CHARS], self._kinds).text
        return f"<{type(value).__name__}>"

    def format(self, record: logging.LogRecord) -> str:
        entry: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, UTC).isoformat(timespec="milliseconds"),
            "level": record.levelname.lower(),
            "logger": record.name,
            "msg": self._clean(record.getMessage()),
            "request_id": request_id_var.get(),
        }
        for key, value in vars(record).items():
            if key not in _STANDARD and not key.startswith("_"):
                entry[key] = self._clean(value)
        if record.exc_info and record.exc_info[0] is not None:
            entry["error"] = record.exc_info[0].__name__
            entry["error_detail"] = self._clean(str(record.exc_info[1]))
        return json.dumps(entry, ensure_ascii=True, separators=(",", ":"))


def configure_logging(level: str = "INFO", *, redaction_kinds: frozenset[str] = ALL_KINDS, stream: Any = None) -> None:
    """Send every log record of the process through the JSON formatter, once."""
    handler = logging.StreamHandler(stream or sys.stdout)
    handler.setFormatter(JsonFormatter(redaction_kinds=redaction_kinds))
    root = logging.getLogger()
    for existing in list(root.handlers):
        root.removeHandler(existing)
    root.addHandler(handler)
    root.setLevel(level.upper())
    for noisy in ("uvicorn", "uvicorn.error", "uvicorn.access", "httpx", "httpcore", "psycopg", "psycopg.pool"):
        logging.getLogger(noisy).handlers.clear()
        logging.getLogger(noisy).propagate = True
    logging.getLogger("uvicorn.access").disabled = True  # the request log line below replaces it, without the path
