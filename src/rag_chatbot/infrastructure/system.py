"""The clock and the id generator of the running service."""

from __future__ import annotations

import secrets
import time
from datetime import UTC, datetime


class SystemClock:
    def now(self) -> datetime:
        return datetime.now(UTC)

    def monotonic(self) -> float:
        return time.monotonic()


class RandomIds:
    """Unguessable ids from the operating system's random source. A session id is a bearer secret for that
    conversation: 128 bits."""

    def new_session_id(self) -> str:
        return "s_" + secrets.token_urlsafe(16)

    def new_request_id(self) -> str:
        return "r_" + secrets.token_hex(8)

    def new_nonce(self) -> str:
        return secrets.token_hex(12)
