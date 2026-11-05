"""A token-bucket rate limiter with bounded memory.

One bucket per key (a session id, a client address). A bucket holds up to `capacity` tokens and gets `refill_per_s`
tokens per second; a request takes one. The table of buckets never grows past `max_keys`: when it is full the least
recently used bucket is dropped, so an attacker who sends from millions of addresses costs a fixed amount of memory,
at the price of those addresses getting a fresh bucket after eviction (SECURITY.md names this limit).

Pure: the caller passes the time (a monotonic clock) in. In-process on purpose: one instance per worker process, so
with N workers a client can send N times the limit (docs/deployment.md says how to put a limit in front instead).
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass

from rag_chatbot.domain.errors import ConfigurationError


@dataclass(frozen=True, slots=True)
class Decision:
    allowed: bool
    retry_after_s: float = 0.0


class RateLimiter:
    def __init__(self, *, capacity: float, refill_per_s: float, max_keys: int = 10_000) -> None:
        if capacity < 1 or refill_per_s <= 0 or max_keys < 1:
            raise ConfigurationError("rate limiter: capacity >= 1, refill_per_s > 0 and max_keys >= 1 are required")
        self._capacity = float(capacity)
        self._refill = float(refill_per_s)
        self._max_keys = max_keys
        self._buckets: OrderedDict[str, tuple[float, float]] = OrderedDict()  # key -> (tokens, last seen)

    def __len__(self) -> int:
        return len(self._buckets)

    def acquire(self, key: str, now: float) -> Decision:
        tokens, last = self._buckets.pop(key, (self._capacity, now))
        tokens = min(self._capacity, tokens + max(0.0, now - last) * self._refill)
        if tokens >= 1.0:
            decision = Decision(True)
            tokens -= 1.0
        else:
            decision = Decision(False, (1.0 - tokens) / self._refill)
        self._buckets[key] = (tokens, now)  # most recently used goes last
        while len(self._buckets) > self._max_keys:
            self._buckets.popitem(last=False)
        return decision
