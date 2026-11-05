"""Limits and switches of the HTTP interface. Plain data; composition fills it from the environment."""

from __future__ import annotations

from dataclasses import dataclass

from rag_chatbot.domain.errors import ConfigurationError


@dataclass(frozen=True, slots=True)
class HttpSettings:
    max_body_bytes: int = 16 * 1024  # a chat message is at most 2,000 characters; the rest is JSON framing
    ip_burst: float = 20.0  # requests an address may make at once ...
    ip_per_minute: float = 30.0  # ... and per minute on average
    session_burst: float = 5.0  # messages one conversation may send at once ...
    session_per_minute: float = 6.0  # ... and per minute on average
    max_tracked_keys: int = 10_000  # buckets kept in memory per limiter; the least recently used is dropped
    trusted_proxy_hops: int = 0  # how many reverse proxies in front append to X-Forwarded-For; 0 = none, use the peer
    request_timeout_s: float = 60.0  # one chat request, retrieval and model call together
    max_token_ttl_s: int = 3600  # the longest embed token this deployment will mint
    enable_docs: bool = False  # the interactive API docs and the OpenAPI document
    log_ip_salt: str = "unset"  # the salt of the address hash in the logs

    def __post_init__(self) -> None:
        if self.max_body_bytes < 256 or self.max_tracked_keys < 1 or self.request_timeout_s <= 0:
            raise ConfigurationError("http settings: body size, tracked keys and timeout must be positive and sensible")
        if min(self.ip_burst, self.session_burst) < 1:
            raise ConfigurationError("http settings: bursts must be at least 1")
        if min(self.ip_per_minute, self.session_per_minute) <= 0 or not 0 <= self.trusted_proxy_hops <= 5:
            raise ConfigurationError("http settings: rates must be positive and trusted_proxy_hops between 0 and 5")
        if not 1 <= self.max_token_ttl_s <= 24 * 3600:
            raise ConfigurationError("http settings: max_token_ttl_s must be between 1 second and 24 hours")
