"""Pure ASGI middleware: request context and log line, CORS by exact origin, body size limit, per-address rate limit.

Order, outermost first: request context (id, headers, the log line, a safety net for unexpected errors) -> CORS ->
address rate limit -> body size limit -> the application. Each layer answers for itself, so a rejected request still
gets the request id and the CORS headers a browser needs to show the error.
"""

from __future__ import annotations

import ipaddress
import json
import logging
import re
from collections.abc import Callable, MutableMapping
from typing import Any

from starlette.datastructures import Headers, MutableHeaders
from starlette.exceptions import HTTPException
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from rag_chatbot.application.ports import Clock, IdGenerator
from rag_chatbot.domain.errors import ChatbotError, RateLimitedError
from rag_chatbot.domain.ratelimit import RateLimiter
from rag_chatbot.infrastructure.http.logging import ip_hash, request_id_var
from rag_chatbot.infrastructure.http.settings import HttpSettings

log = logging.getLogger("rag_chatbot.http")

_REQUEST_ID = re.compile(r"^[A-Za-z0-9._-]{8,64}\Z")
STATUS_FOR_CODE = {
    "invalid_request": 400,
    "unauthorized": 401,
    "origin_not_allowed": 403,
    "forbidden": 403,
    "session_not_found": 404,
    "session_full": 409,
    "payload_too_large": 413,
    "unsupported_media_type": 415,
    "rate_limited": 429,
    "budget_exhausted": 503,
    "provider_unavailable": 503,
    "storage_unavailable": 503,
    "internal_error": 500,
}


def error_body(code: str, message: str) -> bytes:
    return json.dumps(
        {"error": {"code": code, "message": message, "request_id": request_id_var.get()}}, separators=(",", ":")
    ).encode()


async def send_error(send: Send, status: int, code: str, message: str, extra: dict[str, str] | None = None) -> None:
    body = error_body(code, message)
    headers = [(b"content-type", b"application/json"), (b"content-length", str(len(body)).encode())]
    headers += [(k.lower().encode(), v.encode()) for k, v in (extra or {}).items()]
    await send({"type": "http.response.start", "status": status, "headers": headers})
    await send({"type": "http.response.body", "body": body})


def client_address(scope: Scope, trusted_hops: int) -> str:
    """The address the request really came from: the peer, or, behind `trusted_hops` reverse proxies that append to
    X-Forwarded-For, the entry that many places from the right (what the nearest trusted proxy saw)."""
    client = scope.get("client")
    peer = str(client[0]) if client else "unknown"
    if trusted_hops <= 0:
        return peer
    forwarded = Headers(scope=scope).get("x-forwarded-for", "")
    entries = [e.strip() for e in forwarded.split(",") if e.strip()]
    if len(entries) < trusted_hops:
        return peer
    candidate = entries[-trusted_hops]
    try:
        return str(ipaddress.ip_address(candidate))
    except ValueError:
        return peer


class RequestContextMiddleware:
    def __init__(self, app: ASGIApp, *, ids: IdGenerator, clock: Clock, settings: HttpSettings) -> None:
        self.app = app
        self._ids = ids
        self._clock = clock
        self._settings = settings

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        inbound = Headers(scope=scope).get("x-request-id", "")
        request_id = inbound if _REQUEST_ID.match(inbound) else self._ids.new_request_id()
        token = request_id_var.set(request_id)
        started = self._clock.monotonic()
        status = 500
        started_response = False

        async def send_wrapper(message: Message) -> None:
            nonlocal status, started_response
            if message["type"] == "http.response.start":
                started_response = True
                status = message["status"]
                headers = MutableHeaders(scope=message)
                headers["X-Request-Id"] = request_id
                headers["X-Content-Type-Options"] = "nosniff"
                headers.setdefault("Cache-Control", "no-store")
                headers.setdefault("Referrer-Policy", "no-referrer")
                headers.setdefault("Content-Security-Policy", "default-src 'none'; frame-ancestors 'none'")
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        except Exception:
            # the application is meant to turn every failure into a response; this is the net for what it did not
            log.exception("unhandled error", extra={"event": "unhandled_error"})
            if not started_response:
                status = 500
                await send_error(send_wrapper, 500, "internal_error", ChatbotError.public_message)
        finally:
            route = scope.get("route")
            client = client_address(scope, self._settings.trusted_proxy_hops)
            log.info(
                "request",
                extra={
                    "event": "request",
                    "method": scope.get("method", "-"),
                    "route": getattr(route, "path", "unmatched"),
                    "status": status,
                    "latency_ms": max(0, int((self._clock.monotonic() - started) * 1000)),
                    "client": ip_hash(client, self._settings.log_ip_salt),
                },
            )
            request_id_var.reset(token)


class CorsMiddleware:
    """CORS by exact origin. A preflight carries no credential, so it can only be checked against the origins of every
    tenant; the request that follows is checked again against the origins of the tenant its credential names."""

    def __init__(self, app: ASGIApp, *, origins: Callable[[], frozenset[str]]) -> None:
        self.app = app
        self._origins = origins

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        headers = Headers(scope=scope)
        origin = headers.get("origin")
        if origin is None:
            await self.app(scope, receive, send)
            return
        allowed = origin in self._origins()  # an exact comparison with the configured, normalised origins
        if scope["method"] == "OPTIONS" and "access-control-request-method" in headers:
            if not allowed:
                await send_error(
                    send, 403, "origin_not_allowed", "This site may not use the assistant.", {"Vary": "Origin"}
                )
                return
            await send(
                {
                    "type": "http.response.start",
                    "status": 204,
                    "headers": [
                        (b"access-control-allow-origin", origin.encode()),
                        (b"access-control-allow-methods", b"POST, OPTIONS"),
                        (b"access-control-allow-headers", b"authorization, content-type"),
                        (b"access-control-max-age", b"600"),
                        (b"vary", b"Origin"),
                    ],
                }
            )
            await send({"type": "http.response.body", "body": b""})
            return

        async def send_wrapper(message: Message) -> None:
            if message["type"] == "http.response.start":
                out = MutableHeaders(scope=message)
                out.append("Vary", "Origin")
                if allowed:
                    out["Access-Control-Allow-Origin"] = origin
                    out["Access-Control-Expose-Headers"] = "Retry-After, X-Request-Id"
            await send(message)

        await self.app(scope, receive, send_wrapper)


class RateLimitMiddleware:
    """A token bucket per client address on the paths that cost something, before anything is parsed."""

    def __init__(
        self,
        app: ASGIApp,
        *,
        limiter: RateLimiter,
        paths: frozenset[str],
        clock: Clock,
        trusted_hops: int,
    ) -> None:
        self.app = app
        self._limiter = limiter
        self._paths = paths
        self._clock = clock
        self._hops = trusted_hops

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope["path"] not in self._paths or scope["method"] == "OPTIONS":
            await self.app(scope, receive, send)
            return
        decision = self._limiter.acquire(client_address(scope, self._hops), self._clock.monotonic())
        if decision.allowed:
            await self.app(scope, receive, send)
            return
        retry = RateLimitedError("address", int(decision.retry_after_s) + 1)
        log.warning("rate limited", extra={"event": "rate_limited", "scope": "address"})
        await send_error(send, 429, retry.code, retry.public_message, {"Retry-After": str(retry.retry_after_s)})


class BodyLimitMiddleware:
    def __init__(self, app: ASGIApp, *, max_bytes: int) -> None:
        self.app = app
        self._max = max_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        declared = Headers(scope=scope).get("content-length")
        if declared is not None:
            try:
                length = int(declared)
            except ValueError:
                await send_error(send, 400, "invalid_request", "The request is not valid.")
                return
            if length < 0 or length > self._max:
                await send_error(send, 413, "payload_too_large", "The request is too large.")
                return
        received = 0

        async def counting_receive() -> MutableMapping[str, Any]:
            nonlocal received
            message: MutableMapping[str, Any] = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self._max:
                    # an HTTPException is what the framework lets through its body-reading error handling, and what
                    # the application's handler turns into the 413 answer
                    raise HTTPException(status_code=413)
            return message

        await self.app(scope, counting_receive, send)
