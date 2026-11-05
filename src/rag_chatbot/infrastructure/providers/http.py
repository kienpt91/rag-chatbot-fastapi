"""Clients for an HTTP chat/embedding API, configured by a base URL and a key.

The wire format is the common JSON one: `POST {base}/chat/completions` and `POST {base}/embeddings` with a bearer key.
Any service that speaks it can be used (a hosted one, or one you run yourself behind `http://localhost`).

What the client guarantees: the key goes only to the configured base URL, which must be https (plain http is accepted
for a loopback address only) and carry no credentials; redirects are never followed (a redirect could send the key
elsewhere); the response is read up to a size cap and parsed strictly; every failure is a `ProviderError` that names
the class of the failure (a status code, a timeout) and never contains the response body, the request or the key;
`429` and `5xx` answers and connection failures are retried a few times with a short, capped back-off.
"""

from __future__ import annotations

import asyncio
import http.client
import json
import math
import ssl
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit

from rag_chatbot.domain.errors import ConfigurationError, ProviderError
from rag_chatbot.domain.model import ChatCompletion, ChatTurn, EmbeddingResult, Usage
from rag_chatbot.domain.text import estimate_tokens

_LOOPBACK = frozenset({"localhost", "127.0.0.1", "::1"})
_RETRY_STATUSES = frozenset({429, 500, 502, 503, 504})
_MAX_RETRY_AFTER_S = 5.0
_READ_CHUNK = 64 * 1024

Sleep = Callable[[float], Awaitable[None]]


@dataclass(frozen=True, slots=True)
class ApiEndpoint:
    scheme: str
    host: str
    port: int
    prefix: str  # path prefix of the base URL without a trailing slash, e.g. "/v1"

    @classmethod
    def parse(cls, base_url: str) -> ApiEndpoint:
        try:
            parts = urlsplit(base_url)
            port = parts.port
        except ValueError:
            raise ConfigurationError("the API base URL is not a valid URL") from None
        host = parts.hostname or ""
        if parts.scheme not in ("http", "https") or not host:
            raise ConfigurationError("the API base URL must be an http(s) URL with a host")
        if parts.username is not None or parts.password is not None:
            raise ConfigurationError("the API base URL must not contain credentials")
        if parts.query or parts.fragment:
            raise ConfigurationError("the API base URL must not contain a query or a fragment")
        if parts.scheme == "http" and host not in _LOOPBACK:
            raise ConfigurationError("the API base URL must be https (http is accepted for localhost only)")
        return cls(parts.scheme, host, port or (443 if parts.scheme == "https" else 80), parts.path.rstrip("/"))


class HttpApiClient:
    """POSTs JSON and returns the parsed JSON object. Holds the key; never exposes or logs it."""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        *,
        timeout_s: float = 30.0,
        max_response_bytes: int = 8_000_000,
        retries: int = 2,
        sleep: Sleep = asyncio.sleep,
        ssl_context: ssl.SSLContext | None = None,
    ) -> None:
        if not api_key or any(not " " < c <= "~" for c in api_key):
            raise ConfigurationError("the API key must be non-empty printable ASCII without spaces")
        self._endpoint = ApiEndpoint.parse(base_url)
        self._key = api_key
        self._timeout_s = timeout_s
        self._max_bytes = max_response_bytes
        self._retries = retries
        self._sleep = sleep
        self._ssl = ssl_context

    def __repr__(self) -> str:
        return f"HttpApiClient({self._endpoint.scheme}://{self._endpoint.host}:{self._endpoint.port}{self._endpoint.prefix})"

    async def post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        body = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode()
        last = "unavailable"
        for attempt in range(self._retries + 1):
            try:
                status, retry_after, data = await asyncio.to_thread(self._post_blocking, path, body)
            except (OSError, http.client.HTTPException, TimeoutError) as error:
                last = f"network failure ({type(error).__name__})"
                status, retry_after, data = 0, 0.0, b""
            if status == 200:
                return _parse_object(data)
            if status and status not in _RETRY_STATUSES:
                raise ProviderError(f"the API answered with status {status}")
            if status:
                last = f"status {status}"
            if attempt < self._retries:
                await self._sleep(min(_MAX_RETRY_AFTER_S, retry_after or 0.5 * (2**attempt)))
        raise ProviderError(f"the API did not answer: {last}")

    def _post_blocking(self, path: str, body: bytes) -> tuple[int, float, bytes]:
        endpoint = self._endpoint
        connection: http.client.HTTPConnection
        if endpoint.scheme == "https":
            connection = http.client.HTTPSConnection(
                endpoint.host, endpoint.port, timeout=self._timeout_s, context=self._ssl or ssl.create_default_context()
            )
        else:
            connection = http.client.HTTPConnection(endpoint.host, endpoint.port, timeout=self._timeout_s)
        try:
            connection.request(
                "POST",
                f"{endpoint.prefix}{path}",
                body=body,
                headers={
                    "Authorization": f"Bearer {self._key}",
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                    "Accept-Encoding": "identity",
                    "User-Agent": "rag-chatbot-fastapi",
                },
            )
            response = connection.getresponse()
            retry_after = _retry_after(response.getheader("Retry-After"))
            if response.status != 200:
                response.read(1024)  # drain a little; the body of an error is never used
                return response.status, retry_after, b""
            chunks: list[bytes] = []
            total = 0
            while True:
                chunk = response.read(_READ_CHUNK)
                if not chunk:
                    break
                total += len(chunk)
                if total > self._max_bytes:
                    raise ProviderError("the API answer is larger than the allowed size")
                chunks.append(chunk)
            return 200, retry_after, b"".join(chunks)
        finally:
            connection.close()


def _retry_after(value: str | None) -> float:
    try:
        seconds = float(value) if value else 0.0
    except ValueError:
        return 0.0
    return seconds if math.isfinite(seconds) and seconds > 0 else 0.0


def _no_constants(name: str) -> Any:
    raise ValueError(f"constant {name} is not valid JSON")


def _parse_object(data: bytes) -> dict[str, Any]:
    try:
        parsed = json.loads(data, parse_constant=_no_constants)
    except (ValueError, RecursionError):
        raise ProviderError("the API answer is not valid JSON") from None
    if not isinstance(parsed, dict):
        raise ProviderError("the API answer is not a JSON object")
    return parsed


def _usage(parsed: dict[str, Any]) -> Usage:
    usage = parsed.get("usage")
    if not isinstance(usage, dict):
        return Usage()

    def count(key: str) -> int:
        value = usage.get(key)
        return value if isinstance(value, int) and not isinstance(value, bool) and 0 <= value < 10**9 else 0

    return Usage(count("prompt_tokens"), count("completion_tokens"))


class HttpChatProvider:
    def __init__(self, client: HttpApiClient, model: str, *, temperature: float = 0.0) -> None:
        self.model = model
        self._client = client
        self._temperature = temperature

    async def complete(self, system: str, messages: Sequence[ChatTurn], *, max_output_tokens: int) -> ChatCompletion:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                *({"role": m.role, "content": m.content} for m in messages),
            ],
            "max_tokens": max_output_tokens,
            "temperature": self._temperature,
        }
        parsed = await self._client.post("/chat/completions", payload)
        choices = parsed.get("choices")
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
            raise ProviderError("the chat answer has no choices")
        message = choices[0].get("message")
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, str):
            raise ProviderError("the chat answer has no text")
        return ChatCompletion(content, _usage(parsed), self.model)


class HttpEmbeddingProvider:
    def __init__(
        self, client: HttpApiClient, model: str, *, dimensions: int, send_dimensions: bool = False, max_batch: int = 64
    ) -> None:
        self.model = model
        self.dimensions = dimensions
        self._client = client
        self._send_dimensions = send_dimensions
        self._max_batch = max_batch

    async def embed(self, texts: Sequence[str]) -> EmbeddingResult:
        vectors: list[tuple[float, ...]] = []
        usage = Usage()
        for start in range(0, len(texts), self._max_batch):
            batch = list(texts[start : start + self._max_batch])
            payload: dict[str, Any] = {"model": self.model, "input": batch}
            if self._send_dimensions:
                payload["dimensions"] = self.dimensions
            parsed = await self._client.post("/embeddings", payload)
            vectors.extend(self._vectors(parsed, len(batch)))
            reported = _usage(parsed)
            # an answer without usage must not make the question free: estimate it, and say so
            usage += (
                reported if reported.input_tokens else Usage(sum(estimate_tokens(t) for t in batch), 0, estimated=True)
            )
        return EmbeddingResult(tuple(vectors), usage, self.model)

    def _vectors(self, parsed: dict[str, Any], expected: int) -> list[tuple[float, ...]]:
        data = parsed.get("data")
        if not isinstance(data, list) or len(data) != expected:
            raise ProviderError("the embedding answer has the wrong number of vectors")
        ordered: dict[int, tuple[float, ...]] = {}
        for position, item in enumerate(data):
            index = item.get("index", position) if isinstance(item, dict) else None
            vector = item.get("embedding") if isinstance(item, dict) else None
            if (
                isinstance(index, bool)
                or not isinstance(index, int)
                or not isinstance(vector, list)
                or len(vector) != self.dimensions
                or index in ordered
                or not all(isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) for x in vector)
            ):
                raise ProviderError("an embedding in the answer is malformed or has the wrong size")
            ordered[index] = tuple(float(x) for x in vector)
        if set(ordered) != set(range(expected)):
            raise ProviderError("the embedding answer has unexpected indexes")
        return [ordered[i] for i in range(expected)]
