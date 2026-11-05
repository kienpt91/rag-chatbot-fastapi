"""The ports: everything the use cases need from the outside world, as protocols.

The application layer never imports a driver, a socket, the clock or a random number generator. It receives
implementations of these protocols from the composition layer (Postgres, an HTTP chat/embedding API, the system
clock) or from tests (in-memory fakes).
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Collection, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from typing import Protocol

from rag_chatbot.domain.model import (
    ChatCompletion,
    ChatTurn,
    Chunk,
    EmbeddingResult,
    RawPage,
    SearchHit,
    Usage,
)


class Clock(Protocol):
    def now(self) -> datetime:
        """The current time, timezone-aware, in UTC."""
        ...

    def monotonic(self) -> float:
        """Seconds from an arbitrary origin, never going backwards: for rate limiting and latency."""
        ...


class IdGenerator(Protocol):
    def new_session_id(self) -> str: ...

    def new_request_id(self) -> str: ...

    def new_nonce(self) -> str:
        """A fresh unguessable token (hex): delimiters in a prompt, so retrieved text cannot forge them."""
        ...


class EmbeddingProvider(Protocol):
    model: str
    dimensions: int

    async def embed(self, texts: Sequence[str]) -> EmbeddingResult:
        """One vector per text, in order. Raises `ProviderError` for any failure or unusable answer."""
        ...


class ChatProvider(Protocol):
    model: str

    async def complete(self, system: str, messages: Sequence[ChatTurn], *, max_output_tokens: int) -> ChatCompletion:
        """The model's answer. The model has no tools. Raises `ProviderError` for any failure."""
        ...


class ChunkStore(Protocol):
    async def known_chunk_ids(self, tenant_id: str, document_id: str) -> frozenset[str]:
        """Ids of the chunks already stored for a document (their text, and so their embedding, is unchanged)."""
        ...

    async def upsert_document(
        self,
        tenant_id: str,
        document_id: str,
        *,
        url: str,
        title: str,
        chunks: Sequence[Chunk],
        embeddings: Mapping[str, tuple[float, ...]],
    ) -> None:
        """Make `chunks` the whole content of the document, atomically: chunks of the document that are not in
        `chunks` are removed, new ones are inserted with `embeddings[chunk_id]`, existing ones keep their embedding
        and get the new position, title and url. Every chunk id that is not stored yet must have an embedding."""
        ...

    async def prune_documents(self, tenant_id: str, keep_document_ids: Collection[str]) -> int:
        """Remove the documents of a tenant that are not listed. Returns how many were removed."""
        ...

    async def vector_search(self, tenant_id: str, embedding: Sequence[float], limit: int) -> list[SearchHit]:
        """Nearest chunks by cosine distance, best first. Only the tenant's own chunks."""
        ...

    async def keyword_search(
        self, tenant_id: str, terms: Sequence[str], embedding: Sequence[float], limit: int
    ) -> list[SearchHit]:
        """Chunks that contain any of the terms, best text match first; `similarity` is still the cosine similarity
        to `embedding` so that confidence means the same for both searches. Only the tenant's own chunks."""
        ...

    async def count_chunks(self, tenant_id: str) -> int: ...


@dataclass(frozen=True, slots=True)
class SessionInfo:
    session_id: str
    tenant_id: str
    message_count: int


class SessionStore(Protocol):
    async def create_session(self, tenant_id: str, session_id: str, owner_id: str | None) -> None: ...

    async def get_session(self, tenant_id: str, session_id: str) -> SessionInfo | None:
        """None when the session does not exist *for this tenant*."""
        ...

    async def append_messages(
        self,
        tenant_id: str,
        session_id: str,
        turns: Sequence[ChatTurn],
        *,
        grounded: bool | None,
        chunk_ids: Sequence[str],
    ) -> None:
        """Append turns in order. `grounded` and `chunk_ids` describe the last (assistant) turn."""
        ...

    async def recent_messages(self, tenant_id: str, session_id: str, limit: int) -> list[ChatTurn]:
        """The last `limit` turns, oldest first."""
        ...


@dataclass(frozen=True, slots=True)
class UsageEntry:
    tenant_id: str
    session_id: str
    request_id: str
    at: datetime
    chat_model: str
    embedding_model: str
    usage: Usage  # chat tokens
    embedding_tokens: int
    cost_micros: int
    cost_estimated: bool
    latency_ms: int
    outcome: str  # answered, refused, blocked, budget, error


class UsageLedger(Protocol):
    async def spent_micros(self, tenant_id: str, day: date) -> int:
        """Cost of the tenant's requests of one UTC day, in millionths of the price table's currency."""
        ...

    async def record(self, entry: UsageEntry) -> None: ...


class ContentSource(Protocol):
    def pages(self) -> AsyncIterator[RawPage]:
        """Every page of the source, one at a time (never the whole site in memory)."""
        ...


@dataclass(frozen=True, slots=True)
class FetchResult:
    url: str  # the URL that was requested (redirects are not followed: a 3xx answer is returned as it is)
    status: int
    content_type: str  # the media type, lower case, without parameters; "" when the answer names none
    charset: str  # the charset parameter, lower case; "" when absent
    body: bytes  # at most `max_bytes` of the body
    truncated: bool  # the body was longer than `max_bytes`
    location: str = ""  # the Location header of a 3xx answer, "" otherwise


class Fetcher(Protocol):
    async def fetch(self, url: str, *, max_bytes: int) -> FetchResult:
        """GET one URL. Raises `SourceError` when the address may not be contacted, on connection and protocol
        failures and on timeouts. It never follows a redirect and never executes anything it receives."""
        ...
