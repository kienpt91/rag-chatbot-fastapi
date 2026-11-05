"""In-memory implementations of the stores, for the evaluation run and for tests of the use cases.

They follow the same contract as the PostgreSQL stores (`tests/infrastructure/test_store_contract.py` runs one suite
against both): the same ranking by cosine similarity, the same keyword matching on words, the same tenant
isolation. Nothing is persisted, nothing is shared between processes.
"""

from __future__ import annotations

import math
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, date, datetime

from rag_chatbot.application.ports import SessionInfo, UsageEntry
from rag_chatbot.domain.errors import InvalidRequestError, StorageError
from rag_chatbot.domain.model import ChatTurn, Chunk, SearchHit
from rag_chatbot.domain.text import words


def cosine(left: Sequence[float], right: Sequence[float]) -> float:
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    norm = math.sqrt(sum(a * a for a in left)) * math.sqrt(sum(b * b for b in right))
    if norm == 0.0:
        return 0.0
    return min(1.0, max(0.0, dot / norm))


class InMemoryChunkStore:
    def __init__(self, *, dimensions: int) -> None:
        self._dimensions = dimensions
        self._chunks: dict[tuple[str, str], tuple[Chunk, tuple[float, ...]]] = {}
        self._documents: dict[tuple[str, str], tuple[str, str]] = {}

    async def known_chunk_ids(self, tenant_id: str, document_id: str) -> frozenset[str]:
        return frozenset(
            c.chunk_id for (t, _), (c, _) in self._chunks.items() if t == tenant_id and c.document_id == document_id
        )

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
        for chunk in chunks:
            if chunk.tenant_id != tenant_id or chunk.document_id != document_id:
                raise StorageError("a chunk does not belong to the document it is stored with")
        stored = {k: v for k, v in self._chunks.items() if k[0] == tenant_id and v[0].document_id == document_id}
        new = [c for c in chunks if (tenant_id, c.chunk_id) not in stored]
        if any(c.chunk_id not in embeddings for c in new):
            raise StorageError("new chunks have no embedding")
        if any(len(v) != self._dimensions for v in embeddings.values()):
            raise StorageError("an embedding has the wrong size")
        keep = {c.chunk_id for c in chunks}
        for key in [k for k in stored if k[1] not in keep]:
            del self._chunks[key]
        for chunk in chunks:
            key = (tenant_id, chunk.chunk_id)
            vector = self._chunks[key][1] if key in self._chunks else embeddings[chunk.chunk_id]
            self._chunks[key] = (chunk, vector)
        self._documents[(tenant_id, document_id)] = (url, title)

    async def prune_documents(self, tenant_id: str, keep_document_ids: Collection[str]) -> int:
        keep = set(keep_document_ids)
        if not keep:
            raise InvalidRequestError("refusing to remove every document of a tenant")
        gone = [k for k in self._documents if k[0] == tenant_id and k[1] not in keep]
        for key in gone:
            del self._documents[key]
        for key in [k for k, v in self._chunks.items() if k[0] == tenant_id and v[0].document_id not in keep]:
            del self._chunks[key]
        return len(gone)

    def _tenant(self, tenant_id: str) -> list[tuple[Chunk, tuple[float, ...]]]:
        return [v for (t, _), v in self._chunks.items() if t == tenant_id]

    async def vector_search(self, tenant_id: str, embedding: Sequence[float], limit: int) -> list[SearchHit]:
        scored = [SearchHit(c, cosine(embedding, v)) for c, v in self._tenant(tenant_id)]
        scored.sort(key=lambda h: (-h.similarity, h.chunk.chunk_id))
        return scored[:limit]

    async def keyword_search(
        self, tenant_id: str, terms: Sequence[str], embedding: Sequence[float], limit: int
    ) -> list[SearchHit]:
        wanted = set(terms)
        matches: list[tuple[int, SearchHit]] = []
        for chunk, vector in self._tenant(tenant_id):
            counts = sum(1 for w in words(chunk.text) if w in wanted)
            if counts:
                matches.append((counts, SearchHit(chunk, cosine(embedding, vector))))
        matches.sort(key=lambda m: (-m[0], m[1].chunk.chunk_id))
        return [hit for _, hit in matches[:limit]]

    async def count_chunks(self, tenant_id: str) -> int:
        return len(self._tenant(tenant_id))


@dataclass
class _Session:
    owner_id: str | None
    messages: list[ChatTurn] = field(default_factory=list)
    last_active: datetime = field(default_factory=lambda: datetime.now(UTC))


class InMemorySessionStore:
    def __init__(self) -> None:
        self._sessions: dict[tuple[str, str], _Session] = {}
        # what was stored about each assistant turn: (grounded, cited chunk ids), for tests to look at
        self.annotations: dict[tuple[str, str], list[tuple[bool | None, tuple[str, ...]]]] = {}

    async def create_session(self, tenant_id: str, session_id: str, owner_id: str | None) -> None:
        if (tenant_id, session_id) in self._sessions:
            raise StorageError("duplicate session")
        self._sessions[(tenant_id, session_id)] = _Session(owner_id)

    async def get_session(self, tenant_id: str, session_id: str) -> SessionInfo | None:
        session = self._sessions.get((tenant_id, session_id))
        if session is None:
            return None
        return SessionInfo(session_id, tenant_id, len(session.messages), session.owner_id)

    async def append_messages(
        self,
        tenant_id: str,
        session_id: str,
        turns: Sequence[ChatTurn],
        *,
        grounded: bool | None,
        chunk_ids: Sequence[str],
    ) -> None:
        session = self._sessions[(tenant_id, session_id)]
        session.messages.extend(turns)
        session.last_active = datetime.now(UTC)
        self.annotations.setdefault((tenant_id, session_id), []).append((grounded, tuple(chunk_ids)))

    async def recent_messages(self, tenant_id: str, session_id: str, limit: int) -> list[ChatTurn]:
        session = self._sessions.get((tenant_id, session_id))
        return [] if session is None else session.messages[-limit:]

    async def purge_sessions(self, tenant_id: str, older_than: datetime) -> int:
        gone = [k for k, s in self._sessions.items() if k[0] == tenant_id and s.last_active < older_than]
        for key in gone:
            del self._sessions[key]
        return len(gone)


class InMemoryUsageLedger:
    def __init__(self) -> None:
        self.entries: list[UsageEntry] = []

    async def spent_micros(self, tenant_id: str, day: date) -> int:
        return sum(
            e.cost_micros for e in self.entries if e.tenant_id == tenant_id and e.at.astimezone(UTC).date() == day
        )

    async def record(self, entry: UsageEntry) -> None:
        self.entries.append(entry)
