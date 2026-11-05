"""One interface over the in-memory and the PostgreSQL stores, so that a single suite tests both."""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager
from dataclasses import dataclass

import pytest

from rag_chatbot.application.ports import ChunkStore, SessionStore, UsageLedger
from rag_chatbot.domain.chunking import chunk_id_for
from rag_chatbot.domain.model import Chunk
from rag_chatbot.infrastructure.memory import InMemoryChunkStore, InMemorySessionStore, InMemoryUsageLedger
from rag_chatbot.infrastructure.postgres.chunks import PostgresChunkStore
from rag_chatbot.infrastructure.postgres.conversations import PostgresSessionStore, PostgresUsageLedger
from rag_chatbot.infrastructure.providers.fake import FakeEmbeddingProvider
from support.database import ScratchDatabase, app_pool

DIMENSIONS = 1536


@dataclass
class World:
    kind: str
    chunks: ChunkStore
    sessions: SessionStore
    ledger: UsageLedger
    embedder: FakeEmbeddingProvider

    async def embed(self, chunks: list[Chunk]) -> dict[str, tuple[float, ...]]:
        result = await self.embedder.embed([c.text for c in chunks])
        return {c.chunk_id: v for c, v in zip(chunks, result.vectors, strict=True)}

    async def index(
        self, tenant: str, document: str, texts: list[str], *, title: str = "Title", url: str = ""
    ) -> list[Chunk]:
        chunks = [
            Chunk(chunk_id_for(tenant, document, text, 0), tenant, document, i, title, url, text)
            for i, text in enumerate(texts)
        ]
        await self.chunks.upsert_document(
            tenant, document, url=url, title=title, chunks=chunks, embeddings=await self.embed(chunks)
        )
        return chunks


@asynccontextmanager
async def memory_world() -> AsyncIterator[World]:
    yield World(
        "memory",
        InMemoryChunkStore(dimensions=DIMENSIONS),
        InMemorySessionStore(),
        InMemoryUsageLedger(),
        FakeEmbeddingProvider(DIMENSIONS),
    )


@asynccontextmanager
async def postgres_world(database: ScratchDatabase, *, max_size: int = 4) -> AsyncIterator[World]:
    async with app_pool(database, max_size=max_size) as pool:
        yield World(
            "postgres",
            PostgresChunkStore(pool, dimensions=DIMENSIONS),
            PostgresSessionStore(pool),
            PostgresUsageLedger(pool),
            FakeEmbeddingProvider(DIMENSIONS),
        )


def kinds() -> Iterator[object]:
    yield "memory"
    yield pytest.param("postgres", marks=pytest.mark.db)
