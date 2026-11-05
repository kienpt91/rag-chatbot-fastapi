"""Small builders shared by the tests."""

from __future__ import annotations

from rag_chatbot.domain.model import Chunk, Retrieval, RetrievedChunk, SearchHit


def chunk(
    chunk_id: str = "ck_a",
    text: str = "We open at nine.",
    *,
    tenant: str = "acme",
    document: str = "hours",
    title: str = "Opening hours",
    url: str = "https://acme.example/hours",
    ordinal: int = 0,
) -> Chunk:
    return Chunk(chunk_id, tenant, document, ordinal, title, url, text)


def hit(chunk_id: str, similarity: float, text: str = "text") -> SearchHit:
    return SearchHit(chunk(chunk_id, text), similarity)


def retrieved(
    chunk_id: str,
    similarity: float = 0.8,
    text: str = "We open at nine.",
    *,
    title: str = "Opening hours",
    url: str = "https://acme.example/hours",
) -> RetrievedChunk:
    return RetrievedChunk(chunk(chunk_id, text, title=title, url=url), similarity, 1, 1, 0.03)


def retrieval(*items: RetrievedChunk) -> Retrieval:
    return Retrieval(tuple(items), max((i.similarity for i in items), default=0.0))
