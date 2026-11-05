"""Hybrid retrieval: embed the question, search by meaning and by words, fuse the rankings, judge the result."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from rag_chatbot.application.ports import ChunkStore, EmbeddingProvider
from rag_chatbot.application.settings import ChatSettings
from rag_chatbot.domain.model import ChatTurn, Retrieval, Usage
from rag_chatbot.domain.retrieval import build_retrieval, fuse
from rag_chatbot.domain.text import search_terms, words

SHORT_QUESTION_WORDS = 6


@dataclass(frozen=True, slots=True)
class RetrievalOutcome:
    retrieval: Retrieval
    embedding_usage: Usage
    embedding_model: str


def search_query(message: str, history: Sequence[ChatTurn]) -> str:
    """The text to search with. A very short follow-up ("and on weekends?") is searched together with the visitor's
    previous question, which is what it refers to."""
    if len(words(message)) <= SHORT_QUESTION_WORDS:
        for turn in reversed(history):
            if turn.role == "user":
                return f"{turn.content} {message}"
    return message


class Retriever:
    def __init__(self, store: ChunkStore, embedder: EmbeddingProvider, settings: ChatSettings) -> None:
        self._store = store
        self._embedder = embedder
        self._settings = settings

    async def retrieve(self, tenant_id: str, query: str) -> RetrievalOutcome:
        embedded = await self._embedder.embed([query])
        vector = embedded.vectors[0]
        limit = self._settings.candidates_per_search
        terms = search_terms(query)
        # one after the other: two short queries, and no half-finished task to clean up when one of them fails
        vector_hits = await self._store.vector_search(tenant_id, vector, limit)
        keyword_hits = await self._store.keyword_search(tenant_id, terms, vector, limit) if terms else []
        retrieval = build_retrieval(
            fuse(vector_hits, keyword_hits),
            max_chunks=self._settings.context_chunks,
            max_chars=self._settings.context_chars,
            relative_cutoff=self._settings.context_relative_cutoff,
        )
        return RetrievalOutcome(retrieval, embedded.usage, embedded.model)
