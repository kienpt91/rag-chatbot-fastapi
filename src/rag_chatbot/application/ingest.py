"""The ingestion use case: pages in, embedded chunks stored.

A page goes through the same steps whatever its source: text extraction, chunking with stable ids, embedding of the
chunks the database does not already hold, and one atomic replacement of the document's chunks. Re-running ingestion
on an unchanged site therefore embeds nothing and costs nothing.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field

from rag_chatbot.application.ports import ChunkStore, ContentSource, EmbeddingProvider
from rag_chatbot.application.settings import IngestSettings
from rag_chatbot.domain.chunking import chunk_document
from rag_chatbot.domain.errors import ProviderError
from rag_chatbot.domain.extraction import extract_page
from rag_chatbot.domain.model import Chunk, RawPage, SourceDocument, Usage
from rag_chatbot.domain.pricing import PriceTable
from rag_chatbot.domain.quarantine import quarantine
from rag_chatbot.domain.scope import page_instruction_reason
from rag_chatbot.domain.text import normalise_text
from rag_chatbot.domain.urlpolicy import document_id_for_path, document_id_for_url

log = logging.getLogger("rag_chatbot.ingest")
_MAX_TITLE_CHARS = 200


MAX_REPORTED_QUARANTINED = 200


@dataclass(frozen=True, slots=True)
class Quarantined:
    document_id: str
    sentence: str


@dataclass(slots=True)
class IngestReport:
    pages_read: int = 0
    documents_indexed: int = 0
    documents_skipped: dict[str, int] = field(default_factory=dict)  # reason -> count: noindex, empty
    chunks_total: int = 0
    chunks_embedded: int = 0
    chunks_reused: int = 0
    chunks_dropped: int = 0  # over the per-document cap
    quarantined_total: int = 0  # instruction-like sentences left out of the index
    quarantined: list[Quarantined] = field(default_factory=list)  # the first of them, for the operator to read
    embedding_usage: Usage = field(default_factory=Usage)
    cost_micros: int = 0
    document_ids: set[str] = field(default_factory=set)

    def skip(self, reason: str) -> None:
        self.documents_skipped[reason] = self.documents_skipped.get(reason, 0) + 1


def document_id_for(page: RawPage) -> str:
    if page.location.startswith(("http://", "https://")):
        return document_id_for_url(page.location)
    return document_id_for_path(page.location)


def _fallback_title(document_id: str) -> str:
    last = document_id.rstrip("/").rsplit("/", 1)[-1] or document_id
    return last.replace("-", " ").replace("_", " ").strip().title() or document_id


class IngestService:
    def __init__(
        self, store: ChunkStore, embedder: EmbeddingProvider, prices: PriceTable, settings: IngestSettings | None = None
    ) -> None:
        self._store = store
        self._embedder = embedder
        self._prices = prices
        self._settings = settings or IngestSettings()

    async def run(self, tenant_id: str, source: ContentSource) -> IngestReport:
        report = IngestReport()
        async for page in source.pages():
            report.pages_read += 1
            await self._ingest_page(tenant_id, page, report)
        report.cost_micros = self._prices.cost(self._embedder.model, report.embedding_usage).micros
        return report

    async def prune(self, tenant_id: str, report: IngestReport) -> int:
        """Remove the tenant's documents that the run did not see. Only call it after a *complete* run: a crawl that
        stopped at a limit or hit errors has not seen the whole site."""
        return await self._store.prune_documents(tenant_id, report.document_ids)

    async def _ingest_page(self, tenant_id: str, page: RawPage, report: IngestReport) -> None:
        extracted = extract_page(page)
        document_id = document_id_for(page)
        if extracted.noindex:
            report.skip("noindex")
            return
        text = normalise_text(extracted.text)
        title = (extracted.title or _fallback_title(document_id))[:_MAX_TITLE_CHARS]
        if self._settings.quarantine:
            text, removed = quarantine(text)
            if page_instruction_reason(title) is not None:  # a title goes into every prompt that cites the page
                removed.append(title)
                title = _fallback_title(document_id)
            report.quarantined_total += len(removed)
            report.quarantined.extend(
                Quarantined(document_id, s[:300]) for s in removed[: MAX_REPORTED_QUARANTINED - len(report.quarantined)]
            )
        if not text:
            report.skip("empty")
            return
        document = SourceDocument(
            tenant_id=tenant_id,
            document_id=document_id,
            url=page.url,
            title=title,
            text=text,
        )
        chunks = chunk_document(document, self._settings.chunking)
        if len(chunks) > self._settings.max_chunks_per_document:
            report.chunks_dropped += len(chunks) - self._settings.max_chunks_per_document
            chunks = chunks[: self._settings.max_chunks_per_document]
        known = await self._store.known_chunk_ids(tenant_id, document_id)
        new = [c for c in chunks if c.chunk_id not in known]
        embeddings = await self._embed(new, report)
        await self._store.upsert_document(
            tenant_id, document_id, url=document.url, title=document.title, chunks=chunks, embeddings=embeddings
        )
        report.document_ids.add(document_id)
        report.documents_indexed += 1
        report.chunks_total += len(chunks)
        report.chunks_reused += len(chunks) - len(new)
        report.chunks_embedded += len(new)
        log.info("document indexed", extra={"tenant": tenant_id, "chunks": len(chunks), "embedded": len(new)})

    async def _embed(self, chunks: Sequence[Chunk], report: IngestReport) -> dict[str, tuple[float, ...]]:
        embeddings: dict[str, tuple[float, ...]] = {}
        size = self._settings.embed_batch_size
        for start in range(0, len(chunks), size):
            batch = chunks[start : start + size]
            result = await self._embedder.embed([c.text for c in batch])
            if len(result.vectors) != len(batch) or any(len(v) != self._embedder.dimensions for v in result.vectors):
                raise ProviderError("the embedding service returned vectors of an unexpected number or size")
            report.embedding_usage += result.usage
            for chunk, vector in zip(batch, result.vectors, strict=True):
                embeddings[chunk.chunk_id] = vector
        return embeddings
