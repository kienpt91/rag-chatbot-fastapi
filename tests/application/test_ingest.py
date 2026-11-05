from __future__ import annotations

import asyncio

import pytest

from rag_chatbot.application.ingest import IngestService, document_id_for
from rag_chatbot.application.settings import IngestSettings
from rag_chatbot.domain.chunking import ChunkingConfig
from rag_chatbot.domain.errors import ProviderError
from rag_chatbot.domain.model import EmbeddingResult, RawPage, Usage
from rag_chatbot.infrastructure.memory import InMemoryChunkStore
from rag_chatbot.infrastructure.providers.fake import FakeEmbeddingProvider
from support.fakes import RecordingEmbedder
from support.service import DIMENSIONS, PAGES, PRICES, ListSource, build_service


def page(name: str, text: str, kind: str = "text") -> RawPage:
    return RawPage(name, kind, text, f"https://www.acme.example/{name}")  # type: ignore[arg-type]


def test_pages_become_chunks_with_a_title_a_url_and_a_report_with_cost() -> None:
    service = build_service()
    report = asyncio.run(service.load())
    assert report.pages_read == 3 and report.documents_indexed == 3 and report.chunks_total == 3
    assert report.chunks_embedded == 3 and report.chunks_reused == 0 and report.embedding_usage.input_tokens > 0
    assert report.cost_micros == PRICES.cost("fake-embedding-1", report.embedding_usage).micros > 0
    assert report.document_ids == {"hours", "menu", "delivery"}
    assert asyncio.run(service.chunks.count_chunks("acme")) == 3


def test_ingesting_an_unchanged_site_again_embeds_nothing_and_costs_nothing() -> None:
    service = build_service()
    asyncio.run(service.load())
    embedded_before = len(service.embedder.texts)
    again = asyncio.run(service.load())
    assert again.chunks_embedded == 0 and again.chunks_reused == 3 and again.cost_micros == 0
    assert len(service.embedder.texts) == embedded_before


def test_an_edit_embeds_only_the_chunks_that_changed() -> None:
    service = build_service()
    long_text = "\n\n".join(f"Paragraph {i} about bread and flour and yeast. " * 12 for i in range(12))
    asyncio.run(service.load([page("story", long_text)]))
    edited = long_text.replace("Paragraph 11 about", "Paragraph 11 was rewritten about")
    report = asyncio.run(service.load([page("story", edited)]))
    assert 1 <= report.chunks_embedded <= 2 and report.chunks_reused >= report.chunks_total - 2


def test_noindex_and_empty_pages_are_skipped_and_counted() -> None:
    service = build_service()
    pages = [
        RawPage("secret.html", "html", '<meta name="robots" content="noindex"><main><p>Private text here.</p></main>'),
        RawPage("blank.html", "html", "<main>   </main>"),
        page("ok", "A real page with some text."),
    ]
    report = asyncio.run(service.load(pages))
    assert report.documents_skipped == {"noindex": 1, "empty": 1} and report.documents_indexed == 1
    assert asyncio.run(service.chunks.count_chunks("acme")) == 1


def test_the_title_comes_from_the_page_or_from_its_name() -> None:
    service = build_service()
    asyncio.run(service.load([page("a", "text"), RawPage("opening-hours.txt", "text", "Some text")]))
    hits = asyncio.run(
        service.chunks.vector_search("acme", asyncio.run(service.embedder.embed(["text"])).vectors[0], 5)
    )
    titles = {h.chunk.document_id: h.chunk.title for h in hits}
    assert titles["opening-hours"] == "Opening Hours"


def test_a_giant_page_is_capped_per_document() -> None:
    service = build_service()
    service.ingest = IngestService(service.chunks, service.embedder, PRICES, IngestSettings(max_chunks_per_document=5))
    text = "\n\n".join(f"Paragraph number {i} " + "word " * 150 for i in range(200))
    report = asyncio.run(service.load([page("giant", text)]))
    assert report.chunks_total == 5 and report.chunks_dropped > 100
    assert asyncio.run(service.chunks.count_chunks("acme")) == 5


def test_a_wrong_sized_vector_from_the_provider_is_refused_and_nothing_is_stored() -> None:
    class Bad:
        model = "bad"
        dimensions = DIMENSIONS

        async def embed(self, texts):  # type: ignore[no-untyped-def]
            return EmbeddingResult(tuple((0.1,) * 3 for _ in texts), Usage(1), "bad")

    store = InMemoryChunkStore(dimensions=DIMENSIONS)
    ingest = IngestService(store, Bad(), PRICES)  # type: ignore[arg-type]
    with pytest.raises(ProviderError):
        asyncio.run(ingest.run("acme", ListSource([page("a", "some text")])))
    assert asyncio.run(store.count_chunks("acme")) == 0


def test_the_wrong_number_of_vectors_is_refused() -> None:
    class Short:
        model = "short"
        dimensions = DIMENSIONS

        async def embed(self, texts):  # type: ignore[no-untyped-def]
            return EmbeddingResult((), Usage(1), "short")

    ingest = IngestService(InMemoryChunkStore(dimensions=DIMENSIONS), Short(), PRICES)  # type: ignore[arg-type]
    with pytest.raises(ProviderError):
        asyncio.run(ingest.run("acme", ListSource([page("a", "some text")])))


def test_embedding_is_done_in_batches() -> None:
    store = InMemoryChunkStore(dimensions=DIMENSIONS)
    calls: list[int] = []

    class Counting(RecordingEmbedder):
        async def embed(self, texts):  # type: ignore[no-untyped-def]
            calls.append(len(texts))
            return await super().embed(texts)

    embedder = Counting(FakeEmbeddingProvider(DIMENSIONS))
    settings = IngestSettings(chunking=ChunkingConfig(40, 60, 0, 10), embed_batch_size=4)
    ingest = IngestService(store, embedder, PRICES, settings)  # type: ignore[arg-type]
    text = "\n\n".join(f"Paragraph {i} " + "word " * 30 for i in range(10))
    report = asyncio.run(ingest.run("acme", ListSource([page("p", text)])))
    assert sum(calls) == report.chunks_embedded and max(calls) <= 4 and len(calls) >= 2


def test_pruning_removes_what_a_complete_run_did_not_see() -> None:
    service = build_service()
    asyncio.run(service.load())
    report = asyncio.run(service.load(PAGES[:2]))
    assert asyncio.run(service.ingest.prune("acme", report)) == 1
    assert asyncio.run(service.chunks.count_chunks("acme")) == 2


@pytest.mark.parametrize(
    ("location", "expected"),
    [
        ("hours.html", "hours"),
        ("policies/returns.md", "policies/returns"),
        ("https://www.Acme.example/menu?x=1#top", "www.acme.example/menu?x=1"),
        ("https://www.acme.example/", "www.acme.example"),
    ],
)
def test_the_document_id_follows_the_location(location: str, expected: str) -> None:
    assert document_id_for(RawPage(location, "text", "x")) == expected


def test_the_sample_pages_are_what_the_other_tests_assume() -> None:
    assert [p.location for p in PAGES] == ["hours.html", "menu.md", "delivery.txt"]
