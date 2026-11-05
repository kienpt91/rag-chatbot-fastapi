"""`--exclude` applies to crawled pages too: patterns match the URL path, case-insensitively, ignoring the query."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

import pytest

from rag_chatbot.domain.model import RawPage
from rag_chatbot.infrastructure.sources import SourceReport
from rag_chatbot.infrastructure.sources.filtered import FilteredSource

HOST = "https://www.client.example"


class Inner:
    def __init__(self, urls: list[str]) -> None:
        self._urls = urls
        self.report = SourceReport()

    async def pages(self) -> AsyncIterator[RawPage]:
        for url in self._urls:
            yield RawPage(url, "html", "<main>text</main>", url)


def excluded(pattern: str, url: str) -> bool:
    source = FilteredSource(Inner([url]), [pattern])

    async def collect() -> list[str]:
        return [p.location async for p in source.pages()]

    kept = asyncio.run(collect())
    return not kept


@pytest.mark.parametrize(
    ("pattern", "path"),
    [
        ("reviews/*", "/reviews/x"),
        ("/reviews/*", "/reviews/x"),
        ("*/reviews/*", "/reviews/x"),
        ("reviews/*", "/Reviews/X"),
        ("Reviews/*", "/reviews/x"),
        ("reviews/*", "/reviews/x?page=2"),
        ("*/reviews/all", "/reviews/all?a=1"),
        ("reviews/*", "/reviews%2Fx"),
        ("reviews", "/reviews/"),
        ("www.client.example/reviews/*", "/reviews/x"),
    ],
)
def test_a_pattern_written_for_a_folder_or_a_path_excludes_the_crawled_page(pattern: str, path: str) -> None:
    assert excluded(pattern, HOST + path)


@pytest.mark.parametrize(
    ("pattern", "path"),
    [
        ("reviews/*", "/blog/reviews/x"),
        ("forum/*", "/reviews/x"),
        ("reviews/*", "/review/x"),
        ("*/reviews/all", "/reviews/all-time"),
    ],
)
def test_other_pages_are_kept(pattern: str, path: str) -> None:
    assert not excluded(pattern, HOST + path)
