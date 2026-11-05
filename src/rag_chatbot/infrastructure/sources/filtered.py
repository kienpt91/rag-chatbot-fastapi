"""A content source that leaves out the pages the operator excluded."""

from __future__ import annotations

import fnmatch
from collections.abc import AsyncIterator, Sequence
from typing import Protocol
from urllib.parse import unquote, urlsplit

from rag_chatbot.application.ingest import document_id_for
from rag_chatbot.domain.model import RawPage
from rag_chatbot.infrastructure.sources import SourceReport


class ReportingSource(Protocol):
    report: SourceReport

    def pages(self) -> AsyncIterator[RawPage]: ...


def names_of(page: RawPage) -> set[str]:
    """Every name a pattern may refer to a page by, lower case: its location, its document id, and for a page with a URL
    its path (percent-decoded, without the query) with and without the leading and trailing slash and with the host."""
    names = {page.location.lower(), document_id_for(page).lower()}
    names.add(document_id_for(page).lower().split("?", 1)[0])
    if page.location.startswith(("http://", "https://")):
        parts = urlsplit(page.location)
        path = unquote(parts.path).lower()
        host = (parts.hostname or "").lower()
        for variant in {path, path.rstrip("/")}:
            names |= {variant, variant.lstrip("/"), f"{host}{variant}"}
    return {n for n in names if n}


class FilteredSource:
    """Wraps a source and skips every page that a glob pattern names. A pattern is matched, case-insensitively,
    against the page's relative path (a folder), or the path of its URL without the query (a crawl): `reviews/*`,
    `/reviews/*` and `*/reviews/*` all exclude `https://host/Reviews/x?page=2`; the host form `host/reviews/*` works
    too. This is how a site keeps user-written content (reviews, comments, a forum) out of what the assistant may quote.
    `report` is the wrapped source's report and `excluded` counts the pages left out."""

    def __init__(self, inner: ReportingSource, patterns: Sequence[str]) -> None:
        self._inner = inner
        self._patterns = tuple(p.lower() for p in patterns)
        self.excluded = 0

    @property
    def report(self) -> SourceReport:
        return self._inner.report  # the wrapped source starts a new report with every run

    async def pages(self) -> AsyncIterator[RawPage]:
        async for page in self._inner.pages():
            names = names_of(page)
            if any(fnmatch.fnmatchcase(name, pattern) for name in names for pattern in self._patterns):
                self.excluded += 1
                continue
            yield page
