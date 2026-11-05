"""A content source that leaves out the pages the operator excluded."""

from __future__ import annotations

import fnmatch
from collections.abc import AsyncIterator, Sequence
from typing import Protocol

from rag_chatbot.application.ingest import document_id_for
from rag_chatbot.domain.model import RawPage
from rag_chatbot.infrastructure.sources import SourceReport


class ReportingSource(Protocol):
    report: SourceReport

    def pages(self) -> AsyncIterator[RawPage]: ...


class FilteredSource:
    """Wraps a source and skips every page whose location (`reviews/all.html`, or the URL on a crawl) or document id
    (`reviews/all`, `www.example.com/reviews/all`) matches one of the glob patterns (`reviews/*`, `*/comments/*`,
    `community-notes`). This is how a site keeps user-written content
    (reviews, comments, a forum) out of what the assistant may quote. `report` is the wrapped source's report and
    `excluded` counts the pages left out."""

    def __init__(self, inner: ReportingSource, patterns: Sequence[str]) -> None:
        self._inner = inner
        self._patterns = tuple(patterns)
        self.excluded = 0

    @property
    def report(self) -> SourceReport:
        return self._inner.report  # the wrapped source starts a new report with every run

    async def pages(self) -> AsyncIterator[RawPage]:
        async for page in self._inner.pages():
            names = (page.location, document_id_for(page))
            if any(fnmatch.fnmatchcase(name, pattern) for name in names for pattern in self._patterns):
                self.excluded += 1
                continue
            yield page
