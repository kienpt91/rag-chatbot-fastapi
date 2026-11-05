"""A polite, allow-listed, bounded site crawler.

Everything a website says is untrusted, so the crawler decides before every request, never after:

* Every URL (start URLs, links and each redirect hop) is normalised and must pass the `HostPolicy` (exact host
  names, https only unless allowed, the default port unless listed) before it is requested. A redirect is never
  followed blindly: the fetcher hands the 3xx answer back and the crawler judges the target like any other link.
* `robots.txt` is fetched once per origin and obeyed, including `Crawl-delay`. When it cannot be read the origin is
  treated as fully disallowed.
* Requests to one origin are spaced by `max(delay_s, Crawl-delay)` (at most 60 s) through the injected `sleep`/`now`.
* The crawl is bounded in depth, in pages (the first `max_pages` distinct URLs discovered), in bytes per page and in
  total bytes. Only pages are requested, never the images, styles or scripts they mention, and nothing is executed.
"""

from __future__ import annotations

import asyncio
import codecs
import time
from collections import deque
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from urllib.parse import urljoin, urlsplit

from rag_chatbot.application.ports import Fetcher, FetchResult
from rag_chatbot.domain.errors import SourceError
from rag_chatbot.domain.extraction import extract_html
from rag_chatbot.domain.model import PageKind, RawPage
from rag_chatbot.domain.robots import DISALLOW_ALL, RobotsRules, rules_for_status
from rag_chatbot.domain.urlpolicy import HostPolicy, normalise_url
from rag_chatbot.infrastructure.sources import SourceReport

MAX_REDIRECTS = 5
MAX_DELAY_S = 60.0
_ROBOTS_BYTES = 512 * 1024
_KINDS: dict[str, PageKind] = {
    "text/html": "html",
    "application/xhtml+xml": "html",
    "text/plain": "text",
    "text/markdown": "markdown",
    "text/x-markdown": "markdown",
}
# Codecs that are not plain text encodings of a page, or that other tools have been tricked by.
_REFUSED_CODECS = frozenset({"unicode_escape", "raw_unicode_escape", "undefined", "punycode", "idna", "utf_7"})
_UTF8_BOM = b"\xef\xbb\xbf"
_SNIFF_BYTES = 1_024

_Frontier = deque[tuple[str, int]]  # (normalised URL, depth)


class Crawler:
    """Breadth-first crawl from `start_urls`; a `ContentSource` yielding one `RawPage` per indexable page.

    `report.skipped` reasons: `robots`, `host` (also `scheme`, `port`, `credentials`, `invalid`, `too_long` for
    links the policy refused), `redirect` (loop, too many hops or an unusable target), `duplicate`, `status`
    (not 2xx), `not_html`, `noindex`, `depth`, `limit` (page cap; counts links dropped), `total_bytes` (byte cap;
    counts URLs never requested), `error` (fetch failed) and `truncated` (page kept, but cut at the byte cap).
    A link found on several pages counts for `depth` and `limit` each time. One crawl at a time per instance."""

    def __init__(
        self,
        fetcher: Fetcher,
        *,
        start_urls: Sequence[str],
        policy: HostPolicy,
        user_agent: str,
        max_depth: int = 3,
        max_pages: int = 200,
        max_page_bytes: int = 1_000_000,
        max_total_bytes: int = 50_000_000,
        delay_s: float = 1.0,
        respect_robots: bool = True,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        now: Callable[[], float] = time.monotonic,
    ) -> None:
        self._fetcher = fetcher
        self._start_urls = tuple(start_urls)
        self._policy = policy
        self._user_agent = user_agent
        self._max_depth = max_depth
        self._max_pages = max_pages
        self._max_page_bytes = max_page_bytes
        self._max_total_bytes = max_total_bytes
        self._delay_s = max(0.0, delay_s)
        self._respect_robots = respect_robots
        self._sleep = sleep
        self._now = now
        self.report = SourceReport()
        self._seen: set[str] = set()
        self._robots: dict[str, RobotsRules] = {}
        self._last_request: dict[str, float] = {}

    async def pages(self) -> AsyncIterator[RawPage]:
        """Raises `SourceError` before any request when there is no start URL or one of them is refused."""
        self.report = SourceReport()
        self._seen = set()
        self._robots = {}
        self._last_request = {}
        frontier = self._start_frontier()
        total_bytes = 0
        while frontier:
            url, depth = frontier.popleft()
            budget = min(self._max_page_bytes, self._max_total_bytes - total_bytes)
            if budget <= 0:
                self.report.skip("total_bytes", 1 + len(frontier))
                return
            fetched = await self._fetch_page(url, budget)
            if fetched is None:
                continue
            final_url, result = fetched
            total_bytes += len(result.body)
            page = self._to_page(final_url, result, depth, frontier)
            if page is not None:
                self.report.pages += 1
                yield page

    # --- frontier -----------------------------------------------------------------------------------------------

    def _start_frontier(self) -> _Frontier:
        if not self._start_urls:
            raise SourceError("no start URL was given")
        frontier: _Frontier = deque()
        for raw in self._start_urls:
            reason = self._policy.check(raw)
            target = normalise_url(raw)
            if reason is not None or target is None:
                raise SourceError(f"a start URL is not allowed ({reason or 'invalid'})")
            if target in self._seen:
                continue
            if len(self._seen) >= self._max_pages:
                self.report.skip("limit")
                continue
            self._seen.add(target)
            frontier.append((target, 0))
        return frontier

    def _enqueue_links(self, links: Sequence[str], depth: int, frontier: _Frontier) -> None:
        for link in links:
            reason = self._policy.check(link)
            if reason is not None:
                self.report.skip(reason)
                continue
            target = normalise_url(link)
            if target is None or target in self._seen:
                continue
            if depth + 1 > self._max_depth:
                self.report.skip("depth")
            elif len(self._seen) >= self._max_pages:
                self.report.skip("limit")
            else:
                self._seen.add(target)
                frontier.append((target, depth + 1))

    # --- requests -----------------------------------------------------------------------------------------------

    async def _fetch_page(self, url: str, max_bytes: int) -> tuple[str, FetchResult] | None:
        try:
            return await self._follow(url, max_bytes, check_robots=True)
        except SourceError as exc:
            self.report.skip("error")
            self.report.add_error(f"{_describe(url)}: {exc.detail}")
            return None

    async def _follow(self, url: str, max_bytes: int, *, check_robots: bool) -> tuple[str, FetchResult] | None:
        """GET `url`, following up to five redirects hop by hop. Every hop is policy-checked, and for a page also
        checked against robots.txt. Returns the final URL and its answer, or None when something was skipped."""
        current = url
        hops = {url}
        for _ in range(MAX_REDIRECTS + 1):
            if check_robots and not await self._robots_allow(current):
                self.report.skip("robots")
                return None
            result = await self._request(current, max_bytes)
            if not 300 <= result.status < 400:
                return current, result
            target = self._redirect_target(current, result.location)
            if target is None:
                return None
            if target in hops:
                self.report.skip("redirect")
                return None
            if check_robots:
                if target in self._seen:  # it is, or will be, crawled under its own entry
                    self.report.skip("duplicate")
                    return None
                self._seen.add(target)
            hops.add(target)
            current = target
        self.report.skip("redirect")
        return None

    def _redirect_target(self, current: str, location: str) -> str | None:
        if not location:
            self.report.skip("status")
            return None
        try:
            joined = urljoin(current, location)
        except ValueError:
            self.report.skip("redirect")
            return None
        reason = self._policy.check(joined)
        target = normalise_url(joined)
        if reason is not None or target is None:
            self.report.skip(reason or "redirect")
            return None
        return target

    async def _request(self, url: str, max_bytes: int) -> FetchResult:
        origin = _origin(url)
        await self._pace(origin)
        try:
            return await self._fetcher.fetch(url, max_bytes=max_bytes)
        finally:
            self._last_request[origin] = self._now()

    async def _pace(self, origin: str) -> None:
        """Wait until `max(delay_s, Crawl-delay)` has passed since the last request to this origin ended."""
        last = self._last_request.get(origin)
        if last is None:
            return
        wait = last + self._delay_for(origin) - self._now()
        if wait > 0:
            await self._sleep(wait)

    def _delay_for(self, origin: str) -> float:
        rules = self._robots.get(origin)
        delay = self._delay_s
        if rules is not None and rules.crawl_delay_s is not None:
            delay = max(delay, rules.crawl_delay_s)
        return min(delay, MAX_DELAY_S)

    # --- robots.txt ---------------------------------------------------------------------------------------------

    async def _robots_allow(self, url: str) -> bool:
        if not self._respect_robots:
            return True
        origin = _origin(url)
        rules = self._robots.get(origin)
        if rules is None:
            rules = await self._load_robots(origin)
            self._robots[origin] = rules
        parts = urlsplit(url)
        return rules.allows(parts.path + (f"?{parts.query}" if parts.query else ""))

    async def _load_robots(self, origin: str) -> RobotsRules:
        robots_url = f"{origin}/robots.txt"
        reason = self._policy.check(robots_url)
        if reason is not None:
            self.report.skip(reason)
            return DISALLOW_ALL
        try:
            followed = await self._follow(robots_url, _ROBOTS_BYTES, check_robots=False)
        except SourceError as exc:
            self.report.add_error(f"robots.txt of {_describe(origin)} could not be read: {exc.detail}")
            return DISALLOW_ALL
        if followed is None:
            return DISALLOW_ALL
        _, result = followed
        return rules_for_status(result.status, result.body.decode("utf-8", errors="replace"), self._user_agent)

    # --- answers ------------------------------------------------------------------------------------------------

    def _to_page(self, url: str, result: FetchResult, depth: int, frontier: _Frontier) -> RawPage | None:
        if not 200 <= result.status < 300:
            self.report.skip("status")
            return None
        kind = _kind_of(result)
        if kind is None:
            self.report.skip("not_html")
            return None
        if result.truncated:
            self.report.skip("truncated")
        text = _decode(result.body, result.charset)
        if kind == "html":
            # Links are followed even from a page that asks not to be indexed.
            extracted = extract_html(text, base_url=url)
            self._enqueue_links(extracted.links, depth, frontier)
            if extracted.noindex:
                self.report.skip("noindex")
                return None
        return RawPage(location=url, kind=kind, content=text, url=url)


def _origin(url: str) -> str:
    parts = urlsplit(url)
    return f"{parts.scheme}://{parts.netloc}"


def _describe(url: str) -> str:
    """Host and the start of the path: enough to find the page, nothing a query string could carry."""
    parts = urlsplit(url)
    return f"{parts.netloc}{parts.path[:80]}"


def _kind_of(result: FetchResult) -> PageKind | None:
    if result.content_type:
        return _KINDS.get(result.content_type)
    head = result.body[:_SNIFF_BYTES].removeprefix(_UTF8_BOM).lstrip()
    return "html" if head.startswith(b"<") else None


def _decode(body: bytes, charset: str) -> str:
    """The body as text in the declared charset (UTF-8 when none, unknown or unsuitable), BOM removed."""
    encoding = charset if _usable_charset(charset) else "utf-8"
    try:
        text = body.decode(encoding, errors="replace")
    except (LookupError, UnicodeError):  # not a text codec after all
        text = body.decode("utf-8", errors="replace")
    return text.removeprefix("\ufeff")


def _usable_charset(charset: str) -> bool:
    if not charset:
        return False
    try:
        return codecs.lookup(charset).name.replace("-", "_") not in _REFUSED_CODECS  # the registry names them with '-'
    except LookupError:
        return False
