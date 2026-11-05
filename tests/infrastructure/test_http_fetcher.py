"""The HTTP fetcher: SSRF refusal, redirects, size and time bounds, and error reporting.

Real requests only ever go to servers this test starts on 127.0.0.1 (port 0); everything else is refused or served
by an injected resolver and connect hook, so no test touches the network.
"""

from __future__ import annotations

import asyncio
import itertools
import socket
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import cast

import pytest

from rag_chatbot.application.ports import FetchResult
from rag_chatbot.domain.errors import ConfigurationError, SourceError
from rag_chatbot.infrastructure.sources.http_fetcher import (
    Connector,
    HttpFetcher,
    Resolver,
    _Deadline,
    _DeadlineSocket,
    is_public_address,
    parse_content_type,
    tcp_connect,
)

AGENT = "TestCrawler/1.0 (+https://example.com/bot)"


@dataclass
class _Route:
    status: int = 200
    headers: dict[str, str] = field(default_factory=dict)
    body: bytes = b""
    declared_length: int | None = None  # Content-Length when it should differ from the body
    chunks: list[bytes] | None = None  # send chunked instead
    stall_after_body: bool = False  # keep the connection open and silent after the body
    silent: bool = False  # accept the request and never answer


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, format: str, *args: object) -> None:
        return

    def do_GET(self) -> None:
        server = cast("_Server", self.server)
        server.seen.append((self.path, {name.lower(): value for name, value in self.headers.items()}))
        route = server.routes.get(self.path, _Route(status=404, body=b"missing"))
        if route.silent:
            server.release.wait(30)
            return
        self.send_response(route.status)
        for name, value in route.headers.items():
            self.send_header(name, value)
        if route.chunks is not None:
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            for chunk in route.chunks:
                self.wfile.write(f"{len(chunk):x}\r\n".encode("ascii") + chunk + b"\r\n")
            self.wfile.write(b"0\r\n\r\n")
            return
        length = len(route.body) if route.declared_length is None else route.declared_length
        self.send_header("Content-Length", str(length))
        self.end_headers()
        self.wfile.write(route.body)
        self.wfile.flush()
        if route.stall_after_body:
            server.release.wait(30)


class _Server(ThreadingHTTPServer):
    def __init__(self) -> None:
        super().__init__(("127.0.0.1", 0), _Handler)
        self.routes: dict[str, _Route] = {}
        self.seen: list[tuple[str, dict[str, str]]] = []
        self.release = threading.Event()  # lets handlers that stall on purpose finish

    @property
    def port(self) -> int:
        return int(self.server_address[1])

    def url(self, path: str = "/") -> str:
        return f"http://127.0.0.1:{self.port}{path}"


@pytest.fixture
def server() -> Iterator[_Server]:
    instance = _Server()
    thread = threading.Thread(target=instance.serve_forever, kwargs={"poll_interval": 0.05})
    thread.start()
    try:
        yield instance
    finally:
        instance.release.set()
        instance.shutdown()
        instance.server_close()
        thread.join()


@contextmanager
def _raw_server(reply: bytes) -> Iterator[int]:
    """A one-connection TCP server that reads the first bytes the client sends, answers `reply` and hangs up."""
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    listener.settimeout(10)

    def serve() -> None:
        try:
            connection, _ = listener.accept()
        except OSError:
            return
        with connection:
            connection.settimeout(10)
            try:
                connection.recv(4096)
                connection.sendall(reply)
            except OSError:
                return

    thread = threading.Thread(target=serve)
    thread.start()
    try:
        yield int(listener.getsockname()[1])
    finally:
        listener.close()
        thread.join()


def _local_fetcher(
    *,
    timeout_s: float = 10.0,
    resolver: Resolver | None = None,
    connect: Connector | None = None,
    clock: Callable[[], float] = time.monotonic,
) -> HttpFetcher:
    return HttpFetcher(
        user_agent=AGENT,
        timeout_s=timeout_s,
        allow_private_addresses=True,
        resolver=resolver,
        connect=connect,
        clock=clock,
    )


def _fetch(fetcher: HttpFetcher, url: str, max_bytes: int = 1_000_000) -> FetchResult:
    return asyncio.run(fetcher.fetch(url, max_bytes=max_bytes))


class _Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


# --- the happy path -------------------------------------------------------------------------------------------


def test_a_page_is_fetched_with_its_media_type_charset_and_status(server: _Server) -> None:
    server.routes["/page"] = _Route(headers={"Content-Type": "Text/HTML; charset=UTF-8"}, body=b"<p>hi</p>")
    result = _fetch(_local_fetcher(), server.url("/page"))
    assert result == FetchResult(
        url=server.url("/page"),
        status=200,
        content_type="text/html",
        charset="utf-8",
        body=b"<p>hi</p>",
        truncated=False,
        location="",
    )


def test_the_request_names_us_asks_for_plain_text_and_closes_the_connection(server: _Server) -> None:
    server.routes["/q"] = _Route(body=b"ok")
    _fetch(_local_fetcher(), server.url("/q?a=1&b=%7E#frag"))
    ((path, headers),) = server.seen
    assert path == "/q?a=1&b=~"  # the normalised target; the fragment never leaves the client
    assert headers["user-agent"] == AGENT
    assert headers["accept"] == "text/html,text/plain,text/markdown;q=0.9,*/*;q=0.1"
    assert headers["accept-encoding"] == "identity"
    assert headers["connection"] == "close"
    assert headers["host"] == f"127.0.0.1:{server.port}"


def test_the_connection_goes_to_the_validated_address_while_the_host_header_keeps_the_name(server: _Server) -> None:
    server.routes["/"] = _Route(body=b"named")
    resolved: list[tuple[str, int]] = []
    connected: list[str] = []

    def resolver(host: str, port: int) -> list[str]:
        resolved.append((host, port))
        return ["127.0.0.1"]

    def connect(address: str, port: int, timeout_s: float) -> socket.socket:
        connected.append(address)
        return tcp_connect(address, port, timeout_s)

    fetcher = _local_fetcher(resolver=resolver, connect=connect)
    result = _fetch(fetcher, f"http://Site.Example.:{server.port}/")
    assert result.body == b"named"
    assert resolved == [("site.example", server.port)]
    assert connected == ["127.0.0.1"]
    assert server.seen[0][1]["host"] == f"site.example:{server.port}"


def test_error_statuses_are_returned_not_raised(server: _Server) -> None:
    server.routes["/gone"] = _Route(status=404, body=b"nothing here", headers={"Content-Type": "text/plain"})
    server.routes["/boom"] = _Route(status=503)
    fetcher = _local_fetcher()
    assert _fetch(fetcher, server.url("/gone")).status == 404
    assert _fetch(fetcher, server.url("/boom")).status == 503


def test_a_chunked_answer_is_reassembled(server: _Server) -> None:
    server.routes["/c"] = _Route(chunks=[b"ab", b"cde", b"f"])
    assert _fetch(_local_fetcher(), server.url("/c")).body == b"abcdef"


# --- redirects, compression -----------------------------------------------------------------------------------


@pytest.mark.parametrize("status", [301, 302, 303, 307, 308])
def test_a_redirect_is_returned_with_its_location_and_never_followed(server: _Server, status: int) -> None:
    server.routes["/old"] = _Route(status=status, headers={"Location": "/new"}, body=b"moved")
    server.routes["/new"] = _Route(body=b"target")
    result = _fetch(_local_fetcher(), server.url("/old"))
    assert (result.status, result.location, result.body) == (status, "/new", b"")
    assert [path for path, _ in server.seen] == ["/old"]


def test_a_redirect_to_a_private_host_is_returned_as_data_not_followed(server: _Server) -> None:
    target = "http://169.254.169.254/latest/meta-data/"
    server.routes["/r"] = _Route(status=302, headers={"Location": target})
    result = _fetch(_local_fetcher(), server.url("/r"))
    assert result.location == target
    assert len(server.seen) == 1


def test_an_absurdly_long_location_is_dropped(server: _Server) -> None:
    server.routes["/r"] = _Route(status=302, headers={"Location": "/" + "a" * 9_000})
    assert _fetch(_local_fetcher(), server.url("/r")).location == ""


def test_a_compressed_answer_is_refused_because_none_was_requested(server: _Server) -> None:
    server.routes["/z"] = _Route(headers={"Content-Encoding": "gzip"}, body=b"\x1f\x8b\x08")
    with pytest.raises(SourceError, match="compressed"):
        _fetch(_local_fetcher(), server.url("/z"))


# --- size bounds ----------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("max_bytes", "expected_length", "truncated"),
    [(1_000, 1_000, True), (9_999, 9_999, True), (10_000, 10_000, False), (10_001, 10_000, False), (0, 0, True)],
)
def test_the_body_is_cut_at_the_cap_and_flagged(
    server: _Server, max_bytes: int, expected_length: int, truncated: bool
) -> None:
    server.routes["/big"] = _Route(body=b"x" * 10_000)
    result = _fetch(_local_fetcher(), server.url("/big"), max_bytes=max_bytes)
    assert len(result.body) == expected_length
    assert result.truncated is truncated


def test_a_huge_declared_length_is_not_trusted_and_not_waited_for(server: _Server) -> None:
    server.routes["/lie"] = _Route(body=b"y" * 5_000, declared_length=10**9)
    result = _fetch(_local_fetcher(), server.url("/lie"), max_bytes=1_000)
    assert len(result.body) == 1_000
    assert result.truncated is True


def test_a_chunked_body_is_cut_at_the_cap_too(server: _Server) -> None:
    server.routes["/c"] = _Route(chunks=[b"a" * 400] * 10)
    result = _fetch(_local_fetcher(), server.url("/c"), max_bytes=1_000)
    assert (len(result.body), result.truncated) == (1_000, True)


def test_oversized_headers_are_a_protocol_failure(server: _Server) -> None:
    server.routes["/h"] = _Route(headers={"X-Big": "a" * 70_000}, body=b"x")
    with pytest.raises(SourceError, match="protocol failure"):
        _fetch(_local_fetcher(), server.url("/h"))


# --- time bounds ----------------------------------------------------------------------------------------------


def test_a_server_that_stalls_after_a_few_bytes_ends_in_a_timeout(server: _Server) -> None:
    # The server never sends the rest, so the outcome does not depend on how fast this machine is.
    server.routes["/slow"] = _Route(body=b"abc", declared_length=100, stall_after_body=True)
    with pytest.raises(SourceError, match=r"^timeout$"):
        _fetch(_local_fetcher(timeout_s=0.3), server.url("/slow"))


def test_a_server_that_never_answers_ends_in_a_timeout(server: _Server) -> None:
    server.routes["/mute"] = _Route(silent=True)
    with pytest.raises(SourceError, match=r"^timeout$"):
        _fetch(_local_fetcher(timeout_s=0.3), server.url("/mute"))


def test_a_spent_deadline_stops_the_fetch_before_any_connection() -> None:
    times = itertools.chain([0.0], itertools.repeat(1_000.0))
    attempts: list[str] = []

    def connect(address: str, port: int, timeout_s: float) -> socket.socket:
        attempts.append(address)
        raise AssertionError("must not connect")

    fetcher = _local_fetcher(clock=lambda: next(times), connect=connect)
    with pytest.raises(SourceError, match=r"^timeout$"):
        _fetch(fetcher, "http://127.0.0.1:9/")
    assert attempts == []


def test_the_connect_timeout_is_what_is_left_of_the_budget() -> None:
    clock = _Clock()
    seen: list[float] = []

    def connect(address: str, port: int, timeout_s: float) -> socket.socket:
        seen.append(timeout_s)
        raise ConnectionRefusedError

    clock.now = 100.0
    fetcher = _local_fetcher(clock=clock, connect=connect, timeout_s=8.0)
    with pytest.raises(SourceError, match="could not connect"):
        _fetch(fetcher, "http://127.0.0.1:9/")
    assert seen == [8.0]


def test_the_deadline_counts_down_and_raises_at_zero() -> None:
    clock = _Clock()
    deadline = _Deadline(clock, 10.0)
    assert deadline.remaining() == 10.0
    clock.now = 3.5
    assert deadline.remaining() == 6.5
    clock.now = 10.0
    with pytest.raises(TimeoutError):
        deadline.remaining()
    clock.now = 11.0
    with pytest.raises(TimeoutError):
        deadline.remaining()


def test_every_socket_operation_gets_only_the_time_that_is_left() -> None:
    clock = _Clock()
    deadline = _Deadline(clock, 10.0)
    ours, theirs = socket.socketpair()
    try:
        wrapper = _DeadlineSocket(ours, deadline)
        clock.now = 4.0
        wrapper.sendall(b"x")
        assert ours.gettimeout() == pytest.approx(6.0)
        assert theirs.recv(1) == b"x"
        reader = wrapper.makefile("rb")
        theirs.sendall(b"y")
        clock.now = 9.5
        assert reader.read1(10) == b"y"
        assert ours.gettimeout() == pytest.approx(0.5)
        clock.now = 10.0
        with pytest.raises(TimeoutError):
            reader.read1(1)
        with pytest.raises(TimeoutError):
            wrapper.sendall(b"z")
    finally:
        ours.close()
        theirs.close()


# --- SSRF -----------------------------------------------------------------------------------------------------

REFUSED_LITERALS = [
    "http://127.0.0.1:8080/",
    "http://127.1.2.3/",
    "http://10.0.0.1/",
    "http://172.16.5.4/",
    "http://192.168.1.1/",
    "http://169.254.169.254/latest/meta-data/",
    "http://0.0.0.0/",
    "http://100.64.0.1/",
    "http://224.0.0.1/",
    "http://240.0.0.1/",
    "http://255.255.255.255/",
    "http://192.0.2.1/",
    "http://[::1]/",
    "http://[::]/",
    "http://[::ffff:127.0.0.1]/",
    "http://[::ffff:10.0.0.1]/",
    "http://[::ffff:169.254.169.254]/",
    "http://[::127.0.0.1]/",
    "http://[64:ff9b::7f00:1]/",
    "http://[2002:7f00:1::]/",
    "http://[fe80::1]/",
    "http://[fc00::1]/",
    "http://[fd12:3456::1]/",
    "http://[ff02::1]/",
    "http://[2001:db8::1]/",
]


def _refusing_fetcher(
    resolver: Callable[[str, int], list[str]] | None = None,
) -> tuple[HttpFetcher, list[str]]:
    attempts: list[str] = []

    def connect(address: str, port: int, timeout_s: float) -> socket.socket:
        attempts.append(address)
        raise ConnectionRefusedError

    def default_resolver(host: str, port: int) -> list[str]:
        raise AssertionError("an IP literal must not be resolved")

    fetcher = HttpFetcher(user_agent=AGENT, resolver=resolver or default_resolver, connect=connect)
    return fetcher, attempts


@pytest.mark.parametrize("url", REFUSED_LITERALS)
def test_non_public_ip_literals_are_refused_without_a_connection_attempt(url: str) -> None:
    fetcher, attempts = _refusing_fetcher()
    with pytest.raises(SourceError, match="non-public"):
        _fetch(fetcher, url)
    assert attempts == []


@pytest.mark.parametrize(
    "addresses",
    [
        ["127.0.0.1"],
        ["::1"],
        ["::ffff:10.0.0.1"],
        ["169.254.169.254"],
        ["93.184.216.34", "10.0.0.5"],  # one private answer among public ones is enough to refuse
        ["10.0.0.5", "93.184.216.34"],
        ["93.184.216.34", "2606:2800:220:1::1", "192.168.0.9"],
        ["fe80::1%en0"],
    ],
)
def test_a_name_with_any_non_public_address_is_refused_without_a_connection_attempt(addresses: list[str]) -> None:
    fetcher, attempts = _refusing_fetcher(lambda host, port: addresses)
    with pytest.raises(SourceError, match="non-public"):
        _fetch(fetcher, "https://www.example.com/")
    assert attempts == []


def test_a_loopback_server_is_refused_unless_private_addresses_are_allowed(server: _Server) -> None:
    server.routes["/"] = _Route(body=b"internal")
    with pytest.raises(SourceError, match="non-public"):
        _fetch(HttpFetcher(user_agent=AGENT), server.url("/"))
    assert server.seen == []
    assert _fetch(_local_fetcher(), server.url("/")).body == b"internal"


@pytest.mark.parametrize(
    "url",
    [
        "",
        "/relative",
        "ftp://example.com/",
        "file:///etc/passwd",
        "gopher://example.com/",
        "http://user:pw@example.com/",
        "http://user@127.0.0.1/",
        "http://example.com/\r\nX-Injected: 1",
        "http://example.com\\@evil.test/",
        "http://exa mple.com/",
    ],
)
def test_only_plain_http_urls_without_credentials_are_fetched(url: str) -> None:
    fetcher, attempts = _refusing_fetcher(lambda host, port: ["93.184.216.34"])
    with pytest.raises(SourceError):
        _fetch(fetcher, url)
    assert attempts == []


def test_public_addresses_are_connected_to_by_address_never_by_name() -> None:
    fetcher, attempts = _refusing_fetcher(lambda host, port: ["93.184.216.34", "2606:2800:220:1::1"])
    with pytest.raises(SourceError, match="could not connect"):
        _fetch(fetcher, "http://www.example.com/")
    assert attempts == ["93.184.216.34", "2606:2800:220:1::1"]


def test_at_most_four_addresses_are_tried() -> None:
    many = [f"93.184.216.{index}" for index in range(1, 10)]
    fetcher, attempts = _refusing_fetcher(lambda host, port: many)
    with pytest.raises(SourceError, match="could not connect"):
        _fetch(fetcher, "http://www.example.com/")
    assert attempts == many[:4]


def test_the_next_address_is_tried_when_the_first_does_not_answer(server: _Server) -> None:
    server.routes["/"] = _Route(body=b"second")
    attempts: list[str] = []

    def connect(address: str, port: int, timeout_s: float) -> socket.socket:
        attempts.append(address)
        if address == "10.255.255.1":
            raise ConnectionRefusedError
        return tcp_connect("127.0.0.1", port, timeout_s)

    fetcher = _local_fetcher(resolver=lambda host, port: ["10.255.255.1", "127.0.0.1"], connect=connect)
    assert _fetch(fetcher, f"http://site.example:{server.port}/").body == b"second"
    assert attempts == ["10.255.255.1", "127.0.0.1"]


@pytest.mark.parametrize("answer", [[], ["not-an-address"], ["93.184.216.34", "example.com"]])
def test_an_empty_or_nonsense_resolution_is_an_error(answer: list[str]) -> None:
    fetcher, attempts = _refusing_fetcher(lambda host, port: answer)
    with pytest.raises(SourceError, match="name resolution"):
        _fetch(fetcher, "http://www.example.com/")
    assert attempts == []


def test_nonsense_resolution_is_refused_even_when_private_addresses_are_allowed() -> None:
    fetcher = _local_fetcher(resolver=lambda host, port: ["localhost"])
    with pytest.raises(SourceError, match="not an address"):
        _fetch(fetcher, "http://www.example.com/")


def test_a_resolver_failure_is_a_source_error() -> None:
    def resolver(host: str, port: int) -> list[str]:
        raise socket.gaierror(8, "nodename nor servname provided")

    fetcher, attempts = _refusing_fetcher(resolver)
    with pytest.raises(SourceError, match=r"name resolution failed \(gaierror\)"):
        _fetch(fetcher, "http://www.example.com/")
    assert attempts == []


@pytest.mark.parametrize(
    ("address", "public"),
    [
        ("93.184.216.34", True),
        ("8.8.8.8", True),
        ("1.1.1.1", True),
        ("2606:4700:4700::1111", True),
        ("2001:4860:4860::8888", True),
        ("::ffff:8.8.8.8", True),
        ("64:ff9b::808:808", True),
        ("127.0.0.1", False),
        ("10.1.2.3", False),
        ("172.31.255.255", False),
        ("192.168.0.1", False),
        ("169.254.169.254", False),
        ("100.64.0.1", False),
        ("100.127.255.254", False),
        ("0.0.0.0", False),
        ("224.0.0.251", False),
        ("240.0.0.1", False),
        ("::1", False),
        ("::", False),
        ("::ffff:127.0.0.1", False),
        ("::ffff:0:0", False),
        ("::7f00:1", False),
        ("64:ff9b::7f00:1", False),
        ("64:ff9b::a00:1", False),
        ("2002:a00:1::", False),
        ("fe80::1", False),
        ("fe80::1%eth0", False),
        ("fc00::1", False),
        ("ff02::1", False),
        ("2001:db8::1", False),
        ("example.com", False),
        ("", False),
    ],
)
def test_is_public_address(address: str, public: bool) -> None:
    assert is_public_address(address) is public


# --- failures -------------------------------------------------------------------------------------------------


def test_a_closed_port_is_a_connect_failure() -> None:
    probe = socket.socket()
    probe.bind(("127.0.0.1", 0))
    port = int(probe.getsockname()[1])
    probe.close()  # nothing listens here any more
    with pytest.raises(SourceError, match="could not connect"):
        _fetch(_local_fetcher(), f"http://127.0.0.1:{port}/")


@pytest.mark.parametrize("reply", [b"NOT HTTP AT ALL\r\n\r\n", b"HTTP/9.9 200 OK\r\n\r\n", b""])
def test_a_server_that_does_not_speak_http_is_a_source_error(reply: bytes) -> None:
    with _raw_server(reply) as port, pytest.raises(SourceError, match="failure"):
        _fetch(_local_fetcher(), f"http://127.0.0.1:{port}/")


def test_https_to_a_server_that_does_not_speak_tls_fails_the_handshake() -> None:
    # The server answers after reading the hello and hangs up; a reset racing the garbage is the same refusal.
    with _raw_server(b"this is not a TLS record\r\n") as port, pytest.raises(SourceError, match=r"tls|network"):
        _fetch(_local_fetcher(), f"https://127.0.0.1:{port}/")


def test_error_messages_never_carry_the_response_body(server: _Server) -> None:
    secret = "SECRET-BODY-TEXT"
    server.routes["/z"] = _Route(headers={"Content-Encoding": "br"}, body=secret.encode("ascii"))
    with pytest.raises(SourceError) as caught:
        _fetch(_local_fetcher(), server.url("/z"))
    assert secret not in str(caught.value)
    assert secret not in caught.value.detail


@pytest.mark.parametrize("agent", ["", "bad\r\nX-Injected: 1", "caf\u00e9-bot", "tab\tbot"])
def test_a_user_agent_that_could_break_the_request_is_a_configuration_error(agent: str) -> None:
    with pytest.raises(ConfigurationError):
        HttpFetcher(user_agent=agent)


@pytest.mark.parametrize(
    ("header", "expected"),
    [
        ("text/html; charset=UTF-8", ("text/html", "utf-8")),
        ("TEXT/HTML", ("text/html", "")),
        ('text/html; charset="utf-8"', ("text/html", "utf-8")),
        ("text/html;charset=iso-8859-1;format=flowed", ("text/html", "iso-8859-1")),
        ("text/html; boundary=x; charset=latin1", ("text/html", "latin1")),
        ("application/xhtml+xml", ("application/xhtml+xml", "")),
        ("", ("", "")),
        ("garbage", ("", "")),
        ("text/html, text/plain", ("", "")),
        ("text/", ("", "")),
        ("/html", ("", "")),
        ("text/html/x", ("", "")),
        ("text/html; charset=" + "a" * 50, ("text/html", "")),
        ("text/html; charset=utf 8", ("text/html", "")),
        ("text/html; charset=", ("text/html", "")),
        (";charset=utf-8", ("", "utf-8")),
    ],
)
def test_content_type_parsing(header: str, expected: tuple[str, str]) -> None:
    assert parse_content_type(header) == expected
