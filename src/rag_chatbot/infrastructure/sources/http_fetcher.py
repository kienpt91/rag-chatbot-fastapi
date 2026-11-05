"""An HTTP(S) page fetcher built for a hostile network: standard library only, SSRF-safe, bounded in time and size.

What it guarantees, in the order a reviewer should check it:

* Only `http` and `https` URLs without credentials are fetched.
* The host name is resolved once, every resolved address must be globally routable (loopback, private, link-local
  including 169.254.169.254, carrier-grade NAT, multicast, reserved and unspecified addresses are refused, and an
  IPv4 address wrapped in IPv6 is judged as the IPv4 address it is), and the connection goes to that validated
  address, not to the name, so a second DNS answer cannot redirect it. The `Host` header, the TLS server name and the
  certificate check still use the name.
* Redirects are never followed; a 3xx answer is returned with its `Location`.
* Compression is not requested and a compressed answer is refused, so no decompression bomb can be unpacked.
* The body is read in chunks up to `max_bytes`; one monotonic deadline covers connecting, the request, the headers
  and the body, and every socket operation gets only the time that is left of it (a server that trickles a byte at
  a time cannot hold a fetch longer than the deadline). Name resolution itself cannot be interrupted.
* Every failure is a `SourceError` naming the class of the failure, never a response body.
"""

from __future__ import annotations

import asyncio
import http.client
import io
import ipaddress
import socket
import ssl
import string
import time
from collections.abc import Callable
from typing import Any, cast
from urllib.parse import urlsplit

from rag_chatbot.application.ports import FetchResult
from rag_chatbot.domain.errors import ConfigurationError, SourceError
from rag_chatbot.domain.urlpolicy import normalise_url

Resolver = Callable[[str, int], list[str]]  # host, port -> IP address literals
Connector = Callable[[str, int, float], socket.socket]  # validated IP literal, port, timeout -> connected socket
MonotonicClock = Callable[[], float]

_CHUNK_BYTES = 64 * 1024
_MAX_LOCATION_CHARS = 8_192
_MAX_ADDRESSES_TRIED = 4
_ACCEPT = "text/html,text/plain,text/markdown;q=0.9,*/*;q=0.1"
_DEFAULT_PORTS = {"http": 80, "https": 443}
_MEDIA_CHARS = frozenset(string.ascii_lowercase + string.digits + "!#$&-^_.+")
_CHARSET_CHARS = frozenset(string.ascii_lowercase + string.digits + "-_.:")
_NAT64 = ipaddress.IPv6Network("64:ff9b::/96")
_IPV4_COMPATIBLE = ipaddress.IPv6Network("::/96")


def system_resolver(host: str, port: int) -> list[str]:
    """Every address of `host` as the operating system's resolver reports it."""
    return [str(info[4][0]) for info in socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)]


def tcp_connect(address: str, port: int, timeout_s: float) -> socket.socket:
    return socket.create_connection((address, port), timeout_s)


def is_public_address(address: str) -> bool:
    """True only for an address on the public internet. IPv4-mapped, 6to4 and NAT64 addresses are judged by the IPv4
    address inside them; the deprecated IPv4-compatible `::/96` block is refused outright."""
    try:
        ip = ipaddress.ip_address(address.partition("%")[0])
    except ValueError:
        return False
    if isinstance(ip, ipaddress.IPv6Address):
        embedded = _embedded_ipv4(ip)
        if embedded is not None:
            return _is_routable(embedded)
        if ip in _IPV4_COMPATIBLE:
            return False
    return _is_routable(ip)


def parse_content_type(value: str) -> tuple[str, str]:
    """`text/html; charset=UTF-8` -> (`text/html`, `utf-8`). Either part is "" when absent or implausible."""
    media, _, parameters = value.partition(";")
    media = media.strip().lower()
    if not _plausible_media_type(media):
        media = ""
    charset = ""
    for parameter in parameters.split(";"):
        name, _, text = parameter.partition("=")
        if name.strip().lower() == "charset":
            charset = text.strip().strip("\"'").lower()
            break
    if not charset or len(charset) > 40 or not _CHARSET_CHARS.issuperset(charset):
        charset = ""
    return media, charset


class HttpFetcher:
    """`Fetcher` over `http.client`. `resolver` and `connect` are test hooks (the defaults use the system resolver and
    a plain TCP connection); `clock` is the monotonic clock the deadline runs on."""

    def __init__(
        self,
        *,
        user_agent: str,
        timeout_s: float = 10.0,
        allow_private_addresses: bool = False,
        resolver: Resolver | None = None,
        connect: Connector | None = None,
        clock: MonotonicClock = time.monotonic,
        ssl_context: ssl.SSLContext | None = None,
    ) -> None:
        if not user_agent or any(not " " <= char <= "~" for char in user_agent):
            raise ConfigurationError("the user agent must be non-empty printable ASCII")
        self._user_agent = user_agent
        self._timeout_s = timeout_s
        self._allow_private = allow_private_addresses
        self._resolver: Resolver = resolver or system_resolver
        self._connect: Connector = connect or tcp_connect
        self._clock = clock
        self._ssl_context = ssl_context

    async def fetch(self, url: str, *, max_bytes: int) -> FetchResult:
        """GET `url` without following redirects; at most `max_bytes` of the body. Raises `SourceError`."""
        return await asyncio.to_thread(self._fetch_blocking, url, max_bytes)

    def _fetch_blocking(self, url: str, max_bytes: int) -> FetchResult:
        try:
            return self._fetch(url, max_bytes)
        except SourceError:
            raise
        except ssl.SSLError as exc:
            raise SourceError(f"tls failure ({type(exc).__name__})") from exc
        except TimeoutError as exc:
            raise SourceError("timeout") from exc
        except OSError as exc:
            raise SourceError(f"network failure ({type(exc).__name__})") from exc
        except http.client.HTTPException as exc:
            raise SourceError(f"protocol failure ({type(exc).__name__})") from exc
        except ValueError as exc:  # includes UnicodeError and header validation errors
            raise SourceError(f"invalid exchange ({type(exc).__name__})") from exc

    def _fetch(self, url: str, max_bytes: int) -> FetchResult:
        deadline = _Deadline(self._clock, self._timeout_s)
        normalised = normalise_url(url)
        if normalised is None:
            raise SourceError("the URL is not a plain absolute http(s) URL without credentials")
        parts = urlsplit(normalised)
        host = parts.hostname
        if host is None:
            raise SourceError("the URL has no host")
        port = parts.port or _DEFAULT_PORTS[parts.scheme]
        addresses = self._resolve(host, port)
        plain = self._open(addresses, port, deadline)
        stream = plain
        try:
            if parts.scheme == "https":
                stream = self._tls_context().wrap_socket(plain, server_hostname=host)
            return self._exchange(
                stream,
                deadline,
                url=url,
                host_header=_host_header(host, port, parts.scheme),
                target=parts.path + (f"?{parts.query}" if parts.query else ""),
                max_bytes=max_bytes,
            )
        finally:
            stream.close()
            plain.close()

    def _resolve(self, host: str, port: int) -> list[str]:
        """The addresses to connect to, every one of them checked before any connection is made."""
        if _is_ip_literal(host):
            candidates = [host]
        else:
            try:
                candidates = self._resolver(host, port)
            except OSError as exc:
                raise SourceError(f"name resolution failed ({type(exc).__name__})") from exc
        addresses = list(dict.fromkeys(candidates))
        if not addresses:
            raise SourceError("name resolution returned no address")
        for address in addresses:
            if not _is_ip_literal(address.partition("%")[0]):
                raise SourceError("name resolution returned something that is not an address")
            if not self._allow_private and not is_public_address(address):
                raise SourceError("refused: the host resolves to a non-public address")
        return addresses

    def _open(self, addresses: list[str], port: int, deadline: _Deadline) -> socket.socket:
        for address in addresses[:_MAX_ADDRESSES_TRIED]:
            timeout_s = deadline.remaining()  # raises TimeoutError once the budget is spent
            try:
                return self._connect(address, port, timeout_s)
            except OSError:
                continue
        raise SourceError("could not connect to any address of the host")

    def _tls_context(self) -> ssl.SSLContext:
        if self._ssl_context is None:
            self._ssl_context = ssl.create_default_context()  # verifies the certificate chain and the host name
        return self._ssl_context

    def _exchange(
        self,
        stream: socket.socket,
        deadline: _Deadline,
        *,
        url: str,
        host_header: str,
        target: str,
        max_bytes: int,
    ) -> FetchResult:
        connection = http.client.HTTPConnection(host_header, timeout=self._timeout_s)
        # http.client sends on, and reads from, whatever object sits in `sock`; ours enforces the deadline.
        connection.sock = cast("socket.socket", _DeadlineSocket(stream, deadline))
        try:
            connection.request(
                "GET",
                target,
                headers={
                    "Host": host_header,
                    "User-Agent": self._user_agent,
                    "Accept": _ACCEPT,
                    "Accept-Encoding": "identity",
                    "Connection": "close",
                },
            )
            response = connection.getresponse()
            return _result(url, response, max_bytes)
        finally:
            connection.close()


class _Deadline:
    def __init__(self, clock: MonotonicClock, budget_s: float) -> None:
        self._clock = clock
        self._expires = clock() + budget_s

    def remaining(self) -> float:
        """The seconds left; raises `TimeoutError` when there are none."""
        left = self._expires - self._clock()
        if left <= 0:
            raise TimeoutError("the fetch deadline passed")
        return left


class _DeadlineReader(io.RawIOBase):
    def __init__(self, sock: socket.socket, deadline: _Deadline) -> None:
        super().__init__()
        self._sock = sock
        self._deadline = deadline

    def readable(self) -> bool:
        return True

    def readinto(self, buffer: Any, /) -> int:
        self._sock.settimeout(self._deadline.remaining())
        return self._sock.recv_into(buffer)


class _DeadlineSocket:
    """The three socket methods `http.client` uses, each limited to the time left of the fetch."""

    def __init__(self, sock: socket.socket, deadline: _Deadline) -> None:
        self._sock = sock
        self._deadline = deadline

    def sendall(self, data: bytes) -> None:
        self._sock.settimeout(self._deadline.remaining())
        self._sock.sendall(data)

    def makefile(self, _mode: str = "rb", _buffering: int | None = None) -> io.BufferedReader:
        return io.BufferedReader(_DeadlineReader(self._sock, self._deadline), _CHUNK_BYTES)

    def close(self) -> None:
        self._sock.close()


def _result(url: str, response: http.client.HTTPResponse, max_bytes: int) -> FetchResult:
    media, charset = parse_content_type(response.getheader("Content-Type") or "")
    status = response.status
    if 300 <= status < 400:
        location = (response.getheader("Location") or "").strip()
        if len(location) > _MAX_LOCATION_CHARS:
            location = ""
        return FetchResult(url, status, media, charset, b"", False, location)
    if (response.getheader("Content-Encoding") or "").strip().lower() not in {"", "identity"}:
        raise SourceError("refused: the answer is compressed although compression was not requested")
    body, truncated = _read_body(response, max_bytes)
    return FetchResult(url, status, media, charset, body, truncated)


def _read_body(response: http.client.HTTPResponse, max_bytes: int) -> tuple[bytes, bool]:
    """Up to `max_bytes` of the body, and whether there was more. `read1` returns after one receive, so the deadline
    is checked between receives and a slow sender cannot make a single call wait for a whole chunk."""
    max_bytes = max(max_bytes, 0)
    chunks: list[bytes] = []
    total = 0
    while True:
        data = response.read1(min(_CHUNK_BYTES, max_bytes + 1 - total))
        if not data:
            return b"".join(chunks), False
        total += len(data)
        if total > max_bytes:
            chunks.append(data[: len(data) - (total - max_bytes)])
            return b"".join(chunks), True
        chunks.append(data)


def _embedded_ipv4(address: ipaddress.IPv6Address) -> ipaddress.IPv4Address | None:
    if address.ipv4_mapped is not None:
        return address.ipv4_mapped
    if address.sixtofour is not None:
        return address.sixtofour
    if address in _NAT64:
        return ipaddress.IPv4Address(int(address) & 0xFFFFFFFF)
    return None


def _is_routable(address: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    return address.is_global and not (
        address.is_private
        or address.is_loopback
        or address.is_link_local
        or address.is_multicast
        or address.is_reserved
        or address.is_unspecified
    )


def _is_ip_literal(text: str) -> bool:
    try:
        ipaddress.ip_address(text)
    except ValueError:
        return False
    return True


def _host_header(host: str, port: int, scheme: str) -> str:
    name = f"[{host}]" if ":" in host else host
    return name if port == _DEFAULT_PORTS[scheme] else f"{name}:{port}"


def _plausible_media_type(media: str) -> bool:
    kind, slash, subtype = media.partition("/")
    if not slash or not kind or not subtype or len(media) > 100:
        return False
    return _MEDIA_CHARS.issuperset(kind + subtype)
