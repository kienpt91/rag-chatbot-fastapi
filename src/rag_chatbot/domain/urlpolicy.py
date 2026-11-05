"""Which URLs may be fetched, and what a page is called.

A crawler follows links written by strangers, so every URL is parsed once into a canonical form and judged on its
parsed host, never on the raw text: `www.example.com.evil.test`, `www.example.com@evil.test`,
`www.example.com%2eevil.test`, full-width dots, a trailing dot, upper case and user-info tricks all end up either
as the one canonical host or rejected.
"""

from __future__ import annotations

import ipaddress
import posixpath
import string
from dataclasses import dataclass
from urllib.parse import urlsplit

MAX_URL_CHARS = 2_048

_DEFAULT_PORTS = {"http": 80, "https": 443}
_UNRESERVED = frozenset(string.ascii_letters + string.digits + "-._~")
_HEX = frozenset("0123456789abcdefABCDEF")
_LABEL_CHARS = frozenset(string.ascii_lowercase + string.digits + "-")
_EDGE_WHITESPACE = " \t\n\r\f"
_MAX_HOST_CHARS = 253
_MAX_LABEL_CHARS = 63
_MAX_PORT = 65_535


@dataclass(frozen=True, slots=True)
class HostPolicy:
    """The hosts a crawl may contact: exact names only (no wildcards, no suffix matching)."""

    allowed_hosts: frozenset[str]  # lower case, IDNA (punycode) form; see `normalise_host`
    allow_http: bool = False  # plain http only when True
    allowed_ports: frozenset[int] = frozenset()  # empty: the default port of the scheme only; else exactly these

    def check(self, url: str) -> str | None:
        """None when `url` may be fetched, else a short reason code: `too_long` (over 2,048 characters), `invalid`,
        `scheme`, `credentials`, `host` or `port`."""
        parsed = _parse(url)
        if isinstance(parsed, str):
            return parsed
        if parsed.scheme == "http" and not self.allow_http:
            return "scheme"
        if parsed.host not in self.allowed_hosts:
            return "host"
        ports = self.allowed_ports or frozenset({_DEFAULT_PORTS[parsed.scheme]})
        if parsed.port not in ports:
            return "port"
        return None


@dataclass(frozen=True, slots=True)
class _Url:
    scheme: str
    host: str  # canonical: ASCII lower case, or a bracketed compressed IPv6 literal
    port: int
    path: str
    query: str

    def serialise(self) -> str:
        port = "" if self.port == _DEFAULT_PORTS[self.scheme] else f":{self.port}"
        query = f"?{self.query}" if self.query else ""
        return f"{self.scheme}://{self.host}{port}{self.path}{query}"


def normalise_percent_encoding(text: str) -> str:
    """Escapes of unreserved characters are decoded, every other escape is upper-cased, a `%` that starts no escape
    becomes `%25`, and characters outside printable ASCII are encoded as UTF-8 escapes. `%2F` stays `%2F`, so an
    encoded slash never turns into a path separator. Idempotent."""
    out: list[str] = []
    index = 0
    length = len(text)
    while index < length:
        char = text[index]
        if char == "%":
            digits = text[index + 1 : index + 3]
            if len(digits) == 2 and _HEX.issuperset(digits):
                decoded = chr(int(digits, 16))
                out.append(decoded if decoded in _UNRESERVED else "%" + digits.upper())
                index += 3
                continue
            out.append("%25")
        elif char <= " " or char >= "\x7f":
            out.extend(f"%{byte:02X}" for byte in char.encode("utf-8", "surrogatepass"))
        else:
            out.append(char)
        index += 1
    return "".join(out)


def normalise_host(host: str) -> str | None:
    """The canonical form of a host name or bracketed IPv6 literal (for building a `HostPolicy`), or None when it is
    not a plausible host: IDNA-encoded, lower case, one trailing dot removed, letters, digits and hyphens only."""
    try:
        if host.startswith("["):
            return _normalise_ipv6(host)
        return _normalise_name(host)
    except ValueError:  # UnicodeError (a bad IDNA label) is a ValueError
        return None


def normalise_url(url: str) -> str | None:
    """The canonical form of an absolute http(s) URL, or None for anything else (relative, other schemes,
    credentials, an invalid host or port, more than 2,048 characters). Scheme and host are lower case, the default
    port and the fragment are dropped, an empty path becomes `/`, dot segments are resolved and percent-encoding is
    normalised. The query is kept, in its original order."""
    parsed = _parse(url)
    return None if isinstance(parsed, str) else parsed.serialise()


def document_id_for_url(url: str) -> str:
    """The stable identity of a page: its normalised URL without the scheme, `www.example.com/menu` (the query is
    kept, a trailing slash is kept, except for the root which is just `www.example.com`)."""
    normalised = normalise_url(url)
    if normalised is None:
        raise ValueError("not an absolute http(s) URL")
    rest = normalised.partition("://")[2]
    if rest.count("/") == 1 and rest.endswith("/"):
        rest = rest[:-1]
    return rest


def document_id_for_path(relative_path: str) -> str:
    """The identity of a file: its relative path with posix separators and without the extension
    (`policies/shipping.md` -> `policies/shipping`). Case is kept; `index` files keep their name."""
    path = posixpath.normpath(relative_path.replace("\\", "/"))
    if path == "." or path.startswith("/") or path == ".." or path.startswith("../"):
        raise ValueError("not a relative path inside the root")
    return posixpath.splitext(path)[0]


def _parse(url: str) -> _Url | str:
    """A parsed URL, or the reason code for rejecting it."""
    if len(url) > MAX_URL_CHARS:
        return "too_long"
    url = url.strip(_EDGE_WHITESPACE)
    if any(char <= " " or char in "\x7f\\" for char in url):
        return "invalid"
    try:
        parsed = _build(url)
    except ValueError:  # urlsplit, IDNA and UTF-8 encoding failures are all ValueErrors
        return "invalid"
    if isinstance(parsed, _Url) and len(parsed.serialise()) > MAX_URL_CHARS:
        return "too_long"
    return parsed


def _build(url: str) -> _Url | str:
    parts = urlsplit(url)
    if not parts.scheme:
        return "invalid"
    if parts.scheme not in _DEFAULT_PORTS:
        return "scheme"
    netloc = parts.netloc
    if "@" in netloc:
        return "credentials"
    host_text, port_text = _split_netloc(netloc)
    host = normalise_host(host_text)
    port = _parse_port(port_text, parts.scheme)
    if host is None or port is None:
        return "invalid"
    path = _remove_dot_segments(_rooted(normalise_percent_encoding(parts.path)))
    return _Url(parts.scheme, host, port, path, normalise_percent_encoding(parts.query))


def _split_netloc(netloc: str) -> tuple[str, str]:
    """`host`, `port` text; an empty host fails normalisation later. A second colon is left in the port text, where
    it fails the digit check."""
    if netloc.startswith("["):
        end = netloc.find("]")
        if end < 0:
            raise ValueError("unterminated IPv6 literal")
        rest = netloc[end + 1 :]
        if rest and not rest.startswith(":"):
            raise ValueError("junk after the IPv6 literal")
        return netloc[: end + 1], rest[1:]
    host, _, port = netloc.partition(":")
    return host, port


def _parse_port(text: str, scheme: str) -> int | None:
    if not text:
        return _DEFAULT_PORTS[scheme]
    if not (text.isascii() and text.isdigit()) or len(text) > 5:
        return None
    port = int(text)
    return port if 1 <= port <= _MAX_PORT else None


def _normalise_ipv6(host: str) -> str | None:
    if not host.endswith("]"):
        return None
    inner = host[1:-1]
    if "%" in inner:  # no zone identifiers
        return None
    return f"[{ipaddress.IPv6Address(inner).compressed}]"


def _normalise_name(host: str) -> str | None:
    ascii_host = host.encode("idna").decode("ascii").lower()
    ascii_host = ascii_host.removesuffix(".")
    if not ascii_host or len(ascii_host) > _MAX_HOST_CHARS:
        return None
    labels = ascii_host.split(".")
    for label in labels:
        if not label or len(label) > _MAX_LABEL_CHARS or label[0] == "-" or label[-1] == "-":
            return None
        if not _LABEL_CHARS.issuperset(label):
            return None
    last = labels[-1]
    if last.isdigit() or last.startswith("0x"):
        # Looks like an IPv4 address; only the strict dotted-quad form is accepted (no `2130706433`, `0x7f.1`, `0177.0.0.1`).
        return str(ipaddress.IPv4Address(ascii_host))
    return ascii_host


def _rooted(path: str) -> str:
    return path if path.startswith("/") else "/" + path


def _remove_dot_segments(path: str) -> str:
    """RFC 3986 section 5.2.4 for a path that starts with `/`."""
    out: list[str] = []
    segments = path.split("/")[1:]
    for index, segment in enumerate(segments):
        last = index == len(segments) - 1
        if segment == "..":
            if out:
                out.pop()
            if last:
                out.append("")
        elif segment == ".":
            if last:
                out.append("")
        else:
            out.append(segment)
    return "/" + "/".join(out)
