"""URL normalisation, the host allow-list and the identities of pages and files."""

from __future__ import annotations

import pytest

from rag_chatbot.domain.urlpolicy import (
    MAX_URL_CHARS,
    HostPolicy,
    document_id_for_path,
    document_id_for_url,
    normalise_host,
    normalise_percent_encoding,
    normalise_url,
)

HOST = "www.example.com"
POLICY = HostPolicy(allowed_hosts=frozenset({HOST}))


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("HTTPS://WWW.EXAMPLE.COM", "https://www.example.com/"),
        ("https://example.com:443/a", "https://example.com/a"),
        ("http://example.com:80/", "http://example.com/"),
        ("https://example.com:8443/a", "https://example.com:8443/a"),
        ("https://example.com:80/a", "https://example.com:80/a"),  # 80 is only the default of http
        ("https://example.com/a#section", "https://example.com/a"),
        ("https://example.com/a/./b/../c", "https://example.com/a/c"),
        ("https://example.com/a/b/..", "https://example.com/a/"),
        ("https://example.com/a/b/.", "https://example.com/a/b/"),
        ("https://example.com/../../x", "https://example.com/x"),
        ("https://example.com/%7Euser/%2e%2e/x", "https://example.com/x"),
        ("https://example.com/a%2fb", "https://example.com/a%2Fb"),
        ("https://example.com/%61%7e%2D", "https://example.com/a~-"),
        ("https://example.com/%e2%82%ac", "https://example.com/%E2%82%AC"),
        ("https://example.com/caf\u00e9", "https://example.com/caf%C3%A9"),
        ("https://example.com/50%", "https://example.com/50%25"),
        ("https://example.com//a//b", "https://example.com//a//b"),
        ("https://example.com/?b=2&a=1&b=%7e", "https://example.com/?b=2&a=1&b=~"),
        ("https://example.com?x=1", "https://example.com/?x=1"),
        ("https://example.com/a?", "https://example.com/a"),
        ("https://example.com.", "https://example.com/"),
        ("https://www\u3002example\u3002com/", "https://www.example.com/"),
        ("https://b\u00fccher.example/", "https://xn--bcher-kva.example/"),
        ("  https://example.com/  ", "https://example.com/"),
        ("http://[::1]:8080/x", "http://[::1]:8080/x"),
        ("http://[0:0:0:0:0:0:0:1]/", "http://[::1]/"),
        ("http://127.0.0.1:8000/", "http://127.0.0.1:8000/"),
    ],
)
def test_normalise_url_gives_the_canonical_form(url: str, expected: str) -> None:
    assert normalise_url(url) == expected
    assert normalise_url(expected) == expected  # idempotent


@pytest.mark.parametrize(
    "url",
    [
        "",
        "/a/b",
        "a/b",
        "//example.com/x",
        "mailto:a@example.com",
        "javascript:alert(1)",
        "ftp://example.com/",
        "file:///etc/passwd",
        "data:text/html,x",
        "https://user@example.com/",
        "https://user:pw@example.com/",
        "https://@example.com/",
        "https://example.com@evil.test/",
        "https:///x",
        "https://",
        "https:/example.com/",
        "http://exa mple.com/",
        "https://example.com/a b",
        "https://example.com/\x00",
        "https://example.com/\nx",
        "https://example.com\\@evil.test/",
        "https://example.com:0/",
        "https://example.com:99999/",
        "https://example.com:abc/",
        "https://example.com:\u0661\u0662/",
        "https://example.com:80:80/",
        "https://exa%6dple.com/",
        "https://example.com%2eevil.test/",
        "https://-bad.example/",
        "https://bad-.example/",
        "https://a..b/",
        "https://.example.com/",
        "https://example.com../",
        "https://under_score.example/",
        "https://2130706433/",
        "https://0x7f.1/",
        "https://127.1/",
        "https://0177.0.0.1/",
        "https://999.1.1.1/",
        "http://[::1/",
        "http://[::1]x/",
        "http://[fe80::1%25eth0]/",
        "http://[not-an-address]/",
        "https://www.example.com\uff20evil.test/",  # a full-width @ folds to @ under NFKC
        "https://" + "a" * 64 + ".example/",
        "https://example.com/" + "a" * MAX_URL_CHARS,
    ],
)
def test_normalise_url_rejects_anything_that_is_not_a_plain_absolute_http_url(url: str) -> None:
    assert normalise_url(url) is None


def test_a_url_that_grows_past_the_limit_when_encoded_is_rejected() -> None:
    assert normalise_url("https://example.com/" + "\u20ac" * 700) is None


@pytest.mark.parametrize(
    "url",
    [
        "https://www.example.com/",
        "https://WWW.EXAMPLE.COM./x",
        "https://www\u3002example\u3002com/",
        "https://\uff57\uff57\uff57.example.com/",
        "https://www.example.com:443/",
        "https://www.example.com/a?b=c#d",
        "https://www.example.com#@evil.test/",
        "https://www.example.com/redirect?to=https://evil.test/",
    ],
)
def test_the_allowed_host_in_any_spelling_passes(url: str) -> None:
    assert POLICY.check(url) is None


@pytest.mark.parametrize(
    ("url", "reason"),
    [
        ("http://www.example.com/", "scheme"),
        ("ftp://www.example.com/", "scheme"),
        ("javascript:alert(1)", "scheme"),
        ("mailto:me@www.example.com", "scheme"),
        ("https://www.example.com.evil.test/", "host"),
        ("https://evil.test/www.example.com", "host"),
        ("https://evil.test/?u=https://www.example.com/", "host"),
        ("https://evil.test#@www.example.com/", "host"),
        ("https://evil.test?@www.example.com/", "host"),
        ("https://example.com/", "host"),
        ("https://sub.www.example.com/", "host"),
        ("https://xwww.example.com/", "host"),
        ("https://www.example.com.evil.test./", "host"),
        ("https://www.example.com\u3002evil\u3002test/", "host"),
        ("https://ww\u0213.example.com/", "host"),
        ("https://[::1]/", "host"),
        ("https://127.0.0.1/", "host"),
        ("https://www.example.com@evil.test/", "credentials"),
        ("https://evil.test@www.example.com/", "credentials"),
        ("https://user:pw@www.example.com/", "credentials"),
        ("https://www.example.com%2eevil.test/", "invalid"),
        ("https://www.example.com\\.evil.test/", "invalid"),
        ("https://www.example.com\\@evil.test/", "invalid"),
        ("https://www.example.com\uff20evil.test/", "invalid"),
        ("/relative", "invalid"),
        ("", "invalid"),
        ("https://www.example.com:8443/", "port"),
        ("https://www.example.com:80/", "port"),
        ("https://www.example.com/" + "a" * 3_000, "too_long"),
    ],
)
def test_everything_else_is_refused_with_a_short_reason(url: str, reason: str) -> None:
    assert POLICY.check(url) == reason


def test_plain_http_needs_the_explicit_switch() -> None:
    policy = HostPolicy(allowed_hosts=frozenset({HOST}), allow_http=True)
    assert policy.check("http://www.example.com/") is None
    assert policy.check("http://www.example.com:80/") is None
    assert policy.check("http://www.example.com:443/") == "port"
    assert policy.check("https://www.example.com/") is None


def test_listed_ports_replace_the_default_port() -> None:
    policy = HostPolicy(allowed_hosts=frozenset({HOST}), allowed_ports=frozenset({8443}))
    assert policy.check("https://www.example.com:8443/") is None
    assert policy.check("https://www.example.com/") == "port"
    assert policy.check("https://www.example.com:9000/") == "port"


def test_an_internationalised_host_is_matched_in_its_punycode_form() -> None:
    policy = HostPolicy(allowed_hosts=frozenset({"xn--bcher-kva.example"}))
    assert policy.check("https://b\u00fccher.example/") is None
    assert policy.check("https://B\u00dcCHER.example./x") is None
    assert policy.check("https://xn--bcher-kva.example/") is None
    assert policy.check("https://bucher.example/") == "host"


def test_an_empty_allow_list_allows_nothing() -> None:
    assert HostPolicy(allowed_hosts=frozenset()).check("https://www.example.com/") == "host"


@pytest.mark.parametrize(
    ("host", "expected"),
    [
        ("EXAMPLE.com.", "example.com"),
        ("example.com", "example.com"),
        ("B\u00fccher.Example", "xn--bcher-kva.example"),
        ("[0:0:0:0:0:0:0:1]", "[::1]"),
        ("", None),
        (".", None),
        ("a b", None),
        ("a..b", None),
        ("a/b", None),
        ("a@b", None),
        ("a:80", None),
        ("2130706433", None),
        ("[::1", None),
        ("[fe80::1%eth0]", None),
        ("a" * 300, None),
    ],
)
def test_normalise_host(host: str, expected: str | None) -> None:
    assert normalise_host(host) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("%41%62%7e%2d%2E%5f", "Ab~-._"),
        ("%2f%3a%3F", "%2F%3A%3F"),
        ("%zz%4", "%25zz%254"),
        ("%", "%25"),
        ("a b", "a%20b"),
        ("\u00e9", "%C3%A9"),
        ("\ud800", "%ED%A0%80"),  # a lone surrogate must not raise
    ],
)
def test_percent_encoding_normalisation(text: str, expected: str) -> None:
    assert normalise_percent_encoding(text) == expected
    assert normalise_percent_encoding(expected) == expected


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("https://www.example.com/", "www.example.com"),
        ("https://www.example.com", "www.example.com"),
        ("https://www.example.com/menu", "www.example.com/menu"),
        ("https://www.example.com/menu/", "www.example.com/menu/"),
        ("HTTP://WWW.Example.com:80/menu#x", "www.example.com/menu"),
        ("https://www.example.com/?q=1", "www.example.com/?q=1"),
        ("https://www.example.com:8443/a", "www.example.com:8443/a"),
        ("https://www.example.com/a?x=/", "www.example.com/a?x=/"),
        ("https://www.example.com/a/../b", "www.example.com/b"),
    ],
)
def test_document_id_for_url_is_the_normalised_url_without_its_scheme(url: str, expected: str) -> None:
    assert document_id_for_url(url) == expected


def test_both_schemes_of_one_page_share_an_identity() -> None:
    assert document_id_for_url("http://www.example.com/a") == document_id_for_url("https://www.example.com/a")


@pytest.mark.parametrize("url", ["", "/menu", "mailto:a@b.c", "https://user@www.example.com/"])
def test_document_id_for_url_refuses_what_is_not_a_url(url: str) -> None:
    with pytest.raises(ValueError, match="URL"):
        document_id_for_url(url)


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("policies/shipping.md", "policies/shipping"),
        ("a.b.txt", "a.b"),
        ("docs/index.html", "docs/index"),
        ("index.html", "index"),
        ("Docs/FAQ.HTML", "Docs/FAQ"),
        ("a\\b.md", "a/b"),
        ("./x.md", "x"),
        ("a//b/../c.md", "a/c"),
        ("noextension", "noextension"),
    ],
)
def test_document_id_for_path_drops_the_extension_and_keeps_the_case(path: str, expected: str) -> None:
    assert document_id_for_path(path) == expected


@pytest.mark.parametrize("path", ["", ".", "..", "../x.md", "a/../../x.md", "/abs.md"])
def test_document_id_for_path_refuses_paths_outside_the_root(path: str) -> None:
    with pytest.raises(ValueError, match="relative path"):
        document_id_for_path(path)
