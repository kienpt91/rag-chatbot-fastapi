"""The folder source: what it reads, what it refuses, and how it is bounded."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

import pytest

from rag_chatbot.domain.errors import SourceError
from rag_chatbot.domain.model import RawPage
from rag_chatbot.infrastructure.sources.folder import FolderSource


def _put(root: Path, relative: str, content: str | bytes = "x") -> Path:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(content, str):
        path.write_text(content, encoding="utf-8")
    else:
        path.write_bytes(content)
    return path


def _collect(source: FolderSource) -> list[RawPage]:
    async def run() -> list[RawPage]:
        return [page async for page in source.pages()]

    return asyncio.run(run())


@pytest.fixture
def root(tmp_path: Path) -> Path:
    folder = tmp_path / "content"
    folder.mkdir()
    return folder


def test_pages_come_in_sorted_depth_first_order_with_posix_locations(root: Path) -> None:
    for name in ["e.html", "b/d.txt", "a.md", "b/c.md", "b/z/deep.markdown", "B.md"]:
        _put(root, name, name)
    pages = _collect(FolderSource(root))
    assert [p.location for p in pages] == ["B.md", "a.md", "b/c.md", "b/d.txt", "b/z/deep.markdown", "e.html"]
    assert [p.content for p in pages] == [p.location for p in pages]


@pytest.mark.parametrize(
    ("name", "kind"),
    [
        ("a.html", "html"),
        ("a.htm", "html"),
        ("a.xhtml", "html"),
        ("a.md", "markdown"),
        ("a.markdown", "markdown"),
        ("a.txt", "text"),
        ("UPPER.HTML", "html"),
        ("Mixed.Md", "markdown"),
    ],
)
def test_the_extension_decides_the_page_kind(root: Path, name: str, kind: str) -> None:
    _put(root, name)
    assert [p.kind for p in _collect(FolderSource(root))] == [kind]


def test_other_files_and_hidden_entries_are_ignored(root: Path) -> None:
    for name in ["keep.md", "image.png", "data.json", "noext", "script.js", "archive.md.bak"]:
        _put(root, name)
    for name in [".hidden.md", ".git/config.txt", "docs/.secret.txt", ".cache/a/b.md"]:
        _put(root, name)
    source = FolderSource(root)
    assert [p.location for p in _collect(source)] == ["keep.md"]
    assert source.report.pages == 1
    assert source.report.skipped == {}


def test_text_is_decoded_as_utf8_with_replacement_and_the_bom_is_removed(root: Path) -> None:
    _put(root, "bom.txt", b"\xef\xbb\xbfhello \xe2\x82\xac")
    _put(root, "bad.txt", b"ok \xff\xfe bytes")
    pages = {p.location: p.content for p in _collect(FolderSource(root))}
    assert pages["bom.txt"] == "hello \u20ac"
    assert pages["bad.txt"] == "ok \ufffd\ufffd bytes"


def test_an_empty_file_and_an_empty_folder_are_fine(root: Path) -> None:
    assert _collect(FolderSource(root)) == []
    _put(root, "empty.txt", "")
    assert [p.content for p in _collect(FolderSource(root))] == [""]


@pytest.mark.parametrize(
    ("base_url", "relative", "url"),
    [
        ("https://x.test", "about.html", "https://x.test/about"),
        ("https://x.test/", "about.md", "https://x.test/about"),
        ("https://x.test///", "docs/faq.txt", "https://x.test/docs/faq"),
        ("https://x.test", "index.html", "https://x.test/"),
        ("https://x.test", "docs/index.html", "https://x.test/docs/"),
        ("https://x.test", "docs/INDEX.md", "https://x.test/docs/"),
        ("https://x.test", "docs/index-old.html", "https://x.test/docs/index-old"),
        ("https://x.test", "a b/caf\u00e9 menu.md", "https://x.test/a%20b/caf%C3%A9%20menu"),
        ("https://x.test", "q?x#y.md", "https://x.test/q%3Fx%23y"),
        ("", "about.html", ""),
    ],
)
def test_the_public_url_is_the_base_plus_the_path_without_extension(
    root: Path, base_url: str, relative: str, url: str
) -> None:
    _put(root, relative)
    (page,) = _collect(FolderSource(root, base_url=base_url))
    assert page.url == url
    assert page.location == relative


def test_symbolic_links_to_files_and_folders_are_never_followed(root: Path, tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    _put(outside, "secret.txt", "TOP SECRET")
    _put(outside, "dir/more.md", "MORE")
    _put(root, "real.md", "real")
    os.symlink(outside / "secret.txt", root / "link.txt")
    os.symlink(outside / "dir", root / "linked-dir")
    os.symlink(root / "real.md", root / "alias.md")
    source = FolderSource(root)
    pages = _collect(source)
    assert [p.location for p in pages] == ["real.md"]
    assert all("SECRET" not in p.content and "MORE" not in p.content for p in pages)
    assert source.report.skipped == {"symlink": 3}


@pytest.mark.parametrize("kind", ["symlink", "file", "missing"])
def test_a_root_that_is_not_a_plain_directory_is_refused(tmp_path: Path, kind: str) -> None:
    real = tmp_path / "real"
    real.mkdir()
    target = {"symlink": tmp_path / "link", "file": tmp_path / "file.txt", "missing": tmp_path / "nope"}[kind]
    if kind == "symlink":
        os.symlink(real, target)
    elif kind == "file":
        target.write_text("x", encoding="utf-8")
    with pytest.raises(SourceError):
        _collect(FolderSource(target))


def test_files_over_the_byte_cap_are_skipped_and_counted(root: Path) -> None:
    _put(root, "exact.txt", b"a" * 100)
    _put(root, "over.txt", b"a" * 101)
    _put(root, "way-over.md", b"a" * 10_000)
    source = FolderSource(root, max_file_bytes=100)
    assert [p.location for p in _collect(source)] == ["exact.txt"]
    assert source.report.skipped == {"too_large": 2}


def test_the_file_cap_stops_the_walk_and_is_recorded(root: Path) -> None:
    for index in range(5):
        _put(root, f"f{index}.txt")
    source = FolderSource(root, max_files=3)
    assert [p.location for p in _collect(source)] == ["f0.txt", "f1.txt", "f2.txt"]
    assert source.report.pages == 3
    assert source.report.skipped == {"limit": 1}


def test_exactly_as_many_files_as_the_cap_is_not_a_limit_hit(root: Path) -> None:
    for index in range(3):
        _put(root, f"f{index}.txt")
    source = FolderSource(root, max_files=3)
    assert len(_collect(source)) == 3
    assert source.report.skipped == {}


def test_files_skipped_for_size_do_not_count_against_the_file_cap(root: Path) -> None:
    _put(root, "a-big.txt", b"x" * 50)
    for index in range(2):
        _put(root, f"f{index}.txt", "ok")
    source = FolderSource(root, max_files=2, max_file_bytes=10)
    assert [p.location for p in _collect(source)] == ["f0.txt", "f1.txt"]
    assert source.report.skipped == {"too_large": 1}


def test_the_report_starts_afresh_on_every_run(root: Path) -> None:
    _put(root, "a.txt")
    source = FolderSource(root)
    _collect(source)
    _collect(source)
    assert source.report.pages == 1


@pytest.mark.skipif(os.name != "posix" or os.geteuid() == 0, reason="needs permissions that bind the current user")
def test_an_unreadable_file_is_reported_and_the_rest_is_still_read(root: Path) -> None:
    locked = _put(root, "a-locked.txt", "nope")
    _put(root, "b.txt", "fine")
    locked.chmod(0)
    try:
        source = FolderSource(root)
        assert [p.location for p in _collect(source)] == ["b.txt"]
    finally:
        locked.chmod(0o600)
    assert source.report.skipped == {"error": 1}
    assert source.report.errors == ["could not read a-locked.txt"]


def test_a_very_deep_tree_does_not_exhaust_the_recursion_limit(root: Path) -> None:
    deep = root
    for _ in range(150):
        deep = deep / "d"
    _put(deep, "bottom.txt", "bottom")
    (page,) = _collect(FolderSource(root))
    assert page.content == "bottom"
    assert page.location == "d/" * 150 + "bottom.txt"


def test_pages_is_an_async_generator_that_can_be_abandoned_early(root: Path) -> None:
    for index in range(4):
        _put(root, f"f{index}.txt")

    async def first() -> RawPage:
        iterator = FolderSource(root).pages()
        page = await anext(iterator)
        await iterator.aclose()  # type: ignore[attr-defined]
        return page

    assert asyncio.run(first()).location == "f0.txt"
