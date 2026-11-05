"""A folder of exported pages and notes as a content source (the CLI's way of ingesting local files).

The walk is deterministic (names in sorted order, depth first), skips hidden entries, never follows a symbolic
link (a link inside the folder could point anywhere on the machine) and reads at most a fixed number of bytes
from at most a fixed number of files.
"""

from __future__ import annotations

import asyncio
import os
import posixpath
import stat
from collections.abc import AsyncIterator, Iterator
from pathlib import Path
from urllib.parse import quote

from rag_chatbot.domain.errors import SourceError
from rag_chatbot.domain.model import PageKind, RawPage
from rag_chatbot.infrastructure.sources import SourceReport

_KINDS: dict[str, PageKind] = {
    ".html": "html",
    ".htm": "html",
    ".xhtml": "html",
    ".md": "markdown",
    ".markdown": "markdown",
    ".txt": "text",
}
_NO_FOLLOW = getattr(os, "O_NOFOLLOW", 0)  # also refuse a file swapped for a link after the walk saw it


class FolderSource:
    """Every `.html`/`.htm`/`.xhtml`, `.md`/`.markdown` and `.txt` file under `root`, as `RawPage`s.

    `location` is the posix path relative to the root. `url` is `base_url` plus that path without its extension
    (`index` files map to their directory, `docs/index.html` -> `<base>/docs/`) when `base_url` is given, else "".
    After (or during) iteration `report` says what was delivered and what was skipped: `too_large`, `limit`
    (the `max_files` cap was reached), `symlink` and `error` (unreadable)."""

    def __init__(
        self,
        root: Path,
        *,
        base_url: str = "",
        max_files: int = 5_000,
        max_file_bytes: int = 2_000_000,
    ) -> None:
        self._root = root
        self._base_url = base_url
        self._max_files = max_files
        self._max_file_bytes = max_file_bytes
        self.report = SourceReport()

    async def pages(self) -> AsyncIterator[RawPage]:
        """Raises `SourceError` when the root is a symbolic link or not a directory, before yielding anything."""
        report = self.report = SourceReport()
        _check_root(self._root)
        for entry, relative in _walk(self._root, report):
            kind = _kind_of(entry.name)
            if kind is None:
                continue
            if report.pages >= self._max_files:
                report.skip("limit")
                return
            try:
                data = await asyncio.to_thread(_load, entry.path, self._max_file_bytes)
            except OSError:
                report.skip("error")
                report.add_error(f"could not read {relative}")
                continue
            if data is None:
                report.skip("too_large")
                continue
            report.pages += 1
            content = data.decode("utf-8", errors="replace").removeprefix("\ufeff")
            yield RawPage(location=relative, kind=kind, content=content, url=_url_for(self._base_url, relative))


def _kind_of(name: str) -> PageKind | None:
    return _KINDS.get(posixpath.splitext(name)[1].lower())


def _check_root(root: Path) -> None:
    if root.is_symlink():
        raise SourceError("the content folder is a symbolic link")
    if not root.is_dir():
        raise SourceError("the content folder does not exist or is not a directory")


def _walk(root: Path, report: SourceReport) -> Iterator[tuple[os.DirEntry[str], str]]:
    """Regular files below `root` with their posix relative paths, depth first, names sorted. An explicit stack
    keeps a deeply nested tree from exhausting the interpreter's recursion limit."""
    try:
        root_entries = _entries(root)
    except OSError as exc:
        raise SourceError("the content folder cannot be listed") from exc
    stack: list[tuple[Iterator[os.DirEntry[str]], str]] = [(iter(root_entries), "")]
    while stack:
        entries, prefix = stack[-1]
        entry = next(entries, None)
        if entry is None:
            stack.pop()
            continue
        if entry.name.startswith("."):
            continue
        entry_type = _entry_type(entry)
        if entry_type == "symlink":
            report.skip("symlink")
        elif entry_type == "dir":
            try:
                stack.append((iter(_entries(Path(entry.path))), f"{prefix}{entry.name}/"))
            except OSError:
                report.skip("error")
                report.add_error(f"could not list {prefix}{entry.name}")
        elif entry_type == "file":
            yield entry, prefix + entry.name


def _entries(directory: Path) -> list[os.DirEntry[str]]:
    with os.scandir(directory) as scanned:
        return sorted(scanned, key=lambda entry: entry.name)


def _entry_type(entry: os.DirEntry[str]) -> str:
    try:
        if entry.is_symlink():
            return "symlink"
        if entry.is_dir(follow_symlinks=False):
            return "dir"
        if entry.is_file(follow_symlinks=False):
            return "file"
    except OSError:
        return "other"
    return "other"


def _load(path: str, max_bytes: int) -> bytes | None:
    """The file's bytes, or None when it is larger than `max_bytes` (checked up front and again while reading, in
    case the file grows). Raises OSError when it cannot be opened or read."""
    descriptor = os.open(path, os.O_RDONLY | _NO_FOLLOW)
    with os.fdopen(descriptor, "rb") as handle:
        status = os.fstat(handle.fileno())
        if not stat.S_ISREG(status.st_mode):
            raise OSError("not a regular file")
        if status.st_size > max_bytes:
            return None
        data = handle.read(max_bytes + 1)
    return None if len(data) > max_bytes else data


def _url_for(base_url: str, relative: str) -> str:
    if not base_url:
        return ""
    stem = posixpath.splitext(relative)[0]
    directory, _, name = stem.rpartition("/")
    if name.lower() == "index":
        stem = f"{directory}/" if directory else ""
    return f"{base_url.rstrip('/')}/{quote(stem, safe='/')}"
