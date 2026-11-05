"""Shared pieces of text extraction: the result type, the limits, character hygiene and the paragraph writer.

Everything a website or a file contains is untrusted. The helpers here are what keeps that safe and bounded: invisible
and bidirectional characters are removed (they are a hiding place for instructions aimed at a model), numeric
character references cannot raise, and the output never grows past `MAX_TEXT_CHARS`.
"""

from __future__ import annotations

import html
import re
import unicodedata
from dataclasses import dataclass

MAX_INPUT_CHARS = 2_000_000  # longer input is cut here
MAX_TEXT_CHARS = 400_000  # longer output is cut at a paragraph or line boundary where possible
MAX_NESTING_DEPTH = 200  # elements nested deeper are not tracked: no recursion, no unbounded stack
MAX_TITLE_CHARS = 300
MAX_LINKS = 500

# A hard line break inside a paragraph (`<br>`, two trailing spaces). A private-use character is safe as an in-band
# marker because every private-use character is removed from the text it travels with.
BREAK = "\ue000"

_MIN_PARTIAL_PARAGRAPH = 2_000  # a paragraph that does not fit is cut mid-way only when at least this much room is left
_TABLE_CACHE_LIMIT = 4_096  # distinct characters remembered by the character table: bounds its memory


@dataclass(frozen=True, slots=True)
class ExtractedPage:
    """The plain text of one page plus what a crawler and an indexer need to know about it."""

    title: str  # <title>, else the first <h1> / '# ' heading, else ""
    text: str  # paragraphs separated by one blank line; headings as '## x', list items as '- x', table rows as 'a | b'
    links: tuple[str, ...]  # absolute http(s) URLs, fragment-free, de-duplicated, document order (HTML only)
    noindex: bool  # the page asks not to be indexed
    truncated: bool  # the input or the output hit a cap


class _CharTable(dict[int, str | None]):
    """`str.translate` table: drops control, format, surrogate and private-use characters; separators become newlines.

    Looked up lazily so it never materialises the whole Unicode range; the first few thousand answers are cached.
    """

    def __init__(self) -> None:
        super().__init__()
        for codepoint in range(0x20, 0x7F):
            self[codepoint] = chr(codepoint)
        self[0x09] = "\t"
        self[0x0A] = "\n"
        for codepoint in (0x0B, 0x0C, 0x85):  # vertical tab, form feed, next line: line separators
            self[codepoint] = "\n"

    def __missing__(self, codepoint: int) -> str | None:
        char = chr(codepoint)
        category = unicodedata.category(char)
        result: str | None
        if category in ("Cc", "Cf", "Cs", "Co"):
            result = None
        elif category in ("Zl", "Zp"):
            result = "\n"
        else:
            result = char
        if len(self) < _TABLE_CACHE_LIMIT:
            self[codepoint] = result
        return result


_CHAR_TABLE = _CharTable()
_LONG_CHARREF = re.compile(r"&#[0-9]{12,};?")


def clean_chars(text: str) -> str:
    """Remove invisible and dangerous characters (all of Cc except newline and tab, Cf, Cs, Co); NUL goes too.

    Line and paragraph separators, form feed, vertical tab and next-line become newlines.
    """
    if text.isascii() and text.isprintable():
        return text
    return text.translate(_CHAR_TABLE)


def decode_entities(text: str) -> str:
    """`html.unescape`, made safe: a decimal reference of a dozen digits or more is an invalid code point anyway,
    and `int()` refuses absurdly long digit strings, so those are replaced before decoding."""
    if "&" not in text:
        return text
    return html.unescape(_LONG_CHARREF.sub("\ufffd", text))


def collapse(text: str) -> str:
    """Runs of any Unicode whitespace become one space; the ends are trimmed."""
    return " ".join(text.split())


def normalize_newlines(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def bound_input(text: str) -> tuple[str, bool]:
    """Cut the input at `MAX_INPUT_CHARS`; the flag says whether anything was cut."""
    if len(text) > MAX_INPUT_CHARS:
        return text[:MAX_INPUT_CHARS], True
    return text, False


def limit_title(text: str) -> str:
    """A title as it is stored: cleaned, whitespace-collapsed, at most `MAX_TITLE_CHARS` characters."""
    return collapse(clean_chars(text))[:MAX_TITLE_CHARS].rstrip()


def join_inline(parts: list[str], *, keep_breaks: bool) -> str:
    """Join already-cleaned inline fragments into one paragraph.

    Whitespace collapses to single spaces. `BREAK` markers become newlines when `keep_breaks`, spaces otherwise
    (list items, table cells and headings are always one line).
    """
    joined = "".join(parts)
    if keep_breaks:
        lines = (" ".join(piece.split()) for piece in joined.split(BREAK))
        return "\n".join(line for line in lines if line)
    return " ".join(joined.replace(BREAK, " ").split())


def tidy_preformatted(lines: list[str]) -> str:
    """Code and preformatted lines: indentation kept, trailing space removed, runs of blank lines become one blank
    line, no blank lines at either end."""
    out: list[str] = []
    for raw in lines:
        line = raw.rstrip()
        if line:
            out.append(line)
        elif out and out[-1]:
            out.append("")
    while out and not out[-1]:
        out.pop()
    return "\n".join(out)


def _cut_whole_lines(text: str, room: int) -> str:
    """The longest run of whole lines of `text` that fits in `room` characters ('' when the first line is too long)."""
    if text[room] == "\n":
        return text[:room]
    newline = text.rfind("\n", 0, room)
    return text[:newline] if newline > 0 else ""


def _cut_at_word(text: str, room: int) -> str:
    head = text[:room]
    space = head.rfind(" ")
    return head[:space] if space > room // 2 else head


class ParagraphWriter:
    """Collects paragraphs (joined by one blank line) and refuses to grow past the limit.

    Cuts happen at a line boundary where possible, otherwise a paragraph that does not fit is dropped when
    there is already text, and cut at a word when it is (nearly) all there is.
    """

    __slots__ = ("_chunks", "_limit", "_size", "full", "truncated")

    def __init__(self, limit: int = MAX_TEXT_CHARS) -> None:
        self._chunks: list[str] = []
        self._limit = limit
        self._size = 0
        self.full = False
        self.truncated = False

    def add(self, paragraph: str) -> None:
        text = unicodedata.normalize("NFC", paragraph).rstrip().lstrip("\n")
        if not text:
            return
        if self.full:
            self.truncated = True
            return
        separator = 2 if self._chunks else 0
        room = self._limit - self._size - separator
        if len(text) <= room:
            self._chunks.append(text)
            self._size += separator + len(text)
            return
        self.truncated = True
        self.full = True
        cut = ""
        if room > 0:
            cut = _cut_whole_lines(text, room)
            if not cut and (not self._chunks or room >= _MIN_PARTIAL_PARAGRAPH):
                cut = _cut_at_word(text, room)
            cut = cut.rstrip()
        if cut:
            self._chunks.append(cut)
            self._size += separator + len(cut)

    def text(self) -> str:
        return "\n\n".join(self._chunks)
