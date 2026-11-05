"""Markdown and plain text to the shared plain-text format.

The Markdown reader is a line-based block parser plus a single-pass inline scanner. It is not a full CommonMark
implementation; it removes the syntax and keeps the words. Everything is linear in the input: no regular expression
here repeats over a class that the following token can also match, and every look-ahead is bounded by a window.
"""

from __future__ import annotations

import re
import string
from dataclasses import dataclass

from rag_chatbot.domain.textnorm import (
    BREAK,
    ExtractedPage,
    ParagraphWriter,
    bound_input,
    clean_chars,
    decode_entities,
    limit_title,
    normalize_newlines,
    tidy_preformatted,
)

_SPECIAL = re.compile(r"[\\!\[\]*_~`<]")
_HIDDEN_START = re.compile(r"<!--|<(script|style)(?![A-Za-z0-9])", re.IGNORECASE)
_CLOSERS = {
    "script": re.compile(r"</script", re.IGNORECASE),
    "style": re.compile(r"</style", re.IGNORECASE),
}
_TAG = re.compile(r"</?([A-Za-z][A-Za-z0-9-]{0,31})(?:[ \t\n/][^<>]*)?>")
_AUTOLINK = re.compile(r"<(?:[A-Za-z][A-Za-z0-9+.-]{1,31}:[^<>\s]*|[^<>\s@]+@[^<>\s]+)>")
_INLINE_TAGS = frozenset(
    {
        "a", "abbr", "b", "bdi", "bdo", "cite", "code", "data", "del", "dfn", "em", "font", "i", "ins", "kbd",
        "mark", "q", "s", "samp", "small", "span", "strike", "strong", "sub", "sup", "time", "tt", "u", "var",
    }
)  # fmt: skip
_ESCAPABLE = frozenset(string.punctuation)
_PIPE = "\ue001"  # stands for an escaped pipe while a table row is split into cells
_MAX_OPEN_BRACKETS = 256
_LINK_WINDOW = 2_000
_CODE_WINDOW = 5_000
_MAX_BACKTICK_RUN = 32
_MAX_PAREN_RETRIES = 8
_FRONT_MATTER_TRUE = frozenset({"1", "on", "true", "yes"})


@dataclass(slots=True)
class _FrontMatter:
    title: str = ""
    noindex: bool = False


# --- plain text ---------------------------------------------------------------------------------------------------


def extract_plain(text: str) -> ExtractedPage:
    """Plain text: line endings normalised, dangerous characters removed, trailing space stripped, runs of blank lines
    collapsed to one. Line breaks inside a paragraph are kept. There is no title and there are no links."""
    bounded, cut = bound_input(text)
    writer = ParagraphWriter()
    block: list[str] = []
    for raw in clean_chars(normalize_newlines(bounded)).split("\n"):
        line = raw.rstrip()
        if line:
            block.append(line)
        elif block:
            writer.add("\n".join(block))
            block = []
    if block:
        writer.add("\n".join(block))
    return ExtractedPage(title="", text=writer.text(), links=(), noindex=False, truncated=cut or writer.truncated)


# --- markdown: inline ---------------------------------------------------------------------------------------------


def _dest_end(text: str, open_paren: int) -> int | None:
    """Index just after the ')' that closes the '(' at `open_paren`, balancing parentheses a few levels deep."""
    limit = open_paren + 1 + _LINK_WINDOW
    close = text.find(")", open_paren + 1, limit)
    if close < 0:
        return None
    opened = text.count("(", open_paren + 1, close)
    closed = 1
    retries = 0
    while closed <= opened and retries < _MAX_PAREN_RETRIES:
        following = text.find(")", close + 1, limit)
        if following < 0:
            break
        opened += text.count("(", close + 1, following)
        closed += 1
        close = following
        retries += 1
    return close + 1


def _link_end(text: str, pos: int) -> int | None:
    """After a closing ']': the end of '(destination "title")' or '[reference]', None when neither follows."""
    if text.startswith("(", pos):
        return _dest_end(text, pos)
    if text.startswith("[", pos):
        close = text.find("]", pos + 1, pos + 1 + _LINK_WINDOW)
        return None if close < 0 else close + 1
    return None


def _emphasis(text: str, start: int, char: str) -> tuple[str, int]:
    size = len(text)
    end = start
    while end < size and text[end] == char:
        end += 1
    before = text[start - 1] if start else " "
    after = text[end] if end < size else " "
    space_before = before.isspace() or before == BREAK
    space_after = after.isspace() or after == BREAK
    if char == "~":
        delimiter = end - start >= 2 and not (space_before and space_after)
    elif space_before and space_after:
        delimiter = False
    elif char == "_" and before.isalnum() and after.isalnum():
        delimiter = False  # snake_case is not emphasis
    else:
        delimiter = True
    return ("" if delimiter else text[start:end]), end


def _code_span(text: str, start: int) -> tuple[str, int]:
    """A backtick run: its content up to the run of the same length is kept verbatim, the backticks are dropped."""
    size = len(text)
    end = start
    while end < size and text[end] == "`":
        end += 1
    run = end - start
    if run > _MAX_BACKTICK_RUN:
        return "", end
    marker = "`" * run
    search = end
    limit = end + _CODE_WINDOW
    for _ in range(4):
        found = text.find(marker, search, limit)
        if found < 0:
            break
        stop = found + run
        if text[found - 1 : found] != "`" and text[stop : stop + 1] != "`":
            return text[end:found], stop
        search = stop
        while search < size and text[search] == "`":
            search += 1
    return "", end


def _angle(text: str, start: int) -> tuple[str, int]:
    """'<' starts an autolink (dropped), an HTML tag (dropped, its text kept) or is just a less-than sign."""
    autolink = _AUTOLINK.match(text, start)
    if autolink:
        return "", autolink.end()
    tag = _TAG.match(text, start)
    if tag:
        name = tag.group(1).lower()
        if name in _INLINE_TAGS:
            return "", tag.end()
        return (BREAK if name == "br" else " "), tag.end()
    return "<", start + 1


def _inline(text: str) -> str:
    """Remove inline Markdown syntax in one pass: escapes, links and images (their text stays), emphasis, code spans,
    HTML tags and autolinks."""
    out: list[str] = []
    openers: list[int] = []  # positions in `out` of pending '[' / '![' pieces
    size = len(text)
    pos = 0
    while pos < size:
        found = _SPECIAL.search(text, pos)
        if found is None:
            out.append(text[pos:])
            break
        index = found.start()
        if index > pos:
            out.append(text[pos:index])
        char = text[index]
        pos = index + 1
        if char == "\\":
            if pos < size and text[pos] in _ESCAPABLE:
                out.append(text[pos])
                pos += 1
            else:
                out.append("\\")
        elif char == "!":
            if text.startswith("[", pos) and len(openers) < _MAX_OPEN_BRACKETS:
                openers.append(len(out))
                out.append("![")
                pos += 1
            else:
                out.append("!")
        elif char == "[":
            if len(openers) < _MAX_OPEN_BRACKETS:
                openers.append(len(out))
            out.append("[")
        elif char == "]":
            end = _link_end(text, pos) if openers else None
            if openers:
                opener = openers.pop()
                if end is not None:
                    out[opener] = ""
                    pos = end
                    continue
            out.append("]")
        elif char == "`":
            piece, pos = _code_span(text, index)
            out.append(piece)
        elif char == "<":
            piece, pos = _angle(text, index)
            out.append(piece)
        else:
            piece, pos = _emphasis(text, index, char)
            out.append(piece)
    return "".join(out)


def _render(text: str, *, keep_breaks: bool) -> str:
    """Inline syntax removed, entities decoded, dangerous characters removed, whitespace collapsed."""
    decoded = decode_entities(_inline(text))
    if keep_breaks:
        lines = (" ".join(clean_chars(part).split()) for part in decoded.split(BREAK))
        return "\n".join(line for line in lines if line)
    return " ".join(clean_chars(decoded.replace(BREAK, " ")).split())


# --- markdown: blocks ---------------------------------------------------------------------------------------------


def _front_matter(lines: list[str]) -> tuple[_FrontMatter, int]:
    """YAML front matter at the very start: the metadata we use and the index of the first body line."""
    meta = _FrontMatter()
    if not lines or lines[0].rstrip() != "---":
        return meta, 0
    end = 0
    for index in range(1, len(lines)):
        if lines[index].rstrip() in ("---", "..."):
            end = index
            break
    if not end:
        return meta, 0
    for line in lines[1:end]:
        key, separator, value = line.partition(":")
        if not separator or line[:1] in " \t#-":
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        name = key.strip().lower()
        if name == "title":
            meta.title = limit_title(value)
        elif (name == "robots" and "noindex" in value.lower()) or (
            name == "noindex" and value.lower() in _FRONT_MATTER_TRUE
        ):
            meta.noindex = True
    return meta, end + 1


def _fence_open(line: str) -> tuple[str, int] | None:
    stripped = line.lstrip(" ")
    char = stripped[:1]
    if char not in ("`", "~"):
        return None
    run = 0
    while run < len(stripped) and stripped[run] == char:
        run += 1
    if run < 3 or (char == "`" and "`" in stripped[run:]):
        return None
    return char, run


def _strip_quote(line: str) -> str:
    """Remove leading blockquote markers ('> > text')."""
    index = 0
    size = len(line)
    while True:
        probe = index
        while probe < size and probe - index < 3 and line[probe] == " ":
            probe += 1
        if probe < size and line[probe] == ">":
            index = probe + 1
            if index < size and line[index] == " ":
                index += 1
        else:
            break
    return line[index:] if index else line


def _atx_heading(body: str) -> tuple[int, str] | None:
    level = 0
    while level < len(body) and body[level] == "#":
        level += 1
    if level == 0 or level > 6 or (level < len(body) and body[level] not in " \t"):
        return None
    text = body[level:].strip()
    trimmed = text.rstrip("#")
    if not trimmed or trimmed.endswith((" ", "\t")):
        text = trimmed.rstrip()
    return level, text


def _is_rule(body: str) -> bool:
    compact = body.replace(" ", "").replace("\t", "")
    return len(compact) >= 3 and compact[0] in "-*_" and not compact.strip(compact[0])


def _is_underline(body: str, char: str, minimum: int) -> bool:
    return len(body) >= minimum and not body.strip(char)


def _list_item(body: str) -> str | None:
    """The text of a list item line ('- a', '* a', '+ a', '1. a', '1) a'), None when it is not one."""
    if body[0] in "-*+":
        if len(body) == 1:
            return ""
        return body[2:].strip() if body[1] in " \t" else None
    digits = 0
    while digits < len(body) and digits < 10 and body[digits] in "0123456789":
        digits += 1
    if 1 <= digits <= 9 and digits + 1 < len(body) and body[digits] in ".)" and body[digits + 1] in " \t":
        return body[digits + 2 :].strip()
    return None


def _is_reference_definition(body: str) -> bool:
    if not body.startswith("[") or body.startswith("[^"):
        return False
    close = body.find("]:", 1, 1_000)
    if close <= 1 or body.find("]", 1, close) >= 0:
        return False
    parts = body[close + 2 :].split(None, 1)
    return bool(parts) and any(char in parts[0] for char in ":/#.")


def _is_table_separator(line: str) -> bool:
    stripped = line.strip()
    if "|" not in stripped or "-" not in stripped:
        return False
    stripped = stripped.removeprefix("|").removesuffix("|")
    cells = stripped.split("|")
    return all(cell.strip(" \t:") and not cell.strip(" \t:").strip("-") for cell in cells)


def _split_row(row: str) -> list[str]:
    stripped = row.strip().replace("\\|", _PIPE).removeprefix("|").removesuffix("|")
    return [cell.replace(_PIPE, "|") for cell in stripped.split("|")]


class _Markdown:
    def __init__(self) -> None:
        self.writer = ParagraphWriter()
        self.first_h1 = ""
        self._para: list[str] = []
        self._items: list[list[str]] = []
        self._markup = ""  # "comment", "script" or "style" while inside one that spans lines
        self._fence: tuple[str, int] | None = None
        self._code: list[str] = []

    def run(self, lines: list[str], start: int) -> None:
        index = start
        total = len(lines)
        while index < total:
            raw = lines[index]
            index += 1
            if self._fence is not None:
                self._code_line(raw, self._fence)
                continue
            if not self._markup:
                opener = _fence_open(raw)
                if opener is not None:
                    self._flush()
                    self._fence = opener
                    continue
            line = _strip_quote(self._strip_markup(raw))
            body = line.strip()
            if not body:
                self._flush()
                continue
            if "|" in body and index < total and _is_table_separator(lines[index]):
                index = self._table(line, lines, index)
                continue
            if self._block_line(line, body):
                continue
            if self._items:
                self._items[-1].append(body)
            else:
                self._para.append(line)
        if self._fence is not None:
            self._end_fence()
        self._flush()

    def _block_line(self, line: str, body: str) -> bool:
        """Handle a line that is a heading, rule, list item or reference definition; False for ordinary text."""
        indent = len(line) - len(line.lstrip(" "))
        if indent < 4:
            heading = _atx_heading(body)
            if heading is not None:
                self._flush()
                self._heading(heading[0], heading[1])
                return True
            if self._para and _is_underline(body, "=", 1):
                self._setext(1)
                return True
            if self._para and _is_underline(body, "-", 2):
                self._setext(2)
                return True
        if _is_rule(body):
            self._flush()
            return True
        item = _list_item(body)
        if item is not None:
            self._flush_para()
            if item:
                self._items.append([item])
            return True
        return _is_reference_definition(body)

    # code fences

    def _code_line(self, raw: str, fence: tuple[str, int]) -> None:
        char, size = fence
        stripped = raw.strip()
        if len(stripped) >= size and not stripped.strip(char):
            self._end_fence()
        else:
            self._code.append(raw)

    def _end_fence(self) -> None:
        self.writer.add(tidy_preformatted(self._code))
        self._code = []
        self._fence = None

    # comments, scripts and styles (which may span lines)

    def _strip_markup(self, line: str) -> str:
        if not self._markup and "<" not in line:
            return line
        out: list[str] = []
        pos = 0
        size = len(line)
        while pos < size:
            if self._markup:
                pos = self._skip_hidden(line, pos)
                continue
            found = _HIDDEN_START.search(line, pos)
            if found is None:
                out.append(line[pos:])
                break
            out.append(line[pos : found.start()])
            pos = found.end()
            element = found.group(1)
            if element is None:
                self._markup = "comment"
            else:
                self._markup = element.lower()
                close = line.find(">", pos)
                pos = size if close < 0 else close + 1
        return "".join(out)

    def _skip_hidden(self, line: str, pos: int) -> int:
        if self._markup == "comment":
            end = line.find("-->", pos)
            if end < 0:
                return len(line)
            self._markup = ""
            return end + 3
        closer = _CLOSERS[self._markup].search(line, pos)
        if closer is None:
            return len(line)
        self._markup = ""
        close = line.find(">", closer.end())
        return len(line) if close < 0 else close + 1

    # emitting

    def _flush(self) -> None:
        self._flush_para()
        self._flush_items()

    def _para_text(self) -> str:
        parts: list[str] = []
        last = len(self._para) - 1
        for index, line in enumerate(self._para):
            text = line.strip()
            hard = line.endswith("  ")
            if text.endswith("\\") and not text.endswith("\\\\"):
                text = text[:-1]
                hard = True
            parts.append(text)
            if index < last:
                parts.append(BREAK if hard else " ")
        return "".join(parts)

    def _flush_para(self) -> None:
        if self._para:
            text = self._para_text()
            self._para = []
            self.writer.add(_render(text, keep_breaks=True))

    def _flush_items(self) -> None:
        if self._items:
            rendered = [_render(" ".join(item), keep_breaks=False) for item in self._items]
            self._items = []
            self.writer.add("\n".join("- " + text for text in rendered if text))

    def _heading(self, level: int, raw: str) -> None:
        text = _render(raw, keep_breaks=False)
        if text:
            if level == 1 and not self.first_h1:
                self.first_h1 = text
            self.writer.add("#" * level + " " + text)

    def _setext(self, level: int) -> None:
        text = self._para_text()
        self._para = []
        self._heading(level, text)

    def _table(self, header: str, lines: list[str], index: int) -> int:
        """Consume a table whose separator row is lines[index]; return the index after its last row."""
        self._flush()
        rows = [header]
        index += 1
        while index < len(lines) and "|" in lines[index] and lines[index].strip():
            rows.append(_strip_quote(self._strip_markup(lines[index])))
            index += 1
        rendered: list[str] = []
        for row in rows:
            cells = [_render(cell, keep_breaks=False) for cell in _split_row(row)]
            line = " | ".join(cell for cell in cells if cell)
            if line:
                rendered.append(line)
        self.writer.add("\n".join(rendered))
        return index


def extract_markdown(text: str) -> ExtractedPage:
    """Markdown source to plain text: syntax removed, words kept; comments and script/style elements dropped;
    code blocks verbatim; tables as one line per row. Title from front matter or the first level-1 heading."""
    bounded, cut = bound_input(text)
    lines = clean_chars(normalize_newlines(bounded)).split("\n")
    meta, start = _front_matter(lines)
    converter = _Markdown()
    converter.run(lines, start)
    return ExtractedPage(
        title=meta.title or limit_title(converter.first_h1),
        text=converter.writer.text(),
        links=(),
        noindex=meta.noindex,
        truncated=cut or converter.writer.truncated,
    )
