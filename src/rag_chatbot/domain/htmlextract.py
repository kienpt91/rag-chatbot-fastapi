"""HTML to plain text.

A small, linear-time tokenizer drives an explicit element stack (no recursion). `html.parser.HTMLParser` is not used
because several of its start-tag and declaration paths rescan the rest of the input for every stray `<`, which turns
a hostile page (`<a<a<a...`) into quadratic work. Here every token consumes input and every scan is bounded by the
token, so work is proportional to the input length.

What is dropped (never indexed): scripts, styles and other non-content elements, page chrome (nav, footer, aside,
forms, ...), a site `header`, cookie/newsletter/ad/social boilerplate by class or id, comments and declarations, and
everything a page hides from its visitors (`hidden`, `aria-hidden`, inline styles that make text invisible), since
hidden text is the classic place to plant instructions for a model.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from urllib.parse import urldefrag, urljoin, urlsplit

from rag_chatbot.domain.textnorm import (
    BREAK,
    MAX_LINKS,
    MAX_NESTING_DEPTH,
    MAX_TITLE_CHARS,
    ExtractedPage,
    ParagraphWriter,
    bound_input,
    clean_chars,
    decode_entities,
    join_inline,
    limit_title,
    normalize_newlines,
    tidy_preformatted,
)

_VOID = frozenset(
    {
        "area",
        "base",
        "basefont",
        "bgsound",
        "br",
        "col",
        "embed",
        "frame",
        "hr",
        "img",
        "input",
        "keygen",
        "link",
        "meta",
        "param",
        "source",
        "track",
        "wbr",
    }
)
# Elements whose content is not markup: it is skipped up to the matching end tag (title is read first).
_RAW_TEXT = frozenset({"iframe", "noembed", "noframes", "script", "style", "textarea", "title", "xmp"})
_RAW_END = {name: re.compile("</" + name + "(?=[ \t\n\f\r/>])", re.IGNORECASE) for name in _RAW_TEXT}
_HEADINGS = frozenset({"h1", "h2", "h3", "h4", "h5", "h6"})
_LISTS = frozenset({"ol", "ul"})
_CELLS = frozenset({"td", "th"})
_DROP_ALWAYS = frozenset(
    {
        "aside",
        "audio",
        "button",
        "canvas",
        "datalist",
        "dialog",
        "embed",
        "footer",
        "form",
        "frame",
        "frameset",
        "head",
        "iframe",
        "map",
        "menu",
        "nav",
        "noscript",
        "object",
        "script",
        "select",
        "style",
        "svg",
        "template",
        "textarea",
        "video",
    }
)
_HEAD_OK = frozenset({"base", "head", "html", "link", "meta", "noscript", "script", "style", "template", "title"})
_BLOCK = frozenset(
    {
        "address",
        "article",
        "aside",
        "blockquote",
        "body",
        "center",
        "dd",
        "details",
        "dialog",
        "div",
        "dl",
        "dt",
        "fieldset",
        "figcaption",
        "figure",
        "footer",
        "form",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "header",
        "hgroup",
        "hr",
        "html",
        "legend",
        "li",
        "main",
        "menu",
        "nav",
        "ol",
        "p",
        "pre",
        "search",
        "section",
        "summary",
        "table",
        "td",
        "th",
        "tr",
        "ul",
    }
)
_P_CLOSERS = frozenset(
    {
        "address",
        "article",
        "aside",
        "blockquote",
        "center",
        "details",
        "dd",
        "dialog",
        "div",
        "dl",
        "dt",
        "fieldset",
        "figcaption",
        "figure",
        "footer",
        "form",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "header",
        "hgroup",
        "hr",
        "li",
        "main",
        "menu",
        "nav",
        "ol",
        "p",
        "pre",
        "search",
        "section",
        "summary",
        "table",
        "ul",
    }
)
_P = frozenset({"p"})
_P_BOUNDARY = frozenset({"button", "caption", "html", "object", "table", "td", "template", "th"})
_LI = frozenset({"li"})
_LIST_BOUNDARY = frozenset({"menu", "ol", "ul"})
_DEFINITION = frozenset({"dd", "dt"})
_DEFINITION_BOUNDARY = frozenset({"dl"})
_CELL_SCOPE = frozenset({"table", "tr"})
_ROW = frozenset({"tr"})
_TABLE_ONLY = frozenset({"table"})
_SECTIONS = frozenset({"tbody", "tfoot", "thead"})
_OPTION = frozenset({"option"})
_OPTION_BOUNDARY = frozenset({"datalist", "select"})
# Class and id tokens of boilerplate. `sr-only` and `screenreader` are deliberately absent: that text is visible
# to assistive technology and is often real content.
_BOILERPLATE = frozenset(
    {
        "ads",
        "adsense",
        "advert",
        "banner",
        "breadcrumb",
        "breadcrumbs",
        "consent",
        "cookie",
        "cookies",
        "gdpr",
        "modal",
        "newsletter",
        "popup",
        "promo",
        "share",
        "sidebar",
        "skip",
        "social",
        "subscribe",
    }
)
_CLASS_EXEMPT = frozenset({"body", "html", "main"})  # `body class="has-sidebar"` is a page, not a sidebar
_KEPT_ATTRS = frozenset(
    {"alt", "aria-hidden", "class", "content", "hidden", "href", "id", "name", "rel", "role", "style"}
)

_WS = re.compile(r"[ \t\n\f\r]*")
_SLASH_WS = re.compile(r"[ \t\n\f\r/]*")
_TAG_NAME = re.compile(r"[^ \t\n\f\r/>]*")
_ATTR_NAME = re.compile(r"[^ \t\n\f\r/>][^ \t\n\f\r/>=]*")
_UNQUOTED = re.compile(r"[^ \t\n\f\r>]*")
_TOKEN_SPLIT = re.compile(r"[\s_-]+")

_MAX_NAME = 32
_MAX_ATTR_NAME = 64
_MAX_HREF = 2_048
_IMPLIED_SCAN = 24  # how far down the stack an implied end tag looks for the element it closes
_HIDDEN_OFFSET = 999.0
_URL_STRIP = {9: None, 10: None, 13: None}
_URL_TRIM = "".join(chr(code) for code in range(33))

_NO_REGION = 0
_MAIN_REGION = 1
_ARTICLE_REGION = 2


def _scan(pattern: re.Pattern[str], text: str, pos: int) -> int:
    match = pattern.match(text, pos)
    return match.end() if match else pos


# --- inline styles that hide text ---------------------------------------------------------------------------------


def _strip_css_comments(style: str) -> str:
    out: list[str] = []
    pos = 0
    while True:
        start = style.find("/*", pos)
        if start < 0:
            out.append(style[pos:])
            break
        out.append(style[pos:start])
        end = style.find("*/", start + 2)
        if end < 0:
            break
        pos = end + 2
    return "".join(out)


def _number(value: str, *, signed: bool) -> float | None:
    """The leading number of a CSS length ('-9999px' -> -9999.0), None when there is none or the unit is odd."""
    index = 0
    sign = 1.0
    if signed and value[:1] == "-":
        sign = -1.0
        index = 1
    end = index
    while end < len(value) and value[end] in "0123456789.":
        end += 1
    digits = value[index:end]
    unit = value[end:]
    if not digits or digits.count(".") > 1 or digits == "." or not (unit in {"", "%"} or unit.isalpha()):
        return None
    try:
        return sign * float(digits)
    except ValueError:
        return None


def _is_zero(value: str | None) -> bool:
    if not value:
        return False
    number = _number(value.replace(" ", ""), signed=False)
    return number == 0.0


def _is_far_negative(value: str | None) -> bool:
    if not value:
        return False
    number = _number(value.replace(" ", ""), signed=True)
    return number is not None and number <= -_HIDDEN_OFFSET


def _clip_is_empty(value: str) -> bool:
    if not value.startswith("rect("):
        return False
    rest = value[5:].lstrip()
    return rest.startswith("0") and rest[1:2] not in tuple("0123456789.")


def _style_hides(style: str) -> bool:
    declarations: dict[str, str] = {}
    for declaration in clean_chars(_strip_css_comments(style)).lower().split(";"):
        prop, separator, value = declaration.partition(":")
        if not separator:
            continue
        value = " ".join(value.split()).removesuffix("!important").strip()
        declarations["".join(prop.split())] = value
    get = declarations.get
    return (
        get("display") == "none"
        or get("visibility") == "hidden"
        or _is_zero(get("font-size"))
        or _is_zero(get("opacity"))
        or (get("overflow") == "hidden" and (_is_zero(get("height")) or _is_zero(get("max-height"))))
        or _is_far_negative(get("text-indent"))
        or (get("position") == "absolute" and (_is_far_negative(get("left")) or _is_far_negative(get("top"))))
        or _clip_is_empty(get("clip", ""))
        or get("color") == "transparent"
    )


def _has_boilerplate_token(value: str) -> bool:
    return not _BOILERPLATE.isdisjoint(_TOKEN_SPLIT.split(value.lower()))


# --- the paragraph builder ----------------------------------------------------------------------------------------


class _Sink:
    """Turns start/end/text events into paragraphs for one text region (the whole body, the main area, the articles).

    Lists, tables, headings and preformatted blocks are 'contexts'; at most one is active at a time. Anything that
    would nest a different context inside the active one is flattened into it (a table in a list item becomes list
    text, a heading in a table cell is cell text).
    """

    def __init__(self) -> None:
        self.writer = ParagraphWriter()
        self.first_h1 = ""
        self._buf: list[str] = []
        self._lines: list[str] = []
        self._cells: list[str] = []
        self._pre: list[str] = []
        self._heading = 0
        self._list_depth = 0
        self._implicit_list = False
        self._table_depth = 0
        self._pre_depth = 0
        self._pre_lead = False

    # events from the extractor

    def text(self, text: str) -> None:
        if self._pre_depth:
            if self._pre_lead:
                self._pre_lead = False
                if text.startswith("\n"):
                    text = text[1:]
            self._pre.append(text)
        else:
            self._buf.append(text)

    def br(self) -> None:
        if self._pre_depth:
            self._pre.append("\n")
        else:
            self._buf.append(BREAK)

    def boundary(self) -> None:
        """A block element starts or ends."""
        if self._pre_depth:
            if self._pre and not self._pre[-1].endswith("\n"):
                self._pre.append("\n")
        elif self._heading or self._table_depth:
            self._buf.append(" ")
        elif self._list_depth:
            self._flush_item()
            if self._implicit_list:
                self._end_list()
        else:
            self._flush_paragraph()

    def start(self, name: str) -> None:
        if name in _HEADINGS:
            if self._busy():
                self.boundary()
            else:
                self._flush_paragraph()
                self._heading = int(name[1])
        elif name in _LISTS:
            if self._table_depth or self._heading or self._pre_depth:
                self.boundary()
            elif self._list_depth:
                self._flush_item()
                self._list_depth += 1
            else:
                self._flush_paragraph()
                self._list_depth = 1
        elif name == "li":
            if self._list_depth:
                self._flush_item()
            elif self._busy():
                self.boundary()
            else:
                self._flush_paragraph()
                self._list_depth = 1
                self._implicit_list = True
        elif name == "table":
            if self._heading or self._list_depth or self._pre_depth:
                self.boundary()
            else:
                if not self._table_depth:
                    self._flush_paragraph()
                self._table_depth += 1
        elif name == "tr":
            if self._table_depth:
                self._flush_row()
            else:
                self.boundary()
        elif name in _CELLS:
            if self._table_depth:
                self._flush_cell()
            else:
                self.boundary()
        elif name == "pre":
            if self._busy():
                if self._pre_depth:
                    self._pre_depth += 1
                else:
                    self.boundary()
            else:
                self._flush_paragraph()
                self._pre_depth = 1
                self._pre_lead = True
                self._pre = []
        elif name in _BLOCK:
            self.boundary()

    def end(self, name: str) -> None:
        if name in _HEADINGS:
            if self._heading:
                self._flush_heading()
            else:
                self.boundary()
        elif name in _LISTS:
            if self._list_depth:
                self._flush_item()
                self._list_depth -= 1
                if not self._list_depth:
                    self._implicit_list = False
                    self._emit_lines()
            else:
                self.boundary()
        elif name == "li":
            if self._list_depth:
                self._flush_item()
                if self._implicit_list:
                    self._end_list()
            else:
                self.boundary()
        elif name == "table":
            if self._table_depth:
                self._flush_row()
                self._table_depth -= 1
                if not self._table_depth:
                    self._emit_lines()
            else:
                self.boundary()
        elif name == "tr" or name in _CELLS:
            if not self._table_depth:
                self.boundary()
            elif name == "tr":
                self._flush_row()
            else:
                self._flush_cell()
        elif name == "pre":
            if self._pre_depth:
                self._pre_depth -= 1
                if not self._pre_depth:
                    self._emit_pre()
            else:
                self.boundary()
        elif name in _BLOCK:
            self.boundary()

    def finish(self) -> None:
        if self._heading:
            self._flush_heading()
        if self._pre_depth:
            self._pre_depth = 0
            self._emit_pre()
        if self._list_depth:
            self._flush_item()
            self._list_depth = 0
            self._implicit_list = False
        if self._table_depth:
            self._flush_row()
            self._table_depth = 0
        self._emit_lines()
        self._flush_paragraph()

    # internals

    def _busy(self) -> bool:
        return bool(self._heading or self._table_depth or self._list_depth or self._pre_depth)

    def _flush_paragraph(self) -> None:
        if self._buf:
            text = join_inline(self._buf, keep_breaks=True)
            self._buf = []
            self.writer.add(text)

    def _flush_heading(self) -> None:
        text = join_inline(self._buf, keep_breaks=False)
        self._buf = []
        level = self._heading
        self._heading = 0
        if text:
            if level == 1 and not self.first_h1:
                self.first_h1 = text[:MAX_TITLE_CHARS]
            self.writer.add("#" * level + " " + text)

    def _flush_item(self) -> None:
        if self._buf:
            text = join_inline(self._buf, keep_breaks=False)
            self._buf = []
            if text:
                self._lines.append("- " + text)

    def _end_list(self) -> None:
        self._flush_item()
        self._list_depth = 0
        self._implicit_list = False
        self._emit_lines()

    def _flush_cell(self) -> None:
        if self._buf:
            text = join_inline(self._buf, keep_breaks=False)
            self._buf = []
            if text:
                self._cells.append(text)

    def _flush_row(self) -> None:
        self._flush_cell()
        if self._cells:
            self._lines.append(" | ".join(self._cells))
            self._cells = []

    def _emit_lines(self) -> None:
        if self._lines:
            lines = self._lines
            self._lines = []
            self.writer.add("\n".join(lines))

    def _emit_pre(self) -> None:
        text = "".join(self._pre)
        self._pre = []
        self.writer.add(tidy_preformatted(text.split("\n")))


# --- the extractor ------------------------------------------------------------------------------------------------


@dataclass(slots=True)
class _Frame:
    name: str
    drops: bool  # this element and everything inside it is dropped
    region: int  # _MAIN_REGION / _ARTICLE_REGION for the elements that select where text is taken from


class _Extractor:
    def __init__(self, base_url: str) -> None:
        self._base_url = base_url
        self._stack: list[_Frame] = []
        self._open: dict[str, int] = {}  # open elements by name: an end tag with no open match is rejected in O(1)
        self._drop_depth = 0
        self._overflow = 0  # elements opened beyond MAX_NESTING_DEPTH (their tags are not tracked individually)
        self._overflow_drop_level = 0  # the overflow level at which a dropped element opened, 0 when none
        self._body = _Sink()
        self._main = _Sink()
        self._article = _Sink()
        self._active = [self._body]
        self._main_seen = False
        self._main_open = False
        self._article_seen = False
        self._article_depth = 0
        self.title = ""
        self.noindex = False
        self.links: list[str] = []
        self._seen_links: set[str] = set()

    def selected(self) -> _Sink:
        if self._main_seen:
            return self._main
        if self._article_seen:
            return self._article
        return self._body

    def first_heading(self) -> str:
        return self._body.first_h1

    # tokenizer

    def feed(self, html: str) -> None:
        size = len(html)
        pos = 0
        while pos < size:
            lt = html.find("<", pos)
            if lt < 0:
                self._on_text(html[pos:])
                return
            if lt > pos:
                self._on_text(html[pos:lt])
            pos = self._markup(html, lt, size)

    def finish(self) -> None:
        while self._stack:
            self._pop()
        for sink in (self._body, self._main, self._article):
            sink.finish()

    def _markup(self, html: str, lt: int, size: int) -> int:
        """Handle the construct starting at html[lt] == '<'; return where scanning resumes."""
        following = html[lt + 1 : lt + 2]
        if following.isascii() and following.isalpha():
            return self._start_tag(html, lt + 1, size)
        if following == "/":
            if html[lt + 2 : lt + 3].isascii() and html[lt + 2 : lt + 3].isalpha():
                return self._end_tag(html, lt + 2, size)
            return self._after(html, ">", lt + 2, size)
        if following == "!":
            if html.startswith("--", lt + 2):
                if html.startswith(">", lt + 4):
                    return lt + 5
                if html.startswith("->", lt + 4):
                    return lt + 6
                end = html.find("-->", lt + 4)
                return size if end < 0 else end + 3
            if html.startswith("[CDATA[", lt + 2):
                end = html.find("]]>", lt + 9)
                return size if end < 0 else end + 3
            return self._after(html, ">", lt + 2, size)
        if following == "?":
            return self._after(html, ">", lt + 2, size)
        self._on_text("<")
        return lt + 1

    @staticmethod
    def _after(html: str, marker: str, start: int, size: int) -> int:
        end = html.find(marker, start)
        return size if end < 0 else end + len(marker)

    def _start_tag(self, html: str, start: int, size: int) -> int:
        pos = _scan(_TAG_NAME, html, start)
        name = html[start : min(pos, start + _MAX_NAME)].lower()
        attrs: dict[str, str] = {}
        while True:
            pos = _scan(_SLASH_WS, html, pos)
            if pos >= size:
                return size  # end of input inside a tag: the tag is dropped, as browsers do
            if html[pos] == ">":
                pos += 1
                break
            end = max(_scan(_ATTR_NAME, html, pos), pos + 1)
            attr = html[pos:end].lower() if end - pos <= _MAX_ATTR_NAME else ""
            pos = _scan(_WS, html, end)
            value_start = value_end = pos
            if pos < size and html[pos] == "=":
                pos = _scan(_WS, html, pos + 1)
                if pos < size and html[pos] in "\"'":
                    close = html.find(html[pos], pos + 1)
                    if close < 0:
                        return size
                    value_start, value_end = pos + 1, close
                    pos = close + 1
                else:
                    value_start = pos
                    pos = value_end = _scan(_UNQUOTED, html, pos)
            if attr in _KEPT_ATTRS and attr not in attrs:
                attrs[attr] = html[value_start:value_end]
        if name in _RAW_TEXT:
            return self._raw_text(name, html, pos, size)
        if name in _VOID:
            self._void(name, attrs)
        else:
            self._open_element(name, attrs)
        return pos

    def _end_tag(self, html: str, start: int, size: int) -> int:
        pos = _scan(_TAG_NAME, html, start)
        name = html[start : min(pos, start + _MAX_NAME)].lower()
        self._close(name)
        return self._after(html, ">", pos, size)

    def _raw_text(self, name: str, html: str, start: int, size: int) -> int:
        match = _RAW_END[name].search(html, start)
        if match is None:
            if name == "title":
                # An unterminated title is a typo, not a reason to read the rest of the page as its text.
                end = html.find("<", start)
                end = size if end < 0 else end
                self._set_title(html[start:end])
                return end
            return size
        if name == "title":
            self._set_title(html[start : match.start()])
        return self._after(html, ">", match.end(), size)

    def _set_title(self, raw: str) -> None:
        if not self.title and "svg" not in self._open:
            self.title = limit_title(decode_entities(raw))

    # events

    def _on_text(self, raw: str) -> None:
        if "head" in self._open and not raw.isspace() and self._stack[-1].name == "head":
            self._close_head()
        if self._dropping():
            return
        text = clean_chars(decode_entities(raw))
        if text:
            for sink in self._active:
                sink.text(text)

    def _void(self, name: str, attrs: dict[str, str]) -> None:
        if name not in _HEAD_OK and "head" in self._open:
            self._close_head()
        if name == "meta":
            self._read_meta(attrs)
        elif self._dropping():
            return
        elif name == "br":
            if not self._drops(name, attrs):
                for sink in self._active:
                    sink.br()
        elif name == "hr":
            for sink in self._active:
                sink.boundary()
        elif name == "img":
            alt = attrs.get("alt")
            if alt and not self._drops(name, attrs):
                text = clean_chars(decode_entities(alt))
                if text.strip():
                    for sink in self._active:
                        sink.text(" " + text + " ")

    def _read_meta(self, attrs: dict[str, str]) -> None:
        if (
            attrs.get("name", "").strip().lower() in ("robots", "googlebot")
            and "noindex" in decode_entities(attrs.get("content", "")).lower()
        ):
            self.noindex = True

    def _dropping(self) -> bool:
        return self._drop_depth > 0 or self._overflow_drop_level > 0

    def _drops(self, name: str, attrs: dict[str, str]) -> bool:
        """Would this element, with these attributes, be dropped (with everything inside it)?"""
        if name in _DROP_ALWAYS:
            return True
        if name == "header" and not (self._main_open or self._article_depth):
            return True
        if "hidden" in attrs:
            return True
        if attrs.get("aria-hidden", "").strip().lower() == "true":
            return True
        style = attrs.get("style")
        if style and _style_hides(decode_entities(style)):
            return True
        if name not in _CLASS_EXEMPT and attrs.get("role", "").strip().lower() != "main":
            for key in ("class", "id"):
                value = attrs.get(key)
                if value and _has_boilerplate_token(value):
                    return True
        return False

    def _open_element(self, name: str, attrs: dict[str, str]) -> None:
        if name == "a" and "href" in attrs:
            self._collect_link(attrs["href"], attrs.get("rel", ""))
        if "head" in self._open and name not in _HEAD_OK:
            self._close_head()
        if self._overflow:
            self._overflow += 1
            if not self._dropping() and self._drops(name, attrs):
                self._overflow_drop_level = self._overflow
            return
        self._implied_ends(name)
        if len(self._stack) >= MAX_NESTING_DEPTH:
            self._overflow = 1
            if not self._dropping() and self._drops(name, attrs):
                self._overflow_drop_level = 1
            return
        was_dropping = self._dropping()
        drops = False if was_dropping else self._drops(name, attrs)
        region = _NO_REGION
        if not was_dropping and not drops:
            if not self._main_seen and (name == "main" or attrs.get("role", "").strip().lower() == "main"):
                region = _MAIN_REGION
            elif name == "article":
                region = _ARTICLE_REGION
        self._stack.append(_Frame(name, drops, region))
        self._open[name] = self._open.get(name, 0) + 1
        if drops:
            self._drop_depth += 1
        if region == _MAIN_REGION:
            self._main_seen = True
            self._main_open = True
            self._rebuild_active()
        elif region == _ARTICLE_REGION:
            self._article_seen = True
            self._article_depth += 1
            self._rebuild_active()
        if was_dropping:
            return
        if drops:
            if name in _BLOCK:
                self._boundary()
        else:
            for sink in self._active:
                sink.start(name)

    def _pop(self) -> None:
        frame = self._stack.pop()
        name = frame.name
        remaining = self._open[name] - 1
        if remaining:
            self._open[name] = remaining
        else:
            del self._open[name]
        if frame.drops:
            self._drop_depth -= 1
        if self._dropping():
            return
        if frame.drops:
            if name in _BLOCK:
                self._boundary()
            return
        for sink in self._active:
            sink.end(name)
        if frame.region == _MAIN_REGION:
            self._main_open = False
            self._rebuild_active()
        elif frame.region == _ARTICLE_REGION:
            self._article_depth -= 1
            self._rebuild_active()

    def _close(self, name: str) -> None:
        if self._overflow:
            if self._overflow == self._overflow_drop_level:
                self._overflow_drop_level = 0
            self._overflow -= 1
            return
        if name not in self._open:
            return  # stray end tag
        stack = self._stack
        for index in range(len(stack) - 1, -1, -1):
            if stack[index].name == name:
                while len(stack) > index:
                    self._pop()
                return

    def _boundary(self) -> None:
        for sink in self._active:
            sink.boundary()

    def _rebuild_active(self) -> None:
        active = [self._body]
        if self._main_open:
            active.append(self._main)
        if self._article_depth:
            active.append(self._article)
        self._active = active

    # implied end tags

    def _implied_ends(self, name: str) -> None:
        if name in _P_CLOSERS and "p" in self._open:
            self._close_scope(_P, _P_BOUNDARY)
        if name == "li":
            self._close_scope(_LI, _LIST_BOUNDARY)
        elif name in _DEFINITION:
            self._close_scope(_DEFINITION, _DEFINITION_BOUNDARY)
        elif name in _CELLS:
            self._close_scope(_CELLS, _CELL_SCOPE)
        elif name == "tr":
            self._close_scope(_ROW, _TABLE_ONLY)
        elif name in _SECTIONS:
            self._close_scope(_SECTIONS, _TABLE_ONLY)
        elif name == "option":
            self._close_scope(_OPTION, _OPTION_BOUNDARY)
        elif name in _HEADINGS and self._stack and self._stack[-1].name in _HEADINGS:
            self._pop()

    def _close_scope(self, targets: frozenset[str], boundary: frozenset[str]) -> None:
        """Pop up to and including the nearest open element named in `targets`, looking no further than a boundary
        element or `_IMPLIED_SCAN` frames down."""
        if self._open.keys().isdisjoint(targets):
            return
        stack = self._stack
        for index in range(len(stack) - 1, max(-1, len(stack) - 1 - _IMPLIED_SCAN), -1):
            frame_name = stack[index].name
            if frame_name in targets:
                while len(stack) > index:
                    self._pop()
                return
            if frame_name in boundary:
                return

    def _close_head(self) -> None:
        while "head" in self._open:
            self._pop()

    # links

    def _collect_link(self, href: str, rel: str) -> None:
        if len(self.links) >= MAX_LINKS or "nofollow" in rel.lower().split():
            return
        raw = decode_entities(href).translate(_URL_STRIP).strip(_URL_TRIM)
        if not raw or len(raw) > _MAX_HREF:
            return
        try:
            absolute, _fragment = urldefrag(urljoin(self._base_url, raw))
            parts = urlsplit(absolute)
        except ValueError:
            return
        if parts.scheme not in ("http", "https") or not parts.netloc or absolute in self._seen_links:
            return
        self._seen_links.add(absolute)
        self.links.append(absolute)


def extract_html(html: str, *, base_url: str = "") -> ExtractedPage:
    """The readable text, title, links and robots directive of an HTML page. Never raises, never exceeds the caps."""
    bounded, cut = bound_input(html)
    extractor = _Extractor(base_url)
    extractor.feed(normalize_newlines(bounded))
    extractor.finish()
    sink = extractor.selected()
    return ExtractedPage(
        title=extractor.title or extractor.first_heading(),
        text=sink.writer.text(),
        links=tuple(extractor.links),
        noindex=extractor.noindex,
        truncated=cut or sink.writer.truncated,
    )
