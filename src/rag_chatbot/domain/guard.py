"""Checking what the model said before a visitor sees it.

A model that was fooled by text on a page, or by the visitor, produces an answer that looks fine. These checks look
for the ways such an answer shows itself, and they fail closed: an answer that fails any of them is replaced by a
refusal and never shown, stored or cited.

* the answer must carry at least one citation, and every cited id must be one of the sources this request was given
  (a model cannot cite a document it was not shown, nor invent one). A sentence without a citation next to a cited one
  is not detected;
* the answer must not contain the system prompt, the internal marker or a long run of its words;
* the answer may link only to the site's own hosts (the tenant's origins and its declared link hosts, plus the hosts
  of the pages it cites), even when a page of the site lists some other address: a link or an image whose address
  carries the conversation is an exfiltration channel, and a poisoned review that advertises a phishing address must
  not be passed on. `javascript:` and `data:` addresses never pass;
* the answer is plain text: markup tags are removed (a page may quote `<script>`; an integration that forgets to
  escape the answer must not be handed it), citation markers are removed, control characters stripped, length bounded.

What it cannot see: a wrong or misleading statement that is correctly cited, and a leak in a paraphrase or another
language. Those are the evaluation set's job (see eval/).
"""

from __future__ import annotations

import re
from collections.abc import Collection, Mapping
from dataclasses import dataclass
from urllib.parse import urlsplit

from rag_chatbot.domain.prompt import NO_ANSWER
from rag_chatbot.domain.text import fold, strip_invisible, words

MAX_ANSWER_CHARS = 4000
SHINGLE_WORDS = 6

_CITE_GROUP = re.compile(r"\[\s*(s\d{1,3}(?:\s*[,;]\s*s\d{1,3})*)\s*\]", re.IGNORECASE)
_CITE_ID = re.compile(r"s\d{1,3}", re.IGNORECASE)
_URL = re.compile(r"(?:https?|ftp)://[^\s<>\"'`)\]]{1,2000}", re.IGNORECASE)
_BAD_SCHEME = re.compile(r"\b(?:javascript|data|vbscript|file|mailto|tel|sms)\s*:", re.IGNORECASE)
_PROTOCOL_RELATIVE = re.compile(r"(?<![:\w/.])//[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}[^\s<>\"'`)\]]{0,2000}")
_MARKDOWN_IMAGE = re.compile(r"!\[[^\]\n]{0,200}\]\(")
_MARKDOWN_TARGET = re.compile(r"\]\(\s*<?([^)\s>]{1,2000})")
_TAG = re.compile(r"<!--.{0,2000}?-->|</?[A-Za-z!?][^<>\n]{0,300}>", re.DOTALL)
_NO_ANSWER = re.compile(rf"^\W*{NO_ANSWER}\W*$", re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class GuardResult:
    ok: bool
    text: str = ""  # the cleaned answer, only when ok
    cited_aliases: tuple[str, ...] = ()  # in order of first use, only when ok
    reason: str | None = None  # no_answer, empty, no_citation, invalid_citation, system_prompt_leak, unverified_link


def _shingles(text: str, size: int) -> set[tuple[str, ...]]:
    tokens = words(text)
    return {tuple(tokens[i : i + size]) for i in range(len(tokens) - size + 1)}


def leaks_system_prompt(answer: str, system_prompt: str, canary: str) -> bool:
    if canary and canary.casefold() in fold(answer):
        return True
    return bool(_shingles(answer, SHINGLE_WORDS) & _shingles(system_prompt, SHINGLE_WORDS))


def strip_markup(text: str) -> str:
    """Remove HTML tags and comments, repeatedly, so that removing one cannot assemble another (`<scr<b>ipt>`)."""
    for _ in range(5):
        stripped = _TAG.sub("", text)
        if stripped == text:
            break
        text = stripped
    return text


def scrub(text: str, allowed_hosts: Collection[str]) -> str:
    """Text taken from a page (a citation's snippet) made safe to hand on: markup removed, and every address on a host
    the site does not own replaced by `[link removed]`. The same rule as for the answer, so that a poisoned page cannot
    get its address through the side door of a quotation."""
    allowed = {h.rstrip(".").casefold() for h in allowed_hosts}

    def replace(match: re.Match[str]) -> str:
        core = _strip_url(match.group(0))
        tail = match.group(0)[len(core) :]
        return (core if host_of(core) in allowed else "[link removed]") + tail

    return _URL.sub(replace, strip_markup(text))


def _strip_url(url: str) -> str:
    return url.rstrip(".,;:!?")


def urls_in(text: str) -> set[str]:
    return {_strip_url(match.group(0)).casefold() for match in _URL.finditer(text)}


def host_of(url: str) -> str | None:
    """The lower-case host of an address, without a trailing dot; None when it has none or cannot be read."""
    try:
        host = urlsplit(url).hostname
    except ValueError:
        return None
    return host.rstrip(".").casefold() if host else None


def check_answer(
    raw: str,
    *,
    aliases: Mapping[str, str],
    system_prompt: str,
    canary: str,
    allowed_hosts: Collection[str],
) -> GuardResult:
    """`aliases` maps `s1` to the chunk id it stands for; `allowed_hosts` are the hosts the answer may link to."""
    text = strip_invisible(raw).strip()
    if not text:
        return GuardResult(False, reason="empty")
    if _NO_ANSWER.match(text):
        return GuardResult(False, reason="no_answer")
    text = strip_markup(text).strip()
    if len(text) > MAX_ANSWER_CHARS:
        text = text[:MAX_ANSWER_CHARS].rstrip()
    if leaks_system_prompt(text, system_prompt, canary):
        return GuardResult(False, reason="system_prompt_leak")
    allowed = {h.rstrip(".").casefold() for h in allowed_hosts}
    if (
        _BAD_SCHEME.search(text)
        or _MARKDOWN_IMAGE.search(text)  # an image has no business in a plain-text answer, and its address leaks
        or any(host_of(url) not in allowed for url in urls_in(text))
        or any(host_of(m.group(0)) not in allowed for m in _PROTOCOL_RELATIVE.finditer(text))
        or any(host_of(m.group(1)) not in allowed for m in _MARKDOWN_TARGET.finditer(text))
    ):
        return GuardResult(False, reason="unverified_link")
    cited: dict[str, None] = {}
    for group in _CITE_GROUP.finditer(text):
        for alias in _CITE_ID.findall(group.group(1)):
            key = alias.lower()
            if key not in aliases:
                return GuardResult(False, reason="invalid_citation")
            cited[key] = None
    # a bracket that looks like a citation but is malformed (`[s]`, `[s1 s2]`) is not accepted as a citation, and is
    # removed below like the others, so it cannot count as support for anything
    if not cited:
        return GuardResult(False, reason="no_citation")
    cleaned = _CITE_GROUP.sub("", text)
    cleaned = re.sub(r"[ \t]+([.,;:!?])", r"\1", cleaned)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned).strip()
    if not cleaned:
        return GuardResult(False, reason="empty")
    return GuardResult(True, cleaned, tuple(cited))
