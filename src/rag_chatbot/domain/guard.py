"""Checking what the model said before a visitor sees it.

A model that was fooled by text on a page, or by the visitor, produces an answer that looks fine. These checks look
for the ways such an answer shows itself, and they fail closed: an answer that fails any of them is replaced by a
refusal and never shown, stored or cited.

* every statement must carry a citation, and every cited id must be one of the sources this request was given
  (a model cannot cite a document it was not shown, nor invent one);
* the answer must not contain the system prompt, the internal marker or a long run of its words;
* the answer must not contain a web address that is not in the sources the request was given (an exfiltration
  channel: a link or an image whose address carries the conversation) or a `javascript:`/`data:` address;
* the answer is plain text: markers are removed, control characters stripped, length bounded.

What it cannot see: a wrong or misleading statement that is correctly cited, and a leak in a paraphrase or another
language. Those are the evaluation set's job (see eval/).
"""

from __future__ import annotations

import re
from collections.abc import Collection, Mapping
from dataclasses import dataclass

from rag_chatbot.domain.prompt import NO_ANSWER
from rag_chatbot.domain.text import fold, strip_invisible, words

MAX_ANSWER_CHARS = 4000
SHINGLE_WORDS = 6

_CITE_GROUP = re.compile(r"\[\s*(s\d{1,3}(?:\s*[,;]\s*s\d{1,3})*)\s*\]", re.IGNORECASE)
_CITE_ID = re.compile(r"s\d{1,3}", re.IGNORECASE)
_URL = re.compile(r"(?:https?|ftp)://[^\s<>\"'`)\]]{1,2000}", re.IGNORECASE)
_BAD_SCHEME = re.compile(r"\b(?:javascript|data|vbscript|file)\s*:", re.IGNORECASE)
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


def _strip_url(url: str) -> str:
    return url.rstrip(".,;:!?")


def urls_in(text: str) -> set[str]:
    return {_strip_url(match.group(0)).casefold() for match in _URL.finditer(text)}


def check_answer(
    raw: str,
    *,
    aliases: Mapping[str, str],
    system_prompt: str,
    canary: str,
    allowed_urls: Collection[str],
) -> GuardResult:
    """`aliases` maps `s1` to the chunk id it stands for; `allowed_urls` are the addresses that appear in the
    sources (and their pages' own addresses)."""
    text = strip_invisible(raw).strip()
    if not text:
        return GuardResult(False, reason="empty")
    if _NO_ANSWER.match(text):
        return GuardResult(False, reason="no_answer")
    if len(text) > MAX_ANSWER_CHARS:
        text = text[:MAX_ANSWER_CHARS].rstrip()
    if leaks_system_prompt(text, system_prompt, canary):
        return GuardResult(False, reason="system_prompt_leak")
    allowed = {u.casefold() for u in allowed_urls}
    if _BAD_SCHEME.search(text) or not urls_in(text) <= allowed:
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
