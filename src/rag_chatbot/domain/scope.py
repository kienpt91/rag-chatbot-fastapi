"""A cheap check of the visitor's message before any money is spent.

It refuses messages that are plainly attempts to rewrite the assistant's rules or to make it recite them. This is a
*filter, not a defence*: a determined visitor rephrases, in another language or in a code. The defences that count
are structural (the model gets no tools, retrieved text is delimited data, the answer is checked, see `prompt.py` and
`guard.py`); this filter saves the cost of the obvious attempts and keeps them out of the model's context.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from rag_chatbot.domain.text import fold

MAX_MESSAGE_CHARS = 2000

_VERBS_OVERRIDE = r"(?:ignore|disregard|forget|override|bypass|drop|discard)"
_TARGETS = (
    r"(?:instructions?|rules?|prompts?|guidelines?|policy|policies|directives?|constraints?|restrictions?|programming)"
)
_VERBS_REVEAL = r"(?:reveal|show|print|repeat|display|tell|output|leak|dump|share|recite|reproduce|copy|translate)"
_KINDS = r"(?:system|initial|hidden|secret|original|developer|internal|first|above|previous)"
_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("instruction_override", re.compile(rf"\b{_VERBS_OVERRIDE}\b(?:\W+\w+){{0,5}}?\W+{_TARGETS}\b")),
    (
        "prompt_extraction",
        re.compile(
            rf"\b{_VERBS_REVEAL}\b(?:\W+\w+){{0,6}}?\W+{_KINDS}\W+(?:prompt|message|instructions?|rules?|text)\b"
        ),
    ),
    ("prompt_extraction", re.compile(r"\bsystem\s+prompt\b")),
    ("prompt_extraction", re.compile(r"\bwhat\s+(?:are|were)\s+your\s+(?:instructions|rules|guidelines)\b")),
    ("role_hijack", re.compile(r"\byou\s+are\s+now\b")),
    ("role_hijack", re.compile(r"\b(?:developer|god|debug|jailbreak|dan)\s+mode\b")),
    (
        "role_hijack",
        re.compile(
            r"\b(?:pretend|act|behave)\s+(?:to\s+be|as|like)\b(?:\W+\w+){0,3}?\W+(?:dan|unrestricted|jailbroken|unfiltered|evil)\b"
        ),
    ),
    ("role_hijack", re.compile(r"\bdo\s+anything\s+now\b")),
)


def check_message(message: str, extra_phrases: Sequence[str] = ()) -> str | None:
    """None when the message may be answered, else a short reason code (`empty`, `too_long`, a pattern name or
    `blocked_phrase`)."""
    if not message.strip():
        return "empty"
    if len(message) > MAX_MESSAGE_CHARS:
        return "too_long"
    folded = fold(message)
    for reason, pattern in _PATTERNS:
        if pattern.search(folded):
            return reason
    for phrase in extra_phrases:
        needle = fold(phrase)
        if needle and needle in folded:
            return "blocked_phrase"
    return None
