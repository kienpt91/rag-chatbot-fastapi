"""Removing personal data and credentials from text that is stored or logged.

What it finds, and what it does not (see SECURITY.md): e-mail addresses, phone numbers (7 to 15 digits as a visitor
writes them), payment-card numbers (13 to 19 digits that pass the Luhn check) and the credential shapes of this
service plus common bearer tokens. It does not find names, street addresses or free-text secrets. The patterns are
bounded (no nested quantifiers) so hostile input cannot make them slow.
"""

from __future__ import annotations

import re
from collections.abc import Collection
from dataclasses import dataclass, field

ALL_KINDS = frozenset({"emails", "phones", "cards", "secrets"})

_EMAIL = re.compile(r"[A-Za-z0-9._%+\-]{1,64}@[A-Za-z0-9\-]{1,63}(?:\.[A-Za-z0-9\-]{1,63}){1,8}")
# digits with single separators between them, 7 to 19 digits in all
_NUMBER = re.compile(r"(?<![\w.])\+?\(?\d(?:[ .()\-]{0,2}\d){6,18}(?![\w])")
_SECRETS = (
    re.compile(r"\bragk_[A-Za-z0-9_\-]{16,128}"),
    re.compile(r"\bragt1\.[A-Za-z0-9_\-]{8,1024}\.[A-Za-z0-9_\-]{8,128}"),
    re.compile(r"\b[Bb]earer[ \t]+[A-Za-z0-9._~+/=\-]{16,2048}"),
    re.compile(r"\beyJ[A-Za-z0-9_\-]{10,2048}\.eyJ[A-Za-z0-9_\-]{10,2048}\.[A-Za-z0-9_\-]{10,512}"),
    re.compile(r"\b(?:sk|pk|rk)[_\-][A-Za-z0-9_\-]{24,256}"),
    re.compile(r"-----BEGIN [A-Z ]{0,30}PRIVATE KEY-----"),
    re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"),
)
MAX_SCAN_CHARS = 100_000  # beyond this, text is cut: a log line or message is never this long legitimately


@dataclass(frozen=True, slots=True)
class Redacted:
    text: str
    counts: dict[str, int] = field(default_factory=dict)

    @property
    def changed(self) -> bool:
        return bool(self.counts)


def luhn_valid(digits: str) -> bool:
    total = 0
    for index, ch in enumerate(reversed(digits)):
        value = ord(ch) - 48
        if index % 2 == 1:
            value *= 2
            if value > 9:
                value -= 9
        total += value
    return total % 10 == 0


def redact(text: str, kinds: Collection[str] = ALL_KINDS) -> Redacted:
    """Replace every finding by `[email]`, `[phone]`, `[card]` or `[secret]`. Text beyond MAX_SCAN_CHARS is dropped
    (replaced by an ellipsis marker) rather than left unscanned."""
    counts: dict[str, int] = {}

    def bump(kind: str) -> None:
        counts[kind] = counts.get(kind, 0) + 1

    if len(text) > MAX_SCAN_CHARS:
        text = text[:MAX_SCAN_CHARS] + " [truncated]"
    if "secrets" in kinds:

        def secret(_: re.Match[str]) -> str:
            bump("secrets")
            return "[secret]"

        for pattern in _SECRETS:
            text = pattern.sub(secret, text)
    if "emails" in kinds:

        def email(_: re.Match[str]) -> str:
            bump("emails")
            return "[email]"

        text = _EMAIL.sub(email, text)
    if "cards" in kinds or "phones" in kinds:

        def number(match: re.Match[str]) -> str:
            raw = match.group(0)
            digits = "".join(ch for ch in raw if ch.isdigit())
            if "cards" in kinds and 13 <= len(digits) <= 19 and luhn_valid(digits):
                bump("cards")
                return "[card]"
            if "phones" in kinds and 7 <= len(digits) <= 15 and _looks_like_phone(raw):
                bump("phones")
                return "[phone]"
            return raw

        text = _NUMBER.sub(number, text)
    return Redacted(text, counts)


def _looks_like_phone(raw: str) -> bool:
    # an ISO date (2026-11-05) is not a phone number
    return re.fullmatch(r"\d{4}[-.]\d{2}[-.]\d{2}", raw.strip()) is None
