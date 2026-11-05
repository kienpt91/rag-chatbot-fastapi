"""Removing personal data and credentials from text that is stored or logged.

What it finds, and what it does not (see SECURITY.md): e-mail addresses, phone numbers (7 to 15 digits as a visitor
writes them), payment-card numbers (13 to 19 digits that pass the Luhn check) and the credential shapes of this
service plus common bearer tokens. It does not find names, street addresses or free-text secrets. The patterns are
bounded (no nested quantifiers) so hostile input cannot make them slow.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Collection
from dataclasses import dataclass, field

ALL_KINDS = frozenset({"emails", "phones", "cards", "secrets"})

_EMAIL = re.compile(r"[A-Za-z0-9._%+\-]{1,64}@[A-Za-z0-9\-]{1,63}(?:\.[A-Za-z0-9\-]{1,63}){1,8}")
# a run of digits with separators between them (a number, or several written one after the other), 7 to 60 digits
_NUMBER = re.compile(r"(?<![\w.])\+?\(?\d(?:[ .()\-]{0,2}\d){6,59}(?![\w])")
_GROUP = re.compile(r"\d+")
_CARD_LENGTHS = (16, 15, 14, 13, 17, 18, 19)  # the usual ones first: a long run may contain a valid-by-chance window
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
        text = _NUMBER.sub(lambda match: _redact_run(match.group(0), kinds, bump), text)
    return Redacted(text, counts)


def _redact_run(raw: str, kinds: Collection[str], bump: Callable[[str], None]) -> str:
    """A run of digit groups: card numbers are found first (a window of groups of 13 to 19 digits that passes the Luhn
    check), then what is left on either side is a phone number when it has 7 or more digits. Anything long that is
    neither is redacted as a phone number too: a digit run that cannot be classified is still not something to store.
    Numbers written one after the other (`555 0177 4111 1111 1111 1111`) are therefore each found."""
    groups = [(m.start(), m.end()) for m in _GROUP.finditer(raw)]
    if re.fullmatch(r"\d{4}[-.]\d{2}[-.]\d{2}", raw.strip()):
        return raw  # an ISO date (2026-11-05) is not a phone number
    sizes = [end - start for start, end in groups]
    pieces: list[tuple[int, int, str]] = []  # (start, end, replacement) in raw, in order

    def digits(first: int, last: int) -> str:
        return "".join(raw[groups[i][0] : groups[i][1]] for i in range(first, last + 1))

    def classify(first: int, last: int) -> None:
        if first > last:
            return
        if "cards" in kinds:
            for length in _CARD_LENGTHS:
                for a in range(first, last + 1):
                    total = 0
                    for b in range(a, last + 1):
                        total += sizes[b]
                        if total > length:
                            break
                        if total == length and luhn_valid(digits(a, b)):
                            classify(first, a - 1)
                            pieces.append((groups[a][0], groups[b][1], "[card]"))
                            bump("cards")
                            classify(b + 1, last)
                            return
        if "phones" in kinds and sum(sizes[first : last + 1]) >= 7:
            pieces.append((groups[first][0], groups[last][1], "[phone]"))
            bump("phones")

    classify(0, len(groups) - 1)
    if not pieces:
        return raw
    pieces.sort()
    out: list[str] = []
    cursor = 0
    for index, (begin, end, replacement) in enumerate(pieces):
        start = 0 if index == 0 and begin == groups[0][0] else begin  # a leading + or ( belongs to the first number
        out.append(raw[cursor:start])
        out.append(replacement)
        cursor = end
    out.append(raw[cursor:])
    return "".join(out)
