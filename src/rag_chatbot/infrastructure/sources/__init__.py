"""Content sources: a folder reader and a polite, allow-listed site crawler."""

from __future__ import annotations

from dataclasses import dataclass, field

MAX_REPORTED_ERRORS = 50
_MAX_ERROR_CHARS = 200


@dataclass
class SourceReport:
    """What a source did: pages delivered, what it skipped and why, and the failures it survived."""

    pages: int = 0
    skipped: dict[str, int] = field(default_factory=dict)  # reason -> count
    errors: list[str] = field(default_factory=list)  # short messages without secrets, at most 50

    def skip(self, reason: str, count: int = 1) -> None:
        self.skipped[reason] = self.skipped.get(reason, 0) + count

    def add_error(self, message: str) -> None:
        if len(self.errors) < MAX_REPORTED_ERRORS:
            self.errors.append(message[:_MAX_ERROR_CHARS])
