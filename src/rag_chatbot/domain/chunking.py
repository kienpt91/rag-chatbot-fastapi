"""Splitting a document into chunks for retrieval.

Chunks follow the document's own structure (paragraphs, then sentences, then words), carry the heading of their
section so that a chunk makes sense on its own, overlap their neighbours so that an answer that straddles a boundary
is still found, and have *stable ids*: an id is a hash of the tenant, the document and the exact text, so
re-ingesting an unchanged page produces the same ids, an edit changes only the chunks it touches, and an unchanged
chunk is never embedded (and paid for) twice.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterator
from dataclasses import dataclass

from rag_chatbot.domain.errors import ConfigurationError
from rag_chatbot.domain.model import Chunk, SourceDocument
from rag_chatbot.domain.text import normalise_text

_HEADING = re.compile(r"^(#{1,6}) +(\S.*)$")
_SENTENCE_END = re.compile("(?<=[.!?\u3002\uff01\uff1f])\\s+")
CHARS_PER_TOKEN = 4
_MIN_OVERLAP_CHARS = 20  # a smaller carry-over is not worth repeating


@dataclass(frozen=True, slots=True)
class ChunkingConfig:
    target_tokens: int = 220  # a chunk is closed once adding the next piece would pass this
    max_tokens: int = 350  # no chunk is longer than this (a single long sentence or word is cut)
    overlap_tokens: int = 40  # the tail of a chunk that starts the next one (whole pieces only)
    min_tokens: int = 40  # a heading does not close a chunk smaller than this

    def __post_init__(self) -> None:
        if not (0 < self.min_tokens <= self.target_tokens <= self.max_tokens):
            raise ConfigurationError("chunking: need 0 < min_tokens <= target_tokens <= max_tokens")
        if not (0 <= self.overlap_tokens < self.target_tokens):
            raise ConfigurationError("chunking: need 0 <= overlap_tokens < target_tokens")

    @property
    def target_chars(self) -> int:
        return self.target_tokens * CHARS_PER_TOKEN

    @property
    def max_chars(self) -> int:
        return self.max_tokens * CHARS_PER_TOKEN

    @property
    def overlap_chars(self) -> int:
        return self.overlap_tokens * CHARS_PER_TOKEN

    @property
    def min_chars(self) -> int:
        return self.min_tokens * CHARS_PER_TOKEN


def chunk_id_for(tenant_id: str, document_id: str, text: str, occurrence: int) -> str:
    """`occurrence` counts earlier chunks of the same document with exactly the same text (usually 0)."""
    digest = hashlib.sha256(f"{tenant_id}\x00{document_id}\x00{occurrence}\x00{text}".encode()).hexdigest()
    return f"ck_{digest[:24]}"


def _split_long(piece: str, limit: int) -> Iterator[str]:
    """Cut a piece that is longer than `limit` at sentence ends, then spaces, then anywhere."""
    if len(piece) <= limit:
        yield piece
        return
    current = ""
    for sentence in _SENTENCE_END.split(piece):
        if len(sentence) > limit:
            if current:
                yield current
                current = ""
            rest = sentence
            while len(rest) > limit:
                cut = rest.rfind(" ", 0, limit)
                if cut < limit // 2:
                    cut = limit
                yield rest[:cut].rstrip()
                rest = rest[cut:].lstrip()
            current = rest
        elif not current:
            current = sentence
        elif len(current) + 1 + len(sentence) <= limit:
            current = f"{current} {sentence}"
        else:
            yield current
            current = sentence
    if current:
        yield current


@dataclass(slots=True)
class _Piece:
    text: str
    heading: str  # the heading line in force where this piece starts ("" before the first heading)
    is_heading: bool = False


def _pieces(text: str, config: ChunkingConfig) -> list[_Piece]:
    pieces: list[_Piece] = []
    heading = ""
    # a heading line and the paragraph under it stay together as separate pieces; the packer decides
    for raw in normalise_text(text).split("\n\n"):
        paragraph = raw.strip()
        if not paragraph:
            continue
        match = _HEADING.match(paragraph) if "\n" not in paragraph else None
        if match is not None:
            heading = paragraph if len(paragraph) <= config.max_chars // 2 else paragraph[: config.max_chars // 2]
            pieces.append(_Piece(heading, heading, is_heading=True))
            continue
        # leave room for the heading line that opens a chunk
        limit = max(config.max_chars - len(heading) - 1, config.max_chars // 2)
        pieces.extend(_Piece(part, heading) for part in _split_long(paragraph, limit))
    return pieces


def _tail_of(text: str, room: int) -> str:
    """The end of `text` that fits in `room` characters: whole sentences if one fits, else the last words."""
    if room < _MIN_OVERLAP_CHARS:
        return ""
    taken: list[str] = []
    total = 0
    for sentence in reversed(_SENTENCE_END.split(text)):
        if total + len(sentence) > room:
            break
        taken.insert(0, sentence)
        total += len(sentence) + 1
    if taken:
        return " ".join(taken)
    words = text[-room:].split(" ", 1)
    tail = words[1] if len(words) == 2 else ""
    return tail if len(tail) >= _MIN_OVERLAP_CHARS else ""


def _rendered_len(parts: list[_Piece]) -> int:
    """Length of the chunk text `_render` would make (the heading prefix included)."""
    return len(_render(parts))


def _render(parts: list[_Piece]) -> str:
    first = parts[0]
    body = "\n\n".join(p.text for p in parts)
    # every chunk starts with the heading of its section
    if first.heading and not first.is_heading:
        return f"{first.heading}\n\n{body}"
    return body


def split_text(text: str, config: ChunkingConfig | None = None) -> list[str]:
    """The chunk texts of a page's text, in order. Pure; ids are added by `chunk_document`."""
    config = config or ChunkingConfig()
    chunks: list[str] = []
    current: list[_Piece] = []
    fresh = 0  # pieces in `current` that are not overlap carried over from the previous chunk

    def overlap_tail() -> list[_Piece]:
        tail: list[_Piece] = []
        total = 0
        for piece in reversed(current):
            if piece.is_heading:
                break
            if total + len(piece.text) <= config.overlap_chars:
                tail.insert(0, piece)
                total += len(piece.text) + 2
                continue
            part = _tail_of(piece.text, config.overlap_chars - total)  # a long piece: carry its last sentences
            if part:
                tail.insert(0, _Piece(part, piece.heading))
            break
        return tail

    def flush() -> None:
        if all(p.is_heading for p in current) and chunks:
            # a heading with nothing under it joins the previous chunk instead of becoming a chunk of its own
            chunks[-1] = f"{chunks[-1]}\n\n{_render(current)}"
        else:
            chunks.append(_render(current))

    for piece in _pieces(text, config):
        if piece.is_heading:
            if fresh and _rendered_len(current) >= config.min_chars:
                flush()
                current, fresh = [], 0  # a new section starts clean: no overlap across a heading
        elif fresh and _rendered_len([*current, piece]) > config.target_chars:
            tail = overlap_tail()
            flush()
            current, fresh = tail, 0
            if current and _rendered_len([*current, piece]) > config.max_chars:
                current = []  # the overlap would push this chunk over the hard limit: drop it
        current.append(piece)
        fresh += 1
    if fresh:
        flush()
    return [c for c in chunks if c.strip()]


def chunk_document(document: SourceDocument, config: ChunkingConfig | None = None) -> list[Chunk]:
    seen: dict[str, int] = {}
    chunks: list[Chunk] = []
    for ordinal, text in enumerate(split_text(document.text, config)):
        occurrence = seen.get(text, 0)
        seen[text] = occurrence + 1
        chunks.append(
            Chunk(
                chunk_id=chunk_id_for(document.tenant_id, document.document_id, text, occurrence),
                tenant_id=document.tenant_id,
                document_id=document.document_id,
                ordinal=ordinal,
                title=document.title,
                url=document.url,
                text=text,
            )
        )
    return chunks
