"""The vocabulary of the service: documents, chunks, retrieval results, conversations and answers.

Everything here is immutable data. Text that comes from a website or from a visitor is *untrusted*; the types do not
make it trusted, the functions that build prompts and check answers (`prompt.py`, `guard.py`) are where that is
handled.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

Role = Literal["user", "assistant"]
PageKind = Literal["html", "markdown", "text"]
RefusalReason = Literal[
    "out_of_scope",
    "low_confidence",
    "no_answer",
    "unsafe_answer",
]


@dataclass(frozen=True, slots=True)
class RawPage:
    """One page or file as a content source delivers it, before any text extraction."""

    location: str  # identity of the page inside its source: a relative path or an absolute URL
    kind: PageKind
    content: str  # decoded text of the file
    url: str = ""  # where a visitor can open it ("" when unknown)


@dataclass(frozen=True, slots=True)
class SourceDocument:
    """The plain text of one page, ready to be chunked."""

    tenant_id: str
    document_id: str  # stable: derived from the page's location
    url: str
    title: str
    text: str


@dataclass(frozen=True, slots=True)
class Chunk:
    chunk_id: str  # stable: tenant, document, position and text decide it
    tenant_id: str
    document_id: str
    ordinal: int
    title: str
    url: str
    text: str

    @property
    def token_estimate(self) -> int:
        return max(1, (len(self.text) + 3) // 4)


@dataclass(frozen=True, slots=True)
class EmbeddedChunk:
    chunk: Chunk
    embedding: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class Usage:
    """Tokens a provider reported for one call. `estimated` is True when it reported none and we guessed."""

    input_tokens: int = 0
    output_tokens: int = 0
    estimated: bool = False

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            self.input_tokens + other.input_tokens,
            self.output_tokens + other.output_tokens,
            self.estimated or other.estimated,
        )


@dataclass(frozen=True, slots=True)
class EmbeddingResult:
    vectors: tuple[tuple[float, ...], ...]
    usage: Usage
    model: str


@dataclass(frozen=True, slots=True)
class ChatCompletion:
    text: str
    usage: Usage
    model: str


@dataclass(frozen=True, slots=True)
class ChatTurn:
    role: Role
    content: str


@dataclass(frozen=True, slots=True)
class RetrievedChunk:
    chunk: Chunk
    similarity: float  # cosine similarity of the query and the chunk, 0..1 (negative values are clamped to 0)
    vector_rank: int | None  # 1-based rank in the vector search, None when it was not among its hits
    keyword_rank: int | None
    fused_score: float


@dataclass(frozen=True, slots=True)
class Retrieval:
    chunks: tuple[RetrievedChunk, ...]
    confidence: float  # 0..1, see retrieval.confidence_of

    @property
    def chunk_ids(self) -> frozenset[str]:
        return frozenset(c.chunk.chunk_id for c in self.chunks)


@dataclass(frozen=True, slots=True)
class Citation:
    chunk_id: str
    document_id: str
    title: str
    url: str
    snippet: str


@dataclass(frozen=True, slots=True)
class Cost:
    micros: int = 0  # millionths of the price table's currency
    estimated: bool = False


@dataclass(frozen=True, slots=True)
class Answer:
    text: str
    grounded: bool
    citations: tuple[Citation, ...] = ()
    refusal_reason: RefusalReason | None = None
    confidence: float = 0.0


@dataclass(frozen=True, slots=True)
class ChatReply:
    """What the chat use case returns to the transport."""

    session_id: str
    answer: Answer
    usage: Usage = field(default_factory=Usage)
    cost: Cost = field(default_factory=Cost)
    latency_ms: int = 0
