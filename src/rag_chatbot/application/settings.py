"""The tunable behaviour of the chat and ingestion use cases. Plain data; composition fills it from the environment."""

from __future__ import annotations

from dataclasses import dataclass, field

from rag_chatbot.domain.chunking import ChunkingConfig
from rag_chatbot.domain.errors import ConfigurationError
from rag_chatbot.domain.redaction import ALL_KINDS

DEFAULT_REFUSAL = (
    "I don't know the answer to that from this website's content. "
    "Please contact us directly or try rephrasing your question."
)
BLOCKED_MESSAGE = "I can only answer questions about this website's content."


@dataclass(frozen=True, slots=True)
class ChatSettings:
    confidence_threshold: float = 0.30  # below this the best chunk is not good enough: refuse, do not call the model
    candidates_per_search: int = 12
    context_chunks: int = 5
    context_chars: int = 6000
    context_relative_cutoff: float = 0.5  # chunks less than half as similar as the best one are not given to the model
    history_turns: int = 6
    history_token_budget: int = 800
    max_output_tokens: int = 400
    max_sentences: int = 4
    max_messages_per_session: int = 100
    redaction_kinds: frozenset[str] = ALL_KINDS  # what is removed from messages before the model and the database
    blocked_phrases: tuple[str, ...] = ()
    refusal_message: str = DEFAULT_REFUSAL
    default_daily_budget_micros: int | None = None  # per tenant, per UTC day; None means no cap
    canary: str = "CANARY-unset"  # secret marker inside the system prompt; a leak of it is detected
    extra: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise ConfigurationError("confidence_threshold must be between 0 and 1")
        if min(self.candidates_per_search, self.context_chunks, self.history_turns, self.max_output_tokens) < 1:
            raise ConfigurationError("limits must be at least 1")
        if not 0.0 <= self.context_relative_cutoff <= 1.0:
            raise ConfigurationError("context_relative_cutoff must be between 0 and 1")
        if len(self.canary) < 12:
            raise ConfigurationError("the canary must be at least 12 characters")


@dataclass(frozen=True, slots=True)
class IngestSettings:
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    embed_batch_size: int = 32
    max_chunks_per_document: int = 300

    def __post_init__(self) -> None:
        if self.embed_batch_size < 1 or self.max_chunks_per_document < 1:
            raise ConfigurationError("embed_batch_size and max_chunks_per_document must be at least 1")
