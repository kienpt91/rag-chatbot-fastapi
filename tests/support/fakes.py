"""Fakes for the ports that have no real counterpart in the tests: the clock, the ids and scripted providers."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

from rag_chatbot.domain.errors import ProviderError
from rag_chatbot.domain.model import ChatCompletion, ChatTurn, EmbeddingResult, Usage

START = datetime(2026, 11, 5, 12, 0, 0, tzinfo=UTC)


class FakeClock:
    def __init__(self, start: datetime = START) -> None:
        self._now = start
        self._mono = 1000.0

    def now(self) -> datetime:
        return self._now

    def monotonic(self) -> float:
        return self._mono

    def advance(self, seconds: float) -> None:
        self._now += timedelta(seconds=seconds)
        self._mono += seconds


class SequentialIds:
    def __init__(self) -> None:
        self.sessions = 0
        self.requests = 0
        self.nonces = 0

    def new_session_id(self) -> str:
        self.sessions += 1
        return f"s_test{self.sessions:08d}"

    def new_request_id(self) -> str:
        self.requests += 1
        return f"r_{self.requests:08d}"

    def new_nonce(self) -> str:
        self.nonces += 1
        return f"{self.nonces:024x}"


@dataclass
class ScriptedChat:
    """Answers with whatever `reply` returns (a string, or a function of the system prompt and the messages) and
    remembers every call, so a test can look at exactly what the model was given."""

    reply: str | Callable[[str, Sequence[ChatTurn]], str] = "NO_ANSWER"
    model: str = "scripted-chat"
    usage: Usage = field(default_factory=lambda: Usage(100, 20))
    calls: list[tuple[str, tuple[ChatTurn, ...]]] = field(default_factory=list)
    fail: bool = False

    async def complete(self, system: str, messages: Sequence[ChatTurn], *, max_output_tokens: int) -> ChatCompletion:
        self.calls.append((system, tuple(messages)))
        if self.fail:
            raise ProviderError("scripted failure")
        text = self.reply(system, messages) if callable(self.reply) else self.reply
        return ChatCompletion(text, self.usage, self.model)


class RecordingEmbedder:
    """Wraps an embedder, remembers what it was asked to embed, and can be made to fail."""

    def __init__(self, inner: object, *, fail: bool = False, report_tokens: bool = True) -> None:
        self._inner = inner
        self.model: str = getattr(inner, "model", "recording-embedding")
        self.dimensions: int = getattr(inner, "dimensions", 0)
        self.texts: list[str] = []
        self.fail = fail
        self.report_tokens = report_tokens

    async def embed(self, texts: Sequence[str]) -> EmbeddingResult:
        self.texts.extend(texts)
        if self.fail:
            raise ProviderError("scripted failure")
        embed = self._inner.embed  # type: ignore[attr-defined]
        result: EmbeddingResult = await embed(texts)
        return result if self.report_tokens else EmbeddingResult(result.vectors, Usage(), result.model)
