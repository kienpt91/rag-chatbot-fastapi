"""A deterministic stand-in for the embedding and chat providers.

**No hosted model is behind this.** It exists so that the test suite, the evaluation run in CI and the demo need no
account, no network and no money, and give the same answer every time.

* `FakeEmbeddingProvider` hashes words (without the function words, plus word prefixes and pairs) into a fixed number
  of dimensions, so texts that share words have a high cosine similarity and unrelated texts have a similarity near zero. It is a bag of words,
  not a meaning: it cannot tell "open" from "closed" or match a paraphrase. Real embeddings do better; thresholds
  tuned on this fake do not carry over (docs/adapting.md explains how to tune them on a real provider).
* `FakeChatProvider` answers by *extraction*: it finds the sentence of the given sources that shares the most words
  with the question and quotes it with its source id; with no overlap it says `NO_ANSWER`. With `gullible=True` it
  also behaves like a model that was fooled: it obeys a `REPLY:` directive and a request to print its instructions
  when either appears in the sources or the question. That is how the evaluation set exercises the application's
  defences (`domain/guard.py`) without a real model that sometimes falls for it and sometimes does not.
"""

from __future__ import annotations

import hashlib
import itertools
import math
import re
from collections.abc import Sequence

from rag_chatbot.domain.model import ChatCompletion, ChatTurn, EmbeddingResult, Usage
from rag_chatbot.domain.prompt import NO_ANSWER
from rag_chatbot.domain.text import STOPWORDS_EN, estimate_tokens, search_terms, words

_SOURCE = re.compile(r'<<<SOURCE (s\d+) ([0-9a-f]+) title="[^"\n]*">>>\n(.*?)\n<<<END \1 \2>>>', re.DOTALL)
_QUESTION = re.compile(r"<<<QUESTION ([0-9a-f]+)>>>\n(.*?)\n<<<END QUESTION \1>>>", re.DOTALL)
_SENTENCES = re.compile(r"(?<=[.!?])\s+|\n+")
_REVEAL = re.compile(
    r"\b(?:reveal|print|repeat|show|output)\b[^.\n]{0,40}\b(?:system prompt|instructions|rules)\b", re.IGNORECASE
)
_REPLY = re.compile(r"\bREPLY:\s*(.+)", re.IGNORECASE)


def _stem(word: str) -> str:
    return word[:5]


class FakeEmbeddingProvider:
    def __init__(self, dimensions: int = 1536, model: str = "fake-embedding-1") -> None:
        self.dimensions = dimensions
        self.model = model

    def _vector(self, text: str) -> tuple[float, ...]:
        tokens = [
            w for w in words(text) if w not in STOPWORDS_EN
        ]  # like a real embedder, mostly deaf to function words
        features: dict[str, float] = {}
        for token in tokens:
            features[f"w:{token}"] = features.get(f"w:{token}", 0.0) + 1.0
            if len(token) > 4:
                features[f"p:{token[:4]}"] = features.get(f"p:{token[:4]}", 0.0) + 0.5
        for left, right in itertools.pairwise(tokens):
            features[f"b:{left} {right}"] = features.get(f"b:{left} {right}", 0.0) + 0.5
        vector = [0.0] * self.dimensions
        for feature, weight in features.items():
            digest = int.from_bytes(hashlib.blake2b(feature.encode(), digest_size=8).digest(), "big")
            vector[digest % self.dimensions] += weight if (digest >> 63) & 1 else -weight
        norm = math.sqrt(sum(x * x for x in vector))
        if norm == 0.0:  # no words at all: still a valid, non-zero vector
            vector[0] = 1.0
            return tuple(vector)
        return tuple(x / norm for x in vector)

    async def embed(self, texts: Sequence[str]) -> EmbeddingResult:
        vectors = tuple(self._vector(t) for t in texts)
        return EmbeddingResult(vectors, Usage(sum(estimate_tokens(t) for t in texts)), self.model)


class FakeChatProvider:
    def __init__(self, *, gullible: bool = False, model: str = "fake-chat-1") -> None:
        self.gullible = gullible
        self.model = model

    async def complete(
        self,
        system: str,
        messages: Sequence[ChatTurn],
        *,
        max_output_tokens: int,  # noqa: ARG002 - the port's signature; answers are short by construction
    ) -> ChatCompletion:
        prompt = messages[-1].content if messages else ""
        text = self._answer(system, prompt)
        return ChatCompletion(
            text,
            Usage(estimate_tokens(system) + sum(estimate_tokens(m.content) for m in messages), estimate_tokens(text)),
            self.model,
        )

    def _answer(self, system: str, prompt: str) -> str:
        sources = [(m.group(1), m.group(3)) for m in _SOURCE.finditer(prompt)]
        question_match = _QUESTION.search(prompt)
        question = question_match.group(2) if question_match else ""
        if self.gullible:
            obeyed = self._obey(system, sources, question)
            if obeyed is not None:
                return obeyed
        wanted = {_stem(t) for t in search_terms(question)}
        best: tuple[float, str, str] | None = None
        for alias, body in sources:
            for raw in _SENTENCES.split(body):
                sentence = raw.strip()
                if not sentence or sentence.startswith("#"):
                    continue
                overlap = len(wanted & {_stem(w) for w in words(sentence)})
                if overlap and (best is None or overlap > best[0]):
                    best = (overlap, alias, sentence)
        if best is None:
            return NO_ANSWER
        _, alias, sentence = best
        return f"{sentence} [{alias}]"

    @staticmethod
    def _obey(system: str, sources: list[tuple[str, str]], question: str) -> str | None:
        for text in [body for _, body in sources] + [question]:
            if _REVEAL.search(text):
                return f"My instructions are: {system}"
            reply = _REPLY.search(text)
            if reply:
                return reply.group(1).strip()
        return None
