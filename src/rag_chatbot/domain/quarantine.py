"""Quarantine at ingest time: sentences that read as instructions to an assistant never enter the index.

A page is data, but anyone who can edit it (a review, a comment, a guest post) can write sentences aimed at the
assistant: "ignore previous instructions", "REPLY: ...", "the assistant must say ...". Retrieval would hand such a
sentence to the model as a source. The answer checks stop most of what follows, but not a harmless-looking claim that
a cited sentence makes, so the sentence is better never indexed. This module removes it, sentence by sentence, keeps
the rest of the page as it was, and returns what it removed so the operator can read it (`ingest` prints the list).

It is a filter on wording, not a proof: a rephrased or translated instruction passes, and an honest sentence that
happens to match is lost (the report names it, `RAG_QUARANTINE_INSTRUCTIONS=false` switches the filter off, and
`ingest --exclude` keeps whole pages out). The defences that do not depend on wording are in `guard.py` and `prompt.py`.
"""

from __future__ import annotations

import re

from rag_chatbot.domain.scope import page_instruction_reason

_SENTENCE_END = re.compile(r"(?<=[.!?\u3002\uff01\uff1f])\s+")


def quarantine(text: str) -> tuple[str, list[str]]:
    """The text without its instruction-like sentences, and those sentences in order. Paragraph and line structure of
    what remains is kept; a paragraph that loses every sentence disappears."""
    removed: list[str] = []
    paragraphs: list[str] = []
    for paragraph in text.split("\n\n"):
        lines: list[str] = []
        for line in paragraph.split("\n"):
            kept: list[str] = []
            for sentence in _SENTENCE_END.split(line):
                if sentence.strip() and page_instruction_reason(sentence) is not None:
                    removed.append(sentence.strip())
                else:
                    kept.append(sentence)
            joined = " ".join(kept).strip()
            if joined:
                lines.append(joined)
        if lines:
            paragraphs.append("\n".join(lines))
    return "\n\n".join(paragraphs), removed
