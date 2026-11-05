"""Keeping a conversation inside a token budget."""

from __future__ import annotations

from collections.abc import Sequence

from rag_chatbot.domain.model import ChatTurn
from rag_chatbot.domain.text import estimate_tokens


def trim_history(turns: Sequence[ChatTurn], *, budget_tokens: int, max_turns: int) -> tuple[ChatTurn, ...]:
    """The most recent turns that fit `budget_tokens` (estimated) and `max_turns`, oldest first.

    Whole turns only, newest first, so the conversation never starts mid-sentence. The result starts with a visitor
    turn: an assistant turn whose question was dropped is dropped too. A single turn bigger than the whole budget is
    dropped rather than cut."""
    kept: list[ChatTurn] = []
    used = 0
    for turn in reversed(turns):
        cost = estimate_tokens(turn.content) + 4  # a few tokens per message for the role and the framing
        if len(kept) >= max_turns or used + cost > budget_tokens:
            break
        kept.append(turn)
        used += cost
    kept.reverse()
    while kept and kept[0].role != "user":
        kept.pop(0)
    return tuple(kept)
