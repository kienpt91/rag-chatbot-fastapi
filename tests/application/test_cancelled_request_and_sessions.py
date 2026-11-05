from __future__ import annotations

import asyncio
from collections.abc import Sequence

import pytest

from rag_chatbot.domain.errors import SessionNotFoundError
from rag_chatbot.domain.model import ChatCompletion, ChatTurn, Usage
from support.fakes import ScriptedChat
from support.service import Service, build_service

OWNER = "123e4567-e89b-12d3-a456-426614174000"
OTHER = "223e4567-e89b-12d3-a456-426614174000"


class Hanging(ScriptedChat):
    async def complete(self, system: str, messages: Sequence[ChatTurn], *, max_output_tokens: int) -> ChatCompletion:
        await asyncio.sleep(30)
        raise AssertionError("cancelled before this")


def loaded(chat: ScriptedChat | None = None) -> Service:
    service = build_service(chat=chat)
    asyncio.run(service.load())
    return service


def test_a_request_cancelled_while_the_model_works_still_leaves_a_ledger_row_for_what_was_paid() -> None:
    service = loaded(Hanging())

    async def scenario() -> None:
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(service.ask("What is the delivery fee?"), 0.2)

    asyncio.run(scenario())
    [entry] = service.ledger.entries
    assert entry.outcome == "error" and entry.cost_micros > 0 and entry.embedding_tokens > 0


def test_a_session_that_belongs_to_a_user_cannot_be_continued_by_another_user_or_anonymously() -> None:
    service = loaded()
    first = asyncio.run(service.ask("What is the delivery fee?", owner_id=OWNER))
    for stranger in (OTHER, None):
        with pytest.raises(SessionNotFoundError):
            asyncio.run(service.ask("And how many kilometres?", session_id=first.session_id, owner_id=stranger))
    again = asyncio.run(service.ask("And how many kilometres?", session_id=first.session_id, owner_id=OWNER))
    assert again.session_id == first.session_id


def test_an_anonymous_session_can_still_be_continued_with_its_id() -> None:
    service = loaded()
    first = asyncio.run(service.ask("What is the delivery fee?"))
    assert (
        asyncio.run(service.ask("And how many kilometres?", session_id=first.session_id)).session_id == first.session_id
    )


def test_a_strangers_failed_attempt_stores_nothing_in_the_owners_conversation() -> None:
    service = loaded()
    first = asyncio.run(service.ask("What is the delivery fee?", owner_id=OWNER))
    with pytest.raises(SessionNotFoundError):
        asyncio.run(service.ask("poison", session_id=first.session_id, owner_id=OTHER))
    stored = asyncio.run(service.sessions.recent_messages("acme", first.session_id, 10))
    assert len(stored) == 2 and all("poison" not in t.content for t in stored)


def test_usage_reported_as_zero_is_estimated_and_marked() -> None:
    chat = ScriptedChat(reply="The delivery fee is 5 dollars. [s1]", usage=Usage(0, 0))
    service = loaded(chat)
    asyncio.run(service.ask("What is the delivery fee?"))
    assert service.ledger.entries[0].cost_estimated
