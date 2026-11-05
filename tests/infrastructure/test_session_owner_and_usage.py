from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable
from typing import Any

import pytest

from rag_chatbot.infrastructure.providers.http import HttpApiClient, HttpEmbeddingProvider
from support.database import ScratchDatabase
from support.fake_api import FakeApi
from support.stores import World, kinds, memory_world, postgres_world

OWNER = "123e4567-e89b-12d3-a456-426614174000"


@pytest.fixture(params=list(kinds()))
def run(request: pytest.FixtureRequest) -> Callable[[Callable[[World], Awaitable[None]]], None]:
    def runner(scenario: Callable[[World], Awaitable[None]]) -> None:
        async def main() -> None:
            if request.param == "memory":
                async with memory_world() as world:
                    await scenario(world)
            else:
                database: ScratchDatabase = request.getfixturevalue("scratch_db")
                async with postgres_world(database) as world:
                    await scenario(world)

        asyncio.run(main())

    return runner


def test_a_stored_session_reports_its_owner_and_an_anonymous_one_reports_none(run: Callable[..., None]) -> None:
    async def scenario(world: World) -> None:
        await world.sessions.create_session("acme", "s_owned_12345", OWNER)
        await world.sessions.create_session("acme", "s_anon_123456", None)
        owned = await world.sessions.get_session("acme", "s_owned_12345")
        anonymous = await world.sessions.get_session("acme", "s_anon_123456")
        assert owned is not None and owned.owner_id == OWNER
        assert anonymous is not None and anonymous.owner_id is None

    run(scenario)


def test_an_embedding_answer_without_usage_is_estimated_not_free() -> None:
    api = FakeApi(dimensions=8).start()
    try:

        def answer(path: str, body: dict[str, Any]) -> tuple[int, bytes, dict[str, str]]:
            data = [{"index": i, "embedding": [0.1] * 8} for i, _ in enumerate(body["input"])]
            return 200, json.dumps({"data": data}).encode(), {"Content-Type": "application/json"}

        api.override = answer
        provider = HttpEmbeddingProvider(HttpApiClient(api.url, "k" * 20), "m", dimensions=8)
        result = asyncio.run(provider.embed(["a" * 400, "b" * 40]))
        assert result.usage.estimated and result.usage.input_tokens == 110
    finally:
        api.stop()
