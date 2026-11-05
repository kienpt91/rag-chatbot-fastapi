"""The contract of the chunk, session and ledger stores, run against the in-memory and the PostgreSQL implementation."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from datetime import UTC, date, datetime, timedelta

import pytest

from rag_chatbot.application.ports import UsageEntry
from rag_chatbot.domain.errors import InvalidRequestError, StorageError
from rag_chatbot.domain.model import ChatTurn, Chunk, Usage
from rag_chatbot.domain.text import search_terms
from support.database import ScratchDatabase
from support.stores import World, kinds, memory_world, postgres_world

BAKERY = [
    "We open at nine in the morning and close at five in the afternoon from Monday to Friday.",
    "On Sundays the bakery is closed. The Saturday opening hours are eight to two.",
    "Our sourdough bread is baked every morning with flour from a local mill.",
]
DELIVERY = ["We deliver within the city every weekday. Delivery costs five dollars."]


def run_world(kind: str, request: pytest.FixtureRequest, scenario: Callable[[World], Awaitable[None]]) -> None:
    async def main() -> None:
        if kind == "memory":
            async with memory_world() as world:
                await scenario(world)
        else:
            database: ScratchDatabase = request.getfixturevalue("scratch_db")
            async with postgres_world(database) as world:
                await scenario(world)

    asyncio.run(main())


@pytest.fixture(params=list(kinds()))
def run(request: pytest.FixtureRequest) -> Callable[[Callable[[World], Awaitable[None]]], None]:
    kind = request.param

    def runner(scenario: Callable[[World], Awaitable[None]]) -> None:
        run_world(kind, request, scenario)

    return runner


# --- chunks ---------------------------------------------------------------------------------------------------------


def test_a_document_is_stored_counted_and_its_chunk_ids_are_known(run: Callable[..., None]) -> None:
    async def scenario(world: World) -> None:
        chunks = await world.index("acme", "hours", BAKERY)
        assert await world.chunks.count_chunks("acme") == 3
        assert await world.chunks.known_chunk_ids("acme", "hours") == {c.chunk_id for c in chunks}
        assert await world.chunks.known_chunk_ids("acme", "other") == frozenset()
        assert await world.chunks.count_chunks("beta") == 0

    run(scenario)


def test_replacing_a_document_removes_chunks_that_are_gone_and_keeps_the_others_without_new_embeddings(
    run: Callable[..., None],
) -> None:
    async def scenario(world: World) -> None:
        before = await world.index("acme", "hours", BAKERY)
        kept = [
            Chunk(c.chunk_id, "acme", "hours", i, "New title", "https://x.example/h", c.text)
            for i, c in enumerate(before[:2])
        ]
        fresh = Chunk(
            "ck_" + "9" * 24, "acme", "hours", 2, "New title", "https://x.example/h", "A brand new paragraph about jam."
        )
        await world.chunks.upsert_document(
            "acme",
            "hours",
            url="https://x.example/h",
            title="New title",
            chunks=[*kept, fresh],
            embeddings=await world.embed([fresh]),  # no embeddings for the two that are already stored
        )
        assert await world.chunks.known_chunk_ids("acme", "hours") == {
            kept[0].chunk_id,
            kept[1].chunk_id,
            fresh.chunk_id,
        }
        hits = await world.chunks.vector_search("acme", (await world.embedder.embed([BAKERY[0]])).vectors[0], 5)
        top = hits[0]
        assert top.chunk.chunk_id == kept[0].chunk_id and top.similarity > 0.99  # the old embedding was kept
        assert top.chunk.title == "New title" and top.chunk.url == "https://x.example/h"

    run(scenario)


def test_a_new_chunk_without_an_embedding_and_a_wrong_sized_embedding_are_refused_and_change_nothing(
    run: Callable[..., None],
) -> None:
    async def scenario(world: World) -> None:
        await world.index("acme", "hours", BAKERY)
        chunk = Chunk("ck_" + "8" * 24, "acme", "hours", 0, "T", "", "text")
        with pytest.raises(StorageError):
            await world.chunks.upsert_document("acme", "hours", url="", title="T", chunks=[chunk], embeddings={})
        with pytest.raises(StorageError):
            await world.chunks.upsert_document(
                "acme", "hours", url="", title="T", chunks=[chunk], embeddings={chunk.chunk_id: (0.1, 0.2)}
            )
        assert await world.chunks.count_chunks("acme") == 3  # the old content is untouched

    run(scenario)


def test_a_chunk_of_another_tenant_or_document_cannot_be_stored_here(run: Callable[..., None]) -> None:
    async def scenario(world: World) -> None:
        foreign = Chunk("ck_" + "7" * 24, "beta", "hours", 0, "T", "", "text")
        with pytest.raises(StorageError):
            await world.chunks.upsert_document(
                "acme", "hours", url="", title="T", chunks=[foreign], embeddings=await world.embed([foreign])
            )
        wrong_doc = Chunk("ck_" + "6" * 24, "acme", "menu", 0, "T", "", "text")
        with pytest.raises(StorageError):
            await world.chunks.upsert_document(
                "acme", "hours", url="", title="T", chunks=[wrong_doc], embeddings=await world.embed([wrong_doc])
            )

    run(scenario)


def test_vector_search_ranks_by_meaning_and_reports_cosine_similarity(run: Callable[..., None]) -> None:
    async def scenario(world: World) -> None:
        chunks = await world.index("acme", "hours", BAKERY)
        query = (await world.embedder.embed(["what time do you open in the morning"])).vectors[0]
        hits = await world.chunks.vector_search("acme", query, 3)
        assert hits[0].chunk.chunk_id == chunks[0].chunk_id
        assert [h.similarity for h in hits] == sorted((h.similarity for h in hits), reverse=True)
        assert all(0.0 <= h.similarity <= 1.0 for h in hits)
        exact = (await world.embedder.embed([BAKERY[2]])).vectors[0]
        assert (await world.chunks.vector_search("acme", exact, 1))[0].similarity == pytest.approx(1.0, abs=1e-5)
        unrelated = (await world.embedder.embed(["quantum chromodynamics lagrangian"])).vectors[0]
        assert (await world.chunks.vector_search("acme", unrelated, 1))[0].similarity < 0.2
        assert len(await world.chunks.vector_search("acme", query, 2)) == 2

    run(scenario)


def test_keyword_search_finds_rare_words_and_ranks_more_matches_first(run: Callable[..., None]) -> None:
    async def scenario(world: World) -> None:
        chunks = await world.index("acme", "hours", BAKERY)
        query = (await world.embedder.embed(["sourdough flour"])).vectors[0]
        hits = await world.chunks.keyword_search("acme", ["sourdough", "flour"], query, 5)
        assert [h.chunk.chunk_id for h in hits] == [chunks[2].chunk_id]
        two = await world.chunks.keyword_search("acme", ["sundays", "closed", "bakery"], query, 5)
        assert two[0].chunk.chunk_id == chunks[1].chunk_id  # matches all three words, the others at most one
        assert await world.chunks.keyword_search("acme", ["zeppelin"], query, 5) == []
        assert all(0.0 <= h.similarity <= 1.0 for h in hits)

    run(scenario)


def test_keyword_search_matches_any_script_and_ignores_case(run: Callable[..., None]) -> None:
    async def scenario(world: World) -> None:
        await world.index("acme", "vi", ["Cửa hàng mở cửa lúc tám giờ sáng.", "مرحبا بكم في المخبز"])
        query = (await world.embedder.embed(["x"])).vectors[0]
        assert len(await world.chunks.keyword_search("acme", ["cửa"], query, 5)) == 1
        assert len(await world.chunks.keyword_search("acme", ["المخبز"], query, 5)) == 1

    run(scenario)


def test_one_tenant_never_sees_another_tenants_chunks_in_either_search(run: Callable[..., None]) -> None:
    async def scenario(world: World) -> None:
        await world.index("acme", "hours", BAKERY)
        await world.index("beta", "hours", ["Beta secret pricing sheet with bread and flour."])
        query = (await world.embedder.embed(["bread flour"])).vectors[0]
        for tenant, expected in (("acme", 3), ("beta", 1)):
            vector_hits = await world.chunks.vector_search(tenant, query, 10)
            keyword_hits = await world.chunks.keyword_search(tenant, search_terms("bread flour"), query, 10)
            assert len(vector_hits) == expected
            assert {h.chunk.tenant_id for h in [*vector_hits, *keyword_hits]} == {tenant}
        assert await world.chunks.vector_search("nobody", query, 10) == []

    run(scenario)


def test_pruning_removes_documents_that_are_not_listed_with_their_chunks(run: Callable[..., None]) -> None:
    async def scenario(world: World) -> None:
        await world.index("acme", "hours", BAKERY)
        await world.index("acme", "delivery", DELIVERY)
        await world.index("beta", "hours", BAKERY)
        assert await world.chunks.prune_documents("acme", ["hours"]) == 1
        assert await world.chunks.count_chunks("acme") == 3
        assert await world.chunks.count_chunks("beta") == 3
        with pytest.raises(InvalidRequestError):
            await world.chunks.prune_documents("acme", [])
        assert await world.chunks.count_chunks("acme") == 3

    run(scenario)


def test_hostile_text_is_stored_and_returned_literally(run: Callable[..., None]) -> None:
    async def scenario(world: World) -> None:
        nasty = "'; DROP TABLE chunks; -- \x27 \\ % _ <script>alert(1)</script> $$ ${x} {{y}}"
        await world.index("acme", "nasty'; DROP TABLE documents; --", [nasty], title=nasty[:100])
        query = (await world.embedder.embed([nasty])).vectors[0]
        [hit] = await world.chunks.vector_search("acme", query, 3)
        assert hit.chunk.text == nasty and hit.chunk.document_id.startswith("nasty'")
        assert await world.chunks.count_chunks("acme") == 1

    run(scenario)


# --- sessions ------------------------------------------------------------------------------------------------------


def test_sessions_belong_to_their_tenant(run: Callable[..., None]) -> None:
    async def scenario(world: World) -> None:
        await world.sessions.create_session("acme", "s_one_12345", None)
        info = await world.sessions.get_session("acme", "s_one_12345")
        assert info is not None and info.message_count == 0 and info.tenant_id == "acme"
        assert await world.sessions.get_session("beta", "s_one_12345") is None
        assert await world.sessions.get_session("acme", "s_unknown_1") is None
        assert await world.sessions.recent_messages("beta", "s_one_12345", 10) == []

    run(scenario)


def test_messages_are_kept_in_order_counted_and_the_recent_ones_come_back_oldest_first(
    run: Callable[..., None],
) -> None:
    async def scenario(world: World) -> None:
        await world.sessions.create_session("acme", "s_one_12345", "123e4567-e89b-12d3-a456-426614174000")
        for i in range(4):
            await world.sessions.append_messages(
                "acme",
                "s_one_12345",
                [ChatTurn("user", f"q{i}"), ChatTurn("assistant", f"a{i}")],
                grounded=bool(i % 2),
                chunk_ids=["ck_" + "1" * 24] if i % 2 else [],
            )
        info = await world.sessions.get_session("acme", "s_one_12345")
        assert info is not None and info.message_count == 8
        recent = await world.sessions.recent_messages("acme", "s_one_12345", 3)
        assert recent == [ChatTurn("assistant", "a2"), ChatTurn("user", "q3"), ChatTurn("assistant", "a3")]
        assert len(await world.sessions.recent_messages("acme", "s_one_12345", 100)) == 8

    run(scenario)


def test_purging_removes_old_sessions_with_their_messages_and_only_that_tenants(run: Callable[..., None]) -> None:
    async def scenario(world: World) -> None:
        for tenant in ("acme", "beta"):
            await world.sessions.create_session(tenant, "s_old_123456", None)
            await world.sessions.append_messages(
                tenant, "s_old_123456", [ChatTurn("user", "hi")], grounded=None, chunk_ids=[]
            )
        future = datetime.now(UTC) + timedelta(days=1)
        assert await world.sessions.purge_sessions("acme", datetime.now(UTC) - timedelta(days=1)) == 0
        assert await world.sessions.purge_sessions("acme", future) == 1
        assert await world.sessions.get_session("acme", "s_old_123456") is None
        assert await world.sessions.get_session("beta", "s_old_123456") is not None

    run(scenario)


# --- the ledger ----------------------------------------------------------------------------------------------------


def entry(tenant: str, at: datetime, micros: int, outcome: str = "answered") -> UsageEntry:
    return UsageEntry(tenant, "s_ledger_12", "r_1", at, "chat", "embed", Usage(10, 5), 3, micros, False, 12, outcome)


def test_the_ledger_sums_a_tenants_cost_per_utc_day(run: Callable[..., None]) -> None:
    async def scenario(world: World) -> None:
        day = datetime(2026, 11, 5, tzinfo=UTC)
        await world.ledger.record(entry("acme", day - timedelta(seconds=1), 1000))  # the day before
        await world.ledger.record(entry("acme", day, 7))  # the first second of the day
        await world.ledger.record(entry("acme", day + timedelta(hours=12), 30, "refused"))
        await world.ledger.record(entry("acme", day + timedelta(hours=24) - timedelta(microseconds=1), 5))
        await world.ledger.record(entry("acme", day + timedelta(hours=24), 2000))  # the day after
        await world.ledger.record(entry("beta", day + timedelta(hours=1), 400))
        assert await world.ledger.spent_micros("acme", date(2026, 11, 5)) == 42
        assert await world.ledger.spent_micros("beta", date(2026, 11, 5)) == 400
        assert await world.ledger.spent_micros("acme", date(2026, 11, 4)) == 1000
        assert await world.ledger.spent_micros("acme", date(2026, 11, 6)) == 2000
        assert await world.ledger.spent_micros("nobody", date(2026, 11, 5)) == 0

    run(scenario)


# --- PostgreSQL only -----------------------------------------------------------------------------------------------


@pytest.mark.db
def test_a_tenant_with_few_chunks_is_found_even_when_the_index_is_full_of_another_tenants(
    scratch_db: ScratchDatabase,
) -> None:
    """The approximate index filters after it reads: without iterative scans the 3 chunks of `small` would be missed
    because the 40 nearest candidates all belong to `big`. The database is told not to use a sequential scan, so the
    HNSW index is what answers (checked with EXPLAIN)."""
    with scratch_db.admin() as conn:
        # no sequential scan and no explicit sort: the only plan that orders by distance is the HNSW index scan
        conn.execute(f"ALTER DATABASE {scratch_db.name} SET enable_seqscan = off")  # type: ignore[arg-type]
        conn.execute(f"ALTER DATABASE {scratch_db.name} SET enable_sort = off")  # type: ignore[arg-type]
        conn.execute(
            "INSERT INTO documents (tenant_id, document_id) SELECT t, 'doc' FROM unnest(ARRAY['big', 'small']) t"
        )
        conn.execute(
            "INSERT INTO chunks (tenant_id, chunk_id, document_id, ordinal, content, embedding) "
            "SELECT 'big', 'ck_' || lpad(i::text, 24, '0'), 'doc', i, 'big chunk ' || i, "
            "(SELECT array_agg(sin(i * 1.0 + d))::real[]::vector(1536) FROM generate_series(1, 1536) d) "
            "FROM generate_series(1, 400) i"
        )
        conn.execute(
            "INSERT INTO chunks (tenant_id, chunk_id, document_id, ordinal, content, embedding) "
            "SELECT 'small', 'ck_' || lpad((1000 + i)::text, 24, '0'), 'doc', i, 'small chunk ' || i, "
            "(SELECT array_agg(cos(i * 3.0 + d))::real[]::vector(1536) FROM generate_series(1, 1536) d) "
            "FROM generate_series(1, 3) i"
        )

    async def scenario() -> None:
        from support.database import app_pool

        async with app_pool(scratch_db) as pool:
            from rag_chatbot.infrastructure.postgres.chunks import PostgresChunkStore

            store = PostgresChunkStore(pool, dimensions=1536)
            query = [float(__import__("math").sin(5.0 + d)) for d in range(1, 1537)]  # close to the big tenant
            hits = await store.vector_search("small", query, 3)
            assert len(hits) == 3 and {h.chunk.tenant_id for h in hits} == {"small"}

    asyncio.run(scenario())
    with psycopg_connect(scratch_db) as conn:
        conn.execute("ANALYZE chunks")
        plan = "\n".join(
            str(r[0])
            for r in conn.execute(
                "EXPLAIN SELECT chunk_id FROM chunks WHERE tenant_id = 'small' "
                "ORDER BY embedding <=> array_fill(0.1::real, ARRAY[1536])::vector LIMIT 3"
            ).fetchall()
        )
    assert "chunks_embedding_idx" in plan


def psycopg_connect(database: ScratchDatabase):  # type: ignore[no-untyped-def]
    import psycopg

    return psycopg.connect(database.admin_dsn, autocommit=True)


@pytest.mark.db
def test_a_pool_of_one_connection_never_carries_one_tenants_scope_over_to_the_next(scratch_db: ScratchDatabase) -> None:
    async def scenario() -> None:
        async with postgres_world(scratch_db, max_size=1) as world:
            await world.index("acme", "hours", BAKERY)
            await world.index("beta", "hours", DELIVERY)
            for _ in range(3):
                assert len(await world.chunks.known_chunk_ids("acme", "hours")) == 3
                assert len(await world.chunks.known_chunk_ids("beta", "hours")) == 1
                assert await world.chunks.count_chunks("acme") == 3
                assert await world.chunks.count_chunks("beta") == 1

    asyncio.run(scenario())


@pytest.mark.db
def test_database_failures_become_storage_errors_that_say_nothing_about_the_data(scratch_db: ScratchDatabase) -> None:
    async def scenario() -> None:
        async with postgres_world(scratch_db) as world:
            secret_id = "d" * 3000  # violates a CHECK constraint
            chunk = Chunk("ck_" + "5" * 24, "acme", secret_id, 0, "T", "", "text")
            with pytest.raises(StorageError) as caught:
                await world.chunks.upsert_document(
                    "acme", secret_id, url="", title="T", chunks=[chunk], embeddings=await world.embed([chunk])
                )
            assert "ddd" not in str(caught.value) and "sqlstate" in str(caught.value)

    asyncio.run(scenario())
