"""Sessions, messages and the usage ledger in PostgreSQL."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, date, datetime, timedelta

from rag_chatbot.application.ports import SessionInfo, UsageEntry
from rag_chatbot.domain.model import ChatTurn
from rag_chatbot.infrastructure.postgres.pool import Pool, storage_errors, tenant_transaction


class PostgresSessionStore:
    def __init__(self, pool: Pool) -> None:
        self._pool = pool

    async def create_session(self, tenant_id: str, session_id: str, owner_id: str | None) -> None:
        with storage_errors():
            async with tenant_transaction(self._pool, tenant_id) as conn:
                await conn.execute(
                    "INSERT INTO chat_sessions (tenant_id, session_id, owner_id) VALUES (%s, %s, %s)",
                    (tenant_id, session_id, owner_id),
                )

    async def get_session(self, tenant_id: str, session_id: str) -> SessionInfo | None:
        with storage_errors():
            async with tenant_transaction(self._pool, tenant_id) as conn:
                cursor = await conn.execute(
                    "SELECT message_count FROM chat_sessions WHERE tenant_id = %s AND session_id = %s",
                    (tenant_id, session_id),
                )
                row = await cursor.fetchone()
        return None if row is None else SessionInfo(session_id, tenant_id, int(row[0]))

    async def append_messages(
        self,
        tenant_id: str,
        session_id: str,
        turns: Sequence[ChatTurn],
        *,
        grounded: bool | None,
        chunk_ids: Sequence[str],
    ) -> None:
        if not turns:
            return
        with storage_errors():
            async with tenant_transaction(self._pool, tenant_id) as conn:
                last = len(turns) - 1
                async with conn.cursor() as cur:
                    await cur.executemany(
                        "INSERT INTO chat_messages (tenant_id, session_id, role, content, grounded, chunk_ids) "
                        "VALUES (%s, %s, %s, %s, %s, %s)",
                        [
                            (
                                tenant_id,
                                session_id,
                                turn.role,
                                turn.content,
                                grounded if index == last else None,
                                list(chunk_ids) if index == last else [],
                            )
                            for index, turn in enumerate(turns)
                        ],
                    )
                await conn.execute(
                    "UPDATE chat_sessions SET message_count = message_count + %s, last_active_at = now() "
                    "WHERE tenant_id = %s AND session_id = %s",
                    (len(turns), tenant_id, session_id),
                )

    async def recent_messages(self, tenant_id: str, session_id: str, limit: int) -> list[ChatTurn]:
        with storage_errors():
            async with tenant_transaction(self._pool, tenant_id) as conn:
                cursor = await conn.execute(
                    "SELECT role, content FROM ("
                    "  SELECT id, role, content FROM chat_messages WHERE tenant_id = %s AND session_id = %s "
                    "  ORDER BY id DESC LIMIT %s) recent ORDER BY id",
                    (tenant_id, session_id, limit),
                )
                rows = await cursor.fetchall()
        return [ChatTurn("assistant" if row[0] == "assistant" else "user", str(row[1])) for row in rows]

    async def purge_sessions(self, tenant_id: str, older_than: datetime) -> int:
        """Delete a tenant's sessions (and their messages) that were last active before `older_than`."""
        with storage_errors():
            async with tenant_transaction(self._pool, tenant_id) as conn:
                cursor = await conn.execute(
                    "DELETE FROM chat_sessions WHERE tenant_id = %s AND last_active_at < %s", (tenant_id, older_than)
                )
                return cursor.rowcount


class PostgresUsageLedger:
    def __init__(self, pool: Pool) -> None:
        self._pool = pool

    async def spent_micros(self, tenant_id: str, day: date) -> int:
        start = datetime(day.year, day.month, day.day, tzinfo=UTC)
        with storage_errors():
            async with tenant_transaction(self._pool, tenant_id) as conn:
                cursor = await conn.execute(
                    "SELECT coalesce(sum(cost_micros), 0) FROM usage_events "
                    "WHERE tenant_id = %s AND created_at >= %s AND created_at < %s",
                    (tenant_id, start, start + timedelta(days=1)),
                )
                row = await cursor.fetchone()
        return int(row[0]) if row else 0

    async def record(self, entry: UsageEntry) -> None:
        with storage_errors():
            async with tenant_transaction(self._pool, entry.tenant_id) as conn:
                await conn.execute(
                    "INSERT INTO usage_events (tenant_id, session_id, request_id, created_at, chat_model, "
                    "embedding_model, input_tokens, output_tokens, embedding_tokens, cost_micros, cost_estimated, "
                    "latency_ms, outcome) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
                    (
                        entry.tenant_id,
                        entry.session_id,
                        entry.request_id,
                        entry.at,
                        entry.chat_model,
                        entry.embedding_model,
                        entry.usage.input_tokens,
                        entry.usage.output_tokens,
                        entry.embedding_tokens,
                        entry.cost_micros,
                        entry.cost_estimated or entry.usage.estimated,
                        entry.latency_ms,
                        entry.outcome,
                    ),
                )
