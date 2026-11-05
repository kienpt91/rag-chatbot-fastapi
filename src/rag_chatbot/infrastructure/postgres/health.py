"""Checks the running database against what this version of the service expects."""

from __future__ import annotations

from pathlib import Path

from rag_chatbot.infrastructure.postgres.migrations import expected_version
from rag_chatbot.infrastructure.postgres.pool import Pool, storage_errors


async def schema_problems(pool: Pool, *, migrations: Path, dimensions: int) -> list[str]:
    """Short problem codes, empty when the database is ready: `schema_outdated`, `dimension_mismatch`,
    `vector_missing`. A connection failure raises `StorageError`."""
    problems: list[str] = []
    with storage_errors():
        async with pool.connection() as conn:
            cursor = await conn.execute("SELECT 1 FROM pg_extension WHERE extname = 'vector'")
            if await cursor.fetchone() is None:
                problems.append("vector_missing")
            cursor = await conn.execute("SELECT to_regclass('schema_migrations') IS NOT NULL")
            row = await cursor.fetchone()
            applied = ""
            if row is not None and row[0]:
                cursor = await conn.execute("SELECT max(version) FROM schema_migrations")
                latest = await cursor.fetchone()
                applied = str(latest[0]) if latest and latest[0] else ""
            if applied != expected_version(migrations):
                problems.append("schema_outdated")
            else:
                cursor = await conn.execute(
                    "SELECT atttypmod FROM pg_attribute "
                    "WHERE attrelid = to_regclass('chunks') AND attname = 'embedding' AND NOT attisdropped"
                )
                column = await cursor.fetchone()
                if column is None or int(column[0]) != dimensions:
                    problems.append("dimension_mismatch")
    return problems
