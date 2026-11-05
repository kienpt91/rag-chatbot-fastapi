"""The connection pool, and the one way a tenant's data is touched: a transaction that says whose it is."""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager, contextmanager
from typing import Any

import psycopg
from psycopg import AsyncConnection
from psycopg.rows import TupleRow
from psycopg_pool import AsyncConnectionPool

from rag_chatbot.domain.errors import StorageError

Pool = AsyncConnectionPool[AsyncConnection[TupleRow]]


async def open_pool(
    dsn: str, *, min_size: int = 1, max_size: int = 10, timeout_s: float = 5.0, statement_timeout_ms: int = 10_000
) -> Pool:
    """A pool whose connections give up on a statement after `statement_timeout_ms` and on an idle transaction after
    15 seconds. `timeout_s` bounds how long a request waits for a free connection."""
    pool: Pool = AsyncConnectionPool(
        dsn,
        min_size=min_size,
        max_size=max_size,
        timeout=timeout_s,
        open=False,
        kwargs={
            "autocommit": True,  # transactions are explicit, see tenant_transaction
            "application_name": "rag-chatbot",
            "options": f"-c statement_timeout={int(statement_timeout_ms)} -c idle_in_transaction_session_timeout=15000",
        },
    )
    try:
        await pool.open(wait=True, timeout=timeout_s)
    except Exception as error:
        await pool.close()
        raise StorageError(f"cannot connect to the database: {error.__class__.__name__}") from None
    return pool


@contextmanager
def storage_errors() -> Iterator[None]:
    """Turn any database failure into `StorageError` without the SQL, the parameters or the server's message."""
    try:
        yield
    except psycopg.Error as error:
        state = getattr(error, "sqlstate", None) or "-"
        raise StorageError(f"{error.__class__.__name__} (sqlstate {state})") from None
    except TimeoutError:
        raise StorageError("timeout waiting for a database connection") from None


@asynccontextmanager
async def tenant_transaction(pool: Pool, tenant_id: str) -> AsyncIterator[AsyncConnection[Any]]:
    """A connection inside a transaction that is scoped to one tenant for row-level security: the policies compare
    every row with `rag.tenant_id`, which is set here, for this transaction only."""
    async with pool.connection() as conn, conn.transaction():
        await conn.execute("SELECT set_config('rag.tenant_id', %s, true)", (tenant_id,))
        yield conn
