"""A throwaway PostgreSQL + pgvector database per test, cloned from a migrated template.

The server is the container `scripts/py.sh` starts when `PY_DB=1` is set (no network route out, a random superuser
password). Cloning a template database makes a migrated database cost a few milliseconds instead of a migration run.
"""

from __future__ import annotations

import os
import secrets
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass
from pathlib import Path

import psycopg
import pytest
from psycopg import sql
from psycopg.conninfo import make_conninfo

from rag_chatbot.infrastructure.postgres.migrations import migrate
from rag_chatbot.infrastructure.postgres.pool import Pool, open_pool

ROOT = Path(__file__).resolve().parents[2]
MIGRATIONS = ROOT / "migrations"


@dataclass(frozen=True)
class ScratchDatabase:
    name: str
    admin_dsn: str  # a superuser on this database: bypasses row-level security, like a migration or a DBA would
    app_dsn: str  # the service's login role, a member of rag_app: row-level security applies
    app_role: str

    @contextmanager
    def admin(self) -> Iterator[psycopg.Connection[tuple[object, ...]]]:
        with psycopg.connect(self.admin_dsn, autocommit=True) as conn:
            yield conn


def server_dsn() -> str:
    url = os.environ.get("RAG_TEST_DATABASE_URL")
    if url:
        return url
    if os.environ.get("RAG_REQUIRE_DB"):
        pytest.fail("RAG_TEST_DATABASE_URL is not set but RAG_REQUIRE_DB is: the database tests may not be skipped")
    pytest.skip("no database: run through scripts/py.sh with PY_DB=1 (RAG_TEST_DATABASE_URL)")


@contextmanager
def template_database() -> Iterator[tuple[str, str, str, str]]:
    """Creates the migrated template and a login role for the service; drops both afterwards."""
    base = server_dsn()
    suffix = secrets.token_hex(4)
    template = f"rag_template_{suffix}"
    role = f"rag_svc_{suffix}"
    password = secrets.token_hex(16)
    with psycopg.connect(base, autocommit=True) as admin:
        admin.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(template)))
    try:
        migrate(make_conninfo(base, dbname=template), MIGRATIONS)
        with psycopg.connect(base, autocommit=True) as admin:
            admin.execute(
                sql.SQL("CREATE ROLE {} LOGIN PASSWORD {} IN ROLE rag_app").format(
                    sql.Identifier(role), sql.Literal(password)
                )
            )
        yield base, template, role, password
    finally:
        with psycopg.connect(base, autocommit=True) as admin:
            admin.execute(sql.SQL("DROP DATABASE IF EXISTS {} WITH (FORCE)").format(sql.Identifier(template)))
            admin.execute(sql.SQL("DROP ROLE IF EXISTS {}").format(sql.Identifier(role)))


@contextmanager
def scratch_database(base: str, template: str, role: str, password: str) -> Iterator[ScratchDatabase]:
    name = f"rag_test_{secrets.token_hex(6)}"
    with psycopg.connect(base, autocommit=True) as admin:
        admin.execute(sql.SQL("CREATE DATABASE {} TEMPLATE {}").format(sql.Identifier(name), sql.Identifier(template)))
    try:
        yield ScratchDatabase(
            name=name,
            admin_dsn=make_conninfo(base, dbname=name),
            app_dsn=make_conninfo(base, dbname=name, user=role, password=password),
            app_role=role,
        )
    finally:
        with psycopg.connect(base, autocommit=True) as admin:
            admin.execute(sql.SQL("DROP DATABASE IF EXISTS {} WITH (FORCE)").format(sql.Identifier(name)))


@asynccontextmanager
async def app_pool(db: ScratchDatabase, *, max_size: int = 4) -> AsyncIterator[Pool]:
    pool = await open_pool(db.app_dsn, min_size=1, max_size=max_size, timeout_s=10)
    try:
        yield pool
    finally:
        await pool.close()
