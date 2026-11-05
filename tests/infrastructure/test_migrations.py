from __future__ import annotations

import asyncio
import secrets
from pathlib import Path

import psycopg
import pytest
from psycopg import sql

from rag_chatbot.domain.errors import ConfigurationError, StorageError
from rag_chatbot.infrastructure.postgres.health import schema_problems
from rag_chatbot.infrastructure.postgres.migrations import expected_version, load_migrations, migrate
from rag_chatbot.infrastructure.postgres.pool import open_pool
from support.database import MIGRATIONS, ScratchDatabase, app_pool, server_dsn

# --- the files themselves: no database needed ---------------------------------------------------------------------


def test_the_shipped_migrations_are_numbered_without_gaps_and_have_checksums() -> None:
    migrations = load_migrations(MIGRATIONS)
    assert [m.version[:4] for m in migrations] == ["0001", "0002", "0003"]
    assert len({m.checksum for m in migrations}) == len(migrations)
    assert expected_version(MIGRATIONS) == migrations[-1].version


def test_a_gap_a_bad_name_or_a_missing_directory_is_refused(tmp_path: Path) -> None:
    (tmp_path / "0001_a.sql").write_text("SELECT 1;")
    (tmp_path / "0003_c.sql").write_text("SELECT 1;")
    with pytest.raises(ConfigurationError, match="gaps"):
        load_migrations(tmp_path)
    (tmp_path / "0003_c.sql").rename(tmp_path / "three.sql")
    with pytest.raises(ConfigurationError, match="must look like"):
        load_migrations(tmp_path)
    with pytest.raises(ConfigurationError, match="not found"):
        load_migrations(tmp_path / "missing")
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ConfigurationError, match="numbered"):
        load_migrations(empty)


def test_no_migration_creates_a_table_without_row_level_security() -> None:
    schema = (MIGRATIONS / "0002_content_and_conversations.sql").read_text()
    rls = (MIGRATIONS / "0003_row_level_security.sql").read_text()
    tables = [line.split()[2] for line in schema.splitlines() if line.startswith("CREATE TABLE")]
    assert tables
    for table in tables:
        assert f"ALTER TABLE {table}" in rls and "ENABLE ROW LEVEL SECURITY" in rls
        assert f"{table} FORCE ROW LEVEL SECURITY" in rls.replace("  ", " ") or f"ALTER TABLE {table}" in rls


# --- against a real database ---------------------------------------------------------------------------------------


@pytest.mark.db
def test_migrations_apply_once_and_a_second_run_changes_nothing(scratch_db: ScratchDatabase) -> None:
    assert migrate(scratch_db.admin_dsn, MIGRATIONS) == []  # the template already has them
    with scratch_db.admin() as conn:
        versions = [row[0] for row in conn.execute("SELECT version FROM schema_migrations ORDER BY version")]
    assert versions == [m.version for m in load_migrations(MIGRATIONS)]


@pytest.mark.db
def test_a_fresh_database_is_migrated_completely_and_the_extension_is_installed() -> None:
    from psycopg.conninfo import make_conninfo

    base = server_dsn()
    name = f"rag_fresh_{secrets.token_hex(4)}"
    with psycopg.connect(base, autocommit=True) as admin:
        admin.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(name)))
    try:
        dsn = make_conninfo(base, dbname=name)
        assert migrate(dsn, MIGRATIONS) == [m.version for m in load_migrations(MIGRATIONS)]
        with psycopg.connect(dsn) as conn:
            assert conn.execute("SELECT extversion FROM pg_extension WHERE extname = 'vector'").fetchone() is not None
            tables = {r[0] for r in conn.execute("SELECT tablename FROM pg_tables WHERE schemaname = 'public'")}
        assert {"documents", "chunks", "chat_sessions", "chat_messages", "usage_events", "schema_migrations"} <= tables
    finally:
        with psycopg.connect(base, autocommit=True) as admin:
            admin.execute(sql.SQL("DROP DATABASE IF EXISTS {} WITH (FORCE)").format(sql.Identifier(name)))


@pytest.mark.db
def test_editing_an_applied_migration_is_refused(scratch_db: ScratchDatabase, tmp_path: Path) -> None:
    for migration in load_migrations(MIGRATIONS):
        (tmp_path / f"{migration.version}.sql").write_text(migration.sql)
    (tmp_path / "0001_extensions_and_roles.sql").write_text("SELECT 1;")
    with pytest.raises(ConfigurationError, match="changed after it was applied"):
        migrate(scratch_db.admin_dsn, tmp_path)


@pytest.mark.db
def test_a_failing_migration_rolls_back_and_is_not_recorded(scratch_db: ScratchDatabase, tmp_path: Path) -> None:
    for migration in load_migrations(MIGRATIONS):
        (tmp_path / f"{migration.version}.sql").write_text(migration.sql)
    (tmp_path / "0004_broken.sql").write_text("CREATE TABLE half_done (x int); SELECT 1/0;")
    with pytest.raises(StorageError, match="migration failed"):
        migrate(scratch_db.admin_dsn, tmp_path)
    with scratch_db.admin() as conn:
        assert conn.execute("SELECT to_regclass('half_done')").fetchone() == (None,)
        assert conn.execute("SELECT count(*) FROM schema_migrations WHERE version = '0004_broken'").fetchone() == (0,)


@pytest.mark.db
def test_the_schema_check_accepts_a_migrated_database_and_names_what_is_wrong(scratch_db: ScratchDatabase) -> None:
    async def scenario() -> None:
        async with app_pool(scratch_db) as pool:
            assert await schema_problems(pool, migrations=MIGRATIONS, dimensions=1536) == []
            assert await schema_problems(pool, migrations=MIGRATIONS, dimensions=768) == ["dimension_mismatch"]

    asyncio.run(scenario())
    with scratch_db.admin() as conn:
        conn.execute("DELETE FROM schema_migrations WHERE version = '0003_row_level_security'")

    async def outdated() -> None:
        async with app_pool(scratch_db) as pool:
            assert await schema_problems(pool, migrations=MIGRATIONS, dimensions=1536) == ["schema_outdated"]

    asyncio.run(outdated())


@pytest.mark.db
def test_an_unreachable_database_is_a_storage_error_without_the_address() -> None:
    async def scenario() -> None:
        with pytest.raises(StorageError) as caught:
            await open_pool("host=127.0.0.1 port=1 dbname=x user=y connect_timeout=1", timeout_s=1.5)
        assert "127.0.0.1" not in str(caught.value)

    asyncio.run(scenario())
