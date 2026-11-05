"""On Supabase, default privileges hand new public tables and sequences to anon and authenticated. The migrations must
take back everything that is not meant for them, including the migration ledger and the identity sequences."""

from __future__ import annotations

import secrets

import psycopg
import pytest
from psycopg import errors, sql
from psycopg.conninfo import make_conninfo

from rag_chatbot.infrastructure.postgres.migrations import migrate
from support.database import MIGRATIONS, server_dsn

pytestmark = pytest.mark.db


def test_nothing_is_left_to_anon_and_authenticated_that_supabase_default_privileges_would_grant() -> None:
    base = server_dsn()
    name = f"rag_priv_{secrets.token_hex(4)}"
    with psycopg.connect(base, autocommit=True) as admin:
        admin.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(name)))
    try:
        dsn = make_conninfo(base, dbname=name)
        with psycopg.connect(dsn, autocommit=True) as conn:
            for role in ("anon", "authenticated"):
                exists = conn.execute("SELECT 1 FROM pg_roles WHERE rolname = %s", (role,)).fetchone()
                if exists is None:
                    conn.execute(sql.SQL("CREATE ROLE {} NOLOGIN").format(sql.Identifier(role)))
            conn.execute("ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON TABLES TO anon, authenticated")
            conn.execute("ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON SEQUENCES TO anon, authenticated")
        migrate(dsn, MIGRATIONS)
        with psycopg.connect(dsn, autocommit=True) as conn:
            checks = [
                ("anon", "SELECT 1 FROM schema_migrations"),
                ("authenticated", "SELECT 1 FROM schema_migrations"),
                ("authenticated", "DELETE FROM schema_migrations"),
                ("anon", "SELECT nextval('chat_messages_id_seq')"),
                ("authenticated", "SELECT nextval('chat_messages_id_seq')"),
                ("authenticated", "SELECT nextval('usage_events_id_seq')"),
                ("anon", "SELECT 1 FROM chunks"),
                ("authenticated", "SELECT 1 FROM usage_events"),
            ]
            for role, statement in checks:
                conn.execute("BEGIN")
                conn.execute(sql.SQL("SET LOCAL ROLE {}").format(sql.Identifier(role)))
                with pytest.raises(errors.InsufficientPrivilege):
                    conn.execute(statement.encode())
                conn.execute("ROLLBACK")
            rows = conn.execute(
                "SELECT grantee, table_name, privilege_type FROM information_schema.role_table_grants "
                "WHERE table_schema = 'public' AND grantee IN ('anon', 'authenticated') "
                "AND NOT (grantee = 'authenticated' AND table_name IN ('chat_sessions', 'chat_messages') AND privilege_type = 'SELECT')"
            ).fetchall()
            assert rows == []
    finally:
        with psycopg.connect(base, autocommit=True) as admin:
            admin.execute(sql.SQL("DROP DATABASE IF EXISTS {} WITH (FORCE)").format(sql.Identifier(name)))
