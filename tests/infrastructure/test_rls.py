"""The row-level security policies, exercised on a real PostgreSQL with pgvector.

Everything runs as a superuser connection that *drops to* a role with SET ROLE (superusers bypass row-level security;
a role they have switched to does not). Seed data is written first as the superuser, so every test starts from two
tenants with content, conversations and a ledger.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager

import psycopg
import pytest
from psycopg import errors, sql

from support.database import ScratchDatabase

pytestmark = pytest.mark.db

USER_1 = "11111111-1111-4111-8111-111111111111"
USER_2 = "22222222-2222-4222-8222-222222222222"
CHUNK_A = "ck_" + "a" * 24
CHUNK_B = "ck_" + "b" * 24
TABLES = ("documents", "chunks", "chat_sessions", "chat_messages", "usage_events")
Conn = psycopg.Connection[tuple[object, ...]]


@pytest.fixture
def seeded(scratch_db: ScratchDatabase) -> ScratchDatabase:
    with scratch_db.admin() as conn:
        for tenant, chunk in (("acme", CHUNK_A), ("beta", CHUNK_B)):
            conn.execute("INSERT INTO documents (tenant_id, document_id, title) VALUES (%s, 'doc', 'Doc')", (tenant,))
            conn.execute(
                "INSERT INTO chunks (tenant_id, chunk_id, document_id, ordinal, content, embedding) "
                "VALUES (%s, %s, 'doc', 0, %s, array_fill(0.1::real, ARRAY[1536])::vector)",
                (tenant, chunk, f"secret of {tenant}"),
            )
        for tenant, session, owner in (
            ("acme", "s_acme_user1", USER_1),
            ("acme", "s_acme_user2", USER_2),
            ("acme", "s_acme_anonymous", None),
            ("beta", "s_beta_user1", USER_1),
        ):
            conn.execute(
                "INSERT INTO chat_sessions (tenant_id, session_id, owner_id) VALUES (%s, %s, %s)",
                (tenant, session, owner),
            )
            conn.execute(
                "INSERT INTO chat_messages (tenant_id, session_id, role, content) VALUES (%s, %s, 'user', %s)",
                (tenant, session, f"hello from {session}"),
            )
        for tenant in ("acme", "beta"):
            conn.execute(
                "INSERT INTO usage_events (tenant_id, session_id, request_id, created_at, chat_model, embedding_model, "
                "input_tokens, output_tokens, embedding_tokens, cost_micros, cost_estimated, latency_ms, outcome) "
                "VALUES (%s, 's', 'r', now(), 'c', 'e', 1, 1, 1, 7, false, 1, 'answered')",
                (tenant,),
            )
    return scratch_db


@contextmanager
def as_role(
    database: ScratchDatabase, role: str, *, tenant: str | None = None, sub: str | None = None
) -> Iterator[Conn]:
    """A transaction (rolled back at the end) running as `role`, with the tenant and JWT claims set like the service
    or the API gateway would set them."""
    with database.admin() as conn:
        conn.execute("BEGIN")
        try:
            conn.execute(sql.SQL("SET LOCAL ROLE {}").format(sql.Identifier(role)))
            if tenant is not None:
                conn.execute("SELECT set_config('rag.tenant_id', %s, true)", (tenant,))
            if sub is not None:
                conn.execute("SELECT set_config('request.jwt.claims', %s, true)", (json.dumps({"sub": sub}),))
            yield conn
        finally:
            conn.execute("ROLLBACK")


def count(conn: Conn, table: str) -> int:
    row = conn.execute(sql.SQL("SELECT count(*) FROM {}").format(sql.Identifier(table))).fetchone()
    assert row is not None and isinstance(row[0], int)
    return row[0]


def attempt(conn: Conn, statement: str, params: tuple[object, ...] = ()) -> None:
    """Run a statement that is expected to fail inside a savepoint, so the transaction stays usable."""
    with conn.transaction():
        conn.execute(statement.encode(), params)  # type: ignore[arg-type]


# --- the service role: one tenant per transaction -----------------------------------------------------------------


@pytest.mark.parametrize("tenant", ["acme", "beta"])
def test_the_service_sees_only_the_rows_of_its_tenant_in_every_table(seeded: ScratchDatabase, tenant: str) -> None:
    expected = {
        "documents": 1,
        "chunks": 1,
        "chat_sessions": 3 if tenant == "acme" else 1,
        "chat_messages": 3 if tenant == "acme" else 1,
        "usage_events": 1,
    }
    with as_role(seeded, "rag_app", tenant=tenant) as conn:
        assert {table: count(conn, table) for table in TABLES} == expected
        rows = conn.execute("SELECT content FROM chunks").fetchall()
        assert rows == [(f"secret of {tenant}",)]


@pytest.mark.parametrize("unset", [None, ""])
def test_a_transaction_that_did_not_say_whose_data_it_touches_sees_nothing(
    seeded: ScratchDatabase, unset: str | None
) -> None:
    with as_role(seeded, "rag_app") as conn:
        if unset is not None:
            conn.execute("SELECT set_config('rag.tenant_id', %s, true)", (unset,))
        assert {table: count(conn, table) for table in TABLES} == dict.fromkeys(TABLES, 0)


def test_the_service_cannot_write_rows_for_another_tenant(seeded: ScratchDatabase) -> None:
    statements = {
        "documents": ("INSERT INTO documents (tenant_id, document_id) VALUES ('beta', 'x')", ()),
        "chunks": (
            "INSERT INTO chunks (tenant_id, chunk_id, document_id, ordinal, content, embedding) "
            "VALUES ('beta', %s, 'doc', 1, 'x', array_fill(0.1::real, ARRAY[1536])::vector)",
            ("ck_" + "c" * 24,),
        ),
        "chat_sessions": ("INSERT INTO chat_sessions (tenant_id, session_id) VALUES ('beta', 's_forged_by_acme')", ()),
        "chat_messages": (
            "INSERT INTO chat_messages (tenant_id, session_id, role, content) VALUES ('beta', 's_beta_user1', 'user', 'x')",
            (),
        ),
        "usage_events": (
            "INSERT INTO usage_events (tenant_id, session_id, request_id, created_at, chat_model, embedding_model, "
            "input_tokens, output_tokens, embedding_tokens, cost_micros, cost_estimated, latency_ms, outcome) "
            "VALUES ('beta', 's', 'r', now(), 'c', 'e', 0, 0, 0, 0, false, 0, 'answered')",
            (),
        ),
    }
    for table, (statement, params) in statements.items():
        with as_role(seeded, "rag_app", tenant="acme") as conn, pytest.raises(errors.InsufficientPrivilege):
            attempt(conn, statement, params)
        with seeded.admin() as admin:
            before = {"documents": 2, "chunks": 2, "chat_sessions": 4, "chat_messages": 4, "usage_events": 2}[table]
            assert count(admin, table) == before, table


def test_the_service_can_write_rows_for_its_own_tenant(seeded: ScratchDatabase) -> None:
    with as_role(seeded, "rag_app", tenant="acme") as conn:
        conn.execute("INSERT INTO chat_sessions (tenant_id, session_id) VALUES ('acme', 's_new_session_1')")
        conn.execute(
            "INSERT INTO chat_messages (tenant_id, session_id, role, content) VALUES ('acme', 's_new_session_1', 'user', 'hi')"
        )
        assert count(conn, "chat_sessions") == 4


def test_updates_and_deletes_of_another_tenants_rows_touch_nothing(seeded: ScratchDatabase) -> None:
    with as_role(seeded, "rag_app", tenant="acme") as conn:
        for table in ("documents", "chunks", "chat_sessions", "chat_messages"):
            deleted = conn.execute(
                sql.SQL("DELETE FROM {} WHERE tenant_id = 'beta'").format(sql.Identifier(table))
            ).rowcount
            assert deleted == 0, table
        assert conn.execute("UPDATE chat_sessions SET message_count = 99 WHERE tenant_id = 'beta'").rowcount == 0
        assert conn.execute("UPDATE chunks SET content = 'defaced' WHERE tenant_id = 'beta'").rowcount == 0
    with seeded.admin() as admin:
        assert admin.execute("SELECT content FROM chunks WHERE tenant_id = 'beta'").fetchall() == [("secret of beta",)]
        assert count(admin, "chat_sessions") == 4


def test_a_row_cannot_be_moved_to_another_tenant_by_changing_its_tenant_id(seeded: ScratchDatabase) -> None:
    with as_role(seeded, "rag_app", tenant="acme") as conn, pytest.raises(errors.InsufficientPrivilege):
        attempt(conn, "UPDATE chat_sessions SET tenant_id = 'beta' WHERE session_id = 's_acme_user1'")


@pytest.mark.parametrize("statement", ["UPDATE usage_events SET cost_micros = 0", "DELETE FROM usage_events"])
def test_the_ledger_is_append_only_for_the_service(seeded: ScratchDatabase, statement: str) -> None:
    with as_role(seeded, "rag_app", tenant="acme") as conn, pytest.raises(errors.InsufficientPrivilege):
        attempt(conn, statement)


def test_a_join_across_tables_still_stays_inside_the_tenant(seeded: ScratchDatabase) -> None:
    with as_role(seeded, "rag_app", tenant="beta") as conn:
        rows = conn.execute(
            "SELECT s.session_id, m.content FROM chat_sessions s JOIN chat_messages m USING (tenant_id, session_id)"
        ).fetchall()
        assert rows == [("s_beta_user1", "hello from s_beta_user1")]


def test_the_tenant_setting_does_not_outlive_its_transaction(seeded: ScratchDatabase) -> None:
    with seeded.admin() as conn:
        with conn.transaction():
            conn.execute("SELECT set_config('rag.tenant_id', 'acme', true)")
        assert conn.execute("SELECT rag.current_tenant()").fetchone() == (None,)


# --- Supabase clients: the signed-in user's own conversations, read only -------------------------------------------


def test_a_signed_in_user_reads_only_their_own_sessions_and_messages(seeded: ScratchDatabase) -> None:
    with as_role(seeded, "authenticated", sub=USER_1) as conn:
        sessions = {r[0] for r in conn.execute("SELECT session_id FROM chat_sessions").fetchall()}
        messages = {r[0] for r in conn.execute("SELECT content FROM chat_messages").fetchall()}
    assert sessions == {"s_acme_user1", "s_beta_user1"}
    assert messages == {"hello from s_acme_user1", "hello from s_beta_user1"}
    with as_role(seeded, "authenticated", sub=USER_2) as conn:
        assert {r[0] for r in conn.execute("SELECT session_id FROM chat_sessions").fetchall()} == {"s_acme_user2"}


@pytest.mark.parametrize("sub", [None, "33333333-3333-4333-8333-333333333333"])
def test_a_user_without_conversations_and_a_request_without_claims_see_nothing_not_even_anonymous_sessions(
    seeded: ScratchDatabase, sub: str | None
) -> None:
    with as_role(seeded, "authenticated", sub=sub) as conn:
        assert count(conn, "chat_sessions") == 0 and count(conn, "chat_messages") == 0


def test_a_malformed_user_id_fails_instead_of_returning_rows(seeded: ScratchDatabase) -> None:
    with as_role(seeded, "authenticated", sub="not-a-uuid") as conn, pytest.raises(errors.InvalidTextRepresentation):
        attempt(conn, "SELECT * FROM chat_sessions")


@pytest.mark.parametrize("table", ["documents", "chunks", "usage_events"])
def test_a_signed_in_user_cannot_read_content_embeddings_or_the_ledger(seeded: ScratchDatabase, table: str) -> None:
    with as_role(seeded, "authenticated", sub=USER_1) as conn, pytest.raises(errors.InsufficientPrivilege):
        attempt(conn, f"SELECT * FROM {table}")


@pytest.mark.parametrize(
    "statement",
    [
        "INSERT INTO chat_sessions (tenant_id, session_id, owner_id) VALUES ('acme', 's_planted_1', '" + USER_1 + "')",
        "UPDATE chat_sessions SET owner_id = '" + USER_1 + "'",
        "DELETE FROM chat_sessions",
        "INSERT INTO chat_messages (tenant_id, session_id, role, content) VALUES ('acme', 's_acme_user1', 'assistant', 'forged')",
        "UPDATE chat_messages SET content = 'rewritten'",
        "DELETE FROM chat_messages",
    ],
)
def test_a_signed_in_user_cannot_write_anything(seeded: ScratchDatabase, statement: str) -> None:
    with as_role(seeded, "authenticated", sub=USER_1) as conn, pytest.raises(errors.InsufficientPrivilege):
        attempt(conn, statement)


@pytest.mark.parametrize("table", TABLES)
def test_an_anonymous_client_has_no_access_to_any_table(seeded: ScratchDatabase, table: str) -> None:
    with as_role(seeded, "anon") as conn, pytest.raises(errors.InsufficientPrivilege):
        attempt(conn, f"SELECT * FROM {table}")


def test_a_role_that_is_a_member_of_nothing_has_no_access(seeded: ScratchDatabase) -> None:
    with seeded.admin() as admin:
        admin.execute("CREATE ROLE nobody_rls_probe NOLOGIN")
    with as_role(seeded, "nobody_rls_probe", tenant="acme") as conn, pytest.raises(errors.InsufficientPrivilege):
        attempt(conn, "SELECT * FROM chunks")


# --- the policies themselves ---------------------------------------------------------------------------------------


def test_every_table_has_row_level_security_enabled_and_forced(seeded: ScratchDatabase) -> None:
    with seeded.admin() as conn:
        rows = conn.execute(
            "SELECT relname, relrowsecurity, relforcerowsecurity FROM pg_class "
            "WHERE relname = ANY(%s) AND relkind = 'r' ORDER BY relname",
            (list(TABLES),),
        ).fetchall()
        policies = {r[0] for r in conn.execute("SELECT DISTINCT tablename FROM pg_policies").fetchall()}
    assert [r[0] for r in rows] == sorted(TABLES)
    assert all(r[1] and r[2] for r in rows)
    assert policies == set(TABLES)


def test_forced_row_level_security_applies_to_the_table_owner_too(seeded: ScratchDatabase) -> None:
    with seeded.admin() as admin:
        admin.execute("CREATE ROLE rls_owner_probe NOLOGIN")
        admin.execute("ALTER TABLE chunks OWNER TO rls_owner_probe")
    with as_role(seeded, "rls_owner_probe", tenant="acme") as conn:
        assert count(conn, "chunks") == 0  # an owner is subject to the policies, and none of them names the owner


def test_the_service_role_is_not_a_superuser_and_does_not_bypass_row_level_security(seeded: ScratchDatabase) -> None:
    with seeded.admin() as conn:
        row = conn.execute(
            "SELECT rolsuper, rolbypassrls, rolcreaterole, rolcreatedb FROM pg_roles WHERE rolname IN ('rag_app', %s)",
            (seeded.app_role,),
        ).fetchall()
    assert row == [(False, False, False, False)] * 2
