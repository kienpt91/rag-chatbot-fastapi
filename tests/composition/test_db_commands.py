from __future__ import annotations

import asyncio
import io
import json
import shutil
import threading
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import psycopg
import pytest
from psycopg import sql
from psycopg.conninfo import make_conninfo

from rag_chatbot.composition.main import main
from rag_chatbot.infrastructure.postgres.conversations import PostgresSessionStore
from support.database import MIGRATIONS, ScratchDatabase, app_pool, server_dsn

pytestmark = pytest.mark.db
ROOT = Path(__file__).resolve().parents[2]
SITE = ROOT / "examples" / "site"


def environment(db: ScratchDatabase, tmp_path: Path) -> dict[str, str]:
    tenants = tmp_path / "tenants.json"
    tenants.write_text(json.dumps({"tenants": [{"id": "acme", "name": "Acme"}, {"id": "beta", "name": "Beta"}]}))
    return {
        "RAG_DATABASE_URL": db.app_dsn,
        "RAG_TENANTS_FILE": str(tenants),
        "RAG_PRICE_TABLE_FILE": str(ROOT / "examples" / "prices.example.json"),
        "RAG_PROVIDER": "fake",
    }


def run(env: dict[str, str], *argv: str) -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    return main(list(argv), env, out, err), out.getvalue(), err.getvalue()


def count(db: ScratchDatabase, query: str, *params: object) -> int:
    with db.admin() as conn:
        row = conn.execute(query.encode(), params or None).fetchone()  # type: ignore[arg-type]
    assert row is not None and isinstance(row[0], int)
    return row[0]


# --- migrate ----------------------------------------------------------------------------------------------------------


def test_migrate_applies_everything_to_an_empty_database_and_then_reports_up_to_date() -> None:
    base = server_dsn()
    name = "rag_cmd_migrate"
    with psycopg.connect(base, autocommit=True) as admin:
        admin.execute(sql.SQL("DROP DATABASE IF EXISTS {}").format(sql.Identifier(name)))
        admin.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(name)))
    try:
        env = {"RAG_MIGRATE_DATABASE_URL": make_conninfo(base, dbname=name), "RAG_MIGRATIONS_DIR": str(MIGRATIONS)}
        code, out, _ = run(env, "migrate")
        assert code == 0 and "0001_extensions_and_roles" in out and "0003_row_level_security" in out
        code, out, _ = run(env, "migrate")
        assert code == 0 and "up to date" in out
    finally:
        with psycopg.connect(base, autocommit=True) as admin:
            admin.execute(sql.SQL("DROP DATABASE IF EXISTS {} WITH (FORCE)").format(sql.Identifier(name)))


def test_migrate_without_a_database_url_is_exit_2() -> None:
    assert run({}, "migrate")[0] == 2


# --- ingest from a folder -----------------------------------------------------------------------------------------------


def test_ingesting_the_sample_site_stores_chunks_and_a_second_run_embeds_nothing(
    scratch_db: ScratchDatabase, tmp_path: Path
) -> None:
    env = environment(scratch_db, tmp_path)
    code, out, err = run(
        env,
        "ingest",
        "--tenant",
        "acme",
        "--folder",
        str(SITE),
        "--base-url",
        "https://www.maple-street-bakery.example",
    )
    assert code == 0, out + err
    assert "documents       9 indexed" in out and "skipped         noindex=1" in out and "reused unchanged" in out
    chunks = count(scratch_db, "SELECT count(*) FROM chunks WHERE tenant_id = 'acme'")
    assert chunks > 9 and count(scratch_db, "SELECT count(*) FROM chunks WHERE tenant_id = 'beta'") == 0
    assert count(scratch_db, "SELECT count(*) FROM chunks WHERE content ILIKE '%safe code%'") == 0  # the noindex page
    assert (
        count(
            scratch_db,
            "SELECT count(*) FROM chunks WHERE content ILIKE '%hidden attribute payload%' OR content ILIKE '%comment payload%'",
        )
        == 0
    )
    assert (
        count(
            scratch_db,
            "SELECT count(*) FROM documents WHERE tenant_id = 'acme' AND url = 'https://www.maple-street-bakery.example/hours'",
        )
        == 1
    )
    again = run(
        env,
        "ingest",
        "--tenant",
        "acme",
        "--folder",
        str(SITE),
        "--base-url",
        "https://www.maple-street-bakery.example",
    )
    assert again[0] == 0 and f"({0} embedded, {chunks} reused unchanged)" in again[1]
    assert count(scratch_db, "SELECT count(*) FROM chunks WHERE tenant_id = 'acme'") == chunks


def test_exclude_keeps_user_written_pages_out_of_the_knowledge_base(
    scratch_db: ScratchDatabase, tmp_path: Path
) -> None:
    env = environment(scratch_db, tmp_path)
    code, out, _ = run(
        env,
        "ingest",
        "--tenant",
        "acme",
        "--folder",
        str(SITE),
        "--exclude",
        "community-notes",
        "--exclude",
        "policies/*",
    )
    assert code == 0 and "excluded=2" in out
    assert (
        count(scratch_db, "SELECT count(*) FROM documents WHERE document_id IN ('community-notes', 'policies/returns')")
        == 0
    )


def test_prune_removes_pages_that_left_the_site_but_refuses_after_an_incomplete_run(
    scratch_db: ScratchDatabase, tmp_path: Path
) -> None:
    env = environment(scratch_db, tmp_path)
    site = tmp_path / "site"
    shutil.copytree(SITE, site)
    assert run(env, "ingest", "--tenant", "acme", "--folder", str(site))[0] == 0
    (site / "faq.md").unlink()
    code, out, _ = run(env, "ingest", "--tenant", "acme", "--folder", str(site), "--prune")
    assert code == 0 and "pruned 1 document(s)" in out
    assert count(scratch_db, "SELECT count(*) FROM documents WHERE document_id = 'faq'") == 0
    # a page that was too big to read means the run did not see the whole site: nothing is deleted
    (site / "huge.txt").write_bytes(b"x" * 2_100_000)
    (site / "menu.md").unlink()
    code, _, err = run(env, "ingest", "--tenant", "acme", "--folder", str(site), "--prune")
    assert code == 1 and "not pruned" in err and "too_large" in err
    assert count(scratch_db, "SELECT count(*) FROM documents WHERE document_id = 'menu'") == 1


def test_prune_never_empties_a_tenant_when_the_source_has_nothing(scratch_db: ScratchDatabase, tmp_path: Path) -> None:
    env = environment(scratch_db, tmp_path)
    assert run(env, "ingest", "--tenant", "acme", "--folder", str(SITE))[0] == 0
    empty = tmp_path / "empty"
    empty.mkdir()
    code, _, err = run(env, "ingest", "--tenant", "acme", "--folder", str(empty), "--prune")
    assert code == 1 and "indexed nothing" in err
    assert count(scratch_db, "SELECT count(*) FROM documents WHERE tenant_id = 'acme'") > 0


@pytest.mark.parametrize(
    "argv",
    [
        ["--tenant", "nobody", "--folder", str(SITE)],
        ["--tenant", "acme"],
        ["--tenant", "acme", "--folder", str(SITE), "--crawl", "https://x.example/"],
        ["--tenant", "acme", "--crawl", "https://x.example/"],  # a crawl needs --allow-host
        ["--tenant", "acme", "--folder", "/nonexistent/folder"],
    ],
)
def test_ingest_refuses_bad_arguments_before_touching_anything(
    scratch_db: ScratchDatabase, tmp_path: Path, argv: list[str]
) -> None:
    code, _, _ = run(environment(scratch_db, tmp_path), "ingest", *argv)
    assert code in (2, 3)
    assert count(scratch_db, "SELECT count(*) FROM documents") == 0


# --- ingest by crawling a (local) site ------------------------------------------------------------------------------------


@pytest.fixture
def local_site() -> Iterator[int]:
    pages = {
        "/": b"<html><head><title>Home</title></head><body><nav><a href='/about'>About</a> <a href='/private/x'>x</a> "
        b"<a href='http://other.example/'>out</a></nav><main><h1>Home</h1><p>We bake bread every morning.</p></main></body></html>",
        "/about": b"<html><head><title>About</title></head><body><main><p>The bakery was founded in 1998.</p></main></body></html>",
        "/private/x": b"<html><body><main><p>Private page that robots.txt forbids.</p></main></body></html>",
        "/robots.txt": b"User-agent: *\nDisallow: /private/\n",
    }

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: object) -> None:
            pass

        def do_GET(self) -> None:
            body = pages.get(self.path)
            self.send_response(200 if body else 404)
            self.send_header("Content-Type", "text/plain" if self.path == "/robots.txt" else "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body or b"")))
            self.end_headers()
            self.wfile.write(body or b"")

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=lambda: server.serve_forever(poll_interval=0.02), daemon=True)
    thread.start()
    try:
        yield server.server_address[1]
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_a_crawl_honours_robots_and_the_host_list_and_indexes_what_it_may(
    scratch_db: ScratchDatabase, tmp_path: Path, local_site: int
) -> None:
    env = environment(scratch_db, tmp_path)
    argv = [
        "ingest", "--tenant", "acme", "--crawl", f"http://127.0.0.1:{local_site}/", "--allow-host", "127.0.0.1",
        "--allow-http", "--allow-port", str(local_site), "--allow-private-addresses", "--delay", "0",
    ]  # fmt: skip
    code, out, err = run(env, *argv)
    assert code == 0, out + err
    assert "documents       2 indexed" in out and "source:robots=1" in out and "source:host=1" in out
    with scratch_db.admin() as conn:
        ids = {r[0] for r in conn.execute("SELECT document_id FROM documents WHERE tenant_id = 'acme'")}
    assert ids == {f"127.0.0.1:{local_site}", f"127.0.0.1:{local_site}/about"}
    assert count(scratch_db, "SELECT count(*) FROM chunks WHERE content ILIKE '%Private page%'") == 0


def test_a_crawl_refuses_private_addresses_unless_told_otherwise(
    scratch_db: ScratchDatabase, tmp_path: Path, local_site: int
) -> None:
    env = environment(scratch_db, tmp_path)
    _, out, _ = run(
        env, "ingest", "--tenant", "acme", "--crawl", f"http://127.0.0.1:{local_site}/", "--allow-host", "127.0.0.1",
        "--allow-http", "--allow-port", str(local_site), "--delay", "0",
    )  # fmt: skip
    assert "indexed" in out and "documents       0 indexed" in out
    assert count(scratch_db, "SELECT count(*) FROM documents") == 0


# --- purge --------------------------------------------------------------------------------------------------------------


def test_purge_deletes_conversations_older_than_the_given_days_for_one_or_all_tenants(
    scratch_db: ScratchDatabase, tmp_path: Path
) -> None:
    env = environment(scratch_db, tmp_path)

    async def seed() -> None:
        async with app_pool(scratch_db) as pool:
            sessions = PostgresSessionStore(pool)
            for tenant in ("acme", "beta"):
                await sessions.create_session(tenant, "s_old_123456", None)
                await sessions.create_session(tenant, "s_new_123456", None)
        with scratch_db.admin() as conn:
            conn.execute(
                "UPDATE chat_sessions SET last_active_at = %s WHERE session_id = 's_old_123456'",
                (datetime.now(UTC) - timedelta(days=45),),
            )

    asyncio.run(seed())
    code, out, _ = run(env, "purge", "--older-than-days", "30", "--tenant", "acme")
    assert code == 0 and "acme: removed 1 conversation(s)" in out and "beta" not in out
    assert count(scratch_db, "SELECT count(*) FROM chat_sessions") == 3
    code, out, _ = run(env, "purge", "--older-than-days", "30")
    assert code == 0 and "acme: removed 0" in out and "beta: removed 1" in out
    assert count(scratch_db, "SELECT count(*) FROM chat_sessions") == 2
    assert run(env, "purge", "--older-than-days", "30", "--tenant", "nobody")[0] == 2
    assert run({}, "purge", "--older-than-days", "30")[0] == 2
