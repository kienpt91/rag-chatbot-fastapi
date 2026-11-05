"""A small migration runner for the plain SQL files in `migrations/`.

Files are named `NNNN_description.sql` and applied in order, each in its own transaction together with the row that
records it in `schema_migrations`. The checksum of every applied file is kept: editing a migration that has already
been applied is an error (write a new migration instead). A session-level advisory lock keeps two runners from
applying the same migration at once.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

import psycopg

from rag_chatbot.domain.errors import ConfigurationError, StorageError

NAME = re.compile(r"^(\d{4})_[a-z0-9_]+\.sql$")
LOCK_KEY = 7_203_391_855_144  # an arbitrary constant: all runners of this service take the same lock


@dataclass(frozen=True, slots=True)
class Migration:
    version: str  # the file name without .sql
    checksum: str
    sql: str


def load_migrations(directory: Path) -> list[Migration]:
    if not directory.is_dir():
        raise ConfigurationError(f"migrations directory not found: {directory}")
    found: list[Migration] = []
    for path in sorted(directory.glob("*.sql")):
        match = NAME.match(path.name)
        if match is None:
            raise ConfigurationError(f"migration file name must look like 0001_name.sql: {path.name}")
        sql = path.read_text(encoding="utf-8")
        found.append(Migration(path.stem, hashlib.sha256(sql.encode()).hexdigest(), sql))
    numbers = [int(m.version[:4]) for m in found]
    if not found or numbers != list(range(1, len(found) + 1)):
        raise ConfigurationError("migrations must be numbered 0001, 0002, ... without gaps")
    return found


def expected_version(directory: Path) -> str:
    return load_migrations(directory)[-1].version


def migrate(dsn: str, directory: Path) -> list[str]:
    """Apply what is missing. Returns the versions applied by this call. Connect as the role that owns the schema."""
    migrations = load_migrations(directory)
    applied: list[str] = []
    try:
        with psycopg.connect(dsn, autocommit=True, application_name="rag-chatbot-migrate") as conn:
            conn.execute("SELECT pg_advisory_lock(%s)", (LOCK_KEY,))
            conn.execute(
                "CREATE TABLE IF NOT EXISTS schema_migrations ("
                "version text PRIMARY KEY, checksum text NOT NULL, applied_at timestamptz NOT NULL DEFAULT now())"
            )
            done: dict[str, str] = {
                str(version): str(checksum)
                for version, checksum in conn.execute("SELECT version, checksum FROM schema_migrations").fetchall()
            }
            for migration in migrations:
                if migration.version in done:
                    if done[migration.version] != migration.checksum:
                        raise ConfigurationError(
                            f"migration {migration.version} was changed after it was applied; add a new migration"
                        )
                    continue
                with conn.transaction():
                    conn.execute(migration.sql.encode())  # bytes: sent as is, so '%' in SQL is never a placeholder
                    conn.execute(
                        "INSERT INTO schema_migrations (version, checksum) VALUES (%s, %s)",
                        (migration.version, migration.checksum),
                    )
                applied.append(migration.version)
            # the ledger is created here, outside the migrations: on Supabase, default privileges would hand it to the
            # API roles, which could then forge a row and make the service refuse to start
            conn.execute(
                "DO $$ BEGIN REVOKE ALL ON schema_migrations FROM PUBLIC; "
                "IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'anon') THEN REVOKE ALL ON schema_migrations FROM anon; END IF; "
                "IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'authenticated') THEN "
                "REVOKE ALL ON schema_migrations FROM authenticated; END IF; "
                "IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'rag_app') THEN "
                "GRANT SELECT ON schema_migrations TO rag_app; END IF; END $$"
            )
    except psycopg.Error as error:
        raise StorageError(f"migration failed: {error.__class__.__name__}: {error}") from error
    return applied
