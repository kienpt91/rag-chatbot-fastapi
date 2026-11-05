"""The configuration of the service, read from environment variables, validated once, with clear errors.

Only the composition layer reads the environment. Secrets (the API key, the database URL with its password) may be
given as a file instead (`RAG_API_KEY_FILE`, `RAG_DATABASE_URL_FILE`, ...), which is how container secrets arrive;
nothing in an error message or a `repr` ever contains one.
"""

from __future__ import annotations

import math
import secrets
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

from rag_chatbot.application.settings import DEFAULT_REFUSAL, ChatSettings, IngestSettings
from rag_chatbot.domain.chunking import ChunkingConfig
from rag_chatbot.domain.errors import ConfigurationError
from rag_chatbot.domain.pricing import usd_to_micros
from rag_chatbot.domain.redaction import ALL_KINDS
from rag_chatbot.infrastructure.http.settings import HttpSettings

PREFIX = "RAG_"
KNOWN_REDACTION = frozenset({"emails", "phones", "cards"})
DEFAULT_MIGRATIONS = Path(__file__).resolve().parents[3] / "migrations"


@dataclass(frozen=True)
class Settings:
    database_url: str | None = field(default=None, repr=False)
    migrate_database_url: str | None = field(default=None, repr=False)
    tenants_file: Path | None = None
    price_table_file: Path | None = None
    provider: str = ""  # "http" or "fake"
    api_base_url: str = ""
    api_key: str | None = field(default=None, repr=False)
    chat_model: str = ""
    embedding_model: str = ""
    embedding_dimensions: int = 1536
    send_dimensions: bool = False
    provider_timeout_s: float = 30.0
    fake_gullible: bool = False
    chat: ChatSettings = field(default_factory=ChatSettings)
    ingest: IngestSettings = field(default_factory=IngestSettings)
    http: HttpSettings = field(default_factory=HttpSettings)
    host: str = "127.0.0.1"
    port: int = 8000
    log_level: str = "INFO"
    db_pool_max: int = 10
    db_statement_timeout_ms: int = 10_000
    migrations_dir: Path = DEFAULT_MIGRATIONS

    def require(self, *names: str) -> None:
        """Raises `ConfigurationError` naming the environment variables that are missing for what is about to run."""
        needed: dict[str, tuple[str, object]] = {
            "database_url": ("RAG_DATABASE_URL", self.database_url),
            "tenants_file": ("RAG_TENANTS_FILE", self.tenants_file),
            "price_table_file": ("RAG_PRICE_TABLE_FILE", self.price_table_file),
            "provider": ("RAG_PROVIDER", self.provider),
        }
        missing = [needed[n][0] for n in names if not needed[n][1]]
        if self.provider == "http" and "provider" in names:
            for env, value in (
                ("RAG_API_BASE_URL", self.api_base_url),
                ("RAG_API_KEY", self.api_key),
                ("RAG_CHAT_MODEL", self.chat_model),
                ("RAG_EMBEDDING_MODEL", self.embedding_model),
            ):
                if not value:
                    missing.append(env)
        if missing:
            raise ConfigurationError("missing configuration: " + ", ".join(missing))


class _Reader:
    def __init__(self, env: Mapping[str, str]) -> None:
        self._env = env

    def text(self, name: str, default: str = "") -> str:
        return self._env.get(PREFIX + name, default).strip()

    def secret(self, name: str) -> str | None:
        """The value of RAG_<name>, or the contents of the file named by RAG_<name>_FILE (not both)."""
        direct, path = self._env.get(PREFIX + name), self._env.get(f"{PREFIX}{name}_FILE")
        if direct and path:
            raise ConfigurationError(f"set only one of {PREFIX}{name} and {PREFIX}{name}_FILE")
        if path:
            try:
                content = Path(path).read_text(encoding="utf-8").strip()
            except OSError:
                raise ConfigurationError(f"{PREFIX}{name}_FILE cannot be read") from None
            return content or None
        return direct.strip() if direct and direct.strip() else None

    def integer(self, name: str, default: int, *, low: int = 0, high: int = 10**9) -> int:
        raw = self.text(name)
        if not raw:
            return default
        try:
            value = int(raw)
        except ValueError:
            raise ConfigurationError(f"{PREFIX}{name} must be an integer") from None
        if not low <= value <= high:
            raise ConfigurationError(f"{PREFIX}{name} must be between {low} and {high}")
        return value

    def number(self, name: str, default: float, *, low: float = 0.0, high: float = 1e9) -> float:
        raw = self.text(name)
        if not raw:
            return default
        try:
            value = float(raw)
        except ValueError:
            raise ConfigurationError(f"{PREFIX}{name} must be a number") from None
        if math.isnan(value) or not low <= value <= high:
            raise ConfigurationError(f"{PREFIX}{name} must be between {low} and {high}")
        return value

    def flag(self, name: str, default: bool = False) -> bool:
        raw = self.text(name).lower()
        if not raw:
            return default
        if raw in ("1", "true", "yes", "on"):
            return True
        if raw in ("0", "false", "no", "off"):
            return False
        raise ConfigurationError(f"{PREFIX}{name} must be true or false")

    def path(self, name: str) -> Path | None:
        raw = self.text(name)
        return Path(raw) if raw else None


def _redaction(raw: str) -> frozenset[str]:
    if not raw:
        return ALL_KINDS
    if raw.lower() == "none":
        return frozenset({"secrets"})
    kinds = {k.strip().lower() for k in raw.split(",") if k.strip()}
    unknown = kinds - KNOWN_REDACTION
    if unknown:
        raise ConfigurationError(
            f"RAG_REDACT: unknown kind(s) {', '.join(sorted(unknown))} (emails, phones, cards, none)"
        )
    return frozenset(kinds | {"secrets"})


def parse_settings(env: Mapping[str, str]) -> Settings:
    read = _Reader(env)
    provider = read.text("PROVIDER").lower()
    if provider not in ("", "http", "fake"):
        raise ConfigurationError("RAG_PROVIDER must be http or fake")
    redaction = _redaction(read.text("REDACT"))
    budget = read.text("DAILY_BUDGET")
    threshold = read.number("CONFIDENCE_THRESHOLD", 0.30, low=0.0, high=1.0)
    chat = ChatSettings(
        confidence_threshold=threshold,
        candidates_per_search=read.integer("CANDIDATES_PER_SEARCH", 12, low=1, high=200),
        context_chunks=read.integer("CONTEXT_CHUNKS", 5, low=1, high=50),
        context_chars=read.integer("CONTEXT_CHARS", 6000, low=500, high=100_000),
        context_relative_cutoff=read.number("CONTEXT_RELATIVE_CUTOFF", 0.5, low=0.0, high=1.0),
        history_turns=read.integer("HISTORY_TURNS", 6, low=1, high=50),
        history_token_budget=read.integer("HISTORY_TOKEN_BUDGET", 800, low=50, high=20_000),
        max_output_tokens=read.integer("MAX_OUTPUT_TOKENS", 400, low=16, high=8000),
        max_sentences=read.integer("MAX_SENTENCES", 4, low=1, high=20),
        max_messages_per_session=read.integer("MAX_MESSAGES_PER_SESSION", 100, low=2, high=10_000),
        redaction_kinds=redaction,
        blocked_phrases=tuple(p.strip() for p in read.text("BLOCKED_PHRASES").split(",") if p.strip()),
        refusal_message=read.text("REFUSAL_MESSAGE", DEFAULT_REFUSAL)[:500] or DEFAULT_REFUSAL,
        default_daily_budget_micros=usd_to_micros(budget, "RAG_DAILY_BUDGET") if budget else None,
        canary="CANARY-" + secrets.token_hex(8),  # new at every start: a leak of it can only come from this process
    )
    ingest = IngestSettings(
        chunking=ChunkingConfig(
            target_tokens=read.integer("CHUNK_TARGET_TOKENS", 220, low=20, high=2000),
            max_tokens=read.integer("CHUNK_MAX_TOKENS", 350, low=20, high=4000),
            overlap_tokens=read.integer("CHUNK_OVERLAP_TOKENS", 40, low=0, high=1000),
            min_tokens=read.integer("CHUNK_MIN_TOKENS", 40, low=1, high=1000),
        ),
        embed_batch_size=read.integer("EMBED_BATCH_SIZE", 32, low=1, high=256),
        max_chunks_per_document=read.integer("MAX_CHUNKS_PER_DOCUMENT", 300, low=1, high=10_000),
    )
    http = HttpSettings(
        max_body_bytes=read.integer("MAX_BODY_BYTES", 16 * 1024, low=256, high=1_000_000),
        ip_burst=read.number("IP_BURST", 20.0, low=1.0, high=10_000.0),
        ip_per_minute=read.number("IP_PER_MINUTE", 30.0, low=0.1, high=100_000.0),
        session_burst=read.number("SESSION_BURST", 5.0, low=1.0, high=10_000.0),
        session_per_minute=read.number("SESSION_PER_MINUTE", 6.0, low=0.1, high=100_000.0),
        max_tracked_keys=read.integer("MAX_TRACKED_KEYS", 10_000, low=1, high=10_000_000),
        trusted_proxy_hops=read.integer("TRUSTED_PROXY_HOPS", 0, low=0, high=5),
        request_timeout_s=read.number("REQUEST_TIMEOUT_S", 60.0, low=1.0, high=600.0),
        max_token_ttl_s=read.integer("MAX_TOKEN_TTL_S", 3600, low=1, high=86_400),
        enable_docs=read.flag("ENABLE_DOCS"),
        log_ip_salt=secrets.token_hex(
            8
        ),  # new at every start: the hashes in the logs cannot be matched across restarts
    )
    migrations = read.path("MIGRATIONS_DIR") or DEFAULT_MIGRATIONS
    return Settings(
        database_url=read.secret("DATABASE_URL"),
        migrate_database_url=read.secret("MIGRATE_DATABASE_URL"),
        tenants_file=read.path("TENANTS_FILE"),
        price_table_file=read.path("PRICE_TABLE_FILE"),
        provider=provider,
        api_base_url=read.text("API_BASE_URL"),
        api_key=read.secret("API_KEY"),
        chat_model=read.text("CHAT_MODEL", "fake-chat-1" if provider == "fake" else ""),
        embedding_model=read.text("EMBEDDING_MODEL", "fake-embedding-1" if provider == "fake" else ""),
        embedding_dimensions=read.integer("EMBEDDING_DIMENSIONS", 1536, low=8, high=2000),
        send_dimensions=read.flag("EMBEDDING_SEND_DIMENSIONS"),
        provider_timeout_s=read.number("PROVIDER_TIMEOUT_S", 30.0, low=1.0, high=300.0),
        fake_gullible=read.flag("FAKE_GULLIBLE"),
        chat=chat,
        ingest=ingest,
        http=http,
        host=read.text("HOST", "127.0.0.1"),
        port=read.integer("PORT", 8000, low=1, high=65535),
        log_level=read.text("LOG_LEVEL", "INFO").upper(),
        db_pool_max=read.integer("DB_POOL_MAX", 10, low=1, high=200),
        db_statement_timeout_ms=read.integer("DB_STATEMENT_TIMEOUT_MS", 10_000, low=100, high=600_000),
        migrations_dir=migrations,
    )
