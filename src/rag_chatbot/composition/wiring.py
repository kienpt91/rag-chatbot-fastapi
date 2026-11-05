"""Builds the real objects from the settings: providers, stores, the chat service and the web application."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

from rag_chatbot.application.chat import ChatService
from rag_chatbot.application.ingest import IngestService
from rag_chatbot.application.ports import ChatProvider, ChunkStore, EmbeddingProvider, SessionStore, UsageLedger
from rag_chatbot.application.retrieval import Retriever
from rag_chatbot.composition.config import Settings
from rag_chatbot.domain.errors import ConfigurationError
from rag_chatbot.domain.pricing import PriceTable, parse_price_table
from rag_chatbot.domain.tenants import TenantDirectory, parse_tenants
from rag_chatbot.infrastructure.http.app import AppDeps
from rag_chatbot.infrastructure.memory import InMemoryChunkStore, InMemorySessionStore, InMemoryUsageLedger
from rag_chatbot.infrastructure.postgres.chunks import PostgresChunkStore
from rag_chatbot.infrastructure.postgres.conversations import PostgresSessionStore, PostgresUsageLedger
from rag_chatbot.infrastructure.postgres.health import schema_problems
from rag_chatbot.infrastructure.postgres.pool import Pool
from rag_chatbot.infrastructure.providers.fake import FakeChatProvider, FakeEmbeddingProvider
from rag_chatbot.infrastructure.providers.http import HttpApiClient, HttpChatProvider, HttpEmbeddingProvider
from rag_chatbot.infrastructure.system import RandomIds, SystemClock

log = logging.getLogger("rag_chatbot.wiring")
MAX_CONFIG_FILE_BYTES = 1_000_000


def read_json_file(path: Path, what: str) -> object:
    try:
        if path.stat().st_size > MAX_CONFIG_FILE_BYTES:
            raise ConfigurationError(f"{what} is larger than {MAX_CONFIG_FILE_BYTES} bytes")
        return json.loads(path.read_text(encoding="utf-8"))
    except OSError:
        raise ConfigurationError(f"{what} cannot be read: {path}") from None
    except ValueError:
        raise ConfigurationError(f"{what} is not valid JSON: {path}") from None


def load_tenants(settings: Settings) -> TenantDirectory:
    settings.require("tenants_file")
    assert settings.tenants_file is not None
    return parse_tenants(read_json_file(settings.tenants_file, "the tenants file"))


def load_prices(settings: Settings) -> PriceTable:
    settings.require("price_table_file")
    assert settings.price_table_file is not None
    return parse_price_table(read_json_file(settings.price_table_file, "the price table"))


@dataclass
class Providers:
    embedder: EmbeddingProvider
    chat: ChatProvider


def build_providers(settings: Settings) -> Providers:
    settings.require("provider")
    if settings.provider == "fake":
        log.warning("provider=fake: answers come from a deterministic stand-in, not from a language model")
        return Providers(
            FakeEmbeddingProvider(settings.embedding_dimensions, settings.embedding_model),
            FakeChatProvider(gullible=settings.fake_gullible, model=settings.chat_model),
        )
    assert settings.api_key is not None
    client = HttpApiClient(settings.api_base_url, settings.api_key, timeout_s=settings.provider_timeout_s)
    return Providers(
        HttpEmbeddingProvider(
            client,
            settings.embedding_model,
            dimensions=settings.embedding_dimensions,
            send_dimensions=settings.send_dimensions,
        ),
        HttpChatProvider(client, settings.chat_model),
    )


def check_prices(prices: PriceTable, settings: Settings) -> None:
    """Both configured models must be priced: an unpriced call would be free in the accounting and in the budget."""
    prices.price_of(settings.chat_model)
    prices.price_of(settings.embedding_model)


@dataclass
class Stores:
    chunks: ChunkStore
    sessions: SessionStore
    ledger: UsageLedger


def postgres_stores(pool: Pool, settings: Settings) -> Stores:
    return Stores(
        PostgresChunkStore(pool, dimensions=settings.embedding_dimensions),
        PostgresSessionStore(pool),
        PostgresUsageLedger(pool),
    )


def memory_stores(settings: Settings) -> Stores:
    return Stores(
        InMemoryChunkStore(dimensions=settings.embedding_dimensions), InMemorySessionStore(), InMemoryUsageLedger()
    )


def build_chat_service(settings: Settings, providers: Providers, stores: Stores, prices: PriceTable) -> ChatService:
    return ChatService(
        retriever=Retriever(stores.chunks, providers.embedder, settings.chat),
        chat=providers.chat,
        sessions=stores.sessions,
        ledger=stores.ledger,
        prices=prices,
        embedding_model=providers.embedder.model,
        clock=SystemClock(),
        ids=RandomIds(),
        settings=settings.chat,
    )


def build_ingest_service(settings: Settings, providers: Providers, stores: Stores, prices: PriceTable) -> IngestService:
    return IngestService(stores.chunks, providers.embedder, prices, settings.ingest)


async def build_app_deps(settings: Settings, pool: Pool) -> AppDeps:
    """Everything the web application needs, with a database that has been checked against this version's schema."""
    tenants = load_tenants(settings)
    prices = load_prices(settings)
    check_prices(prices, settings)
    providers = build_providers(settings)
    problems = await schema_problems(pool, migrations=settings.migrations_dir, dimensions=settings.embedding_dimensions)
    if problems:
        raise ConfigurationError(
            "the database does not match this version (" + ", ".join(problems) + "): run `rag-chatbot migrate`, and "
            "check that RAG_EMBEDDING_DIMENSIONS equals the size of the embedding column"
        )
    stores = postgres_stores(pool, settings)

    async def readiness() -> list[str]:
        return await schema_problems(pool, migrations=settings.migrations_dir, dimensions=settings.embedding_dimensions)

    return AppDeps(
        chat=build_chat_service(settings, providers, stores, prices),
        directory=tenants,
        clock=SystemClock(),
        ids=RandomIds(),
        settings=settings.http,
        readiness=readiness,
    )
