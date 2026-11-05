from __future__ import annotations

from pathlib import Path

import pytest

from rag_chatbot.composition.config import parse_settings
from rag_chatbot.domain.errors import ConfigurationError


def test_an_empty_environment_gives_safe_defaults_that_still_need_the_required_settings() -> None:
    settings = parse_settings({})
    assert settings.provider == "" and settings.host == "127.0.0.1" and settings.port == 8000
    assert settings.chat.confidence_threshold == 0.30 and settings.chat.redaction_kinds == {
        "emails",
        "phones",
        "cards",
        "secrets",
    }
    assert (
        settings.http.max_body_bytes == 16384
        and settings.http.enable_docs is False
        and settings.http.trusted_proxy_hops == 0
    )
    with pytest.raises(
        ConfigurationError, match="RAG_DATABASE_URL, RAG_TENANTS_FILE, RAG_PRICE_TABLE_FILE, RAG_PROVIDER"
    ):
        settings.require("database_url", "tenants_file", "price_table_file", "provider")


def test_the_canary_and_the_log_salt_are_new_at_every_start() -> None:
    first, second = parse_settings({}), parse_settings({})
    assert first.chat.canary != second.chat.canary and first.chat.canary.startswith("CANARY-")
    assert first.http.log_ip_salt != second.http.log_ip_salt


def test_the_fake_provider_needs_nothing_else_and_defaults_its_model_names() -> None:
    settings = parse_settings({"RAG_PROVIDER": "fake"})
    settings.require("provider")
    assert settings.chat_model == "fake-chat-1" and settings.embedding_model == "fake-embedding-1"


def test_the_http_provider_lists_every_missing_variable() -> None:
    settings = parse_settings({"RAG_PROVIDER": "http"})
    with pytest.raises(ConfigurationError) as caught:
        settings.require("provider")
    for name in ("RAG_API_BASE_URL", "RAG_API_KEY", "RAG_CHAT_MODEL", "RAG_EMBEDDING_MODEL"):
        assert name in caught.value.detail


def test_every_tunable_is_read_and_typed() -> None:
    settings = parse_settings(
        {
            "RAG_PROVIDER": "HTTP",
            "RAG_API_BASE_URL": "https://api.example.com/v1",
            "RAG_API_KEY": "k" * 20,
            "RAG_CHAT_MODEL": "c",
            "RAG_EMBEDDING_MODEL": "e",
            "RAG_EMBEDDING_DIMENSIONS": "768",
            "RAG_EMBEDDING_SEND_DIMENSIONS": "yes",
            "RAG_CONFIDENCE_THRESHOLD": "0.42",
            "RAG_CONTEXT_CHUNKS": "3",
            "RAG_CONTEXT_RELATIVE_CUTOFF": "0.7",
            "RAG_REDACT": "emails, cards",
            "RAG_BLOCKED_PHRASES": "bitcoin, , lottery",
            "RAG_REFUSAL_MESSAGE": "Call us.",
            "RAG_DAILY_BUDGET": "2.50",
            "RAG_TRUSTED_PROXY_HOPS": "1",
            "RAG_IP_PER_MINUTE": "12.5",
            "RAG_ENABLE_DOCS": "true",
            "RAG_HOST": "0.0.0.0",
            "RAG_PORT": "9000",
            "RAG_LOG_LEVEL": "debug",
            "RAG_CHUNK_TARGET_TOKENS": "100",
            "RAG_CHUNK_MAX_TOKENS": "200",
            "RAG_MIGRATIONS_DIR": "/x/migrations",
        }
    )
    assert settings.provider == "http" and settings.embedding_dimensions == 768 and settings.send_dimensions
    assert settings.chat.confidence_threshold == 0.42 and settings.chat.context_chunks == 3
    assert settings.chat.context_relative_cutoff == 0.7 and settings.chat.redaction_kinds == {
        "emails",
        "cards",
        "secrets",
    }
    assert settings.chat.blocked_phrases == ("bitcoin", "lottery") and settings.chat.refusal_message == "Call us."
    assert settings.chat.default_daily_budget_micros == 2_500_000
    assert settings.http.trusted_proxy_hops == 1 and settings.http.ip_per_minute == 12.5 and settings.http.enable_docs
    assert settings.host == "0.0.0.0" and settings.port == 9000 and settings.log_level == "DEBUG"
    assert settings.ingest.chunking.target_tokens == 100 and settings.migrations_dir == Path("/x/migrations")
    settings.require("provider")


def test_redaction_none_still_redacts_secrets() -> None:
    assert parse_settings({"RAG_REDACT": "none"}).chat.redaction_kinds == {"secrets"}


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("RAG_PROVIDER", "other"),
        ("RAG_PORT", "abc"),
        ("RAG_PORT", "0"),
        ("RAG_PORT", "70000"),
        ("RAG_CONFIDENCE_THRESHOLD", "1.5"),
        ("RAG_CONFIDENCE_THRESHOLD", "nan"),
        ("RAG_CONFIDENCE_THRESHOLD", "x"),
        ("RAG_ENABLE_DOCS", "maybe"),
        ("RAG_REDACT", "emails,faces"),
        ("RAG_DAILY_BUDGET", "lots"),
        ("RAG_DAILY_BUDGET", "-1"),
        ("RAG_TRUSTED_PROXY_HOPS", "9"),
        ("RAG_CONTEXT_CHUNKS", "0"),
        ("RAG_EMBEDDING_DIMENSIONS", "4"),
        ("RAG_CHUNK_OVERLAP_TOKENS", "500"),
        ("RAG_MAX_BODY_BYTES", "10"),
    ],
)
def test_bad_values_are_refused_with_the_name_of_the_variable_and_never_the_value(name: str, value: str) -> None:
    with pytest.raises(ConfigurationError) as caught:
        parse_settings({name: value})
    assert (
        name.removeprefix("RAG_") in caught.value.detail
        or name in caught.value.detail
        or "chunking" in caught.value.detail
    )


def test_a_secret_can_come_from_a_file_but_not_from_both(tmp_path: Path) -> None:
    key_file = tmp_path / "key"
    key_file.write_text("  secret-key-from-a-file\n")
    assert parse_settings({"RAG_API_KEY_FILE": str(key_file)}).api_key == "secret-key-from-a-file"
    with pytest.raises(ConfigurationError, match="only one"):
        parse_settings({"RAG_API_KEY": "x" * 10, "RAG_API_KEY_FILE": str(key_file)})
    with pytest.raises(ConfigurationError, match="cannot be read") as caught:
        parse_settings({"RAG_DATABASE_URL_FILE": str(tmp_path / "missing")})
    assert str(tmp_path) not in caught.value.detail
    (tmp_path / "empty").write_text("\n")
    assert parse_settings({"RAG_API_KEY_FILE": str(tmp_path / "empty")}).api_key is None


def test_secrets_never_appear_in_a_repr_or_an_error() -> None:
    settings = parse_settings({"RAG_API_KEY": "super-secret-key-1234", "RAG_DATABASE_URL": "postgresql://u@h/db"})
    assert "super-secret" not in repr(settings) and "postgresql://" not in repr(settings)
