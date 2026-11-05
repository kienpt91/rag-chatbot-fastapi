from __future__ import annotations

from rag_chatbot.composition.config import parse_settings


def test_an_empty_variable_counts_as_unset_so_compose_can_pass_every_name() -> None:
    settings = parse_settings(
        {
            "RAG_PROVIDER": "fake",
            "RAG_CHAT_MODEL": "",
            "RAG_EMBEDDING_MODEL": " ",
            "RAG_API_KEY": "",
            "RAG_API_BASE_URL": "",
        }
    )
    assert (
        settings.chat_model == "fake-chat-1"
        and settings.embedding_model == "fake-embedding-1"
        and settings.api_key is None
    )
    settings.require("provider")


def test_the_quarantine_switch_is_read() -> None:
    assert parse_settings({}).ingest.quarantine is True
    assert parse_settings({"RAG_QUARANTINE_INSTRUCTIONS": "false"}).ingest.quarantine is False
