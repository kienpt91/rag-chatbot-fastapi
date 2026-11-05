"""Claims in the documentation that were checked against the code and must stay true."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def compose_service(name: str) -> str:
    text = (ROOT / "docker-compose.yml").read_text()
    match = re.search(rf"^  {name}:\n(.*?)(?=^  \w|^volumes:)", text, re.S | re.M)
    assert match is not None
    return match.group(1)


def test_the_internet_facing_service_does_not_receive_the_database_superuser_password() -> None:
    app = compose_service("app")
    assert "env_file" not in app and "RAG_DB_ADMIN_PASSWORD" not in app and "POSTGRES_PASSWORD" not in app


def test_the_security_document_states_its_limits_where_the_review_found_overclaims() -> None:
    text = (ROOT / "SECURITY.md").read_text()
    assert "best-effort" in text  # hidden-text removal
    assert "forge the `Origin` header" in text  # the origin check constrains browsers only
    assert "the budget is shared" in text and "tripwire" in text
    assert "per process; the budget" in text or "rate limits are per process" in text.lower()
    assert "quarantine" in text.lower() and "embedding model" in text.lower()


def test_the_guard_docstring_does_not_promise_a_citation_for_every_statement() -> None:
    assert "every statement must carry a citation" not in (ROOT / "src/rag_chatbot/domain/guard.py").read_text()
