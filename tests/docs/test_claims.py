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


def test_the_service_image_follows_a_python_3_13_slim_tag_pinned_by_digest() -> None:
    bases = re.findall(r"^FROM (\S+)", (ROOT / "Dockerfile").read_text(), re.M)
    assert len(bases) == 2 and bases[0] == bases[1]
    assert re.fullmatch(r"python:3\.13\.\d+-slim-trixie@sha256:[0-9a-f]{64}", bases[0]), bases[0]


def test_dependabot_cannot_move_the_interpreter_to_another_minor_version_or_touch_the_build_backend() -> None:
    text = (ROOT / ".github/dependabot.yml").read_text()
    pip, docker = (
        text.split("package-ecosystem: docker")[0],
        text.split("package-ecosystem: docker")[1].split("package-ecosystem: github-actions")[0],
    )
    assert "dependency-name: setuptools" in pip and "groups:" in pip
    assert (
        "dependency-name: python" in docker
        and "version-update:semver-minor" in docker
        and "version-update:semver-major" in docker
    )


def test_releasing_says_that_locks_are_regenerated_by_our_script_not_merged_from_dependabot() -> None:
    text = (ROOT / "RELEASING.md").read_text()
    assert (
        "Do not merge a Dependabot pull request" in text and "scripts/lock-deps.py" in text and "version bump" in text
    )
