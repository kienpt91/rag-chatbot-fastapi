"""The documentation and the repository's own hygiene, checked against the code and the files."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from rag_chatbot import __version__
from rag_chatbot.composition.config import parse_settings
from rag_chatbot.domain.evalset import parse_dataset
from rag_chatbot.domain.pricing import parse_price_table
from rag_chatbot.domain.tenants import parse_tenants

ROOT = Path(__file__).resolve().parents[2]


def tracked_files() -> list[Path]:
    if shutil.which("git") is not None and (ROOT / ".git").exists():
        out = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=False)
        if out.returncode == 0 and out.stdout.strip():
            return [ROOT / line for line in out.stdout.splitlines() if (ROOT / line).is_file()]
    skip = {".git", "out", "__pycache__", ".ruff_cache", ".mypy_cache", "node_modules"}
    return [p for p in ROOT.rglob("*") if p.is_file() and not skip & set(p.relative_to(ROOT).parts)]


def text_files() -> list[Path]:
    suffixes = {
        ".py",
        ".md",
        ".json",
        ".jsonl",
        ".toml",
        ".yml",
        ".yaml",
        ".sh",
        ".txt",
        ".in",
        ".mjs",
        ".js",
        ".html",
        ".css",
        ".sql",
    }
    return [p for p in tracked_files() if p.suffix in suffixes or p.name in {"LICENSE", ".gitignore", "Dockerfile"}]


def test_security_doc_names_tests_that_exist() -> None:
    text = (ROOT / "SECURITY.md").read_text()
    references = re.findall(r"`(tests/[A-Za-z0-9_/.-]+\.py)(?:::([A-Za-z0-9_]+))?`", text)
    assert len(references) >= 30
    for path, name in references:
        source = (ROOT / path).read_text()
        assert not name or re.search(rf"def {name}\b", source), (path, name)


def test_every_environment_variable_the_code_reads_is_documented() -> None:
    config = (ROOT / "src/rag_chatbot/composition/config.py").read_text()
    names = set(re.findall(r'read\.(?:text|secret|integer|number|flag|path)\(\s*"([A-Z_]+)"', config))
    documented = (ROOT / "docs/configuration.md").read_text()
    assert len(names) >= 40
    for name in sorted(names):
        assert f"RAG_{name}" in documented or (
            "`RAG_CHUNK_TARGET_TOKENS`" in documented and name.startswith("CHUNK_")
        ), name


def test_a_variable_in_the_docs_that_the_code_ignores_does_not_exist() -> None:
    config = (ROOT / "src/rag_chatbot/composition/config.py").read_text()
    read = {f"RAG_{n}" for n in re.findall(r'read\.(?:text|secret|integer|number|flag|path)\(\s*"([A-Z_]+)"', config)}
    in_docs = set(re.findall(r"`(RAG_[A-Z_]+)`", (ROOT / "docs/configuration.md").read_text()))
    odd = {
        n
        for n in in_docs - read
        if not n.endswith("_") and n.removesuffix("_FILE") not in read and n != "RAG_CHUNK_TARGET_TOKENS"
    }
    assert odd <= {"RAG_CHUNK_TARGET_TOKENS"} or not odd, odd


def test_every_runtime_and_development_package_is_justified_in_the_readme() -> None:
    readme = (ROOT / "README.md").read_text().lower()
    for source in ("requirements.in", "requirements-dev.in"):
        for line in (ROOT / source).read_text().splitlines():
            if line.strip() and not line.startswith("#"):
                name = line.split("==")[0].strip().lower()
                assert f"`{name}`" in readme, f"{name} is pinned but not justified in the README"


def test_the_examples_and_the_dataset_are_valid() -> None:
    parse_tenants(json.loads((ROOT / "examples/tenants.example.json").read_text()))
    parse_price_table(json.loads((ROOT / "examples/prices.example.json").read_text()))
    lines = (ROOT / "eval/datasets/site.jsonl").read_text().splitlines()
    assert len(parse_dataset([(f"line {i + 1}", t) for i, t in enumerate(lines)])) == 44
    settings = parse_settings(
        dict(
            line.split("=", 1)
            for line in (ROOT / "eval/fake.settings").read_text().splitlines()
            if line and not line.startswith("#")
        )
    )
    assert settings.provider == "fake" and settings.chat.confidence_threshold == 0.10


def test_the_adrs_are_numbered_without_gaps_and_have_the_three_sections() -> None:
    files = sorted((ROOT / "docs/adr").glob("*.md"))
    assert [f.name[:4] for f in files] == [f"{i:04d}" for i in range(1, len(files) + 1)] and len(files) >= 7
    for file in files:
        text = file.read_text()
        assert all(h in text for h in ("## Context", "## Decision", "## Consequences")) and "Status: accepted" in text


def test_no_file_mentions_an_assistant_or_a_model_vendor() -> None:
    v = [
        "cla" + "ude",
        "anth" + "ropic",
        "open" + "ai",
        "chat" + "g" + "pt",
        "co" + "dex",
        "copi" + "lot",
        "gem" + "ini",
    ]
    pattern = re.compile("|".join(v) + r"|\b" + "G" + "PT" + r"\b", re.IGNORECASE)
    for path in text_files():
        if path.name == "check-commit-msg.sh":
            continue
        assert pattern.search(path.read_text(errors="replace")) is None, path


def test_no_tracked_file_is_a_secret_shaped_file_name() -> None:
    for path in tracked_files():
        assert not re.search(r"(^|/)(\.env(\..*)?|.*\.(pem|key|p12|pfx|jks))$", path.name) or path.name.endswith(
            ".example"
        ), path


def test_the_changelog_has_the_current_version_and_the_license_is_mit() -> None:
    assert f"## {__version__}" in (ROOT / "CHANGELOG.md").read_text()
    assert "Copyright (c) 2026 Kien Phan" in (ROOT / "LICENSE").read_text()


def test_every_workflow_action_is_pinned_by_commit_sha_and_permissions_are_declared() -> None:
    for workflow in (ROOT / ".github/workflows").glob("*.yml"):
        text = workflow.read_text()
        assert re.search(r"^permissions:", text, re.M), workflow
        for use in re.findall(r"uses: ([^\s]+)", text):
            assert re.fullmatch(r"[\w./-]+@[0-9a-f]{40}", use), (workflow.name, use)


@pytest.mark.parametrize("path", ["Dockerfile", "docker-compose.yml", "scripts/py.sh"])
def test_images_are_pinned_by_digest(path: str) -> None:
    text = (ROOT / path).read_text()
    for image in re.findall(r"^(?:FROM|\s*image:)\s+(\S+)", text, re.M):
        assert "@sha256:" in image, (path, image)
