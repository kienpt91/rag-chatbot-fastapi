from __future__ import annotations

import re
from pathlib import Path

from rag_chatbot import __version__


def test_version_matches_pyproject() -> None:
    text = (Path(__file__).resolve().parents[1] / "pyproject.toml").read_text(encoding="utf-8")
    match = re.search(r'^version = "([^"]+)"$', text, re.MULTILINE)
    assert match is not None and match.group(1) == __version__
