"""The architecture, enforced mechanically over the real source tree and checked against hostile snippets."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from support.arch import LAYERS, layer_of, violations

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src" / "rag_chatbot"


def _files() -> list[Path]:
    return sorted(SRC.rglob("*.py"))


def test_every_layer_has_source_files() -> None:
    assert {layer_of(p.relative_to(SRC).as_posix()) for p in _files()} >= set(LAYERS)


@pytest.mark.parametrize("path", _files(), ids=lambda p: p.relative_to(SRC).as_posix())
def test_source_file_obeys_the_dependency_rules(path: Path) -> None:
    rel = path.relative_to(SRC).as_posix()
    assert violations(rel, path.read_text(encoding="utf-8")) == []


def test_every_source_directory_is_a_package() -> None:
    for directory in {p.parent for p in _files()}:
        assert (directory / "__init__.py").exists(), directory


def test_the_declared_runtime_dependencies_are_exactly_the_ones_the_architecture_names() -> None:
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    block = re.search(r"^dependencies = \[(.*?)\]", text, re.S | re.M)
    assert block is not None
    declared = set(re.findall(r'"([a-z_-]+)==', block.group(1)))
    assert declared == {"fastapi", "uvicorn", "pydantic", "psycopg", "psycopg-binary", "psycopg-pool"}


def test_the_widget_is_package_data_and_a_single_file() -> None:
    widget = SRC / "widget"
    assert sorted(p.name for p in widget.iterdir() if p.name != "__pycache__") == ["__init__.py", "rag-chat.js"]


# --- the checker itself: each rule must fire on a snippet that breaks it, and stay quiet on a clean one ---

BAD: list[tuple[str, str, str]] = [
    ("domain/x.py", "import os\n", "domain must be pure"),
    ("domain/x.py", "import time\n", "domain must be pure"),
    ("domain/x.py", "import random\n", "domain must be pure"),
    ("domain/x.py", "import logging\n", "domain must be pure"),
    ("domain/x.py", "import pathlib\n", "domain must be pure"),
    ("domain/x.py", "import datetime\n", "domain must be pure"),
    ("domain/x.py", "import secrets\n", "domain must be pure"),
    ("domain/x.py", "import asyncio\n", "must not schedule anything"),
    ("domain/x.py", "import threading\n", "processes and threads"),
    ("domain/x.py", "import socket\n", "may only be imported by"),
    ("domain/x.py", "import subprocess\n", "processes and threads"),
    ("domain/x.py", "import fastapi\n", "may only be imported by"),
    ("domain/x.py", "import requests\n", "not a declared dependency"),
    ("domain/x.py", "from rag_chatbot.application import ports\n", "domain must not import application"),
    (
        "domain/x.py",
        "from rag_chatbot.infrastructure.system import SystemClock\n",
        "domain must not import infrastructure",
    ),
    ("domain/x.py", "from rag_chatbot import composition\n", "domain must not import composition"),
    ("domain/x.py", "x = open('f')\n", "open() in the domain layer"),
    ("domain/x.py", "print(1)\n", "print()"),
    ("domain/x.py", "x = os.environ\n", "only the composition layer reads the environment"),
    ("domain/x.py", "from datetime import datetime\nx = datetime.now()\n", "domain must be pure"),
    ("application/x.py", "import os\n", "reaches the outside world only through ports"),
    ("application/x.py", "import time\n", "reaches the outside world only through ports"),
    ("application/x.py", "import asyncio\n", "must not schedule anything"),
    ("application/x.py", "import psycopg\n", "may only be imported by"),
    (
        "application/x.py",
        "from rag_chatbot.infrastructure import memory\n",
        "application must not import infrastructure",
    ),
    ("application/x.py", "import datetime\nx = datetime.datetime.now()\n", "the time comes from the Clock port"),
    ("application/x.py", "from datetime import date\nx = date.today()\n", "the time comes from the Clock port"),
    ("application/x.py", "open('f')\n", "open() in the application layer"),
    ("infrastructure/x.py", "from rag_chatbot.composition import main\n", "infrastructure must not import composition"),
    ("infrastructure/x.py", "import uvicorn\n", "may only be imported by"),
    ("infrastructure/http/x.py", "import psycopg\n", "may only be imported by"),
    ("infrastructure/postgres/x.py", "import fastapi\n", "may only be imported by"),
    ("infrastructure/http/x.py", "import httpx\n", "not a declared dependency"),
    ("infrastructure/x.py", "import socket\n", "may only be imported by"),
    ("infrastructure/x.py", "import http.client\n", "may only be imported by"),
    ("infrastructure/x.py", "import urllib.request\n", "may only be imported by"),
    ("infrastructure/x.py", "import ssl\n", "may only be imported by"),
    ("infrastructure/x.py", "import subprocess\n", "processes and threads"),
    ("infrastructure/x.py", "import threading\n", "processes and threads"),
    ("infrastructure/x.py", "import pickle\n", "can execute or load arbitrary code"),
    ("infrastructure/x.py", "import importlib\n", "can execute or load arbitrary code"),
    ("infrastructure/x.py", "x = os.environ['A']\n", "only the composition layer reads the environment"),
    ("infrastructure/x.py", "import os\nos.getenv('A')\n", "only the composition layer reads the environment"),
    ("infrastructure/x.py", "import os as o\no.environ\n", "only the composition layer reads the environment"),
    ("infrastructure/x.py", "from os import environ\n", "only the composition layer reads the environment"),
    ("infrastructure/x.py", "import os\nos.system('x')\n", "runs or replaces a process"),
    ("infrastructure/x.py", "import os\nos.execv('/bin/sh', [])\n", "runs or replaces a process"),
    ("infrastructure/x.py", "eval('1')\n", "reference to eval"),
    ("infrastructure/x.py", "exec('1')\n", "reference to exec"),
    ("infrastructure/x.py", "ev = eval\n", "reference to eval"),
    ("infrastructure/x.py", "f = __import__\n", "reference to __import__"),
    ("infrastructure/x.py", "import builtins\nbuiltins.eval('1')\n", "can execute or load arbitrary code"),
    ("infrastructure/x.py", "import os\ngetattr(os, 'sys' + 'tem')('x')\n", "getattr with a computed name"),
    ("infrastructure/x.py", "import os\ngetattr(os, 'system')\n", "forbidden attribute"),
    ("infrastructure/x.py", "x.__builtins__\n", "forbidden attribute"),
    ("infrastructure/x.py", "().__class__.__bases__[0].__subclasses__()\n", "forbidden attribute"),
    ("infrastructure/x.py", "import sys\nsys.modules['os']\n", "forbidden attribute"),
    ("infrastructure/x.py", "subprocess.run(['a'], shell=True)\n", "shell= must be the literal False"),
    ("composition/x.py", "exec('1')\n", "reference to exec"),
    ("composition/x.py", "from os import *\n", "wildcard import"),
    ("composition/x.py", "import ctypes\n", "can execute or load arbitrary code"),
    ("composition/x.py", "import requests\n", "not a declared dependency"),
    ("composition/x.py", "import fastapi\n", "may only be imported by"),
    ("composition/x.py", "print('x')\n", "print()"),
    ("__main__.py", "x = 1\nexec('1')\n", "reference to exec"),
]


@pytest.mark.parametrize(("rel", "source", "expected"), BAD)
def test_checker_flags_breaking_snippet(rel: str, source: str, expected: str) -> None:
    found = violations(rel, source)
    assert any(expected in line for line in found), found


GOOD: list[tuple[str, str]] = [
    (
        "domain/x.py",
        "import re\nimport json\nimport hashlib\nimport hmac\nfrom dataclasses import dataclass\nfrom urllib.parse import urlsplit\n",
    ),
    ("domain/x.py", "from rag_chatbot.domain import y\nx = getattr(obj, 'literal_name', None)\n"),
    ("application/x.py", "import logging\nfrom datetime import date, datetime\nfrom rag_chatbot.domain import y\n"),
    ("application/x.py", "def f(clock):\n    return clock.now()\n"),
    ("application/x.py", "class C:\n    def f(self):\n        return self._clock.now().date()\n"),
    ("infrastructure/postgres/x.py", "import psycopg\nfrom psycopg_pool import AsyncConnectionPool\n"),
    ("infrastructure/http/x.py", "import fastapi\nfrom starlette.types import ASGIApp\nimport pydantic\n"),
    ("infrastructure/providers/http.py", "import http.client\nimport ssl\nimport asyncio\n"),
    ("infrastructure/sources/http_fetcher.py", "import socket\nimport ssl\nimport http.client\n"),
    (
        "infrastructure/x.py",
        "import os\nimport time\nimport secrets\nimport asyncio\nfrom rag_chatbot.application import ports\n",
    ),
    ("infrastructure/x.py", "from datetime import UTC, datetime\nnow = datetime.now(UTC)\n"),
    ("composition/x.py", "import os\nx = os.environ.get('A')\nimport uvicorn\n".replace("import uvicorn\n", "")),
    ("composition/commands.py", "import uvicorn\nimport os\nfrom rag_chatbot.infrastructure import system\n"),
    ("__main__.py", "from rag_chatbot.composition.main import run\n"),
    ("infrastructure/x.py", "import os\nos.path.join('a', 'b')\nos.replace('a', 'b')\nos.fdopen(1)\n"),
]


@pytest.mark.parametrize(("rel", "source"), GOOD)
def test_checker_accepts_clean_snippet(rel: str, source: str) -> None:
    assert violations(rel, source) == []


def test_relative_imports_are_resolved_against_the_layer() -> None:
    assert violations("domain/x.py", "from ..application import ports\n")
    assert violations("domain/sub/x.py", "from ...infrastructure import system\n")
    assert violations("domain/sub/x.py", "from .. import model\n") == []
