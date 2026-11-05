from __future__ import annotations

import pytest

from support.arch import violations

BAD: list[tuple[str, str, str]] = [
    ("infrastructure/x.py", "import asyncio\nasyncio.create_subprocess_exec('ls')\n", "process"),
    ("infrastructure/x.py", "import asyncio\nasyncio.create_subprocess_shell('ls')\n", "process"),
    ("infrastructure/x.py", "from asyncio import create_subprocess_exec\n", "process"),
    ("infrastructure/x.py", "import asyncio\nasyncio.open_connection('h', 1)\n", "network"),
    ("infrastructure/x.py", "import asyncio\nasyncio.start_server(f, 'h', 1)\n", "network"),
    ("infrastructure/x.py", "from asyncio import open_connection\n", "network"),
    ("infrastructure/x.py", "from sys import modules\nmodules['os'].system('x')\n", "modules"),
    ("infrastructure/x.py", "import os\nvars(os)['system']('x')\n", "vars"),
    ("infrastructure/x.py", "g = globals()\n", "globals"),
    ("infrastructure/x.py", "l = locals()\n", "locals"),
    ("infrastructure/x.py", "import operator\noperator.attrgetter('system')\n", "attrgetter"),
    ("infrastructure/x.py", "from operator import methodcaller\n", "methodcaller"),
]


@pytest.mark.parametrize(("rel", "source", "expected"), BAD)
def test_the_lint_catches_this_evasion(rel: str, source: str, expected: str) -> None:
    found = violations(rel, source)
    assert any(expected in line for line in found), found


def test_the_allowed_uses_stay_allowed() -> None:
    assert (
        violations(
            "infrastructure/providers/http.py", "import asyncio\nawait asyncio.to_thread(f)\n".replace("await ", "")
        )
        == []
    )
    assert (
        violations(
            "infrastructure/http/logging.py", "import logging\nrecord = logging.makeLogRecord({})\nx = vars(record)\n"
        )
        == []
    )
    assert violations("infrastructure/x.py", "import asyncio\nasyncio.sleep(1)\nasyncio.wait_for(f, 1)\n") == []
