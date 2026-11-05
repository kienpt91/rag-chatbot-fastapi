from __future__ import annotations

from collections.abc import Iterator

import pytest

from support.database import ScratchDatabase, scratch_database, template_database


@pytest.fixture(scope="session")
def _template() -> Iterator[tuple[str, str, str, str]]:
    with template_database() as template:
        yield template


@pytest.fixture
def scratch_db(_template: tuple[str, str, str, str]) -> Iterator[ScratchDatabase]:
    with scratch_database(*_template) as database:
        yield database
