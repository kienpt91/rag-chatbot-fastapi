#!/bin/sh
# The whole local gate in the pinned containers: format check, lint, strict type check, the whole suite against a real
# PostgreSQL + pgvector (a throwaway container, no network) with the coverage thresholds, and the widget tests.
# usage: scripts/verify.sh [fast]   (fast skips coverage and the widget tests)
set -eu
here=$(dirname "$0")
export PY_DB=1
if [ "${1:-}" = fast ]; then
  exec sh "$here/py.sh" 'ruff format --check . && ruff check . && python -m mypy && RAG_REQUIRE_DB=1 python -m pytest -q'
fi
sh "$here/py.sh" 'ruff format --check . && ruff check . && python -m mypy && RAG_REQUIRE_DB=1 python scripts/coverage_check.py'
sh "$here/test-widget.sh"
