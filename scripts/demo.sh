#!/bin/sh
# Produces docs/demo-output.txt: REAL output of the evaluation run (fake provider, no network), of the configuration
# check, and of the end-to-end tests that start the server process against PostgreSQL + pgvector. Times vary.
set -eu
here=$(cd "$(dirname "$0")/.." && pwd)
out=$here/docs/demo-output.txt
{
  echo "# captured $(date -u +%Y-%m-%dT%H:%M:%SZ) by scripts/demo.sh; no hosted model was called (deterministic fake provider)"
  echo
  echo '$ sh scripts/eval.sh'
  sh "$here/scripts/eval.sh" || true
  echo
  echo '$ rag-chatbot check-config   (RAG_PROVIDER=fake, example tenants and price table)'
  sh "$here/scripts/py.sh" 'PYTHONPATH=src RAG_PROVIDER=fake RAG_TENANTS_FILE=examples/tenants.example.json RAG_PRICE_TABLE_FILE=examples/prices.example.json python -m rag_chatbot check-config' || true
  echo
  echo '$ python -m pytest tests/e2e -v   (real server process, real PostgreSQL + pgvector, local HTTP provider)'
  PY_DB=1 sh "$here/scripts/py.sh" 'python -m pytest tests/e2e -v -p no:randomly 2>&1 | grep -E "PASSED|FAILED|ERROR|passed|failed" | sed "s#^tests/e2e/##"' || true
} > "$out" 2>&1
cat "$out"
