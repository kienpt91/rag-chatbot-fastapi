#!/bin/sh
# Runs the evaluation set (eval/datasets/site.jsonl) through the whole chat flow against the DETERMINISTIC FAKE provider,
# in the pinned container, with no network. No hosted model is called. Settings: eval/fake.settings. Extra arguments go to
# `rag-chatbot eval` (for example --min-quality 0.9). Exit status: 0 pass, 1 quality below the minimum, 3 the service
# failed, 4 a safety case failed. The report is written to out/eval-report.json.
set -eu
here=$(dirname "$0")
# EVAL_ENV='NAME=value ...' adds settings, for example RAG_QUARANTINE_INSTRUCTIONS=false to test the answer checks alone
exec sh "$here/py.sh" "set -a; . eval/fake.settings; ${EVAL_ENV:+export $EVAL_ENV;} set +a; PYTHONPATH=src python -m rag_chatbot eval $*"
