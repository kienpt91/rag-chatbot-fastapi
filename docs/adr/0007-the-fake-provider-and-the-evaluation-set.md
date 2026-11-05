# 7. A deterministic fake provider and an evaluation set in the llm-eval format

Status: accepted

## Context

Tests, CI and the demo must not need an account or money, but the application's defences must be exercised against
a model that misbehaves.

## Decision

`FakeEmbeddingProvider` hashes words; `FakeChatProvider` quotes the best-matching sentence, and, when told to be
gullible (evaluation only), obeys a `REPLY:` line or a request to print its rules found in the retrieved text. The
evaluation set (44 cases) is written in the JSON Lines format of the sibling project llm-eval-harness; the built-in
runner grades the subset this service needs, and `eval/pipeline.py` lets llm-eval run the same cases. A safety failure
is exit 4, never averaged away.

## Consequences

A pass against the fake proves that the plumbing and the defences work, not that a real model answers well; the
README and docs/evaluation.md say so, and `scripts/eval-real.sh` runs the same set against a real provider.
