# 1. Layers and the dependency rule

Status: accepted

## Context

The service is mostly decisions (is this retrieval good enough, is this answer allowed, is this request over its
budget) wrapped around input and output (HTTP, PostgreSQL, a provider API). The decisions must be testable without
any of that, and the input and output must be unable to leak into the decisions.

## Decision

Four layers in `src/rag_chatbot/`, dependencies pointing inward only: `domain` (chunking, retrieval fusion, prompt
building, answer checks, redaction, authentication, pricing, rate limiting, evaluation grading: pure standard-library
code), `application` (the chat, retrieval and ingestion use cases over ports), `infrastructure` (PostgreSQL, the HTTP
interface, providers, sources) and `composition` (configuration, commands, wiring; the only place that reads the
environment and starts the server). The rule is checked mechanically by `tests/architecture/test_layers.py`, an AST
lint over every source file that also confines each third-party package to the one module that exists for it (the web
framework to `infrastructure/http`, the database driver to `infrastructure/postgres`) and bans dynamic code
execution, process creation and wildcard imports. The checker is tested against a snippet for each rule.

## Consequences

The domain and application layers run in tests with plain objects and in-memory fakes (coverage is reported per
layer in the README). Adding a provider or a store means one adapter and one line of wiring. A convenient `import os` in
a pure layer is a failing test, not a review comment. The lint is not a sandbox: SECURITY.md says what it cannot see.
