# Changelog

## 0.1.0

First release.

- Ingestion CLI: a folder of HTML, Markdown and text, or a crawl restricted to the hosts the operator lists (exact host
  match, https by default, robots.txt and crawl-delay honoured, size, depth and page caps, no JavaScript, private
  addresses refused, redirects re-checked hop by hop); HTML to text with boilerplate and hidden text removed using the
  standard library only; chunking with overlap and stable content-addressed ids; unchanged chunks are never embedded again.
- PostgreSQL with pgvector: hybrid retrieval (vector and keyword search fused by reciprocal rank fusion), confidence
  from the cosine similarity of the best chunk, a configurable threshold below which the model is not called.
- FastAPI service: `POST /chat` with citations and a `grounded` flag, `POST /embed-token`, `/healthz`, `/readyz`,
  `/widget.js`; API keys and signed embed tokens with exact allowed origins; per-address and per-session rate limits
  in bounded memory; request size limits; JSON logs with request ids and no personal data; errors that never leak
  internals.
- Plain SQL migrations with row-level security (tenant scoping for the service, own-conversations-only for Supabase
  clients), tested on a real database, with a mutation check that the tests fail when a policy is weakened.
- Prompt-injection defences: fenced, labelled untrusted text with a per-request nonce, no tools for the model, answers
  checked for valid citations, leaks of the rules, links outside the site's own hosts and markup.
- Cost and latency accounting from provider-reported usage and an operator price table, and a daily budget that fails
  closed with a friendly message.
- Drop-in widget: one vanilla JavaScript file, textContent only, no inline script or style.
- A 44-case evaluation set in the llm-eval format with a built-in runner, and a wiring for llm-eval itself; runs in CI
  against a deterministic fake provider (no hosted model is called).
- Review hardening: ingest-time quarantine of instruction-like sentences with an operator report; hidden-text rules for
  script double-escape, image alt text, more hiding styles and raw HTML in Markdown; `--exclude` for crawled URLs;
  protocol-relative and Markdown links refused in answers; earlier turns fenced as data; sessions of signed-in users
  bound to them; cancelled requests recorded; embedding usage estimated when missing; the migration ledger and identity
  sequences revoked from the API roles; a least-privilege compose service; a stricter architecture lint.
- Dockerfile (non-root, digest-pinned base, read-only filesystem), docker-compose for development, CI, a tag-triggered
  release workflow with build provenance, dependabot with a 7-day cooldown.
