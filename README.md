# rag-chatbot-fastapi

A website chatbot service that answers **only from the site's own content** and says "I don't know" otherwise. Python
3.12+, FastAPI, PostgreSQL with pgvector (Supabase-compatible schema), hybrid retrieval with citations, application-level
prompt-injection defences, cost accounting with a daily budget that fails closed, and a drop-in widget.

**Honesty first.** No hosted model was called while building or testing this repository. The tests, the evaluation set,
CI and the demo use a deterministic fake provider; what a real model does with the same prompts is **not verified here**
(see "What is not verified"). There is no customer data, no real credential and no claim of production use. It is a
reference implementation written so that a freelancer can adapt it for a client in days ([docs/adapting.md](docs/adapting.md)).

## What it does

- **Ingestion** (`rag-chatbot ingest`): a folder of HTML, Markdown and text, or a crawl restricted to hosts you list
  (exact match, https by default, robots.txt and crawl-delay honoured, size/depth/page caps, no JavaScript, private
  addresses refused). HTML becomes text with boilerplate and hidden text removed, using the standard library only.
  Chunks overlap and have stable ids, so re-ingesting an unchanged site embeds nothing.
- **Retrieval**: vector search and keyword search (`tsvector`) in PostgreSQL, fused with reciprocal rank fusion. Confidence
  is the cosine similarity of the best chunk; below `RAG_CONFIDENCE_THRESHOLD` the model is not called.
- **`POST /chat`** returns an answer with citations and a `grounded` flag, refuses when out of scope, keeps a per-session
  history trimmed to a token budget, authenticates API keys and signed embed tokens (exact allowed origins), limits
  requests per address and per session in bounded memory, logs JSON with request ids and no personal data, and never
  leaks internals in an error. `/healthz`, `/readyz`, `/embed-token` and `/widget.js` complete it ([docs/api.md](docs/api.md)).
- **Schema** (`migrations/`, plain SQL on any PostgreSQL + pgvector): row-level security for the service (by tenant) and for
  Supabase clients (own conversations only), tested on a real database.
- **Defences**: untrusted text fenced with a per-request nonce, no tools for the model, every answer checked (valid
  citations, no leak of the rules, links only to the site's own hosts, markup removed), personal data redacted before the
  model and the database. [SECURITY.md](SECURITY.md) names a test per claim and lists what is *not* protected.
- **Widget**: one vanilla JavaScript file (`src/rag_chatbot/widget/rag-chat.js`, served at `/widget.js`), `textContent`
  only, no inline script or style. Demo page: `examples/widget/`.
- **Evaluation**: 44 golden cases in the format of the sibling project llm-eval-harness, a built-in runner, and a wiring
  for llm-eval itself ([docs/evaluation.md](docs/evaluation.md)).
- **Accounting**: tokens as the provider reports them, an operator price table, one `usage_events` row per request, a daily
  budget per tenant.

## Quickstart (containers only; nothing is installed on your machine)

```sh
sh scripts/py.sh install                 # the pinned Python image and hash-checked packages, into a docker volume
sh scripts/verify.sh                     # format, lint, strict types, the suite against a throwaway pgvector database, coverage
sh scripts/eval.sh                       # the evaluation set against the fake provider (no network, no cost)
sh scripts/demo.sh                       # writes docs/demo-output.txt
cp .env.example .env && docker compose up --build   # development stack (see docs/deployment.md)
```

`scripts/py.sh` runs commands in the python image pinned by digest, project mounted read-only, no capabilities, non-root,
no network; with `PY_DB=1` it also starts a throwaway PostgreSQL + pgvector container (pinned by digest, `--internal`
network, nothing leaves the machine) and gives the command its address.

Configuration is `RAG_*` environment variables ([docs/configuration.md](docs/configuration.md)); `rag-chatbot --help`
lists the commands: `serve`, `migrate`, `ingest`, `eval`, `init-tenant`, `new-api-key`, `embed-token`, `purge`, `check-config`.

## Demo output (real; full transcript in [docs/demo-output.txt](docs/demo-output.txt))

The evaluation against the fake provider (a deliberately gullible one, so the cases exercise the application's checks):

```text
dataset   eval/datasets/site.jsonl  sha256:a1cb9c490896…  (44 cases)
provider  fake  chat=fake-chat-1  embedding=fake-embedding-1
quality   30/30 passed
safety    14/14 passed
tokens    in 18046  out 728
cost      0.020337 USD
latency   p50 5 ms  p95 7 ms  (run took 0.3 s)
result    exit 0
```

The end-to-end tests start the real server process against PostgreSQL + pgvector with an HTTP provider:

```text
test_server_process.py::test_the_service_answers_from_the_site_with_citations_over_http_and_stores_the_conversation PASSED [ 11%]
test_server_process.py::test_the_widget_flow_a_token_minted_by_the_backend_works_from_the_allowed_origin_only PASSED [ 22%]
test_server_process.py::test_hostile_content_and_hostile_requests_never_get_through PASSED [ 33%]
test_server_process.py::test_logs_and_the_database_hold_no_personal_data_and_no_secret PASSED [ 44%]
test_server_process.py::test_a_provider_outage_is_a_503_that_names_nothing_and_the_service_stays_up PASSED [ 55%]
test_server_process.py::test_a_tiny_daily_budget_makes_the_bot_fail_closed_with_a_friendly_message PASSED [ 66%]
test_server_process.py::test_rate_limits_answer_429_with_retry_after_before_any_work PASSED [ 77%]
test_server_process.py::test_the_service_refuses_to_start_on_a_database_that_does_not_match PASSED [ 88%]
test_server_process.py::test_sigterm_stops_the_server_cleanly PASSED [100%]
============================== 9 passed in 10.12s ==============================
```

## Verification (real numbers from `sh scripts/verify.sh`, 2026-11-05)

| layer | line coverage | threshold |
| --- | --- | --- |
| domain | 97.6% | 95% |
| application | 98.8% | 90% |
| infrastructure | 97.6% | - |
| composition | 94.6% | - |
| **overall** | **97.3%** (branches 94.0%) | 85% |

Python tests: **1,904** passed (domain 1,078, application 66, infrastructure 425 incl. the real-database RLS, migration and
store-contract suites, API 94, composition 61, end-to-end 9, architecture 146, docs, scripts and version 25) on Python 3.13 in the
pinned container; widget tests under Node 22: **91** passed (`scripts/test-widget.sh`, a fake DOM, no network). The CI
workflow also runs the suite on Python 3.12; that matrix leg has not been run locally. The evaluation set: 30/30 quality and
14/14 safety cases with the fake provider; with the answer checks disabled, six safety and five quality cases fail.
The dataset also validates and runs under llm-eval-harness (`validate`: ok; `run`: 30/30 quality, 14 safety cases, 0
failing) when that project is checked out next to this one.

## Layout

```text
src/rag_chatbot/domain          pure: chunking, extraction, retrieval fusion, prompt, answer checks, redaction, auth, pricing, rate limiting, evaluation
src/rag_chatbot/application     use cases over ports: chat, ingestion, retrieval, evaluation
src/rag_chatbot/infrastructure  postgres (migrations, stores), http (FastAPI, middleware, logging), providers (http, fake), sources (folder, crawler, fetcher)
src/rag_chatbot/composition     configuration, commands, wiring
migrations/                     plain SQL: extensions and roles, content and conversations, row-level security
eval/  examples/site/  examples/widget/   evaluation set, sample site, widget demo page
docs/  docs/adr/                adapting, deployment, configuration, api, evaluation; seven architecture decisions
```

The architecture is enforced by `tests/architecture/test_layers.py` (an AST lint; ADR 0001).

## Dependencies (exact pins, sha256 hashes, every file at least seven days old at the 2026-10-24 cutoff)

`python scripts/check-dep-age.py` verifies the age and that each lock matches its input; both locks are in `requirements*.txt`.

Runtime (`requirements.in`):

- `fastapi`: the web framework the job asks for. `starlette`: its ASGI toolkit, imported directly for the middleware.
- `uvicorn`: the ASGI server. `h11`: its HTTP parser. `click`: its command line dependency.
- `pydantic`: request validation, required by FastAPI; `pydantic-core`, `annotated-types`, `typing-inspection` and
  `typing-extensions` are its closure; `annotated-doc` and `opentelemetry-api` are required by this FastAPI version
  (the service does not use the latter).
- `anyio`: Starlette's async layer. `idna`: host names for `anyio`.
- `psycopg`: the async PostgreSQL driver. `psycopg-binary`: bundles libpq so the slim image needs no system packages.
  `psycopg-pool`: a bounded connection pool.

Everything else (the HTML extractor, crawler, HTTP clients, redaction, token signing, rate limiting) is the standard library.

Development (`requirements-dev.in`; none is installed in the service image):

- `pytest`, `pytest-cov`, `coverage`: tests and the coverage thresholds. `iniconfig`, `packaging`, `pluggy`, `pygments`: pytest's closure.
- `mypy` (strict), with `ast-serialize`, `librt`, `mypy-extensions`, `pathspec`: type checking. `ruff`: format and lint.
- `build`, `setuptools`, `pyproject-hooks`: the reproducible sdist and wheel.
- `httpx2`: Starlette's test client now expects it; `httpcore2` and `truststore` are its closure.

## What is not verified

- **A real model.** Nothing here called a hosted chat or embedding API. The HTTP provider is tested against a local server
  that speaks the common wire format; the defences are tested with scripted and gullible fakes. Run `scripts/eval-real.sh`
  with your provider before trusting a threshold.
- **The compose stack beyond start-up.** `docker compose up` was run: the database, the migrations, the service role and the
  read-only non-root service came up, `/readyz` answered ready and `/chat` answered with the "I don't know" refusal of an
  empty knowledge base. Ingestion inside compose (the `docker compose run ... ingest` command in docs/deployment.md) and a
  grounded answer through it were not run; the same ingestion is covered by the tests against the same database image.
- **Python 3.12** and the **GitHub workflows** were not run (the workflows reuse the local scripts, which were).
- **Supabase itself**: the migrations are plain SQL tested on PostgreSQL + pgvector; the policies use `auth.uid()` and the
  `anon`/`authenticated` roles as Supabase defines them, but a hosted project was not used.
- TLS to a real provider and real DNS (the fetcher's TLS path has no end-to-end test; its SSRF logic is tested with injected resolvers).
- Throughput and load; the per-process limits in docs/deployment.md.

## What a reviewer should attack first

1. **A poisoned page that is retrieved**: the checks stop instructions, rule leaks and foreign links, but a cited sentence that a
   customer wrote is passed on (SECURITY.md, first "not protected" item). Try to get something harmful through that is not a link.
2. **The answer checks** (`domain/guard.py`): leaks by paraphrase, citations that are valid but irrelevant, URLs the regex misses.
3. **The row-level security** (`migrations/0003`): find a statement that crosses tenants; the tests are in `tests/infrastructure/test_rls.py`.
4. **The crawler's SSRF handling** (`infrastructure/sources/http_fetcher.py`, `domain/urlpolicy.py`): host spellings, redirects, DNS.
5. **The HTML extractor's hidden-text rules** (`domain/htmlextract.py`): a hiding style it does not know indexes attacker text.
6. **Embed tokens and CORS** (`domain/auth.py`, `infrastructure/http/middleware.py`): origin handling and the unauthenticated preflight.

## License

MIT, see [LICENSE](LICENSE). Releasing: [RELEASING.md](RELEASING.md). Changes: [CHANGELOG.md](CHANGELOG.md).
