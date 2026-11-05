# Environment variable reference

All settings are `RAG_*` environment variables, read once at start-up and validated (`rag-chatbot check-config` checks a
configuration without connecting to anything or printing a secret). `*_FILE` variants exist for the secrets.

| variable | default | meaning |
| --- | --- | --- |
| `RAG_DATABASE_URL` (`_FILE`) | - | connection string of the service's role (member of `rag_app`) |
| `RAG_MIGRATE_DATABASE_URL` (`_FILE`) | database URL | connection string of the schema owner, for `migrate` |
| `RAG_MIGRATIONS_DIR` | `migrations/` | the SQL files |
| `RAG_TENANTS_FILE` | - | the tenants JSON (examples/tenants.example.json) |
| `RAG_PRICE_TABLE_FILE` | - | the price table JSON (examples/prices.example.json); both models must be in it |
| `RAG_PROVIDER` | - | `http` or `fake` |
| `RAG_API_BASE_URL`, `RAG_API_KEY` (`_FILE`) | - | for `http`: https base URL (http for localhost) and bearer key |
| `RAG_CHAT_MODEL`, `RAG_EMBEDDING_MODEL` | - (fake: fake-chat-1, fake-embedding-1) | model names sent to the API and looked up in the price table |
| `RAG_EMBEDDING_DIMENSIONS` | 1536 | must equal the size of the `embedding` column |
| `RAG_EMBEDDING_SEND_DIMENSIONS` | false | send `dimensions` in embedding requests |
| `RAG_PROVIDER_TIMEOUT_S` | 30 | per provider call |
| `RAG_FAKE_GULLIBLE` | false | fake provider obeys `REPLY:` lines and rule-reveal requests (evaluation only) |
| `RAG_CONFIDENCE_THRESHOLD` | 0.30 | below this similarity of the best chunk: refuse without calling the model |
| `RAG_CONTEXT_CHUNKS`, `RAG_CONTEXT_CHARS` | 5, 6000 | how much retrieved text the model gets |
| `RAG_CONTEXT_RELATIVE_CUTOFF` | 0.5 | drop chunks less than this fraction as similar as the best |
| `RAG_CANDIDATES_PER_SEARCH` | 12 | hits taken from each of the vector and keyword searches |
| `RAG_HISTORY_TURNS`, `RAG_HISTORY_TOKEN_BUDGET` | 6, 800 | earlier turns given to the model, trimmed to the budget |
| `RAG_MAX_OUTPUT_TOKENS`, `RAG_MAX_SENTENCES` | 400, 4 | answer length |
| `RAG_MAX_MESSAGES_PER_SESSION` | 100 | a conversation's length limit |
| `RAG_REDACT` | emails,phones,cards | personal data removed from messages; `none` keeps only credential redaction |
| `RAG_BLOCKED_PHRASES` | - | comma-separated phrases that are refused before any spend |
| `RAG_REFUSAL_MESSAGE` | built-in | default "I don't know" text (a tenant can override) |
| `RAG_DAILY_BUDGET` | none | per tenant per UTC day, in the price table's currency; a tenant's `daily_budget` wins |
| `RAG_CHUNK_TARGET_TOKENS`, `_MAX_`, `_OVERLAP_`, `_MIN_` | 220, 350, 40, 40 | chunking (`RAG_CHUNK_*_TOKENS`) |
| `RAG_EMBED_BATCH_SIZE`, `RAG_MAX_CHUNKS_PER_DOCUMENT` | 32, 300 | ingestion |
| `RAG_QUARANTINE_INSTRUCTIONS` | true | leave instruction-like sentences of a page out of the index, and report them |
| `RAG_HOST`, `RAG_PORT` | 127.0.0.1, 8000 | bind address |
| `RAG_MAX_BODY_BYTES` | 16384 | request body limit |
| `RAG_IP_BURST`, `RAG_IP_PER_MINUTE` | 20, 30 | per-address limit |
| `RAG_SESSION_BURST`, `RAG_SESSION_PER_MINUTE` | 5, 6 | per-conversation limit |
| `RAG_MAX_TRACKED_KEYS` | 10000 | buckets kept per limiter |
| `RAG_TRUSTED_PROXY_HOPS` | 0 | reverse proxies that append to `X-Forwarded-For` |
| `RAG_REQUEST_TIMEOUT_S` | 60 | one chat request |
| `RAG_MAX_TOKEN_TTL_S` | 3600 | longest embed token this deployment mints |
| `RAG_ENABLE_DOCS` | false | `/docs` and `/openapi.json` |
| `RAG_DB_POOL_MAX`, `RAG_DB_STATEMENT_TIMEOUT_MS` | 10, 10000 | connection pool and statement timeout |
| `RAG_LOG_LEVEL` | INFO | |

## Files

`tenants.json`: `{"tenants": [{"id", "name", "allowed_origins", "api_key_sha256", "token_secrets", "daily_budget", "confidence_threshold", "refusal_message", "link_hosts"}]}`.
Origins are `scheme://host[:port]`, https except localhost; unknown keys are errors. `price table`:
`{"currency": "USD", "models": {"name": {"input_per_mtok": 0.15, "output_per_mtok": 0.6}}}`. The numbers in the
examples are illustrative, not current prices.
