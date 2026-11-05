# Adapting this to a client project

A checklist, in the order that works, for putting this service behind a client's website in days.

1. **Content.** Point the ingestion at the client's site: `rag-chatbot ingest --tenant NAME --crawl https://www.client.example/
   --allow-host www.client.example` (robots.txt is honoured, https only, private addresses refused), or `--folder` with
   exported HTML/Markdown/text and `--base-url` so citations link to the real pages. Leave out what visitors write
   themselves: `--exclude 'reviews/*' --exclude 'forum/*'` (a pattern is matched, case-insensitively and without the
   query, against the page's path: `reviews/*` and `*/reviews/*` both exclude `https://www.client.example/Reviews/x?p=2`).
   Read the `quarantined` lines of the ingest report: they are sentences that look like instructions to an assistant and
   were kept out of the index; an honest sentence among them means the wording needs a look (or `RAG_QUARANTINE_INSTRUCTIONS=false`). Re-run it on a schedule (cron, CI): unchanged chunks are
   not embedded again, so a nightly run of an unchanged site costs nothing; add `--prune` to drop pages that left
   the site (it refuses to prune after an incomplete crawl).
2. **Tenant.** `rag-chatbot init-tenant --id client --name "Client Name" --origin https://www.client.example --out tenants.json`
   prints the API key once (store it on the client's server, never in a page) and writes the hash and a token secret.
   Add more sites with `--add`. Per tenant in the file: `confidence_threshold`, `refusal_message` (the "I don't know"
   text in the client's voice and language), `daily_budget`, `link_hosts` (partner sites the assistant may link to).
3. **The page.** The client's backend mints a short-lived token (`POST /embed-token` with the API key, or
   `rag-chatbot embed-token`) and renders it into the script tag:
   `<script src="https://chat.client.example/widget.js" data-api-url="https://chat.client.example" data-token="..." data-title="Ask Client" data-accent="#0a6" defer></script>`.
   Or give `data-token-url` an endpoint on the client's server that returns `{"token": "..."}`. Branding and language are
   `data-*` attributes (`data-title`, `data-greeting`, `data-placeholder`, `data-send-label`, `data-error-text`, ...);
   the widget never needs an inline script or style, so a strict Content-Security-Policy works (`script-src` and
   `connect-src` need the chat host). Allow the page's origin in the tenant (`allowed_origins`, exact).
4. **Provider.** Set `RAG_PROVIDER=http` and the `RAG_API_*` variables for any HTTP chat/embedding API with the common
   wire format (`/chat/completions`, `/embeddings`). Put both models in the price table (`RAG_PRICE_TABLE_FILE`):
   the service refuses to start with an unpriced model. Embedding size: the column is `vector(1536)`; for another
   size edit migration `0002` *before the first migration*, and set `RAG_EMBEDDING_DIMENSIONS` (the service refuses to
   start when they differ).
5. **Threshold.** `RAG_CONFIDENCE_THRESHOLD` (default 0.30) is the lever between "answers too much" and "says I don't
   know too often". Tune it on the evaluation set with the real provider: put the client's real questions in a copy
   of `eval/datasets/site.jsonl` (answerable cases with the facts and the page that must be cited, unanswerable ones),
   run `sh scripts/eval-real.sh`, and pick the highest threshold that still answers every answerable case. Smaller chunks
   (`RAG_CHUNK_TARGET_TOKENS`) suit FAQ-style pages; larger ones suit prose.
6. **Languages.** The keyword search uses the `simple` text-search configuration (any language, no stemming); for a
   single-language site switch `to_tsvector('simple', ...)` in migration `0002` to `english`, `french`, ... before the
   first migration. The stop-word list for the keyword terms (`domain/text.py`) is English: replace it for another
   language. The model answers in the visitor's language by instruction. Write `refusal_message` per tenant in the
   site's language.
7. **Privacy.** `RAG_REDACT` chooses what is removed from messages (`emails,phones,cards`, or `none`; credentials are
   always removed). Set a retention policy: `rag-chatbot purge --older-than-days 30` from cron.
8. **Cost.** Set `RAG_DAILY_BUDGET` (or `daily_budget` per tenant). Cost is computed from the usage the provider reports
   (estimated, and flagged, when it reports none) and your price table; `usage_events` holds one row per request.
9. **Deploy.** `docs/deployment.md`.

## What is deliberately left out

- **Streaming.** Answers are checked as a whole before anyone sees a word (ADR 0005).
- **Tools and actions.** The model can only answer; it cannot call anything. Adding tools changes the threat model.
- **A human hand-off, a CRM, analytics dashboards, an admin UI.** `usage_events` and `chat_messages` are plain tables.
- **Visitor accounts.** Embed tokens carry an optional Supabase user id so that signed-in users can read their own
  history through row-level security; there is no login in this service.
- **Distributed rate limiting.** In-process buckets; put a gateway limit in front for more than one worker.
- **Per-document permissions.** Everything indexed for a tenant is visible to every visitor of that tenant.
- **PDF and JavaScript-rendered pages.** The crawler fetches HTML, Markdown and text and executes nothing.
