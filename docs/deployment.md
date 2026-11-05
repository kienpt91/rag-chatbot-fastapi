# Deployment

## Local development

```sh
cp .env.example .env                                   # placeholders only; edit the two passwords
sh scripts/py.sh install                               # once: the pinned packages in a container volume
docker run --rm -v "$PWD:/work" -w /work python@sha256:d98cc29d3266e535dd9da1099688d9cb3ca4753813571e071763d17725460962 \
  true                                                 # (optional) pull the pinned image
# a tenants file with a fresh key: the key is printed once
PYTHONPATH=src python3 -m rag_chatbot init-tenant --id maple --name "Maple Street Bakery" --origin http://localhost:8080 --out tenants.json
docker compose up --build -d                           # database, migrations, the service role, the service
```

Then ingest the sample site (the ingestion CLI is the same image):
`docker compose run --rm app ingest --tenant maple --folder /app/../site ...` needs the site mounted; the simplest
route is `scripts/py.sh`-style runs shown in the README demo. The service listens on `127.0.0.1:8000`.

## Production notes

- **Run behind a reverse proxy that terminates TLS.** The service speaks plain HTTP and binds `127.0.0.1` by default
  (`RAG_HOST`; the image sets `0.0.0.0` for the container network). Tell it how many proxies are in front of it:
  `RAG_TRUSTED_PROXY_HOPS=1`, and make the proxy *append* the connecting address to `X-Forwarded-For` (nginx:
  `proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;`). Set a body limit in the proxy too
  (`client_max_body_size 32k;`), and pass `/chat` and `/embed-token` only; `/healthz` and `/readyz` are for the
  orchestrator and need not be public.
- **Rate limits are per process.** One process per container is the supported shape; for more, add a limit at the
  proxy or gateway (nginx `limit_req_zone $binary_remote_addr zone=chat:10m rate=30r/m;`).
- **Database roles.** Migrations run as the database owner (`RAG_MIGRATE_DATABASE_URL`); the service connects as a
  login role that is a member of `rag_app` (`docker/create-service-role.sql`), which cannot change the schema and is
  subject to row-level security. Never give the service a superuser or a BYPASSRLS role. On Supabase, run the
  migrations from `migrations/` in order (the roles and `auth.uid()` exist there; the migrations leave them alone) and
  connect with a dedicated role that is a member of `rag_app`.
- **Secrets** come from the environment or `_FILE` variants (`RAG_API_KEY_FILE`, `RAG_DATABASE_URL_FILE`): use your
  platform's secret mounts. The tenants file holds hashes and token secrets: mode 0600, outside the image.
- **Read-only container.** The image writes nothing: run with `read_only: true` and a `tmpfs` on `/tmp`, as the compose
  file does; drop all capabilities; run as the image's non-root user (uid 10001).
- **Probes.** `GET /healthz` is liveness (no dependencies). `GET /readyz` checks the database, the schema version and
  the embedding size and answers 503 with problem codes only.
- **Logs** are one JSON object per line on standard output: `request_id`, route, status, latency, a salted hash of the
  client address, token and cost counts. No message text, no addresses, no tokens.
- **Backups.** The conversations, the usage ledger and the chunks are in PostgreSQL: use your platform's base backups or
  `pg_dump` (the chunks can always be rebuilt by re-running the ingestion, so they are the cheapest thing to lose).
  Restore, then run `rag-chatbot migrate` (a no-op when current). `rag-chatbot purge` implements retention.
- **Schema changes** are new numbered files in `migrations/`; the runner records checksums and refuses to run if an
  applied file was edited.
- **Capacity.** One process handles a request at a time per core in Python terms (async I/O); provider latency
  dominates. HNSW keeps vector search fast into the hundreds of thousands of chunks per deployment.
