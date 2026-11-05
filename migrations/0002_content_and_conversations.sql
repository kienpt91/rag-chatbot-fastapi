-- Content (documents and their chunks with embeddings), conversations and the usage ledger.
-- Every table carries tenant_id: one deployment serves several websites, and nothing is shared between them.
--
-- The embedding column has a fixed size, vector(1536). It must equal the size of the vectors your embedding model
-- returns (the service refuses to start otherwise). To use another size, change the number here BEFORE the first
-- migration of a database and set RAG_EMBEDDING_DIMENSIONS to the same value (docs/adapting.md). An HNSW index can
-- be built for up to 2000 dimensions.

CREATE TABLE documents (
  tenant_id   text        NOT NULL CHECK (tenant_id ~ '^[a-z0-9][a-z0-9_-]{0,62}$'),
  document_id text        NOT NULL CHECK (char_length(document_id) BETWEEN 1 AND 2048),
  url         text        NOT NULL DEFAULT '' CHECK (char_length(url) <= 2048),
  title       text        NOT NULL DEFAULT '' CHECK (char_length(title) <= 300),
  updated_at  timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY (tenant_id, document_id)
);

CREATE TABLE chunks (
  tenant_id   text        NOT NULL,
  chunk_id    text        NOT NULL CHECK (chunk_id ~ '^ck_[0-9a-f]{24}$'),
  document_id text        NOT NULL,
  ordinal     integer     NOT NULL CHECK (ordinal >= 0),
  title       text        NOT NULL DEFAULT '' CHECK (char_length(title) <= 300),
  url         text        NOT NULL DEFAULT '' CHECK (char_length(url) <= 2048),
  content     text        NOT NULL CHECK (char_length(content) BETWEEN 1 AND 20000),
  embedding   vector(1536) NOT NULL,
  -- 'simple' keeps every word and stems none, so it works for any language; use another configuration (english,
  -- french, ...) when the site is in one language and you want stemming. Changing it means re-creating this column.
  search      tsvector GENERATED ALWAYS AS (to_tsvector('simple', content)) STORED,
  PRIMARY KEY (tenant_id, chunk_id),
  FOREIGN KEY (tenant_id, document_id) REFERENCES documents (tenant_id, document_id) ON DELETE CASCADE
);
CREATE INDEX chunks_document_idx ON chunks (tenant_id, document_id, ordinal);
CREATE INDEX chunks_search_idx ON chunks USING gin (search);
-- Approximate nearest neighbours by cosine distance. Filtered by tenant, the search needs iterative scans (the
-- service sets hnsw.iterative_scan per query) or a tenant with few chunks would get too few hits.
CREATE INDEX chunks_embedding_idx ON chunks USING hnsw (embedding vector_cosine_ops);

CREATE TABLE chat_sessions (
  tenant_id      text        NOT NULL CHECK (tenant_id ~ '^[a-z0-9][a-z0-9_-]{0,62}$'),
  session_id     text        NOT NULL CHECK (char_length(session_id) BETWEEN 8 AND 64),
  -- the end user, when the credential carried one: a Supabase user id. NULL for anonymous visitors.
  owner_id       uuid,
  created_at     timestamptz NOT NULL DEFAULT now(),
  last_active_at timestamptz NOT NULL DEFAULT now(),
  message_count  integer     NOT NULL DEFAULT 0 CHECK (message_count >= 0),
  PRIMARY KEY (tenant_id, session_id)
);
CREATE INDEX chat_sessions_owner_idx ON chat_sessions (owner_id) WHERE owner_id IS NOT NULL;
CREATE INDEX chat_sessions_activity_idx ON chat_sessions (tenant_id, last_active_at);

CREATE TABLE chat_messages (
  id         bigint      GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  tenant_id  text        NOT NULL,
  session_id text        NOT NULL,
  role       text        NOT NULL CHECK (role IN ('user', 'assistant')),
  -- already redacted by the service (e-mail addresses, phone and card numbers) unless the operator turned that off
  content    text        NOT NULL CHECK (char_length(content) <= 8000),
  grounded   boolean,
  chunk_ids  text[]      NOT NULL DEFAULT '{}',
  created_at timestamptz NOT NULL DEFAULT now(),
  FOREIGN KEY (tenant_id, session_id) REFERENCES chat_sessions (tenant_id, session_id) ON DELETE CASCADE
);
CREATE INDEX chat_messages_session_idx ON chat_messages (tenant_id, session_id, id);

-- One row per chat request: tokens as the providers reported them, cost from the operator's price table, in
-- millionths of its currency. The daily budget is the sum of a tenant's rows of one UTC day.
CREATE TABLE usage_events (
  id               bigint      GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  tenant_id        text        NOT NULL CHECK (tenant_id ~ '^[a-z0-9][a-z0-9_-]{0,62}$'),
  session_id       text        NOT NULL,
  request_id       text        NOT NULL,
  created_at       timestamptz NOT NULL,
  chat_model       text        NOT NULL,
  embedding_model  text        NOT NULL,
  input_tokens     integer     NOT NULL CHECK (input_tokens >= 0),
  output_tokens    integer     NOT NULL CHECK (output_tokens >= 0),
  embedding_tokens integer     NOT NULL CHECK (embedding_tokens >= 0),
  cost_micros      bigint      NOT NULL CHECK (cost_micros >= 0),
  cost_estimated   boolean     NOT NULL,
  latency_ms       integer     NOT NULL CHECK (latency_ms >= 0),
  outcome          text        NOT NULL CHECK (outcome IN ('answered', 'refused', 'blocked', 'error'))
);
CREATE INDEX usage_events_day_idx ON usage_events (tenant_id, created_at);
