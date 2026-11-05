-- Extensions, roles and the helpers the row-level security policies use.
-- Runs on any PostgreSQL 14+ with pgvector 0.8 or newer, and on Supabase (where the roles and auth.uid() exist).

CREATE EXTENSION IF NOT EXISTS vector;

DO $$
BEGIN
  IF (SELECT extversion FROM pg_extension WHERE extname = 'vector') < '0.8.0' THEN
    RAISE EXCEPTION 'pgvector 0.8.0 or newer is required (iterative index scans keep filtered searches complete)';
  END IF;
END
$$;

-- The roles of a Supabase project. On plain PostgreSQL they are created here; on Supabase they already exist.
DO $$
BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'anon') THEN
    CREATE ROLE anon NOLOGIN;
  END IF;
  IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'authenticated') THEN
    CREATE ROLE authenticated NOLOGIN;
  END IF;
END
$$;

-- The role this service runs as (a login role that is a member of it, see docs/deployment.md). It is not a superuser
-- and does not bypass row-level security: every table below is limited to the tenant the request is for.
DO $$
BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'rag_app') THEN
    CREATE ROLE rag_app NOLOGIN;
  END IF;
END
$$;

-- auth.uid(): the id of the end user, read from the JWT claims the API gateway sets. Supabase provides it; elsewhere
-- this stand-in reads the same setting. It is created only when it does not exist, so on Supabase the real one stays.
CREATE SCHEMA IF NOT EXISTS auth;
DO $$
BEGIN
  IF to_regprocedure('auth.uid()') IS NULL THEN
    CREATE FUNCTION auth.uid() RETURNS uuid
      LANGUAGE sql STABLE
      AS $f$
        SELECT coalesce(
          nullif(current_setting('request.jwt.claim.sub', true), ''),
          (nullif(current_setting('request.jwt.claims', true), '')::jsonb ->> 'sub')
        )::uuid
      $f$;
  END IF;
END
$$;
GRANT USAGE ON SCHEMA auth TO anon, authenticated, rag_app;
GRANT EXECUTE ON FUNCTION auth.uid() TO anon, authenticated, rag_app;

-- rag.current_tenant(): the tenant the current transaction works for, set by the service with
-- set_config('rag.tenant_id', ..., true) at the start of every transaction. NULL when unset, and a NULL never
-- satisfies a policy: a statement that forgot to say whose data it touches sees nothing.
CREATE SCHEMA IF NOT EXISTS rag;
CREATE FUNCTION rag.current_tenant() RETURNS text
  LANGUAGE sql STABLE
  AS $$ SELECT nullif(current_setting('rag.tenant_id', true), '') $$;
GRANT USAGE ON SCHEMA rag TO rag_app;
GRANT EXECUTE ON FUNCTION rag.current_tenant() TO rag_app;
