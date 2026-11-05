# 3. Row-level security by tenant and by user

Status: accepted

## Context

One deployment serves several sites, and a Supabase client may read conversations directly. Mistakes in
application SQL must not cross tenants, and a client must see only its own conversations.

## Decision

Every table carries `tenant_id` and has row-level security forced. The service role `rag_app` (a login role is a
member of it) works inside a transaction that sets `rag.tenant_id`; its policies compare it with the row, and an unset
value matches nothing. The Supabase roles get one narrow policy: `authenticated` may read sessions and messages whose
`owner_id = auth.uid()`; `anon` has nothing. The ledger is append-only for the service. The migrations are plain SQL and
create the Supabase roles and a stand-in `auth.uid()` only when they do not exist.

## Consequences

Row-level security here is defence in depth against bugs in the service: a role that can run arbitrary SQL can set
the tenant itself (SECURITY.md). The tests run on a real database and a mutation check showed that weakening a policy
fails ten of them. Every query also names the tenant explicitly, so correctness never depends on the policies alone.
