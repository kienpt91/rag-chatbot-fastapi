-- Row-level security.
--
-- Two kinds of client, two sets of policies:
--
-- * The service (role rag_app, or a login role that is a member of it) works for one tenant per transaction and
--   says which with set_config('rag.tenant_id', ...). Its policies compare tenant_id to rag.current_tenant(), so a
--   query that forgets its WHERE clause, or a bug that mixes tenants up, returns and changes nothing of another
--   tenant's. This is defence in depth against mistakes in the service. It is NOT a barrier against someone who
--   can run arbitrary SQL as that role: they can set rag.tenant_id themselves. Keep that role's password secret.
--
-- * A Supabase client (roles anon and authenticated, through the API gateway with the user's JWT) may read ONLY the
--   sessions and messages of the signed-in user (owner_id = auth.uid()), and nothing else, and write nothing.
--   anon has no access to any table. Content, embeddings and the usage ledger are not readable by clients at all.
--
-- FORCE makes the policies apply to the table owner too. Superusers and roles with BYPASSRLS (including Supabase's
-- service_role) still bypass row-level security by design: never hand those to the service or to a browser.

ALTER TABLE documents     ENABLE ROW LEVEL SECURITY;
ALTER TABLE chunks        ENABLE ROW LEVEL SECURITY;
ALTER TABLE chat_sessions ENABLE ROW LEVEL SECURITY;
ALTER TABLE chat_messages ENABLE ROW LEVEL SECURITY;
ALTER TABLE usage_events  ENABLE ROW LEVEL SECURITY;
ALTER TABLE documents     FORCE ROW LEVEL SECURITY;
ALTER TABLE chunks        FORCE ROW LEVEL SECURITY;
ALTER TABLE chat_sessions FORCE ROW LEVEL SECURITY;
ALTER TABLE chat_messages FORCE ROW LEVEL SECURITY;
ALTER TABLE usage_events  FORCE ROW LEVEL SECURITY;

-- no PUBLIC access to anything; grant exactly what each role needs
REVOKE ALL ON documents, chunks, chat_sessions, chat_messages, usage_events FROM PUBLIC, anon, authenticated;

GRANT SELECT, INSERT, UPDATE, DELETE ON documents, chunks, chat_sessions, chat_messages TO rag_app;
GRANT SELECT, INSERT ON usage_events TO rag_app;  -- the ledger is append-only for the service
GRANT USAGE ON SEQUENCE chat_messages_id_seq, usage_events_id_seq TO rag_app;

GRANT SELECT ON chat_sessions, chat_messages TO authenticated;

-- the service: one tenant at a time, for every command
CREATE POLICY documents_tenant ON documents FOR ALL TO rag_app
  USING (tenant_id = rag.current_tenant()) WITH CHECK (tenant_id = rag.current_tenant());
CREATE POLICY chunks_tenant ON chunks FOR ALL TO rag_app
  USING (tenant_id = rag.current_tenant()) WITH CHECK (tenant_id = rag.current_tenant());
CREATE POLICY chat_sessions_tenant ON chat_sessions FOR ALL TO rag_app
  USING (tenant_id = rag.current_tenant()) WITH CHECK (tenant_id = rag.current_tenant());
CREATE POLICY chat_messages_tenant ON chat_messages FOR ALL TO rag_app
  USING (tenant_id = rag.current_tenant()) WITH CHECK (tenant_id = rag.current_tenant());
CREATE POLICY usage_events_tenant_read ON usage_events FOR SELECT TO rag_app
  USING (tenant_id = rag.current_tenant());
CREATE POLICY usage_events_tenant_insert ON usage_events FOR INSERT TO rag_app
  WITH CHECK (tenant_id = rag.current_tenant());

-- a signed-in Supabase user: their own conversations, read only
CREATE POLICY chat_sessions_owner_read ON chat_sessions FOR SELECT TO authenticated
  USING (owner_id IS NOT NULL AND owner_id = auth.uid());
CREATE POLICY chat_messages_owner_read ON chat_messages FOR SELECT TO authenticated
  USING (EXISTS (
    SELECT 1 FROM chat_sessions s
    WHERE s.tenant_id = chat_messages.tenant_id
      AND s.session_id = chat_messages.session_id
      AND s.owner_id IS NOT NULL
      AND s.owner_id = auth.uid()
  ));
