-- Run once after the migrations, as the database owner: the login role the service connects as. It is a member of
-- rag_app (created by the migrations), so row-level security applies to it; it is not a superuser and owns nothing.
-- usage: psql -v ON_ERROR_STOP=1 -v service_password="$RAG_DB_PASSWORD" -f docker/create-service-role.sql
SELECT format('CREATE ROLE rag_svc LOGIN PASSWORD %L IN ROLE rag_app', :'service_password')
WHERE NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'rag_svc') \gexec
SELECT format('ALTER ROLE rag_svc PASSWORD %L', :'service_password') \gexec
