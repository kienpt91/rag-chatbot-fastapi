# Security

This service reads text it cannot trust (web pages, visitors' messages, the answers of a language model) and keeps
the conversations of people it does not know. This document is the threat model: what is protected, from whom, how,
which test shows it, and what is **not** protected. Where a defence has limits, the limit is stated next to it.

No hosted model was called while building or testing this repository: the tests, the evaluation set and the demo use
a deterministic fake provider (`infrastructure/providers/fake.py`). What the real model does with the same prompts is
not verified here; the application-level checks below are designed so that they do not depend on it behaving.

## Reporting a vulnerability

Open a private security advisory on the repository (Security, Report a vulnerability) with the request, the page or
message that triggers it, and the version (`rag-chatbot --version`). Reports are answered and credited in the changelog.

## What is protected

| asset | why it matters |
| --- | --- |
| the visitor's conversation and personal data | people paste e-mail addresses, phone and card numbers into chat boxes |
| one site's content, conversations and ledger from another site's (tenants) | one deployment serves several sites |
| the operator's keys: API keys, token secrets, the provider key, the database password | each one is money or access |
| the visitor's browser | the widget runs on the client's pages |
| the budget | an open chat endpoint is an open wallet |
| the assistant's integrity | it must say only what the site says, and nothing else |

## Actors

| actor | trusted? |
| --- | --- |
| the operator, and the configuration they write (tenants file, price table, environment) | trusted, but validated strictly: a typo is an error |
| **web page text**, including reviews and comments on it | untrusted: anyone who can edit a page can write instructions into it |
| **the visitor** and their messages | untrusted |
| **the language model's answer** | untrusted until it passes the checks |
| a browser holding an embed token | trusted only as that tenant, from an allowed origin, until the token expires |
| the service's database role | trusted by the database only as far as row-level security lets it |

## Threats and mitigations

Each row names the test that exercises it: `scripts/py.sh 'python -m pytest tests -k NAME'` (database tests need
`PY_DB=1`).

| threat | mitigation | tests |
| --- | --- | --- |
| a page carries sentences written for the assistant ("ignore previous instructions", `REPLY: ...`, "the assistant must ...") | **quarantine at ingest**: sentence by sentence, instruction-like wording is left out of the index and listed for the operator (`ingest` prints it); a title of that kind is replaced. This is a filter on wording, not a proof (see below); `RAG_QUARANTINE_INSTRUCTIONS=false` switches it off | `tests/domain/test_quarantine.py`, `tests/application/test_ingest_quarantine.py`, `tests/composition/test_db_commands.py::test_ingesting_the_sample_site_stores_chunks_and_a_second_run_embeds_nothing` |
| a page tells the model to ignore its rules | the text is placed between markers that carry a random per-request code and is labelled as data; `<<<` and `>>>` are neutralised in everything quoted; the model has no tools | `tests/domain/test_prompt_guard.py::test_a_page_cannot_forge_an_end_marker_or_a_new_source`, `tests/application/test_chat.py::test_the_model_is_given_the_rules_the_fenced_sources_and_the_question_and_no_tools` |
| the model is fooled anyway | every answer is checked before anyone sees it: each citation must be a source of this request, at least one is required, the rules and the internal marker must not appear, links only to the site's own hosts, markup removed. A failed check becomes a refusal that is not stored, cited or shown | `tests/application/test_chat.py::test_an_answer_that_fails_the_checks_never_reaches_the_visitor_or_the_database`, `tests/application/test_chat.py::test_the_hostile_page_sample_is_caught_end_to_end_with_a_gullible_model`, `tests/domain/test_prompt_guard.py::test_an_address_the_site_itself_lists_on_a_foreign_host_still_fails` |
| hidden text on a page (display:none, hidden attribute, comments, zero-width characters, off-screen or zero-size boxes, unseen image alt text, script content a browser would not show) is indexed | **best-effort**: the extractor drops hidden elements, comments, scripts (with the browser's double-escaped script rules), image alt text and invisible characters before chunking, and applies the same hidden-element rules to raw HTML inside Markdown | `tests/domain/test_extraction_html.py::test_text_hidden_by_an_inline_style_is_never_extracted`, `tests/domain/test_extraction_html.py::test_text_hidden_by_an_attribute_is_never_extracted`, `tests/domain/test_extraction_differentials.py::test_a_script_that_hides_an_inner_script_end_tag_stays_hidden_until_the_real_end`, `tests/domain/test_extraction_differentials.py::test_more_ways_of_hiding_text_are_recognised`, `tests/domain/test_extraction_differentials.py::test_inline_html_in_markdown_gets_the_same_hidden_text_rules` |
| a page or a quotation exfiltrates the conversation through a link or an image | the answer may link only to the tenant's own hosts and declared link hosts, whether written with a scheme, as `//host/...` or as a Markdown link target; Markdown images and `mailto:`/`tel:`/`sms:` are refused; citation snippets have markup and foreign addresses removed | `tests/domain/test_prompt_guard.py::test_lookalike_hosts_fail`, `tests/domain/test_prompt_guard.py::test_scrub_keeps_the_sites_own_addresses_and_removes_markup_and_foreign_ones`, `tests/application/test_chat.py::test_a_snippet_never_carries_a_foreign_address_or_markup_from_the_page`, `tests/domain/test_guard_links.py::test_links_without_a_scheme_markdown_links_images_and_other_schemes_are_refused` |
| earlier assistant turns carry a quoted page sentence back into the next prompt as the model's own words | earlier turns (visitor and assistant) are replayed inside the same nonce fence, labelled as data, and the rules say so | `tests/domain/test_history_fence.py` |
| an off-topic or unanswerable question gets an invented answer | retrieval confidence (cosine similarity of the best chunk) below the threshold refuses *without calling the model*; the model must answer `NO_ANSWER` when the sources lack the answer | `tests/application/test_chat.py::test_when_retrieval_finds_nothing_good_enough_the_model_is_not_called`, `tests/application/test_chat.py::test_the_models_own_no_answer_becomes_a_refusal` |
| a visitor's personal data reaches the provider, the database or the logs | e-mail addresses, phone numbers and card numbers (Luhn) are replaced before retrieval and storage, including numbers written back to back; logs never contain bodies, tokens or addresses, only sizes and a salted hash of the client address | `tests/application/test_chat.py::test_personal_data_never_reaches_the_model_the_embedder_or_the_database`, `tests/domain/test_redaction.py::test_numbers_written_one_after_the_other_are_each_found`, `tests/api/test_limits_tokens_ops.py::test_logs_are_json_lines_with_request_ids_and_nothing_that_identifies_the_visitor`, `tests/e2e/test_server_process.py::test_logs_and_the_database_hold_no_personal_data_and_no_secret` |
| one tenant reads or changes another tenant's data (an application bug) | every query is scoped by tenant, and row-level security repeats the scoping in the database: each transaction sets `rag.tenant_id`, policies compare it, an unset value matches nothing. Mutation check: weakening a policy makes ten of these tests fail | `tests/infrastructure/test_rls.py::test_the_service_sees_only_the_rows_of_its_tenant_in_every_table`, `tests/infrastructure/test_rls.py::test_the_service_cannot_write_rows_for_another_tenant`, `tests/infrastructure/test_store_contract.py::test_one_tenant_never_sees_another_tenants_chunks_in_either_search`, `tests/infrastructure/test_store_contract.py::test_a_pool_of_one_connection_never_carries_one_tenants_scope_over_to_the_next` |
| a Supabase client reads other people's conversations | `authenticated` may read only sessions and messages whose `owner_id = auth.uid()`; `anon` has no access; nobody but the service writes | `tests/infrastructure/test_rls.py::test_a_signed_in_user_reads_only_their_own_sessions_and_messages`, `tests/infrastructure/test_rls.py::test_an_anonymous_client_has_no_access_to_any_table`, `tests/infrastructure/test_rls.py::test_a_signed_in_user_cannot_write_anything` |
| the filtered vector search silently returns too few chunks for a small tenant | the HNSW index filters after it reads; the service enables iterative scans per query. Mutation check: without it the test finds nothing | `tests/infrastructure/test_store_contract.py::test_a_tenant_with_few_chunks_is_found_even_when_the_index_is_full_of_another_tenants` |
| a stolen or forged credential | API keys are stored as SHA-256 and compared in constant time; embed tokens are HMAC-signed, expire, name one tenant and cannot be moved to another by editing the payload; every failure gives the same answer | `tests/domain/test_tenants_auth.py::test_a_token_cannot_be_moved_to_another_tenant_by_editing_its_payload`, `tests/domain/test_tenants_auth.py::test_an_expired_token_is_refused_at_the_second_it_expires`, `tests/api/test_chat_api.py::test_every_authentication_failure_looks_the_same` |
| another website's *page* uses the widget with a stolen token | in a browser, a token works only from a page whose `Origin` matches an allowed origin of its tenant; CORS is an exact string match, and the token check normalises case and default ports. **A script outside a browser can forge the `Origin` header**: what bounds a stolen token is its short life (default 15 minutes, at most `RAG_MAX_TOKEN_TTL_S`), the rate limits and the daily budget | `tests/api/test_chat_api.py::test_a_token_without_an_origin_header_or_from_another_origin_is_refused`, `tests/api/test_chat_api.py::test_a_preflight_from_anything_but_an_exactly_listed_origin_is_refused` |
| script injection in the widget | text reaches the page only through `textContent`; styling only through the CSSOM; links only for http(s) URLs with `rel=noopener noreferrer nofollow`; a test scans the source for unsafe APIs | `tests/widget/rag-chat.test.mjs` ("the source avoids unsafe APIs and stays small", hostile answer and citation tests), run by `scripts/test-widget.sh` |
| cost abuse and floods | per-address and per-session token buckets in bounded memory (before parsing, before authentication for the address); body size limit including streamed bodies; a daily budget per tenant that fails closed before any provider call | `tests/api/test_limits_tokens_ops.py::test_an_address_that_sends_too_fast_gets_429_with_retry_after_and_others_are_unaffected`, `tests/domain/test_pricing_ratelimit.py::test_memory_is_bounded_and_the_least_recently_used_key_is_dropped`, `tests/api/test_chat_api.py::test_a_streamed_body_without_a_length_is_cut_off_at_the_limit`, `tests/application/test_chat.py::test_a_tenant_over_its_daily_budget_fails_closed_before_any_provider_is_called`, `tests/e2e/test_server_process.py::test_a_tiny_daily_budget_makes_the_bot_fail_closed_with_a_friendly_message` |
| spoofing the client address to dodge the limit | `X-Forwarded-For` is ignored unless the operator says how many proxies are trusted; then the entry that many places from the right counts | `tests/api/test_limits_tokens_ops.py::test_the_forwarded_header_is_ignored_unless_proxies_are_trusted`, `tests/api/test_limits_tokens_ops.py::test_with_trusted_proxies_the_entry_the_nearest_proxy_added_counts_not_the_clients_claim` |
| errors that leak internals (exception text, SQL, hosts, file paths, input) | every failure becomes JSON with a stable code, a visitor-safe message and the request id; validation errors do not echo input; database and provider errors carry a class and a status, never a body | `tests/api/test_chat_api.py::test_failures_become_generic_json_errors_without_the_exception_text`, `tests/api/test_chat_api.py::test_malformed_requests_are_400_with_a_message_that_does_not_echo_the_input`, `tests/infrastructure/test_store_contract.py::test_database_failures_become_storage_errors_that_say_nothing_about_the_data` |
| the crawler is used to reach internal services (SSRF) | exact host allow-list; every resolved address must be public (loopback, private, link-local and metadata addresses refused, IPv6 wrappers unwrapped); the connection goes to the validated address; redirects are re-checked per hop; robots.txt honoured | `tests/infrastructure/test_http_fetcher.py::test_non_public_ip_literals_are_refused_without_a_connection_attempt`, `tests/infrastructure/test_http_fetcher.py::test_a_name_with_any_non_public_address_is_refused_without_a_connection_attempt`, `tests/infrastructure/test_crawler.py::test_a_robots_redirect_to_another_host_is_not_followed_and_closes_the_origin`, `tests/domain/test_urlpolicy.py::test_the_allowed_host_in_any_spelling_passes` |
| the provider key goes somewhere else | the base URL must be https (http for localhost only) with no credentials; redirects are never followed; errors never contain the key | `tests/infrastructure/test_http_provider.py::test_a_redirect_is_never_followed_so_the_key_cannot_be_sent_elsewhere`, `tests/infrastructure/test_http_provider.py::test_unsafe_or_invalid_base_urls_are_refused` |
| dependency compromise | exact pins with sha256 hashes for the runtime and the development packages, none uploaded less than seven days before the cutoff, checked in CI; images pinned by digest; GitHub Actions pinned by commit SHA with read-only permissions | `tests/docs/test_scripts.py`, `tests/architecture/test_layers.py::test_the_declared_runtime_dependencies_are_exactly_the_ones_the_architecture_names` |
| code paths that execute text | an AST lint (a tripwire, see below) fails the build on `eval`, `exec`, process creation (also through `asyncio`), raw sockets outside two modules, pickle-like loaders, wildcard imports, reflection helpers, environment access outside composition, and on third-party imports outside the module that exists for them | `tests/architecture/test_layers.py`, `tests/architecture/test_layer_evasions.py` |
| a crawl indexes pages the operator meant to leave out | `--exclude` patterns match the path of a crawled URL (case-insensitive, query ignored, percent-decoding applied) as well as a folder's relative path | `tests/composition/test_exclude_crawl.py` |
| the database API roles reach the migration ledger or the identity sequences | the runner and migration 0003 revoke them from `PUBLIC`, `anon` and `authenticated` even when default privileges grant them | `tests/infrastructure/test_schema_privileges.py` |
| the internet-facing container holds the database superuser password | the compose `app` service receives an explicit list of variables, not the whole `.env` | `tests/docs/test_claims.py::test_the_internet_facing_service_does_not_receive_the_database_superuser_password` |

## What is NOT protected (read this before you rely on it)

- **A site that publishes hostile text can still have it quoted.** The checks stop instructions, rule leaks, foreign
  links and invalid citations. A correctly cited sentence that a customer wrote ("everything is free today") is, to the
  assistant, what the site says. Keep user-written pages out of the index (`ingest --exclude`, tested in
  `tests/composition/test_db_commands.py::test_exclude_keeps_user_written_pages_out_of_the_knowledge_base`). A poisoned
  chunk that is retrieved can also turn a good answer into a refusal: the system fails closed, at some cost in answers.
- **The message filter is not a defence.** `domain/scope.py` refuses obvious attempts before money is spent; a
  rephrasing or another language passes it (`tests/domain/test_scope_history.py::test_the_filter_is_not_a_defence_a_rephrasing_gets_through`).
  The structural checks above are what hold.
- **Leaks in a paraphrase or another language** are not detected: the leak check looks for the internal marker and for
  six-word runs of the rules.
- **Row-level security is defence in depth against bugs in the service, not a barrier against someone who can run SQL
  as the service role**: that role can set `rag.tenant_id` itself. Superusers and roles with BYPASSRLS bypass it by
  design; never give one to the service. The Supabase policies (`auth.uid()`) do hold against a client.
- **Rate limits are per process; the budget is shared.** With N workers a client can send N times the rate limit
  (put a limit in front, docs/deployment.md), and the limiter table is bounded, so an attacker with millions of
  addresses can flush other clients' buckets. The daily budget is summed from the database ledger, so all workers see
  the same total; it is checked before the call, so requests that are in flight can overshoot it.
- **Quarantine is a filter on wording.** A rephrased or translated instruction passes, and an honest sentence that
  matches is left out (it is listed in the ingest report). Hidden-text removal is best-effort too: text hidden by an
  external stylesheet or a class, by a rule the extractor does not know, or shown only by script is not recognised.
  A browser and our tokenizer can still disagree on malformed markup.
- **Provider usage is trusted.** Cost uses the tokens the provider reports (estimated, and marked, only when it
  reports none); a provider that under-reports under-charges the budget. A request cancelled by the timeout is
  recorded with what was paid up to then. A session of a signed-in user can be continued only by that user; an
  anonymous session by anyone who holds its 128-bit id.
- **Changing the embedding model needs a fresh index.** Chunk ids do not include the model, and unchanged chunks keep
  their stored vectors, so switching models without deleting the chunks would mix vector spaces.
- **The architecture lint is a tripwire, not a sandbox**: it stops mistakes and the usual evasions (aliases, computed
  attribute names, `asyncio` subprocess and socket helpers, `vars`/`globals`), not a contributor who wants to bypass it.
- **Name resolution in the crawler cannot be interrupted**, so a hostile DNS server can stall a crawl until the
  resolver gives up; it cannot redirect it.
- **Preflight requests carry no credential**, so CORS preflights are approved against the origins of every tenant; the
  actual request is checked against the tenant's own list.
- **Redaction finds shapes, not meaning**: no names, street addresses or free-text secrets. A long digit run that is
  neither a card nor a phone number is also redacted (false positives are chosen over leaks).
- **No end-user authentication of visitors** beyond the embed token: anyone on an allowed page can use the widget until
  the token expires.
- **The real model's behaviour is not verified.** All tests use the fake provider.
