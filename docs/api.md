# HTTP API

All bodies are JSON. Errors are always `{"error": {"code", "message", "request_id"}}`; `message` is written for the
visitor and safe to display; the `X-Request-Id` header carries the same id as the body and the logs.

## `POST /chat`

`Authorization: Bearer <API key or embed token>`, `Content-Type: application/json`, body
`{"message": "...", "session_id": "s_..."}` (`message` 1 to 2,000 characters; `session_id` omitted for the first
message: the server issues it and returns it).

```json
{"session_id": "s_...", "answer": "The bakery opens at 7:00 ...", "grounded": true,
 "citations": [{"id": "ck_...", "title": "Opening hours", "url": "https://www.example.com/hours", "snippet": "..."}],
 "refusal_reason": null, "request_id": "r_..."}
```

`answer` is plain text: render it with `textContent`. `grounded: false` means the assistant refused (`refusal_reason`:
`low_confidence`, `no_answer`, `unsafe_answer` or `out_of_scope`); `answer` then holds the refusal message. A browser
call with an embed token needs an `Origin` header matching the tenant's allowed origins.

| status | code | meaning |
| --- | --- | --- |
| 400 | `invalid_request` | malformed body, empty or too long message, bad session id |
| 401 | `unauthorized` | missing, malformed, wrong or expired credential (all look alike); `WWW-Authenticate: Bearer` |
| 403 | `origin_not_allowed`, `forbidden` | the credential is valid but the page's origin is not allowed / a token tried to mint tokens |
| 404 | `session_not_found` | unknown session, or one of another tenant |
| 409 | `session_full` | the conversation reached its length limit |
| 413, 415 | `payload_too_large`, `unsupported_media_type` | body over the limit / not `application/json` |
| 429 | `rate_limited` | `Retry-After` seconds |
| 503 | `budget_exhausted`, `provider_unavailable`, `storage_unavailable` | fail closed; the budget message tells the visitor to contact the site |

## `POST /embed-token`

API key only. Body `{"ttl_seconds": 900, "subject": "<uuid>", "origin": "https://..."}` (all optional; the lifetime is
capped by `RAG_MAX_TOKEN_TTL_S`). Returns `{"token": "ragt1....", "expires_at": 1800000000}`. A token names one tenant,
expires, and works only from an allowed origin. `subject` is the end user's Supabase user id: the conversation is then
readable by that user through row-level security.

## `GET /healthz`, `GET /readyz`, `GET /widget.js`

`/healthz` is `{"status": "ok"}`. `/readyz` is `{"status": "ready"}` or 503 `{"status": "unavailable", "problems": [...]}`
with codes (`schema_outdated`, `dimension_mismatch`, `vector_missing`, `database_unreachable`). `/widget.js` is the
drop-in script (cacheable, `ETag`).

## CORS

`OPTIONS /chat` is approved for an origin that exactly matches an allowed origin of any tenant (a preflight carries no
credential); the real request is checked against the credential's tenant. No credentials mode, no wildcard.
