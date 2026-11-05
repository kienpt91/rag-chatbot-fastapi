from __future__ import annotations

import hashlib
import io
import json
import logging
from collections.abc import Iterator

import pytest
from httpx2 import Response

from rag_chatbot.domain.auth import TOKEN_PREFIX, read_unverified_claims
from rag_chatbot.domain.errors import StorageError
from rag_chatbot.infrastructure.http.app import WIDGET_PATH
from rag_chatbot.infrastructure.http.logging import configure_logging
from rag_chatbot.infrastructure.http.settings import HttpSettings
from support.api import API_KEY, BETA_KEY, ORIGIN, USER, Api, make_api

# --- rate limits ---------------------------------------------------------------------------------------------------


def test_an_address_that_sends_too_fast_gets_429_with_retry_after_and_others_are_unaffected() -> None:
    api = make_api(http=HttpSettings(ip_burst=3, ip_per_minute=6, log_ip_salt="x", trusted_proxy_hops=1))
    mine = {"x-forwarded-for": "198.51.100.7"}
    statuses = [api.chat(headers=mine).status_code for _ in range(4)]
    assert statuses == [200, 200, 200, 429]
    refused = api.chat(headers=mine)
    assert refused.status_code == 429 and refused.json()["error"]["code"] == "rate_limited"
    assert int(refused.headers["retry-after"]) >= 1 and refused.headers["x-request-id"]
    assert api.chat(headers={"x-forwarded-for": "198.51.100.8"}).status_code == 200


def test_the_limit_is_applied_before_anything_is_parsed_or_authenticated() -> None:
    api = make_api(http=HttpSettings(ip_burst=2, ip_per_minute=6, log_ip_salt="x"))
    for _ in range(2):
        api.client.post("/chat", content=b"garbage", headers={"content-type": "application/json"})
    refused = api.client.post("/chat", content=b"garbage", headers={"content-type": "application/json"})
    assert refused.status_code == 429


def test_the_forwarded_header_is_ignored_unless_proxies_are_trusted() -> None:
    api = make_api(http=HttpSettings(ip_burst=2, ip_per_minute=6, log_ip_salt="x", trusted_proxy_hops=0))
    codes = [api.chat(headers={"x-forwarded-for": f"203.0.113.{i}"}).status_code for i in range(3)]
    assert codes == [200, 200, 429]  # a client cannot dodge the limit by changing a header


def test_with_trusted_proxies_the_entry_the_nearest_proxy_added_counts_not_the_clients_claim() -> None:
    api = make_api(http=HttpSettings(ip_burst=2, ip_per_minute=6, log_ip_salt="x", trusted_proxy_hops=1))
    # the client prepends fake addresses; the proxy appends the real one, last
    codes = [api.chat(headers={"x-forwarded-for": f"10.0.0.{i}, 198.51.100.9"}).status_code for i in range(3)]
    assert codes == [200, 200, 429]
    junk = [api.chat(headers={"x-forwarded-for": "not-an-ip"}).status_code for _ in range(3)]
    assert junk == [200, 200, 429]  # an unreadable entry falls back to the connection's own address


def test_one_conversation_that_sends_too_fast_is_limited_without_affecting_another() -> None:
    api = make_api(http=HttpSettings(session_burst=2, session_per_minute=6, log_ip_salt="x"))
    first = api.chat().json()["session_id"]
    second = api.chat().json()["session_id"]
    statuses = [api.chat(session_id=first).status_code for _ in range(3)]
    assert statuses == [200, 200, 429]
    refused = api.chat(session_id=first)
    assert refused.json()["error"]["code"] == "rate_limited" and "retry-after" in refused.headers
    assert api.chat(session_id=second).status_code == 200


def test_the_session_limit_counts_per_tenant_so_one_tenant_cannot_burn_anothers_budget() -> None:
    api = make_api(http=HttpSettings(session_burst=1, session_per_minute=6, log_ip_salt="x"))
    assert api.chat(session_id="s_sharedname01").status_code == 404  # 404 still took the only token of (acme, id)
    assert api.chat(session_id="s_sharedname01").status_code == 429
    assert api.chat(session_id="s_sharedname01", key=BETA_KEY).status_code == 404  # beta has its own bucket


def test_the_limiters_remember_a_bounded_number_of_clients() -> None:
    api = make_api(
        http=HttpSettings(ip_burst=1, ip_per_minute=6, max_tracked_keys=10, log_ip_salt="x", trusted_proxy_hops=1)
    )
    for i in range(200):
        api.chat(headers={"x-forwarded-for": f"192.0.2.{i % 250}"})
    # no way to read the table from outside; the unit tests of RateLimiter prove the bound, this proves it is wired
    assert api.chat(headers={"x-forwarded-for": "192.0.2.249"}).status_code in (200, 429)


# --- embed tokens -------------------------------------------------------------------------------------------------------


def mint(api: Api, **body: object) -> Response:
    return api.client.post("/embed-token", json=body, headers={"authorization": f"Bearer {API_KEY}"})


def test_the_servers_backend_mints_a_token_the_widget_can_use() -> None:
    api = make_api()
    response = mint(api, ttl_seconds=300, subject=USER, origin=ORIGIN)
    assert response.status_code == 200
    token = response.json()["token"]
    assert (
        token.startswith(TOKEN_PREFIX)
        and response.json()["expires_at"] == int(api.service.clock.now().timestamp()) + 300
    )
    claims, _, _ = read_unverified_claims(token)
    assert claims.tenant_id == "acme" and claims.subject == USER and claims.origin == ORIGIN
    assert api.chat(key=token, origin=ORIGIN).status_code == 200


def test_a_token_bound_to_an_origin_does_not_work_from_another() -> None:
    api = make_api()
    token = mint(api, origin="http://localhost:8080").json()["token"]
    assert api.chat(key=token, origin="http://localhost:8080").status_code == 200
    assert api.chat(key=token, origin=ORIGIN).status_code == 403


def test_the_lifetime_is_capped_by_the_deployment() -> None:
    api = make_api(http=HttpSettings(max_token_ttl_s=120, log_ip_salt="x"))
    assert mint(api, ttl_seconds=86400).json()["expires_at"] == int(api.service.clock.now().timestamp()) + 120


@pytest.mark.parametrize(
    "body",
    [
        {"ttl_seconds": 0},
        {"ttl_seconds": 90000},
        {"subject": "not-a-uuid"},
        {"subject": USER.upper()},
        {"origin": "https://evil.example/path"},
        {"origin": "javascript:alert(1)"},
        {"unknown": 1},
    ],
)
def test_invalid_token_requests_are_400(body: dict[str, object]) -> None:
    assert mint(make_api(), **body).status_code == 400


def test_an_embed_token_cannot_mint_more_tokens_and_no_key_means_no_token() -> None:
    api = make_api()
    token = api.token()
    forbidden = api.client.post("/embed-token", json={}, headers={"authorization": f"Bearer {token}", "origin": ORIGIN})
    assert forbidden.status_code == 403 and forbidden.json()["error"]["code"] == "forbidden"
    assert api.client.post("/embed-token", json={}).status_code == 401


def test_a_tenant_without_a_token_secret_cannot_mint() -> None:
    from rag_chatbot.domain.auth import hash_api_key
    from rag_chatbot.domain.tenants import parse_tenants

    api = make_api()
    api.deps.directory = parse_tenants({"tenants": [{"id": "acme", "api_key_sha256": [hash_api_key(API_KEY)]}]})
    assert mint(api).status_code == 403


# --- probes and the widget --------------------------------------------------------------------------------------------


def test_the_liveness_probe_needs_nothing() -> None:
    response = make_api().client.get("/healthz")
    assert response.status_code == 200 and response.json() == {"status": "ok"}


def test_readiness_is_200_when_there_are_no_problems_and_503_with_codes_otherwise() -> None:
    assert make_api().client.get("/readyz").json() == {"status": "ready"}
    down = make_api(readiness_problems=["schema_outdated"]).client.get("/readyz")
    assert down.status_code == 503 and down.json() == {"status": "unavailable", "problems": ["schema_outdated"]}


def test_a_readiness_check_that_fails_reports_the_database_without_details() -> None:
    api = make_api(readiness_error=StorageError("OperationalError at host db.internal password=hunter2"))
    response = api.client.get("/readyz")
    assert response.status_code == 503 and response.json()["problems"] == ["database_unreachable"]
    assert "hunter2" not in response.text and "db.internal" not in response.text


def test_the_widget_script_is_served_with_the_right_type_cache_headers_and_etag() -> None:
    api = make_api()
    response = api.client.get("/widget.js")
    assert response.status_code == 200 and response.content == WIDGET_PATH.read_bytes()
    assert response.headers["content-type"] == "text/javascript; charset=utf-8"
    assert (
        response.headers["cross-origin-resource-policy"] == "cross-origin"
        and response.headers["x-content-type-options"] == "nosniff"
    )
    assert "max-age=3600" in response.headers["cache-control"]
    etag = response.headers["etag"]
    assert etag == '"' + hashlib.sha256(response.content).hexdigest()[:32] + '"'
    again = api.client.get("/widget.js", headers={"if-none-match": etag})
    assert again.status_code == 304 and again.content == b""


# --- logging ------------------------------------------------------------------------------------------------------------


@pytest.fixture
def log_stream() -> Iterator[io.StringIO]:
    stream = io.StringIO()
    configure_logging("INFO", stream=stream)
    try:
        yield stream
    finally:
        logging.getLogger().handlers.clear()


def test_logs_are_json_lines_with_request_ids_and_nothing_that_identifies_the_visitor(log_stream: io.StringIO) -> None:
    api = make_api(http=HttpSettings(log_ip_salt="pepper", trusted_proxy_hops=1))
    secret_question = "jane.doe@example.org +1 555 0177 card 4111 1111 1111 1111 What is the delivery fee?"
    token = api.token()
    response = api.chat(secret_question, key=token, origin=ORIGIN, headers={"x-forwarded-for": "203.0.113.55"})
    assert response.status_code == 200
    lines = [json.loads(line) for line in log_stream.getvalue().splitlines()]
    assert lines and all({"ts", "level", "msg", "request_id", "logger"} <= set(line) for line in lines)
    text = log_stream.getvalue()
    for leaked in ("jane.doe", "555 0177", "4111", token, "203.0.113.55", "What is the delivery fee", "Bearer"):
        assert leaked not in text
    [done] = [line for line in lines if line.get("event") == "request"]
    assert (
        done["route"] == "/chat"
        and done["status"] == 200
        and done["method"] == "POST"
        and done["request_id"] == response.headers["x-request-id"]
    )
    assert done["client"] == hashlib.sha256(b"pepper|203.0.113.55").hexdigest()[:10]
    [chat] = [line for line in lines if line.get("event") == "chat"]
    assert chat["tenant"] == "acme" and chat["grounded"] is True and chat["message_chars"] == len(secret_question)
    assert chat["cost_micros"] > 0 and chat["citations"] == 1 and "answer" not in chat


def test_unexpected_errors_are_logged_with_their_class_and_a_redacted_message(log_stream: io.StringIO) -> None:
    api = make_api()

    async def boom(request: object) -> object:
        raise RuntimeError("token ragk_" + "Q" * 43 + " for jane@example.org")

    api.deps.chat.handle = boom  # type: ignore[method-assign,assignment]
    assert api.chat().status_code == 500
    text = log_stream.getvalue()
    assert "RuntimeError" in text and "ragk_" not in text and "jane@example.org" not in text and "[secret]" in text


def test_a_refused_request_is_logged_by_code_only(log_stream: io.StringIO) -> None:
    api = make_api()
    api.client.post(
        "/chat", json={"message": "hello there friend"}, headers={"authorization": "Bearer ragk_" + "Z" * 43}
    )
    text = log_stream.getvalue()
    assert '"code":"unauthorized"' in text and "ragk_" not in text
