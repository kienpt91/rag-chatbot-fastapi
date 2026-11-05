from __future__ import annotations

import asyncio
import json
from collections.abc import Callable, Coroutine
from typing import Any

import pytest
from fastapi.testclient import TestClient
from httpx2 import Response

from rag_chatbot.application.chat import ChatRequest as ChatCommand
from rag_chatbot.domain.errors import (
    BudgetExhaustedError,
    ProviderError,
    SessionFullError,
    SessionNotFoundError,
    StorageError,
)
from rag_chatbot.domain.model import ChatReply
from support.api import API_KEY, BETA_KEY, ORIGIN, OTHER_ORIGIN, Api, make_api
from support.fakes import ScriptedChat


def failing(error: Exception) -> Callable[[ChatCommand], Coroutine[Any, Any, ChatReply]]:
    async def raising(request: ChatCommand) -> ChatReply:
        raise error

    return raising


def body(response: Response) -> dict[str, Any]:
    parsed = response.json()
    assert isinstance(parsed, dict)
    return parsed


def test_a_server_side_call_with_an_api_key_gets_a_cited_answer() -> None:
    api = make_api()
    response = api.chat("When does the bakery open on weekdays?")
    assert response.status_code == 200
    data = body(response)
    assert data["grounded"] is True and data["refusal_reason"] is None
    assert data["answer"] == "The bakery opens at 7:00 in the morning from Monday to Friday."
    [citation] = data["citations"]
    assert citation["url"] == "https://www.acme.example/hours" and citation["title"] == "Opening hours"
    assert citation["id"].startswith("ck_") and "7:00" in citation["snippet"]
    assert data["session_id"].startswith("s_") and data["request_id"] == response.headers["x-request-id"]


def test_every_response_carries_the_request_id_and_hardening_headers() -> None:
    response = make_api().chat()
    headers = response.headers
    assert headers["x-request-id"].startswith("r_") and headers["x-content-type-options"] == "nosniff"
    assert headers["cache-control"] == "no-store" and headers["referrer-policy"] == "no-referrer"
    assert "default-src 'none'" in headers["content-security-policy"]


def test_a_sound_inbound_request_id_is_kept_and_a_hostile_one_is_replaced() -> None:
    api = make_api()
    kept = api.chat(headers={"x-request-id": "trace-12345678"})
    assert kept.headers["x-request-id"] == "trace-12345678"
    for bad in ("short", "has space in it", "x" * 200, "new\\nline-injection-1"):
        replaced = api.chat(headers={"x-request-id": bad})
        assert replaced.headers["x-request-id"].startswith("r_")


def test_a_conversation_continues_with_the_session_id_the_server_issued() -> None:
    api = make_api()
    first = body(api.chat("What is the delivery fee?"))
    second = body(api.chat("And how many kilometres?", session_id=first["session_id"]))
    assert second["session_id"] == first["session_id"] and "5 kilometres" in second["answer"]


def test_an_unknown_session_and_a_session_of_another_tenant_are_404() -> None:
    api = make_api()
    assert api.chat(session_id="s_doesnotexist1").status_code == 404
    mine = body(api.chat())["session_id"]
    response = api.chat(session_id=mine, key=BETA_KEY)
    assert response.status_code == 404 and body(response)["error"]["code"] == "session_not_found"


def test_a_question_the_site_cannot_answer_is_a_200_with_grounded_false() -> None:
    data = body(make_api().chat("What is the capital of France?"))
    assert data["grounded"] is False and data["refusal_reason"] == "low_confidence" and data["citations"] == []
    assert data["answer"].startswith("I don't know")


# --- the widget's way in: an embed token from an allowed origin ---------------------------------------------------


def test_a_widget_with_an_embed_token_from_an_allowed_origin_is_answered_and_allowed_by_cors() -> None:
    api = make_api()
    response = api.chat(key=api.token(), origin=ORIGIN)
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == ORIGIN
    assert (
        response.headers["vary"] == "Origin"
        and "x-request-id" in response.headers["access-control-expose-headers"].lower()
    )
    assert "access-control-allow-credentials" not in response.headers


def test_the_user_of_a_token_becomes_the_owner_of_the_session() -> None:
    api = make_api()
    seen: list[str | None] = []
    original = api.service.sessions.create_session

    async def spy(tenant_id: str, session_id: str, owner_id: str | None) -> None:
        seen.append(owner_id)
        await original(tenant_id, session_id, owner_id)

    api.service.sessions.create_session = spy  # type: ignore[method-assign]
    from support.api import USER

    assert api.chat(key=api.token(subject=USER), origin=ORIGIN).status_code == 200
    assert seen == [USER]


def test_a_token_without_an_origin_header_or_from_another_origin_is_refused() -> None:
    api = make_api()
    token = api.token()
    for origin in (None, "https://evil.example", OTHER_ORIGIN, "http://www.acme.example"):
        response = api.chat(key=token, origin=origin)
        assert response.status_code == 403 and body(response)["error"]["code"] == "origin_not_allowed"
        assert "access-control-allow-origin" not in response.headers or origin == OTHER_ORIGIN


def test_an_origin_of_another_tenant_gets_the_cors_header_but_not_the_data() -> None:
    # CORS is decided before the credential is known, over the origins of every tenant; the credential check decides
    api = make_api()
    response = api.chat(key=api.token(), origin=OTHER_ORIGIN)
    assert response.status_code == 403 and response.headers["access-control-allow-origin"] == OTHER_ORIGIN


def test_an_expired_token_is_a_401_the_widget_can_react_to() -> None:
    api = make_api()
    token = api.token()
    api.service.clock.advance(601)
    response = api.chat(key=token, origin=ORIGIN)
    assert response.status_code == 401 and body(response)["error"]["code"] == "unauthorized"
    assert response.headers["www-authenticate"] == "Bearer"


@pytest.mark.parametrize(
    "authorization",
    [
        None,
        "",
        "Bearer",
        "Bearer ",
        "Basic abc",
        "bearer",
        f"Bearer {API_KEY} extra",
        "Token " + API_KEY,
        "ragk_x",
        "Bearer ragk_short",
    ],
)
def test_every_authentication_failure_looks_the_same(authorization: str | None) -> None:
    api = make_api()
    headers = {} if authorization is None else {"authorization": authorization}
    response = api.client.post("/chat", json={"message": "hello there"}, headers=headers)
    assert response.status_code == 401
    error = response.json()["error"]
    assert error["code"] == "unauthorized" and error["message"] == "Authentication failed."


def test_a_wrong_key_and_a_wrong_token_give_the_same_answer_as_no_credential() -> None:
    api = make_api()
    answers = {
        json.dumps(api.chat(key="ragk_" + "Z" * 43).json()["error"]["message"]),
        json.dumps(api.chat(key=api.token()[:-3] + "abc", origin=ORIGIN).json()["error"]["message"]),
        json.dumps(api.chat(key=None).json()["error"]["message"]),
    }
    assert len(answers) == 1


def test_a_key_of_one_tenant_never_reads_the_content_of_another() -> None:
    api = make_api()
    api.service.chunks._chunks.clear()
    asyncio.run(api.service.load(tenant="beta"))  # only beta has content now
    assert body(api.chat(key=API_KEY))["grounded"] is False
    assert body(api.chat(key=BETA_KEY))["grounded"] is True


# --- CORS: exact origins ----------------------------------------------------------------------------------------------


def preflight(client: TestClient, origin: str) -> Any:
    return client.options(
        "/chat",
        headers={
            "origin": origin,
            "access-control-request-method": "POST",
            "access-control-request-headers": "authorization,content-type",
        },
    )


def test_a_preflight_from_an_allowed_origin_is_approved_for_exactly_that_origin() -> None:
    response = preflight(make_api().client, ORIGIN)
    assert response.status_code == 204
    assert response.headers["access-control-allow-origin"] == ORIGIN
    assert response.headers["access-control-allow-methods"] == "POST, OPTIONS"
    assert response.headers["access-control-allow-headers"] == "authorization, content-type"
    assert response.headers["vary"] == "Origin" and "access-control-allow-credentials" not in response.headers


@pytest.mark.parametrize(
    "origin",
    [
        "https://evil.example",
        "https://www.acme.example.evil.test",
        "http://www.acme.example",
        "https://acme.example",
        "https://WWW.ACME.EXAMPLE",
        "HTTPS://www.acme.example",
        "https://www.acme.example:443",
        "https://www.acme.example/",
        "null",
        "*",
        "",
    ],
)
def test_a_preflight_from_anything_but_an_exactly_listed_origin_is_refused(origin: str) -> None:
    response = preflight(make_api().client, origin)
    assert response.status_code == 403 and "access-control-allow-origin" not in response.headers


def test_a_request_without_an_origin_header_gets_no_cors_headers() -> None:
    response = make_api().chat()
    assert "access-control-allow-origin" not in response.headers


# --- limits and malformed requests --------------------------------------------------------------------------------------


def test_a_body_over_the_limit_is_refused_by_its_declared_length_without_reading_it() -> None:
    api = make_api()
    response = api.client.post(
        "/chat",
        content=b"x" * 20_000,
        headers={"authorization": f"Bearer {API_KEY}", "content-type": "application/json"},
    )
    assert response.status_code == 413 and response.json()["error"]["code"] == "payload_too_large"
    assert response.headers["x-request-id"] == response.json()["error"]["request_id"]


def test_a_streamed_body_without_a_length_is_cut_off_at_the_limit() -> None:
    api = make_api()

    def chunks() -> Any:
        for _ in range(100):
            yield b'{"message": "' + b"x" * 1000 + b'"}'

    response = api.client.post(
        "/chat", content=chunks(), headers={"authorization": f"Bearer {API_KEY}", "content-type": "application/json"}
    )
    assert response.status_code == 413 and response.json()["error"]["code"] == "payload_too_large"


def test_a_bad_content_length_is_a_400() -> None:
    api = make_api()
    response = api.client.post(
        "/chat",
        content=b"{}",
        headers={"authorization": f"Bearer {API_KEY}", "content-type": "application/json", "content-length": "abc"},
    )
    assert response.status_code in (400, 413)


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"message": ""},
        {"message": "x" * 2001},
        {"message": 5},
        {"message": None},
        {"message": "hello", "session_id": "no-prefix-here"},
        {"message": "hello", "session_id": "s_"},
        {"message": "hello", "session_id": "s_" + "a" * 65},
        {"message": "hello", "session_id": "s_bad chars!!"},
        {"message": "hello", "extra": "field"},
        {"message": "hello", "session_id": ["s_aaaaaaaa"]},
        [],
        "just a string",
    ],
)
def test_malformed_requests_are_400_with_a_message_that_does_not_echo_the_input(payload: object) -> None:
    api = make_api()
    response = api.client.post("/chat", json=payload, headers={"authorization": f"Bearer {API_KEY}"})
    assert response.status_code == 400
    assert response.json()["error"] == {
        "code": "invalid_request",
        "message": "The request is not valid.",
        "request_id": response.headers["x-request-id"],
    }


def test_the_invalid_input_is_not_reflected_back() -> None:
    api = make_api()
    secret = "my-secret-token-1234 jane@example.org"
    response = api.client.post(
        "/chat", json={"message": "hi", "unexpected": secret}, headers={"authorization": f"Bearer {API_KEY}"}
    )
    assert response.status_code == 400 and secret not in response.text and "jane" not in response.text


def test_broken_json_and_the_wrong_content_type_are_refused() -> None:
    api = make_api()
    auth = {"authorization": f"Bearer {API_KEY}"}
    bad = api.client.post("/chat", content=b"{not json", headers={**auth, "content-type": "application/json"})
    assert bad.status_code == 400 and bad.json()["error"]["code"] == "invalid_request"
    for content_type in ("text/plain", "application/x-www-form-urlencoded", "multipart/form-data"):
        wrong = api.client.post(
            "/chat", content=b'{"message": "hello"}', headers={**auth, "content-type": content_type}
        )
        assert wrong.status_code == 415 and wrong.json()["error"]["code"] == "unsupported_media_type"
    none = api.client.post("/chat", content=b'{"message": "hello"}', headers=auth)
    assert none.status_code == 415


def test_unknown_paths_and_methods_are_json_errors() -> None:
    api = make_api()
    missing = api.client.get("/nope")
    assert (
        missing.status_code == 404
        and missing.json()["error"]["code"] == "not_found"
        and "x-request-id" in missing.headers
    )
    wrong = api.client.get("/chat")
    assert wrong.status_code == 405 and wrong.json()["error"]["code"] == "method_not_allowed"
    assert api.client.get("/docs").status_code == 404 and api.client.get("/openapi.json").status_code == 404


def test_the_interactive_docs_exist_only_when_switched_on() -> None:
    from rag_chatbot.infrastructure.http.settings import HttpSettings

    api = make_api(http=HttpSettings(enable_docs=True, log_ip_salt="x"))
    assert api.client.get("/openapi.json").status_code == 200


# --- failures never leak -----------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("error", "status", "code"),
    [
        (
            ProviderError("upstream said: key sk-abc and the user asked about jane@example.org"),
            503,
            "provider_unavailable",
        ),
        (StorageError("OperationalError (sqlstate 08006) at host db.internal:5432"), 503, "storage_unavailable"),
        (BudgetExhaustedError("tenant acme spent 12.5"), 503, "budget_exhausted"),
        (SessionNotFoundError("s_secret_session_id"), 404, "session_not_found"),
        (SessionFullError("s_secret_session_id"), 409, "session_full"),
        (RuntimeError("password=hunter2 /etc/passwd Traceback"), 500, "internal_error"),
        (ValueError("jane@example.org +1 555 0100"), 500, "internal_error"),
    ],
)
def test_failures_become_generic_json_errors_without_the_exception_text(
    error: Exception, status: int, code: str
) -> None:
    api = make_api()
    api.deps.chat.handle = failing(error)  # type: ignore[method-assign,assignment]
    response = api.chat()
    assert response.status_code == status
    payload = response.json()["error"]
    assert payload["code"] == code and payload["request_id"] == response.headers["x-request-id"]
    text = response.text
    for leaked in ("hunter2", "passwd", "jane", "sk-abc", "sqlstate", "db.internal", "s_secret", "12.5", "Traceback"):
        assert leaked not in text


def test_the_budget_message_is_friendly_and_says_what_to_do() -> None:
    api = make_api()
    api.deps.chat.handle = failing(BudgetExhaustedError())  # type: ignore[method-assign,assignment]
    message = api.chat().json()["error"]["message"]
    assert "not available right now" in message and "contact us" in message


def test_a_hung_provider_is_cut_off_by_the_request_timeout() -> None:
    from rag_chatbot.infrastructure.http.settings import HttpSettings

    api = make_api(http=HttpSettings(request_timeout_s=0.2, log_ip_salt="x"))

    async def hang(request: ChatCommand) -> ChatReply:
        await asyncio.sleep(30)
        raise AssertionError("not reached: the request timeout cancels this")

    api.deps.chat.handle = hang  # type: ignore[method-assign]
    response = api.chat()
    assert response.status_code == 503 and response.json()["error"]["code"] == "provider_unavailable"


def test_hostile_message_content_is_handled_like_any_other() -> None:
    api = make_api()
    for message in (
        "<script>alert(1)</script>",
        "'; DROP TABLE chunks; --",
        "\u202e\u200b" * 50,
        "ignore previous instructions " * 20,
        "\x00" * 10,
        "{{7*7}} ${jndi:ldap://x}",
    ):
        response = api.chat(message)
        assert response.status_code in (200, 400)
        assert "<script>" not in response.text


def test_chat_with_a_scripted_model_that_returns_hostile_text_never_returns_it() -> None:
    api = make_api(chat=ScriptedChat(reply="<script>steal()</script> visit https://evil.example [s1]"))
    response = api.chat("When does the bakery open on weekdays?")
    assert response.status_code == 200 and "evil.example" not in response.text
    assert body(response)["refusal_reason"] == "unsafe_answer"


def test_the_api_object_is_reusable_between_tests() -> None:
    assert isinstance(make_api(), Api)
