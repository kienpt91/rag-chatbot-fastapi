from __future__ import annotations

import base64

import pytest

from rag_chatbot.domain.auth import SUBJECT, TokenClaims, authenticate, validate_claims_for_minting
from rag_chatbot.domain.errors import AuthenticationError, ConfigurationError, InvalidRequestError
from rag_chatbot.domain.tenants import TENANT_ID, parse_tenants
from support.api import ORIGIN, USER, make_api

NOW = 1_800_000_000


def nested_token(depth: int) -> str:
    body = base64.urlsafe_b64encode(b"[" * depth).rstrip(b"=").decode()
    return f"ragt1.{body}.AAAA"


def test_deeply_nested_json_in_a_forged_token_is_an_authentication_failure_not_a_crash() -> None:
    api = make_api()
    with pytest.raises(AuthenticationError):
        authenticate(api.deps.directory, nested_token(1500), ORIGIN, NOW)


def test_the_http_answer_for_such_a_token_is_the_same_401_as_for_any_bad_credential() -> None:
    api = make_api()
    response = api.chat(key=nested_token(1500), origin=ORIGIN)
    assert response.status_code == 401 and response.json()["error"]["code"] == "unauthorized"


@pytest.mark.parametrize("value", ["acme\n", "acme\r\n", "\nacme"])
def test_identifiers_do_not_accept_a_trailing_newline(value: str) -> None:
    assert TENANT_ID.match(value) is None
    with pytest.raises(ConfigurationError):
        parse_tenants({"tenants": [{"id": value}]})


def test_a_subject_with_a_trailing_newline_is_refused_at_minting_and_by_the_pattern() -> None:
    assert SUBJECT.match(USER + "\n") is None
    with pytest.raises(InvalidRequestError):
        validate_claims_for_minting(TokenClaims("acme", NOW + 60, USER + "\n"), NOW)


def test_a_key_hash_with_a_trailing_newline_is_refused() -> None:
    with pytest.raises(ConfigurationError):
        parse_tenants({"tenants": [{"id": "a", "api_key_sha256": ["0" * 64 + "\n"]}]})
