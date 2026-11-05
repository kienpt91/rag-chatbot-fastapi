from __future__ import annotations

import base64
import json

import pytest

from rag_chatbot.domain.auth import (
    API_KEY_PREFIX,
    MAX_TOKEN_TTL_S,
    TOKEN_PREFIX,
    TokenClaims,
    authenticate,
    hash_api_key,
    mint_token,
    read_unverified_claims,
    validate_claims_for_minting,
)
from rag_chatbot.domain.errors import (
    AuthenticationError,
    ConfigurationError,
    InvalidRequestError,
    OriginNotAllowedError,
)
from rag_chatbot.domain.tenants import Tenant, TenantDirectory, normalise_origin, parse_tenants

SECRET = bytes(range(32))
SECRET_B64 = base64.urlsafe_b64encode(SECRET).rstrip(b"=").decode()
OTHER = bytes(range(1, 33))
KEY = API_KEY_PREFIX + "K" * 43
NOW = 1_800_000_000
USER = "123e4567-e89b-12d3-a456-426614174000"


def directory(**overrides: object) -> TenantDirectory:
    entry: dict[str, object] = {
        "id": "acme",
        "name": "Acme Bakery",
        "allowed_origins": ["https://www.acme.example", "http://localhost:8080"],
        "api_key_sha256": [hash_api_key(KEY)],
        "token_secrets": [SECRET_B64],
        **overrides,
    }
    return parse_tenants({"tenants": [entry]})


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("https://www.acme.example", "https://www.acme.example"),
        ("HTTPS://WWW.ACME.EXAMPLE", "https://www.acme.example"),
        ("https://www.acme.example:443", "https://www.acme.example"),
        ("http://localhost:80", "http://localhost"),
        ("http://localhost:8080", "http://localhost:8080"),
        ("https://www.acme.example/", "https://www.acme.example"),
        ("https://bücher.example", "https://xn--bcher-kva.example"),
        ("http://[::1]:3000", "http://[::1]:3000"),
    ],
)
def test_origins_are_normalised(value: str, expected: str) -> None:
    assert normalise_origin(value) == expected


@pytest.mark.parametrize(
    "value",
    [
        "",
        "null",
        "https://www.acme.example/path",
        "https://www.acme.example?x=1",
        "https://www.acme.example#frag",
        "https://user:pw@www.acme.example",
        "ftp://www.acme.example",
        "javascript:alert(1)",
        "https://",
        "https://www.acme.example:notaport",
        "https://www.acme.example:99999",
        " https://www.acme.example",
        "https://exa mple.com",
        "https://-bad.example",
        "https://" + "a" * 300 + ".example",
        "//www.acme.example",
    ],
)
def test_things_that_are_not_origins_are_refused(value: str) -> None:
    assert normalise_origin(value) is None


def test_a_tenant_file_is_parsed_with_all_its_settings() -> None:
    tenants = parse_tenants(
        {
            "tenants": [
                {
                    "id": "acme",
                    "name": "Acme",
                    "allowed_origins": ["https://www.acme.example"],
                    "daily_budget": "2.50",
                    "confidence_threshold": 0.4,
                    "refusal_message": "Call us.",
                }
            ]
        }
    )
    tenant = tenants_get(tenants, "acme")
    assert tenant.daily_budget_micros == 2_500_000 and tenant.confidence_threshold == 0.4
    assert tenant.refusal_message == "Call us." and tenants.all_origins() == {"https://www.acme.example"}


def tenants_get(tenants: TenantDirectory, tenant_id: str) -> Tenant:
    found = tenants.get(tenant_id)
    assert found is not None
    return found


@pytest.mark.parametrize(
    "data",
    [
        [],
        {},
        {"tenants": []},
        {"tenants": "x"},
        {"tenants": [{}]},
        {"tenants": [{"id": "Bad Id"}]},
        {"tenants": [{"id": "a", "unknown": 1}]},
        {"tenants": [{"id": "a"}, {"id": "a"}]},
        {"tenants": [{"id": "a", "allowed_origins": ["https://x.example/p"]}]},
        {"tenants": [{"id": "a", "allowed_origins": ["http://www.acme.example"]}]},
        {"tenants": [{"id": "a", "allowed_origins": "https://x.example"}]},
        {"tenants": [{"id": "a", "api_key_sha256": ["nothex"]}]},
        {"tenants": [{"id": "a", "token_secrets": ["!!!"]}]},
        {"tenants": [{"id": "a", "token_secrets": ["c2hvcnQ"]}]},
        {"tenants": [{"id": "a", "confidence_threshold": 2}]},
        {"tenants": [{"id": "a", "confidence_threshold": True}]},
        {"tenants": [{"id": "a", "daily_budget": "lots"}]},
        {"tenants": [{"id": "a", "daily_budget": True}]},
        {"tenants": [{"id": "a", "refusal_message": ""}]},
        {"tenants": [{"id": "a", "name": "x\u0007"}]},
        {"tenants": [{"id": "a", "api_key_sha256": ["0" * 64]}, {"id": "b", "api_key_sha256": ["0" * 64]}]},
    ],
)
def test_malformed_tenant_files_are_refused(data: object) -> None:
    with pytest.raises(ConfigurationError):
        parse_tenants(data)


def test_origin_allowed_compares_exactly_after_normalisation() -> None:
    tenant = tenants_get(directory(), "acme")
    assert tenant.origin_allowed("https://WWW.acme.example:443")
    assert not tenant.origin_allowed("https://www.acme.example.evil.test")
    assert not tenant.origin_allowed("https://evil.test")
    assert not tenant.origin_allowed("http://www.acme.example")
    assert not tenant.origin_allowed("null")
    assert not tenant.origin_allowed("https://acme.example")


# --- credentials ---------------------------------------------------------------------------------------------


def good_token(**overrides: object) -> str:
    claims = TokenClaims("acme", NOW + 600, **overrides)  # type: ignore[arg-type]
    return mint_token(SECRET, claims)


def test_an_embed_token_from_an_allowed_origin_authenticates_as_its_tenant_and_subject() -> None:
    principal = authenticate(directory(), good_token(subject=USER), "https://www.acme.example", NOW)
    assert principal.tenant.tenant_id == "acme" and principal.method == "embed_token" and principal.subject == USER


def test_a_token_without_an_origin_header_is_refused_because_it_is_not_a_browser() -> None:
    with pytest.raises(OriginNotAllowedError):
        authenticate(directory(), good_token(), None, NOW)


def test_a_token_from_another_origin_is_refused() -> None:
    with pytest.raises(OriginNotAllowedError):
        authenticate(directory(), good_token(), "https://evil.example", NOW)


def test_a_token_bound_to_an_origin_works_only_from_it() -> None:
    both = directory(allowed_origins=["https://www.acme.example", "https://shop.acme.example"])
    token = good_token(origin="https://shop.acme.example")
    assert authenticate(both, token, "https://shop.acme.example", NOW).tenant.tenant_id == "acme"
    with pytest.raises(OriginNotAllowedError):
        authenticate(both, token, "https://www.acme.example", NOW)


def test_an_expired_token_is_refused_at_the_second_it_expires() -> None:
    token = mint_token(SECRET, TokenClaims("acme", NOW + 10))
    assert authenticate(directory(), token, "https://www.acme.example", NOW + 9).tenant.tenant_id == "acme"
    with pytest.raises(AuthenticationError):
        authenticate(directory(), token, "https://www.acme.example", NOW + 10)


def test_a_token_signed_with_another_secret_or_for_another_tenant_is_refused() -> None:
    forged = mint_token(OTHER, TokenClaims("acme", NOW + 600))
    with pytest.raises(AuthenticationError):
        authenticate(directory(), forged, "https://www.acme.example", NOW)
    unknown = mint_token(SECRET, TokenClaims("nobody", NOW + 600))
    with pytest.raises(AuthenticationError):
        authenticate(directory(), unknown, "https://www.acme.example", NOW)


def test_a_token_cannot_be_moved_to_another_tenant_by_editing_its_payload() -> None:
    two = parse_tenants(
        {
            "tenants": [
                {"id": "acme", "allowed_origins": ["https://a.example"], "token_secrets": [SECRET_B64]},
                {
                    "id": "other",
                    "allowed_origins": ["https://a.example"],
                    "token_secrets": [base64.urlsafe_b64encode(OTHER).decode()],
                },
            ]
        }
    )
    token = mint_token(SECRET, TokenClaims("acme", NOW + 600))
    body = json.loads(base64.urlsafe_b64decode(token.split(".")[1] + "=="))
    body["t"] = "other"
    edited = base64.urlsafe_b64encode(json.dumps(body).encode()).rstrip(b"=").decode()
    tampered = f"{TOKEN_PREFIX}{edited}.{token.split('.')[2]}"
    with pytest.raises(AuthenticationError):
        authenticate(two, tampered, "https://a.example", NOW)


def test_rotation_any_listed_secret_verifies() -> None:
    rotated = directory(token_secrets=[base64.urlsafe_b64encode(OTHER).decode(), SECRET_B64])
    assert authenticate(rotated, good_token(), "https://www.acme.example", NOW).tenant.tenant_id == "acme"


@pytest.mark.parametrize(
    "credential",
    [
        "",
        "x",
        "Bearer abc",
        TOKEN_PREFIX,
        TOKEN_PREFIX + "a.b",
        TOKEN_PREFIX + "a.b.c",
        TOKEN_PREFIX + "!!!.???",
        TOKEN_PREFIX + base64.urlsafe_b64encode(b"[1,2]").decode() + ".sig",
        TOKEN_PREFIX + base64.urlsafe_b64encode(b'{"t":"acme","exp":"x"}').decode() + ".sig",
        TOKEN_PREFIX + base64.urlsafe_b64encode(b'{"t":"acme","exp":true}').decode() + ".sig",
        TOKEN_PREFIX + base64.urlsafe_b64encode(b'{"t":"acme","exp":1,"extra":1}').decode() + ".sig",
        TOKEN_PREFIX + base64.urlsafe_b64encode(b'{"t":"A B","exp":1}').decode() + ".sig",
        TOKEN_PREFIX + "é.sig",
        TOKEN_PREFIX + "a" * 5000,
        API_KEY_PREFIX + "short",
        "ragk_" + "x" * 5000,
    ],
)
def test_malformed_credentials_all_fail_the_same_way(credential: str) -> None:
    with pytest.raises(AuthenticationError):
        authenticate(directory(), credential, "https://www.acme.example", NOW)


def test_an_api_key_authenticates_without_an_origin_and_with_an_allowed_one() -> None:
    assert authenticate(directory(), KEY, None, NOW).method == "api_key"
    assert authenticate(directory(), KEY, "https://www.acme.example", NOW).tenant.tenant_id == "acme"


def test_an_api_key_used_from_a_foreign_origin_is_refused() -> None:
    with pytest.raises(OriginNotAllowedError):
        authenticate(directory(), KEY, "https://evil.example", NOW)


def test_an_unknown_api_key_is_refused() -> None:
    with pytest.raises(AuthenticationError):
        authenticate(directory(), API_KEY_PREFIX + "Z" * 43, None, NOW)


def test_a_tenant_without_token_secrets_cannot_be_reached_with_a_token() -> None:
    with pytest.raises(AuthenticationError):
        authenticate(directory(token_secrets=[]), good_token(), "https://www.acme.example", NOW)


def test_minting_validates_lifetime_subject_and_origin() -> None:
    validate_claims_for_minting(TokenClaims("acme", NOW + 60, USER, "https://www.acme.example"), NOW)
    for bad in (
        TokenClaims("acme", NOW),
        TokenClaims("acme", NOW + MAX_TOKEN_TTL_S + 1),
        TokenClaims("Bad Id", NOW + 60),
        TokenClaims("acme", NOW + 60, "x" * 200),
        TokenClaims("acme", NOW + 60, "user-1"),
        TokenClaims("acme", NOW + 60, USER.upper()),
        TokenClaims("acme", NOW + 60, ""),
        TokenClaims("acme", NOW + 60, None, "https://www.acme.example/path"),
    ):
        with pytest.raises(InvalidRequestError):
            validate_claims_for_minting(bad, NOW)


def test_reading_claims_does_not_verify_them() -> None:
    claims, signed, signature = read_unverified_claims(good_token(subject=USER))
    assert (
        claims.tenant_id == "acme"
        and claims.subject == USER
        and signed.startswith(TOKEN_PREFIX)
        and len(signature) == 32
    )
