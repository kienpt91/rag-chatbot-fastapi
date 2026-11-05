"""Authentication: API keys for servers, signed embed tokens for the website widget.

* An **API key** (`ragk_...`, 32 random bytes) is a server-to-server credential. The directory stores only its
  SHA-256; comparing is constant time. It must never be placed in a web page.
* An **embed token** (`ragt1.<payload>.<signature>`) is what a page may carry: HMAC-SHA256 over the payload, which
  names the tenant, an expiry and optionally a subject (the end user's uuid, for row-level security) and an origin. A browser
  that holds one can talk to the API only from an allowed origin, only until it expires, and only as that tenant.

Every failure to authenticate gives the same error, so a caller learns nothing about which part was wrong.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import json
import re
from dataclasses import dataclass
from typing import Literal

from rag_chatbot.domain.errors import AuthenticationError, InvalidRequestError, OriginNotAllowedError
from rag_chatbot.domain.tenants import TENANT_ID, Tenant, TenantDirectory, normalise_origin

API_KEY_PREFIX = "ragk_"
TOKEN_PREFIX = "ragt1."  # noqa: S105 - the visible start of a token, not a password
MAX_CREDENTIAL_CHARS = 2048
MAX_TOKEN_TTL_S = 24 * 3600
# the subject is the end user's Supabase user id: row-level security compares it with auth.uid(), a uuid
SUBJECT = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\Z")

Method = Literal["api_key", "embed_token"]


def hash_api_key(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


def _b64e(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def _b64d(text: str) -> bytes:
    if not text.isascii():
        raise AuthenticationError("token: not ascii")
    try:
        return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))
    except (binascii.Error, ValueError):
        raise AuthenticationError("token: not base64url") from None


@dataclass(frozen=True, slots=True)
class TokenClaims:
    tenant_id: str
    expires_at: int  # seconds since the epoch
    subject: str | None = None
    origin: str | None = None


def mint_token(secret: bytes, claims: TokenClaims) -> str:
    payload: dict[str, object] = {"t": claims.tenant_id, "exp": claims.expires_at}
    if claims.subject is not None:
        payload["sub"] = claims.subject
    if claims.origin is not None:
        payload["o"] = claims.origin
    body = _b64e(json.dumps(payload, separators=(",", ":"), sort_keys=True).encode())
    signature = hmac.new(secret, f"{TOKEN_PREFIX}{body}".encode(), hashlib.sha256).digest()
    return f"{TOKEN_PREFIX}{body}.{_b64e(signature)}"


def validate_claims_for_minting(claims: TokenClaims, now: int) -> None:
    """Refuse to mint a token that could not be verified or that lives too long."""
    if not TENANT_ID.match(claims.tenant_id):
        raise InvalidRequestError("token: bad tenant id")
    if not now < claims.expires_at <= now + MAX_TOKEN_TTL_S:
        raise InvalidRequestError(f"token: the lifetime must be between 1 second and {MAX_TOKEN_TTL_S} seconds")
    if claims.subject is not None and not SUBJECT.match(claims.subject):
        raise InvalidRequestError("token: the subject must be a lower-case uuid (the Supabase user id)")
    if claims.origin is not None and normalise_origin(claims.origin) != claims.origin:
        raise InvalidRequestError("token: the origin must be a normalised origin")


def read_unverified_claims(token: str) -> tuple[TokenClaims, str, bytes]:
    """Parse a token without trusting it: returns the claims, the signed text and the signature."""
    if len(token) > MAX_CREDENTIAL_CHARS or not token.startswith(TOKEN_PREFIX):
        raise AuthenticationError("token: shape")
    parts = token[len(TOKEN_PREFIX) :].split(".")
    if len(parts) != 2:
        raise AuthenticationError("token: shape")
    body, signature = parts
    raw = _b64d(body)
    if (
        len(raw) > 1024 or raw.count(b"{") != 1 or b"[" in raw
    ):  # a payload is one flat object: refuse deep nesting unparsed
        raise AuthenticationError("token: payload")
    try:
        payload = json.loads(raw)
    except (ValueError, UnicodeDecodeError, RecursionError):
        raise AuthenticationError("token: payload") from None
    if not isinstance(payload, dict) or set(payload) - {"t", "exp", "sub", "o"}:
        raise AuthenticationError("token: payload")
    tenant_id, expires = payload.get("t"), payload.get("exp")
    subject, origin = payload.get("sub"), payload.get("o")
    if (
        not isinstance(tenant_id, str)
        or not TENANT_ID.match(tenant_id)
        or isinstance(expires, bool)
        or not isinstance(expires, int)
        or (subject is not None and (not isinstance(subject, str) or not SUBJECT.match(subject)))
        or (origin is not None and not isinstance(origin, str))
    ):
        raise AuthenticationError("token: payload")
    return (
        TokenClaims(tenant_id, expires, subject, origin),
        f"{TOKEN_PREFIX}{body}",
        _b64d(signature),
    )


@dataclass(frozen=True, slots=True)
class Principal:
    tenant: Tenant
    method: Method
    subject: str | None = None


def authenticate(directory: TenantDirectory, credential: str, origin: str | None, now: int) -> Principal:
    """Who is calling. `origin` is the request's Origin header or None. Raises `AuthenticationError` (any credential
    problem) or `OriginNotAllowedError` (the credential is fine, the page it comes from is not allowed)."""
    if not credential or len(credential) > MAX_CREDENTIAL_CHARS:
        raise AuthenticationError("credential: shape")
    if credential.startswith(TOKEN_PREFIX):
        return _authenticate_token(directory, credential, origin, now)
    if credential.startswith(API_KEY_PREFIX):
        return _authenticate_key(directory, credential, origin)
    raise AuthenticationError("credential: unknown kind")


def _authenticate_token(directory: TenantDirectory, token: str, origin: str | None, now: int) -> Principal:
    claims, signed, signature = read_unverified_claims(token)
    tenant = directory.get(claims.tenant_id)
    if tenant is None or not tenant.token_secrets:
        raise AuthenticationError("token: unknown tenant")
    valid = False
    for secret in tenant.token_secrets:  # all of them, so that timing does not say which one matched
        expected = hmac.new(secret, signed.encode(), hashlib.sha256).digest()
        valid = hmac.compare_digest(expected, signature) or valid
    if not valid or claims.expires_at <= now:
        raise AuthenticationError("token: signature or expiry")
    # a token is for browsers: it works only from an allowed origin (a browser always sends Origin on a cross-origin
    # request; a script that leaves it out is not a browser holding a token for its page)
    if origin is None or not tenant.origin_allowed(origin):
        raise OriginNotAllowedError("token: origin")
    if claims.origin is not None and normalise_origin(origin) != claims.origin:
        raise OriginNotAllowedError("token: origin differs from the one the token was issued for")
    return Principal(tenant, "embed_token", claims.subject)


def _authenticate_key(directory: TenantDirectory, key: str, origin: str | None) -> Principal:
    digest = hash_api_key(key)
    found: Tenant | None = None
    for tenant in directory.tenants.values():  # every tenant and every hash: constant work per request
        for stored in tenant.api_key_sha256:
            if hmac.compare_digest(stored, digest):
                found = tenant
    if found is None:
        raise AuthenticationError("api key: unknown")
    if origin is not None and not found.origin_allowed(origin):
        raise OriginNotAllowedError("api key: origin")
    return Principal(found, "api_key")
