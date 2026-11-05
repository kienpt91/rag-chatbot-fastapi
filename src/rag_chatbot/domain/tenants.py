"""Tenants: the websites one deployment serves. Each has its own content, credentials, allowed origins and limits.

The operator describes them in a JSON file (examples/tenants.example.json). The file is read once at start-up and
validated strictly: a typo is an error, not a silently ignored setting.
"""

from __future__ import annotations

import base64
import binascii
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from urllib.parse import urlsplit

from rag_chatbot.domain.errors import ConfigurationError
from rag_chatbot.domain.pricing import usd_to_micros

TENANT_ID = re.compile(r"^[a-z0-9][a-z0-9_-]{0,62}\Z")
_HEX64 = re.compile(r"^[0-9a-f]{64}\Z")
_HOSTNAME = re.compile(r"^[a-z0-9]([a-z0-9-]{0,61}[a-z0-9])?(\.[a-z0-9]([a-z0-9-]{0,61}[a-z0-9])?)*\Z")
_LOOPBACK = frozenset({"localhost", "127.0.0.1", "::1"})
MIN_SECRET_BYTES = 32
_KEYS = frozenset(
    {
        "id",
        "name",
        "allowed_origins",
        "api_key_sha256",
        "token_secrets",
        "daily_budget",
        "confidence_threshold",
        "refusal_message",
        "link_hosts",
    }
)


def normalise_origin(value: str) -> str | None:
    """`scheme://host[:port]`, lower case, default port dropped; None for anything that is not exactly an origin
    (a path, a query, credentials, `null`, an unknown scheme, odd host names)."""
    if not value or len(value) > 300 or value != value.strip():
        return None
    try:
        parts = urlsplit(value)
        port = parts.port
    except ValueError:
        return None
    if parts.scheme not in ("http", "https") or parts.username is not None or parts.password is not None:
        return None
    if parts.path not in ("", "/") or parts.query or parts.fragment or "?" in value or "#" in value:
        return None
    host = parts.hostname
    if not host:
        return None
    try:
        host = host.encode("idna").decode("ascii").lower()
    except UnicodeError:
        return None
    if ":" in host:  # an IPv6 literal
        host_text = f"[{host}]"
    elif _HOSTNAME.match(host):
        host_text = host
    else:
        return None
    default = 443 if parts.scheme == "https" else 80
    suffix = "" if port in (None, default) else f":{port}"
    return f"{parts.scheme}://{host_text}{suffix}"


@dataclass(frozen=True, slots=True)
class Tenant:
    tenant_id: str
    name: str
    allowed_origins: frozenset[str]
    api_key_sha256: frozenset[str] = frozenset()
    token_secrets: tuple[bytes, ...] = ()
    daily_budget_micros: int | None = None
    confidence_threshold: float | None = None
    refusal_message: str | None = None
    link_hosts: frozenset[str] = frozenset()  # hosts, besides the origins', the assistant may link to

    def origin_allowed(self, origin: str) -> bool:
        normalised = normalise_origin(origin)
        return normalised is not None and normalised in self.allowed_origins


def host_of_origin(origin: str) -> str:
    return (urlsplit(origin).hostname or "").casefold()


def linkable_hosts(tenant: Tenant) -> frozenset[str]:
    """The hosts an answer for this tenant may link to: those of its allowed origins and its declared link hosts."""
    return frozenset({host_of_origin(o) for o in tenant.allowed_origins} | set(tenant.link_hosts))


@dataclass(frozen=True, slots=True)
class TenantDirectory:
    tenants: Mapping[str, Tenant] = field(default_factory=dict)

    def get(self, tenant_id: str) -> Tenant | None:
        return self.tenants.get(tenant_id)

    def all_origins(self) -> frozenset[str]:
        return frozenset(origin for tenant in self.tenants.values() for origin in tenant.allowed_origins)


def _decode_secret(value: str, where: str) -> bytes:
    try:
        raw = base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))
    except (binascii.Error, ValueError):
        raise ConfigurationError(f"tenants: {where} is not base64url") from None
    if len(raw) < MIN_SECRET_BYTES:
        raise ConfigurationError(f"tenants: {where} must decode to at least {MIN_SECRET_BYTES} bytes")
    return raw


def _string_list(entry: Mapping[str, object], key: str, where: str) -> list[str]:
    value = entry.get(key, [])
    if not isinstance(value, list) or not all(isinstance(v, str) for v in value) or len(value) > 50:
        raise ConfigurationError(f"tenants: {where}.{key} must be a list of at most 50 strings")
    return [v for v in value if isinstance(v, str)]


def parse_tenants(data: object) -> TenantDirectory:
    """`{"tenants": [{"id": "acme", "name": "Acme", "allowed_origins": ["https://www.acme.example"], ...}]}`"""
    if not isinstance(data, dict) or set(data) != {"tenants"} or not isinstance(data["tenants"], list):
        raise ConfigurationError("tenants: expected an object with a 'tenants' list")
    entries = data["tenants"]
    if not entries or len(entries) > 1000:
        raise ConfigurationError("tenants: between 1 and 1000 tenants are required")
    tenants: dict[str, Tenant] = {}
    seen_hashes: dict[str, str] = {}
    for position, entry in enumerate(entries):
        if not isinstance(entry, dict) or set(entry) - _KEYS or "id" not in entry:
            raise ConfigurationError(f"tenants[{position}]: expected an object with 'id' and only known keys")
        tenant_id = entry["id"]
        if not isinstance(tenant_id, str) or not TENANT_ID.match(tenant_id):
            raise ConfigurationError(f"tenants[{position}].id: lower-case letters, digits, '_' and '-', at most 63")
        if tenant_id in tenants:
            raise ConfigurationError(f"tenants: duplicate id {tenant_id!r}")
        where = f"tenants[{tenant_id}]"
        name = entry.get("name", tenant_id)
        if not isinstance(name, str) or not (1 <= len(name) <= 100) or any(ord(c) < 32 for c in name):
            raise ConfigurationError(f"{where}.name: 1 to 100 printable characters")
        origins: set[str] = set()
        for raw in _string_list(entry, "allowed_origins", where):
            origin = normalise_origin(raw)
            if origin is None:
                raise ConfigurationError(f"{where}.allowed_origins: {raw!r} is not an origin (scheme://host[:port])")
            host = urlsplit(origin).hostname or ""
            if origin.startswith("http://") and host not in _LOOPBACK:
                raise ConfigurationError(f"{where}.allowed_origins: {raw!r} must be https (http only for localhost)")
            origins.add(origin)
        link_hosts: set[str] = set()
        for raw in _string_list(entry, "link_hosts", where):
            host = raw.strip().rstrip(".").casefold()
            try:
                host = host.encode("idna").decode("ascii")
            except UnicodeError:
                host = ""
            if not _HOSTNAME.match(host):
                raise ConfigurationError(f"{where}.link_hosts: {raw!r} is not a host name")
            link_hosts.add(host)
        hashes = set(_string_list(entry, "api_key_sha256", where))
        for digest in hashes:
            if not _HEX64.match(digest):
                raise ConfigurationError(f"{where}.api_key_sha256: entries are 64 lower-case hex digits")
            if digest in seen_hashes:
                raise ConfigurationError(f"{where}.api_key_sha256: a key hash is used by two tenants")
            seen_hashes[digest] = tenant_id
        secrets = tuple(
            _decode_secret(raw, f"{where}.token_secrets[{i}]")
            for i, raw in enumerate(_string_list(entry, "token_secrets", where))
        )
        budget = entry.get("daily_budget")
        if budget is not None and (isinstance(budget, bool) or not isinstance(budget, (int, float, str))):
            raise ConfigurationError(f'{where}.daily_budget: a number or a string such as "5.00"')
        threshold = entry.get("confidence_threshold")
        if threshold is not None and (
            isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not 0.0 <= threshold <= 1.0
        ):
            raise ConfigurationError(f"{where}.confidence_threshold: a number from 0 to 1")
        refusal = entry.get("refusal_message")
        if refusal is not None and (not isinstance(refusal, str) or not (1 <= len(refusal) <= 500)):
            raise ConfigurationError(f"{where}.refusal_message: 1 to 500 characters")
        tenants[tenant_id] = Tenant(
            tenant_id=tenant_id,
            name=name,
            allowed_origins=frozenset(origins),
            api_key_sha256=frozenset(hashes),
            token_secrets=secrets,
            daily_budget_micros=None if budget is None else usd_to_micros(budget, f"{where}.daily_budget"),
            confidence_threshold=None if threshold is None else float(threshold),
            refusal_message=refusal,
            link_hosts=frozenset(link_hosts),
        )
    return TenantDirectory(tenants)
