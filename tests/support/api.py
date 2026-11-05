"""The FastAPI application wired over in-memory stores and the deterministic fakes, with two tenants."""

from __future__ import annotations

import base64
from dataclasses import dataclass

from fastapi.testclient import TestClient
from httpx2 import Response

from rag_chatbot.application.settings import ChatSettings
from rag_chatbot.domain.auth import API_KEY_PREFIX, TokenClaims, hash_api_key, mint_token
from rag_chatbot.domain.tenants import TenantDirectory, parse_tenants
from rag_chatbot.infrastructure.http.app import AppDeps, create_app
from rag_chatbot.infrastructure.http.settings import HttpSettings
from support.fakes import ScriptedChat
from support.service import CANARY, TENANT, Service, build_service

ORIGIN = "https://www.acme.example"
OTHER_ORIGIN = "https://shop.beta.example"
API_KEY = API_KEY_PREFIX + "A" * 43
BETA_KEY = API_KEY_PREFIX + "B" * 43
SECRET = bytes(range(32))
BETA_SECRET = bytes(range(1, 33))
USER = "123e4567-e89b-12d3-a456-426614174000"


def b64(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode()


def directory() -> TenantDirectory:
    return parse_tenants(
        {
            "tenants": [
                {
                    "id": "acme",
                    "name": "Maple Bakery",
                    "allowed_origins": [ORIGIN, "http://localhost:8080"],
                    "api_key_sha256": [hash_api_key(API_KEY)],
                    "token_secrets": [b64(SECRET)],
                },
                {
                    "id": "beta",
                    "name": "Beta Shop",
                    "allowed_origins": [OTHER_ORIGIN],
                    "api_key_sha256": [hash_api_key(BETA_KEY)],
                    "token_secrets": [b64(BETA_SECRET)],
                },
            ]
        }
    )


@dataclass
class Api:
    client: TestClient
    service: Service
    deps: AppDeps

    def token(self, **overrides: object) -> str:
        claims = TokenClaims("acme", int(self.service.clock.now().timestamp()) + 600, **overrides)  # type: ignore[arg-type]
        return mint_token(SECRET, claims)

    def chat(
        self,
        message: str = "What is the delivery fee?",
        *,
        key: str | None = API_KEY,
        origin: str | None = None,
        session_id: str | None = None,
        headers: dict[str, str] | None = None,
    ) -> Response:
        sent = {"authorization": f"Bearer {key}"} if key is not None else {}
        if origin is not None:
            sent["origin"] = origin
        sent.update(headers or {})
        body: dict[str, object] = {"message": message}
        if session_id is not None:
            body["session_id"] = session_id
        return self.client.post("/chat", json=body, headers=sent)


def make_api(
    *,
    http: HttpSettings | None = None,
    chat: ScriptedChat | None = None,
    settings: ChatSettings | None = None,
    load: bool = True,
    readiness_problems: list[str] | None = None,
    readiness_error: Exception | None = None,
) -> Api:
    import asyncio

    service = build_service(settings or ChatSettings(canary=CANARY, confidence_threshold=0.15), chat)
    if load:
        asyncio.run(service.load())
        asyncio.run(service.load(tenant="beta"))

    async def readiness() -> list[str]:
        if readiness_error is not None:
            raise readiness_error
        return list(readiness_problems or [])

    deps = AppDeps(
        chat=service.chat_service,
        directory=directory(),
        clock=service.clock,
        ids=service.ids,
        settings=http or HttpSettings(log_ip_salt="test-salt"),
        readiness=readiness,
    )
    return Api(TestClient(create_app(deps), raise_server_exceptions=False), service, deps)


__all__ = ["API_KEY", "BETA_KEY", "ORIGIN", "OTHER_ORIGIN", "SECRET", "TENANT", "USER", "Api", "make_api"]
