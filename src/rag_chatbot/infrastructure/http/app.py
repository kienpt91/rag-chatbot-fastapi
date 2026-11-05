"""The FastAPI application: POST /chat, POST /embed-token, GET /healthz, /readyz and /widget.js.

The application is a thin shell. Everything that decides something lives in the domain and the use cases; this module
turns an HTTP request into a call and the result (or the error) back into a response, and makes sure that no failure
reaches the client in a shape that leaks anything: errors are JSON with a stable code, a message written for visitors
and the request id, never an exception text, a query or a path.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
from collections.abc import Awaitable, Callable
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass
from pathlib import Path

from fastapi import Depends, FastAPI, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from rag_chatbot.application.chat import ChatRequest as ChatCommand
from rag_chatbot.application.chat import ChatService
from rag_chatbot.application.ports import Clock, IdGenerator
from rag_chatbot.domain.auth import (
    Principal,
    TokenClaims,
    authenticate,
    mint_token,
    validate_claims_for_minting,
)
from rag_chatbot.domain.errors import (
    AuthenticationError,
    ChatbotError,
    ForbiddenError,
    InvalidRequestError,
    ProviderError,
    RateLimitedError,
)
from rag_chatbot.domain.model import ChatReply
from rag_chatbot.domain.ratelimit import RateLimiter
from rag_chatbot.domain.tenants import TenantDirectory
from rag_chatbot.infrastructure.http.logging import request_id_var
from rag_chatbot.infrastructure.http.middleware import (
    STATUS_FOR_CODE,
    BodyLimitMiddleware,
    CorsMiddleware,
    RateLimitMiddleware,
    RequestContextMiddleware,
)
from rag_chatbot.infrastructure.http.schemas import (
    ChatBody,
    ChatOut,
    CitationOut,
    ErrorDetail,
    ErrorOut,
    TokenBody,
    TokenOut,
)
from rag_chatbot.infrastructure.http.settings import HttpSettings

log = logging.getLogger("rag_chatbot.http")
WIDGET_PATH = Path(__file__).resolve().parents[2] / "widget" / "rag-chat.js"


@dataclass
class AppDeps:
    chat: ChatService
    directory: TenantDirectory
    clock: Clock
    ids: IdGenerator
    settings: HttpSettings
    readiness: Callable[[], Awaitable[list[str]]]  # problem codes; empty when ready
    lifespan: Callable[[FastAPI], AbstractAsyncContextManager[None]] | None = None


def _error(status: int, error: ChatbotError, headers: dict[str, str] | None = None) -> JSONResponse:
    body = ErrorOut(error=ErrorDetail(code=error.code, message=error.public_message, request_id=request_id_var.get()))
    return JSONResponse(body.model_dump(), status_code=status, headers=headers)


def create_app(deps: AppDeps) -> FastAPI:
    settings = deps.settings
    app = FastAPI(
        title="rag-chatbot-fastapi",
        docs_url="/docs" if settings.enable_docs else None,
        redoc_url=None,
        openapi_url="/openapi.json" if settings.enable_docs else None,
        lifespan=deps.lifespan,
    )
    ip_limiter = RateLimiter(
        capacity=settings.ip_burst, refill_per_s=settings.ip_per_minute / 60, max_keys=settings.max_tracked_keys
    )
    session_limiter = RateLimiter(
        capacity=settings.session_burst,
        refill_per_s=settings.session_per_minute / 60,
        max_keys=settings.max_tracked_keys,
    )
    widget = WIDGET_PATH.read_bytes()
    widget_etag = '"' + hashlib.sha256(widget).hexdigest()[:32] + '"'

    # --- errors: stable codes, visitor-safe messages, no internals ---------------------------------------------

    @app.exception_handler(ChatbotError)
    async def chatbot_error(_: Request, error: ChatbotError) -> JSONResponse:
        headers: dict[str, str] = {}
        if isinstance(error, RateLimitedError):
            headers["Retry-After"] = str(error.retry_after_s)
        if isinstance(error, AuthenticationError):
            headers["WWW-Authenticate"] = "Bearer"
        status = STATUS_FOR_CODE.get(error.code, 500)
        if status >= 500:
            log.error("request failed", extra={"event": "error", "code": error.code, "detail": error.detail})
        elif status != 404:
            log.info("request refused", extra={"event": "refused", "code": error.code})
        return _error(status, error, headers)

    @app.exception_handler(RequestValidationError)
    async def invalid_request(_: Request, __: RequestValidationError) -> JSONResponse:
        # the default handler echoes the offending input back; this one says nothing about it
        return _error(400, InvalidRequestError())

    @app.exception_handler(StarletteHTTPException)
    async def http_error(_: Request, error: StarletteHTTPException) -> JSONResponse:
        mapping = {
            404: ("not_found", "Not found."),
            405: ("method_not_allowed", "That method is not allowed here."),
            413: ("payload_too_large", "The request is too large."),
            415: ("unsupported_media_type", "Send the request as application/json."),
        }
        code, message = mapping.get(error.status_code, ("invalid_request", "The request is not valid."))
        body = ErrorOut(error=ErrorDetail(code=code, message=message, request_id=request_id_var.get()))
        headers = {"Allow": "POST"} if error.status_code == 405 else None
        return JSONResponse(
            body.model_dump(), status_code=error.status_code if error.status_code in mapping else 400, headers=headers
        )

    # --- helpers --------------------------------------------------------------------------------------------------

    def principal_of(request: Request) -> Principal:
        header = request.headers.get("authorization", "")
        scheme, _, credential = header.partition(" ")
        if scheme.lower() != "bearer" or not credential or " " in credential:
            raise AuthenticationError("missing or malformed Authorization header")
        return authenticate(
            deps.directory, credential, request.headers.get("origin"), int(deps.clock.now().timestamp())
        )

    async def require_json(request: Request) -> None:
        media = request.headers.get("content-type", "").split(";", 1)[0].strip().lower()
        if media != "application/json":
            raise StarletteHTTPException(status_code=415)

    def to_out(reply: ChatReply, request_id: str) -> ChatOut:
        answer = reply.answer
        return ChatOut(
            session_id=reply.session_id,
            answer=answer.text,
            grounded=answer.grounded,
            citations=[
                CitationOut(id=c.chunk_id, title=c.title, url=c.url, snippet=c.snippet) for c in answer.citations
            ],
            refusal_reason=answer.refusal_reason,
            request_id=request_id,
        )

    # --- routes ---------------------------------------------------------------------------------------------------

    @app.post("/chat", response_model=ChatOut, dependencies=[Depends(require_json)])
    async def chat(request: Request, body: ChatBody) -> ChatOut:
        principal = principal_of(request)
        if body.session_id is not None:
            decision = session_limiter.acquire(
                f"{principal.tenant.tenant_id}:{body.session_id}", deps.clock.monotonic()
            )
            if not decision.allowed:
                log.warning("rate limited", extra={"event": "rate_limited", "scope": "session"})
                raise RateLimitedError("session", int(decision.retry_after_s) + 1)
        command = ChatCommand(
            tenant=principal.tenant,
            message=body.message,
            session_id=body.session_id,
            owner_id=principal.subject,
            request_id=request_id_var.get(),
        )
        try:
            async with asyncio.timeout(settings.request_timeout_s):
                reply = await deps.chat.handle(command)
        except TimeoutError:
            raise ProviderError("the request took too long") from None
        log.info(
            "chat answered",
            extra={
                "event": "chat",
                "tenant": principal.tenant.tenant_id,
                "grounded": reply.answer.grounded,
                "refusal": reply.answer.refusal_reason or "-",
                "message_chars": len(body.message),
                "answer_chars": len(reply.answer.text),
                "citations": len(reply.answer.citations),
                "cost_micros": reply.cost.micros,
                "tokens_in": reply.usage.input_tokens,
                "tokens_out": reply.usage.output_tokens,
            },
        )
        return to_out(reply, request_id_var.get())

    @app.post("/embed-token", response_model=TokenOut, dependencies=[Depends(require_json)])
    async def embed_token(request: Request, body: TokenBody) -> TokenOut:
        principal = principal_of(request)
        tenant = principal.tenant
        if principal.method != "api_key":
            raise ForbiddenError("an embed token cannot mint tokens")
        if not tenant.token_secrets:
            raise ForbiddenError("the tenant has no token secret")
        now = int(deps.clock.now().timestamp())
        ttl = min(body.ttl_seconds, settings.max_token_ttl_s)
        claims = TokenClaims(tenant.tenant_id, now + ttl, body.subject, body.origin)
        validate_claims_for_minting(claims, now)
        log.info("embed token minted", extra={"event": "token", "tenant": tenant.tenant_id, "ttl_s": ttl})
        return TokenOut(token=mint_token(tenant.token_secrets[0], claims), expires_at=claims.expires_at)

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/readyz")
    async def readyz() -> JSONResponse:
        try:
            problems = await asyncio.wait_for(deps.readiness(), timeout=5)
        except (ChatbotError, TimeoutError):
            problems = ["database_unreachable"]
        if problems:
            return JSONResponse({"status": "unavailable", "problems": problems}, status_code=503)
        return JSONResponse({"status": "ready"})

    @app.get("/widget.js")
    async def widget_js(request: Request) -> Response:
        headers = {
            "ETag": widget_etag,
            "Cache-Control": "public, max-age=3600",
            "Cross-Origin-Resource-Policy": "cross-origin",
            "Content-Security-Policy": "default-src 'none'",
        }
        if request.headers.get("if-none-match") == widget_etag:
            return Response(status_code=304, headers=headers)
        return Response(widget, media_type="text/javascript; charset=utf-8", headers=headers)

    # --- middleware, innermost first ------------------------------------------------------------------------------

    app.add_middleware(BodyLimitMiddleware, max_bytes=settings.max_body_bytes)
    app.add_middleware(
        RateLimitMiddleware,
        limiter=ip_limiter,  # chat and token minting share the address budget
        paths=frozenset({"/chat", "/embed-token"}),
        clock=deps.clock,
        trusted_hops=settings.trusted_proxy_hops,
    )
    app.add_middleware(CorsMiddleware, origins=deps.directory.all_origins)
    app.add_middleware(RequestContextMiddleware, ids=deps.ids, clock=deps.clock, settings=settings)
    return app
