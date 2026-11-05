"""The shapes of the API's requests and responses."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

SESSION_ID_PATTERN = r"^s_[A-Za-z0-9_-]{8,64}$"


class ChatBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    message: str = Field(min_length=1, max_length=2000)
    session_id: str | None = Field(default=None, pattern=SESSION_ID_PATTERN)


class CitationOut(BaseModel):
    id: str
    title: str
    url: str
    snippet: str


class ChatOut(BaseModel):
    session_id: str
    answer: str
    grounded: bool
    citations: list[CitationOut]
    refusal_reason: str | None
    request_id: str


class ErrorDetail(BaseModel):
    code: str
    message: str
    request_id: str


class ErrorOut(BaseModel):
    error: ErrorDetail


class TokenBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ttl_seconds: int = Field(default=900, ge=1, le=86400)
    subject: str | None = Field(default=None, max_length=64)
    origin: str | None = Field(default=None, max_length=300)


class TokenOut(BaseModel):
    token: str
    expires_at: int
