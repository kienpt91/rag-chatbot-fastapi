"""Errors of the domain. Each carries a stable machine-readable `code` and a message that is safe to show a visitor:
nothing in a message names a file, a query, a host, a secret or a person."""

from __future__ import annotations


class ChatbotError(Exception):
    """Base class. `code` is stable and part of the API; `public_message` is what a visitor may see."""

    code = "internal_error"
    public_message = "Something went wrong. Please try again later."

    def __init__(self, detail: str = "") -> None:
        super().__init__(detail or self.code)
        self.detail = detail


class InvalidRequestError(ChatbotError):
    code = "invalid_request"
    public_message = "The request is not valid."


class AuthenticationError(ChatbotError):
    code = "unauthorized"
    public_message = "Authentication failed."


class OriginNotAllowedError(ChatbotError):
    code = "origin_not_allowed"
    public_message = "This site may not use the assistant."


class SessionNotFoundError(ChatbotError):
    code = "session_not_found"
    public_message = "The conversation was not found. Start a new one."


class SessionFullError(ChatbotError):
    code = "session_full"
    public_message = "This conversation has reached its length limit. Please start a new one."


class RateLimitedError(ChatbotError):
    code = "rate_limited"
    public_message = "Too many messages. Please wait a moment and try again."

    def __init__(self, detail: str = "", retry_after_s: int = 1) -> None:
        super().__init__(detail)
        self.retry_after_s = max(1, retry_after_s)


class BudgetExhaustedError(ChatbotError):
    code = "budget_exhausted"
    public_message = "The assistant is not available right now. Please contact us directly."


class ProviderError(ChatbotError):
    """The language model or embedding service failed or answered something unusable."""

    code = "provider_unavailable"
    public_message = "The assistant could not answer right now. Please try again in a moment."


class StorageError(ChatbotError):
    code = "storage_unavailable"
    public_message = "The assistant could not answer right now. Please try again in a moment."


class ConfigurationError(ChatbotError):
    """The operator's configuration is wrong. Raised at start-up, never during a request."""

    code = "configuration_error"
    public_message = "The service is misconfigured."


class SourceError(ChatbotError):
    """A content source (folder or site) could not be read as asked."""

    code = "source_error"
    public_message = "The content source could not be read."
