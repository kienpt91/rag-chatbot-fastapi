from __future__ import annotations

import asyncio
import json
import threading
from collections.abc import Iterator
from typing import Any

import pytest

from rag_chatbot.domain.errors import ConfigurationError, ProviderError
from rag_chatbot.domain.model import ChatTurn
from rag_chatbot.infrastructure.providers.http import (
    ApiEndpoint,
    HttpApiClient,
    HttpChatProvider,
    HttpEmbeddingProvider,
)
from support.fake_api import FakeApi

KEY = "test-key-" + "k" * 24


class Sleeps:
    def __init__(self) -> None:
        self.calls: list[float] = []

    async def __call__(self, seconds: float) -> None:
        self.calls.append(seconds)


@pytest.fixture
def api() -> Iterator[FakeApi]:
    server = FakeApi(dimensions=8).start()
    try:
        yield server
    finally:
        server.stop()


def client(api: FakeApi, sleeps: Sleeps | None = None, **kwargs: Any) -> HttpApiClient:
    return HttpApiClient(api.url + "/v1", KEY, sleep=sleeps or Sleeps(), **kwargs)


def reply(status: int, body: object, headers: dict[str, str] | None = None) -> tuple[int, bytes, dict[str, str]]:
    raw = body if isinstance(body, bytes) else json.dumps(body).encode()
    return status, raw, {"Content-Type": "application/json", **(headers or {})}


# --- the requests ----------------------------------------------------------------------------------------------------


def test_a_chat_request_has_the_system_prompt_first_the_turns_after_and_the_key_in_the_header(api: FakeApi) -> None:
    provider = HttpChatProvider(client(api), "chat-small")
    messages = [ChatTurn("user", "hi"), ChatTurn("assistant", "hello"), ChatTurn("user", "bye")]
    result = asyncio.run(provider.complete("be brief", messages, max_output_tokens=77))
    [recorded] = api.requests
    assert recorded.path == "/v1/chat/completions"
    assert (
        recorded.headers["authorization"] == f"Bearer {KEY}" and recorded.headers["content-type"] == "application/json"
    )
    assert (
        recorded.body["model"] == "chat-small"
        and recorded.body["max_tokens"] == 77
        and recorded.body["temperature"] == 0.0
    )
    assert [m["role"] for m in recorded.body["messages"]] == ["system", "user", "assistant", "user"]
    assert recorded.body["messages"][0]["content"] == "be brief"
    assert result.model == "chat-small" and result.text and result.usage.input_tokens > 0


def test_embeddings_are_requested_in_batches_and_returned_in_order(api: FakeApi) -> None:
    provider = HttpEmbeddingProvider(client(api), "embed-small", dimensions=8, max_batch=2)
    texts = ["alpha one", "beta two", "gamma three", "delta four", "epsilon five"]
    result = asyncio.run(provider.embed(texts))
    assert len(api.requests) == 3 and [len(r.body["input"]) for r in api.requests] == [2, 2, 1]
    assert result.model == "embed-small" and result.usage.input_tokens > 0 and len(result.vectors) == 5
    assert all(len(v) == 8 for v in result.vectors)
    single = asyncio.run(provider.embed(["gamma three"]))
    assert single.vectors[0] == result.vectors[2]  # the order of the answer matched the order of the texts


def test_dimensions_are_sent_only_when_asked_for(api: FakeApi) -> None:
    asyncio.run(HttpEmbeddingProvider(client(api), "m", dimensions=8).embed(["x"]))
    asyncio.run(HttpEmbeddingProvider(client(api), "m", dimensions=8, send_dimensions=True).embed(["x"]))
    assert "dimensions" not in api.requests[0].body and api.requests[1].body["dimensions"] == 8


def test_no_texts_means_no_request(api: FakeApi) -> None:
    result = asyncio.run(HttpEmbeddingProvider(client(api), "m", dimensions=8).embed([]))
    assert result.vectors == () and api.requests == []


def test_the_answer_may_list_embeddings_out_of_order_and_is_put_in_order(api: FakeApi) -> None:
    api.override = lambda path, body: reply(
        200,
        {
            "data": [{"index": 1, "embedding": [2.0] * 8}, {"index": 0, "embedding": [1.0] * 8}],
            "usage": {"prompt_tokens": 4},
        },
    )
    result = asyncio.run(HttpEmbeddingProvider(client(api), "m", dimensions=8).embed(["a", "b"]))
    assert result.vectors == ((1.0,) * 8, (2.0,) * 8) and result.usage.input_tokens == 4


# --- retries and failures --------------------------------------------------------------------------------------------


def test_a_busy_server_is_retried_with_its_retry_after_capped_and_then_succeeds(api: FakeApi) -> None:
    api.statuses = [429, 503]
    api.retry_after = "30"
    sleeps = Sleeps()
    provider = HttpEmbeddingProvider(client(api, sleeps), "m", dimensions=8)
    assert len(asyncio.run(provider.embed(["x"])).vectors) == 1
    assert len(api.requests) == 3 and sleeps.calls == [5.0, 5.0]  # 30 seconds asked for, 5 allowed


def test_without_retry_after_the_back_off_grows(api: FakeApi) -> None:
    api.statuses = [500, 502]
    sleeps = Sleeps()
    asyncio.run(HttpEmbeddingProvider(client(api, sleeps), "m", dimensions=8).embed(["x"]))
    assert sleeps.calls == [0.5, 1.0]


def test_when_the_retries_run_out_the_error_names_the_status_and_nothing_else(api: FakeApi) -> None:
    api.statuses = [503, 503, 503, 503]
    with pytest.raises(ProviderError) as caught:
        asyncio.run(HttpEmbeddingProvider(client(api), "m", dimensions=8).embed(["secret question"]))
    assert len(api.requests) == 3 and "503" in str(caught.value)
    assert KEY not in str(caught.value) and "secret question" not in str(caught.value)


@pytest.mark.parametrize("status", [400, 401, 403, 404, 422])
def test_a_client_error_is_not_retried(api: FakeApi, status: int) -> None:
    api.statuses = [status]
    with pytest.raises(ProviderError, match=str(status)):
        asyncio.run(HttpEmbeddingProvider(client(api), "m", dimensions=8).embed(["x"]))
    assert len(api.requests) == 1


def test_a_redirect_is_never_followed_so_the_key_cannot_be_sent_elsewhere(api: FakeApi) -> None:
    api.override = lambda path, body: (302, b"", {"Location": "http://127.0.0.1:1/steal"})
    with pytest.raises(ProviderError, match="302"):
        asyncio.run(HttpEmbeddingProvider(client(api), "m", dimensions=8).embed(["x"]))
    assert len(api.requests) == 1


def test_an_unreachable_server_is_a_provider_error_without_the_address() -> None:
    provider = HttpEmbeddingProvider(
        HttpApiClient("http://127.0.0.1:1", KEY, sleep=Sleeps(), timeout_s=1), "m", dimensions=8
    )
    with pytest.raises(ProviderError) as caught:
        asyncio.run(provider.embed(["x"]))
    assert "127.0.0.1" not in str(caught.value) and "network failure" in str(caught.value)


def test_a_server_that_never_answers_ends_in_a_timeout_error(api: FakeApi) -> None:
    release = threading.Event()
    api.override = lambda path, body: (release.wait(10), reply(200, {}))[1]
    try:
        provider = HttpEmbeddingProvider(client(api, retries=0, timeout_s=0.3), "m", dimensions=8)
        with pytest.raises(ProviderError, match="network failure"):
            asyncio.run(provider.embed(["x"]))
    finally:
        release.set()


def test_an_oversized_answer_is_cut_off_and_refused(api: FakeApi) -> None:
    api.override = lambda path, body: reply(200, b'{"data": [' + b"0," * 5000 + b"0]}")
    with pytest.raises(ProviderError, match="larger than"):
        asyncio.run(HttpEmbeddingProvider(client(api, max_response_bytes=1000), "m", dimensions=8).embed(["x"]))
    assert len(api.requests) == 1


@pytest.mark.parametrize(
    "body",
    [b"not json", b"[1, 2]", b'"text"', b"null", b'{"data": NaN}', b'{"data": Infinity}', b"{" * 100_000, b""],
)
def test_malformed_answers_are_provider_errors(api: FakeApi, body: bytes) -> None:
    api.override = lambda path, request: reply(200, body)
    with pytest.raises(ProviderError):
        asyncio.run(HttpEmbeddingProvider(client(api), "m", dimensions=8).embed(["x"]))
    with pytest.raises(ProviderError):
        asyncio.run(HttpChatProvider(client(api), "m").complete("s", [ChatTurn("user", "q")], max_output_tokens=5))


@pytest.mark.parametrize(
    "answer",
    [
        {},
        {"data": []},
        {"data": [{"embedding": [0.0] * 7}]},  # wrong size
        {"data": [{"embedding": [0.0] * 9}]},
        {"data": [{"embedding": ["a"] * 8}]},
        {"data": [{"embedding": [True] * 8}]},
        {"data": [{"embedding": "text"}]},
        {"data": ["x"]},
        {"data": [{"index": 5, "embedding": [0.0] * 8}]},
        {"data": [{"index": True, "embedding": [0.0] * 8}]},
        {"data": [{"embedding": [0.0] * 8}, {"embedding": [0.0] * 8}]},  # one too many for one text
        {"data": {"embedding": [0.0] * 8}},
    ],
)
def test_malformed_embedding_answers_are_refused(api: FakeApi, answer: object) -> None:
    api.override = lambda path, body: reply(200, answer)
    with pytest.raises(ProviderError):
        asyncio.run(HttpEmbeddingProvider(client(api), "m", dimensions=8).embed(["x"]))


def test_duplicate_embedding_indexes_are_refused(api: FakeApi) -> None:
    api.override = lambda path, body: reply(200, {"data": [{"index": 0, "embedding": [0.0] * 8}] * 2})
    with pytest.raises(ProviderError):
        asyncio.run(HttpEmbeddingProvider(client(api), "m", dimensions=8).embed(["a", "b"]))


@pytest.mark.parametrize(
    "answer",
    [
        {},
        {"choices": []},
        {"choices": ["x"]},
        {"choices": [{}]},
        {"choices": [{"message": {}}]},
        {"choices": [{"message": {"content": 5}}]},
    ],
)
def test_malformed_chat_answers_are_refused(api: FakeApi, answer: object) -> None:
    api.override = lambda path, body: reply(200, answer)
    with pytest.raises(ProviderError):
        asyncio.run(HttpChatProvider(client(api), "m").complete("s", [ChatTurn("user", "q")], max_output_tokens=5))


@pytest.mark.parametrize(
    ("usage", "expected"),
    [
        ({"prompt_tokens": 12, "completion_tokens": 5}, (12, 5)),
        ({}, (0, 0)),
        ("lots", (0, 0)),
        ({"prompt_tokens": -3, "completion_tokens": True}, (0, 0)),
        ({"prompt_tokens": 1.5}, (0, 0)),
        ({"prompt_tokens": 10**12}, (0, 0)),
    ],
)
def test_reported_usage_is_read_defensively(api: FakeApi, usage: object, expected: tuple[int, int]) -> None:
    api.override = lambda path, body: reply(200, {"choices": [{"message": {"content": "ok"}}], "usage": usage})
    result = asyncio.run(HttpChatProvider(client(api), "m").complete("s", [ChatTurn("user", "q")], max_output_tokens=5))
    assert (result.usage.input_tokens, result.usage.output_tokens) == expected and not result.usage.estimated


# --- configuration ---------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "url",
    [
        "http://api.example.com/v1",  # plain http to a remote host
        "ftp://api.example.com",
        "https://user:pw@api.example.com/v1",
        "https://api.example.com/v1?x=1",
        "https://api.example.com/v1#frag",
        "https://",
        "not a url",
        "https://api.example.com:notaport",
    ],
)
def test_unsafe_or_invalid_base_urls_are_refused(url: str) -> None:
    with pytest.raises(ConfigurationError):
        ApiEndpoint.parse(url)


def test_acceptable_base_urls_are_parsed() -> None:
    assert ApiEndpoint.parse("https://api.example.com/v1/") == ApiEndpoint("https", "api.example.com", 443, "/v1")
    assert ApiEndpoint.parse("http://localhost:8080") == ApiEndpoint("http", "localhost", 8080, "")
    assert ApiEndpoint.parse("http://127.0.0.1:9/x") == ApiEndpoint("http", "127.0.0.1", 9, "/x")


@pytest.mark.parametrize("key", ["", "with space", "new\nline", "tab\t", "caf\u00e9"])
def test_a_key_that_could_inject_a_header_is_refused(key: str) -> None:
    with pytest.raises(ConfigurationError):
        HttpApiClient("https://api.example.com", key)


def test_the_key_is_never_in_the_representation() -> None:
    assert KEY not in repr(HttpApiClient("https://api.example.com/v1", KEY))
