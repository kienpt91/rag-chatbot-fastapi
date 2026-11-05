"""A loopback HTTP server that speaks the chat/embedding wire format, answering with the deterministic fakes.

Used by the provider tests and by the end-to-end test, where the real server process talks to it over HTTP. It can be
told to misbehave: a scripted list of statuses, a stall, an oversized answer, a redirect, a malformed body.
"""

from __future__ import annotations

import asyncio
import json
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from rag_chatbot.domain.model import ChatTurn
from rag_chatbot.infrastructure.providers.fake import FakeChatProvider, FakeEmbeddingProvider


class QuietServer(ThreadingHTTPServer):
    """A client that gives up early (a timeout, an oversized answer) makes the handler fail: that is expected here."""

    def handle_error(self, request: object, client_address: object) -> None:
        pass


@dataclass
class Recorded:
    path: str
    headers: dict[str, str]
    body: dict[str, Any]


@dataclass
class FakeApi:
    dimensions: int = 1536
    gullible: bool = False
    statuses: list[int] = field(default_factory=list)  # consumed one per request before answering normally
    retry_after: str | None = None
    override: Callable[[str, dict[str, Any]], tuple[int, bytes, dict[str, str]]] | None = None
    requests: list[Recorded] = field(default_factory=list)
    _server: ThreadingHTTPServer | None = None
    _thread: threading.Thread | None = None

    @property
    def url(self) -> str:
        assert self._server is not None
        return f"http://127.0.0.1:{self._server.server_address[1]}"

    def start(self) -> FakeApi:
        api = self
        embedder = FakeEmbeddingProvider(self.dimensions)
        chat = FakeChatProvider(gullible=self.gullible)

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, format: str, *args: object) -> None:
                pass

            def do_POST(self) -> None:
                length = int(self.headers.get("Content-Length", "0"))
                body = json.loads(self.rfile.read(length) or b"{}")
                api.requests.append(Recorded(self.path, {k.lower(): v for k, v in self.headers.items()}, body))
                if api.statuses:
                    status = api.statuses.pop(0)
                    self.send_response(status)
                    if api.retry_after:
                        self.send_header("Retry-After", api.retry_after)
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                if api.override is not None:
                    status, payload, headers = api.override(self.path, body)
                    self.send_response(status)
                    for key, value in headers.items():
                        self.send_header(key, value)
                    self.send_header("Content-Length", str(len(payload)))
                    self.end_headers()
                    self.wfile.write(payload)
                    return
                if self.path.endswith("/embeddings"):
                    result = asyncio.run(embedder.embed(body["input"]))
                    answer: dict[str, Any] = {
                        "data": [{"index": i, "embedding": list(v)} for i, v in enumerate(result.vectors)],
                        "usage": {"prompt_tokens": result.usage.input_tokens},
                    }
                elif self.path.endswith("/chat/completions"):
                    messages = body["messages"]
                    turns = [
                        ChatTurn("assistant" if m["role"] == "assistant" else "user", m["content"])
                        for m in messages[1:]
                    ]
                    completion = asyncio.run(
                        chat.complete(messages[0]["content"], turns, max_output_tokens=body["max_tokens"])
                    )
                    answer = {
                        "choices": [{"message": {"role": "assistant", "content": completion.text}}],
                        "usage": {
                            "prompt_tokens": completion.usage.input_tokens,
                            "completion_tokens": completion.usage.output_tokens,
                        },
                    }
                else:
                    self.send_response(404)
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                payload = json.dumps(answer).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

        self._server = QuietServer(("127.0.0.1", 0), Handler)
        server = self._server
        # a short poll interval keeps shutdown() from waiting half a second at the end of every test
        self._thread = threading.Thread(target=lambda: server.serve_forever(poll_interval=0.02), daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=5)
