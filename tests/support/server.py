"""The real server process, started the way an operator starts it, and a small HTTP client for it."""

from __future__ import annotations

import http.client
import json
import os
import signal
import socket
import subprocess
import sys
import time
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]


@dataclass
class Response:
    status: int
    headers: dict[str, str]
    body: bytes

    def json(self) -> dict[str, Any]:
        parsed = json.loads(self.body)
        assert isinstance(parsed, dict)
        return parsed


def free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


@dataclass
class ServerProcess:
    port: int
    process: subprocess.Popen[bytes]
    output: list[str] = field(default_factory=list)

    def request(
        self,
        method: str,
        path: str,
        *,
        headers: Mapping[str, str] | None = None,
        body: bytes | object | None = None,
    ) -> Response:
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=15)
        try:
            payload: bytes | None
            sent = dict(headers or {})
            if body is None:
                payload = None
            elif isinstance(body, bytes):
                payload = body
            else:
                payload = json.dumps(body).encode()
                sent.setdefault("Content-Type", "application/json")
            connection.request(method, path, body=payload, headers=sent)
            response = connection.getresponse()
            return Response(response.status, {k.lower(): v for k, v in response.getheaders()}, response.read())
        finally:
            connection.close()

    def stop(self) -> None:
        """Terminate the process (a no-op when it is gone), collect everything it printed and close its pipe."""
        if self.process.poll() is None:
            self.process.send_signal(signal.SIGTERM)
            try:
                self.process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        if self.process.stdout is not None and not self.process.stdout.closed:
            self.output.extend(self.process.stdout.read().decode("utf-8", errors="replace").splitlines())
            self.process.stdout.close()

    def chat(self, message: str, key: str, *, origin: str | None = None, session_id: str | None = None) -> Response:
        headers = {"Authorization": f"Bearer {key}"}
        if origin:
            headers["Origin"] = origin
        payload: dict[str, object] = {"message": message}
        if session_id:
            payload["session_id"] = session_id
        return self.request("POST", "/chat", headers=headers, body=payload)


@contextmanager
def running_server(env: Mapping[str, str], *, timeout_s: float = 30.0) -> Iterator[ServerProcess]:
    port = free_port()
    full_env = {
        "PATH": os.environ.get("PATH", ""),
        "PYTHONPATH": str(ROOT / "src"),
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONUNBUFFERED": "1",
        "HOME": "/tmp",
        **env,
        "RAG_HOST": "127.0.0.1",
        "RAG_PORT": str(port),
    }
    process = subprocess.Popen(
        [sys.executable, "-m", "rag_chatbot", "serve"],
        env=full_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        cwd=ROOT,
    )
    server = ServerProcess(port, process)
    try:
        deadline = time.monotonic() + timeout_s
        while True:  # wait for the server to come up: poll the readiness probe, and notice if it died
            if process.poll() is not None:
                server.stop()
                raise RuntimeError("the server exited during start-up:\n" + "\n".join(server.output))
            try:
                if server.request("GET", "/readyz").status == 200:
                    break
            except OSError:
                pass
            if time.monotonic() > deadline:
                raise RuntimeError("the server did not become ready")
            time.sleep(0.05)
        yield server
    finally:
        server.stop()
