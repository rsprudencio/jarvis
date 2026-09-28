"""Local stand-in for jarvis-core's hook endpoints, for hook-client tests.

Binds 127.0.0.1 on an ephemeral port, records every request, and answers
from a per-route table so tests can script 200/400/503 replies and slow
(wedged) handlers without touching a real server.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

HOOKS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "hooks-handlers")
if HOOKS_DIR not in sys.path:
    sys.path.insert(0, HOOKS_DIR)


class FakeCore:
    def __init__(self):
        self.routes: dict[tuple[str, str], tuple[int, object, float]] = {}
        self.requests: list[tuple[str, str, object]] = []
        self._lock = threading.Lock()
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    def respond(self, method: str, path: str, status: int = 200, body=None, delay: float = 0.0):
        """Script the reply for METHOD PATH (body: dict -> JSON, bytes -> raw)."""
        self.routes[(method, path)] = (status, {} if body is None else body, delay)

    def paths(self) -> list[str]:
        with self._lock:
            return [path for _, path, _ in self.requests]

    @property
    def base_url(self) -> str:
        host, port = self._server.server_address[:2]
        return f"http://{host}:{port}"

    def start(self) -> "FakeCore":
        fake = self

        class Handler(BaseHTTPRequestHandler):
            def _handle(self):
                length = int(self.headers.get("Content-Length") or 0)
                raw = self.rfile.read(length) if length else b""
                try:
                    payload = json.loads(raw.decode("utf-8")) if raw else None
                except ValueError:
                    payload = raw
                with fake._lock:
                    fake.requests.append((self.command, self.path, payload))
                status, body, delay = fake.routes.get(
                    (self.command, self.path), (404, {"error": "Not found"}, 0.0)
                )
                if delay:
                    time.sleep(delay)
                data = body if isinstance(body, bytes) else json.dumps(body).encode()
                try:
                    self.send_response(status)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)
                except OSError:
                    pass  # client already gave up (timeout tests)

            do_POST = _handle
            do_PUT = _handle

            def log_message(self, *args):
                pass

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._server.daemon_threads = True
        self._thread = threading.Thread(
            target=self._server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True
        )
        self._thread.start()
        return self

    def stop(self):
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()


@pytest.fixture
def fake_core():
    core = FakeCore().start()
    try:
        yield core
    finally:
        core.stop()


@pytest.fixture
def hook_env(tmp_path, monkeypatch, fake_core):
    """Isolated JARVIS_HOME and the real hook HTTP client aimed at fake_core."""
    import hook_http_client

    jarvis_home = tmp_path / ".jarvis"
    monkeypatch.setenv("JARVIS_HOME", str(jarvis_home))
    monkeypatch.setattr(
        hook_http_client, "resolve_base_url", lambda mcp_json_path=None: fake_core.base_url
    )
    return jarvis_home
