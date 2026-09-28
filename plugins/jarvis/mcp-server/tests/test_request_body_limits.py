"""Request body caps that the client can actually see (review CR-3, SEC-3).

A body over core's 1 MiB hook cap used to get a 413 with the rest of the body
unread; uvicorn then closed a socket with data pending and the client saw a
broken pipe or reset instead of the 413 — classified as "core unavailable",
which silenced every hook for 60s. /mcp had no cap at all (the SDK reads the
whole body with no limit).
"""

from __future__ import annotations

import asyncio
import io
import json
import socket
import sys
import threading
import time

import pytest

from tests.fake_core_server import HOOKS_DIR  # noqa: F401 (puts hooks-handlers on sys.path)

import hook_http_client
from hook_http_client import is_core_degraded, post_json

try:
    import mcp.server.streamable_http_manager  # noqa: F401

    _HAS_STREAMABLE_HTTP = True
except Exception:
    _HAS_STREAMABLE_HTTP = False

needs_http_app = pytest.mark.skipif(
    not _HAS_STREAMABLE_HTTP,
    reason="Streamable HTTP module only available in Docker environment",
)

MiB = 1024 * 1024


def _messages(total: int, chunk: int = 256 * 1024) -> list[dict]:
    out = []
    sent = 0
    while sent < total:
        n = min(chunk, total - sent)
        sent += n
        out.append({"type": "http.request", "body": b"x" * n, "more_body": sent < total})
    return out


def _receiver(messages: list[dict]):
    inbox = list(messages)

    async def receive():
        if inbox:
            return inbox.pop(0)
        await asyncio.Event().wait()

    return receive, inbox


# ── _read_request_body ──────────────────────────────────────────────────


@needs_http_app
def test_oversized_body_is_drained_before_413():
    import http_app

    receive, inbox = _receiver(_messages(3 * MiB))
    with pytest.raises(http_app._BodyTooLarge) as exc:
        asyncio.run(http_app._read_request_body(receive))
    assert exc.value.limit == http_app.MAX_REQUEST_BODY_BYTES
    assert inbox == []  # everything read, so the 413 can be delivered


@needs_http_app
def test_drain_is_bounded():
    import http_app

    total = http_app.MAX_REQUEST_BODY_BYTES + http_app._MAX_DRAIN_BYTES + 4 * MiB
    receive, inbox = _receiver(_messages(total, chunk=MiB))
    with pytest.raises(http_app._BodyTooLarge):
        asyncio.run(http_app._read_request_body(receive))
    assert inbox, "stops reading past the drain bound"


@needs_http_app
def test_disconnect_while_draining_is_a_disconnect():
    import http_app

    messages = _messages(2 * MiB)[:-1] + [{"type": "http.disconnect"}]
    receive, _ = _receiver(messages)
    with pytest.raises(http_app._ClientDisconnected):
        asyncio.run(http_app._read_request_body(receive))


# ── /mcp ────────────────────────────────────────────────────────────────


def _scope(method: str, path: str) -> dict:
    return {
        "type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1",
        "method": method, "scheme": "http", "path": path, "raw_path": path.encode(),
        "query_string": b"", "headers": [(b"content-type", b"application/json")],
        "client": ("127.0.0.1", 50000), "server": ("127.0.0.1", 8741),
    }


@needs_http_app
def test_mcp_post_body_is_capped(monkeypatch):
    import http_app

    monkeypatch.setattr(http_app, "authenticate", lambda scope: ("tester", ""))

    async def sdk_must_not_run(scope, receive, send):
        raise AssertionError("an oversized body must never reach the SDK")

    monkeypatch.setattr(http_app.session_manager, "handle_request", sdk_must_not_run)
    receive, _ = _receiver(_messages(http_app.MAX_MCP_BODY_BYTES + MiB, chunk=MiB))
    sent = []

    async def send(message):
        sent.append(message)

    asyncio.run(http_app.app(_scope("POST", "/mcp"), receive, send))
    assert sent[0]["status"] == 413
    body = json.loads(sent[1]["body"])
    assert str(http_app.MAX_MCP_BODY_BYTES) in body["error"]


@needs_http_app
def test_mcp_post_body_is_replayed_to_the_sdk(monkeypatch):
    """Under the cap the SDK reads exactly the body the client sent."""
    import http_app

    monkeypatch.setattr(http_app, "authenticate", lambda scope: ("tester", ""))
    seen = {}

    async def fake_sdk(scope, receive, send):
        chunks = []
        while True:
            message = await receive()
            chunks.append(message.get("body", b""))
            if not message.get("more_body"):
                break
        seen["body"] = b"".join(chunks)
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"{}"})

    monkeypatch.setattr(http_app.session_manager, "handle_request", fake_sdk)
    payload = json.dumps({"jsonrpc": "2.0", "id": 1, "method": "tools/list", "pad": "p" * 3 * MiB}).encode()
    messages = [
        {"type": "http.request", "body": payload[i:i + MiB], "more_body": i + MiB < len(payload)}
        for i in range(0, len(payload), MiB)
    ]
    receive, _ = _receiver(messages)
    sent = []

    async def send(message):
        sent.append(message)

    asyncio.run(http_app.app(_scope("POST", "/mcp"), receive, send))
    assert sent[0]["status"] == 200
    assert seen["body"] == payload


# ── Real server, real hook client ───────────────────────────────────────


def _free_port() -> int:
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


@needs_http_app
def test_client_sees_413_for_oversized_hook_bodies(tmp_path, monkeypatch):
    """Reviewer's repro: 2-8 MiB bodies read as broken pipe / reset (outage)."""
    import uvicorn

    import http_app
    import tools.hook_endpoints as hook_endpoints

    monkeypatch.setenv("JARVIS_HOME", str(tmp_path / ".jarvis"))
    monkeypatch.setattr(http_app, "authenticate", lambda scope: ("tester", ""))
    monkeypatch.setattr(
        hook_endpoints, "get_prompt_context", lambda prompt: {"success": True, "matches": []}
    )
    port = _free_port()
    srv = uvicorn.Server(uvicorn.Config(
        http_app.app, host="127.0.0.1", port=port, lifespan="off",
        log_level="warning", http="h11",
    ))
    thread = threading.Thread(target=srv.run, daemon=True)
    thread.start()
    try:
        deadline = time.monotonic() + 5
        while not srv.started and time.monotonic() < deadline:
            time.sleep(0.02)
        monkeypatch.setattr(
            hook_http_client, "resolve_base_url",
            lambda mcp_json_path=None: f"http://127.0.0.1:{port}",
        )
        for size in (2 * MiB, 4 * MiB, 8 * MiB):
            for _ in range(3):
                result = post_json(
                    "/hook/prompt-context", {"prompt": "x" * size}, timeout_seconds=10
                )
                assert result["http_status"] == 413, result["error"]
                assert result["permanent"] is True
                assert result["retryable"] is False
                assert not is_core_degraded()
    finally:
        srv.should_exit = True
        thread.join(timeout=5)
        http_app._shutdown_hook_executor()


# ── Client-side truncation ──────────────────────────────────────────────


def test_prompt_context_request_is_truncated(tmp_path, monkeypatch, capsys):
    """Core only uses the head of a prompt; a huge paste is not sent whole."""
    import context_enrichment

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("JARVIS_HOME", str(tmp_path / ".jarvis"))

    sent = []

    def fake_post(path, payload, **kwargs):
        sent.append(payload)
        return {"success": True, "data": {"success": True, "matches": []}, "error": ""}

    monkeypatch.setattr(context_enrichment, "post_json", fake_post)
    prompt = "please review this log " + "y" * (4 * MiB)
    monkeypatch.setattr(sys, "argv", ["context_enrichment.py", prompt])
    monkeypatch.setattr(sys, "stdin", io.StringIO(""))
    with pytest.raises(SystemExit):
        context_enrichment.main()
    assert len(sent) == 1
    assert sent[0]["prompt"] == prompt[: context_enrichment.MAX_PROMPT_CHARS]
    assert len(json.dumps(sent[0]).encode()) < 1024 * 1024
