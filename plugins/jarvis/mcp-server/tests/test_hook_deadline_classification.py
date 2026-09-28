"""A hook deadline miss is a DB outage only when the database is down.

Review finding CR-1: the 2.0s hook deadline answered every slow request with
503 "database unavailable (timeout)", and the hook client turned that into the
60s core_degraded marker. One slow-but-healthy prompt-context (slow model-host
rerank, a queue behind busy hook workers) silenced memory injection in every
session for a minute. Now core says why (``error_kind``), prompt-context
answers a healthy-DB deadline miss with an empty 200, and the client only
backs off for real unavailability.
"""

from __future__ import annotations

import asyncio
import json
import socket
import threading
import time

import pytest

from tests.fake_core_server import fake_core, hook_env  # noqa: F401 (fixtures)

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

PROMPT = "/hook/prompt-context"
INGEST = "/hook/auto-extract/ingest"
CONTEXT = "/hook/auto-extract/context"


# ── Server side (raw ASGI) ──────────────────────────────────────────────


async def _call(app, path: str, body: dict):
    raw = json.dumps(body).encode()
    inbox = [{"type": "http.request", "body": raw, "more_body": False}]
    sent = []

    async def receive():
        if inbox:
            return inbox.pop(0)
        await asyncio.Event().wait()

    async def send(message):
        sent.append(message)

    scope = {
        "type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1",
        "method": "POST", "scheme": "http", "path": path, "raw_path": path.encode(),
        "query_string": b"", "headers": [], "client": ("127.0.0.1", 50000),
        "server": ("127.0.0.1", 8741),
    }
    await app(scope, receive, send)
    start = next(m for m in sent if m["type"] == "http.response.start")
    payload = b"".join(m.get("body", b"") for m in sent if m["type"] == "http.response.body")
    headers = {bytes(k).decode(): bytes(v).decode() for k, v in start["headers"]}
    return start["status"], headers, json.loads(payload)


@pytest.fixture
def slow_app(monkeypatch):
    """http_app whose hooks outlive a 0.2s deadline (released on teardown)."""
    import http_app
    import tools.hook_endpoints as hook_endpoints

    release = threading.Event()

    def slow(*args, **kwargs):
        release.wait(5)
        return {"success": True, "matches": [{"id": "m1"}]}

    real_run = http_app._run_blocking

    async def short_deadline(fn, *args, deadline=None, **kwargs):
        return await real_run(fn, *args, deadline=0.2, **kwargs)

    monkeypatch.setattr(http_app, "authenticate", lambda scope: ("tester", ""))
    monkeypatch.setattr(http_app, "_run_blocking", short_deadline)
    for name in ("get_prompt_context", "get_auto_extract_context", "ingest_auto_extract"):
        monkeypatch.setattr(hook_endpoints, name, slow)
    yield http_app
    release.set()
    http_app._shutdown_hook_executor()


def _run(coro_fn):
    return asyncio.run(coro_fn())


@needs_http_app
def test_deadline_with_healthy_db_is_not_reported_as_db_outage(slow_app, monkeypatch):
    monkeypatch.setattr(slow_app, "_db_available", lambda: True)

    async def main():
        return await asyncio.gather(
            _call(slow_app.app, PROMPT, {"prompt": "what did we decide?"}),
            _call(slow_app.app, INGEST, {"observations": []}),
            _call(slow_app.app, CONTEXT, {"workstream_limit": 5}),
        )

    (p_status, _, p_body), (i_status, i_headers, i_body), (c_status, _, c_body) = _run(main)

    # Prompt-context: nothing injected this once, and no outage signal.
    assert p_status == 200
    assert p_body["success"] is True and p_body["matches"] == []
    assert p_body["timed_out"] is True
    assert "degraded" not in p_body
    # Ingest / context: retry this payload, but it is not a DB outage.
    for status, body in ((i_status, i_body), (c_status, c_body)):
        assert status == 503
        assert body["retryable"] is True
        assert body["error_kind"] == "deadline"
        assert "database" not in body["error"]
    assert i_headers["retry-after"] == "10"


@needs_http_app
def test_deadline_with_breaker_open_is_a_db_outage(slow_app, monkeypatch):
    monkeypatch.setattr(slow_app, "_db_available", lambda: False)

    async def main():
        return await asyncio.gather(
            _call(slow_app.app, PROMPT, {"prompt": "what did we decide?"}),
            _call(slow_app.app, INGEST, {"observations": []}),
        )

    for status, headers, body in _run(main):
        assert status == 503
        assert headers["retry-after"] == "10"
        assert body == {
            "success": False, "retryable": True, "error_kind": "db_unavailable",
            "error": "database unavailable (timeout)",
        }


@needs_http_app
def test_db_unavailable_exception_carries_error_kind(monkeypatch):
    import http_app
    import tools.hook_endpoints as hook_endpoints
    from tools.schema import DatabaseUnavailable

    def down(*args, **kwargs):
        raise DatabaseUnavailable("the database system is in recovery mode")

    monkeypatch.setattr(http_app, "authenticate", lambda scope: ("tester", ""))
    monkeypatch.setattr(hook_endpoints, "ingest_auto_extract", down)
    try:
        status, _, body = _run(lambda: _call(http_app.app, INGEST, {"observations": []}))
    finally:
        http_app._shutdown_hook_executor()
    assert status == 503
    assert body["error_kind"] == "db_unavailable"
    assert body["error"] == "database unavailable: the database system is in recovery mode"


# ── Client side (real hook client against a scripted fake core) ─────────


def test_client_does_not_trip_breaker_on_deadline_503(hook_env, fake_core):
    fake_core.respond("POST", INGEST, 503, {
        "success": False, "retryable": True, "error_kind": "deadline",
        "error": "request timed out after 2s",
    })
    result = post_json(INGEST, {"observations": []})
    assert result["success"] is False
    assert result["retryable"] is True  # still re-queued by the Stop hook
    assert result["error_kind"] == "deadline"
    assert not is_core_degraded()

    # The next hook call still reaches core.
    fake_core.respond("POST", PROMPT, 200, {"success": True, "matches": []})
    assert post_json(PROMPT, {"prompt": "next"})["success"] is True
    assert fake_core.paths() == [INGEST, PROMPT]


def test_client_trips_breaker_on_db_unavailable_503(hook_env, fake_core):
    fake_core.respond("POST", INGEST, 503, {
        "success": False, "retryable": True, "error_kind": "db_unavailable",
        "error": "database unavailable: the database system is in recovery mode",
    })
    result = post_json(INGEST, {"observations": []})
    assert result["error_kind"] == "db_unavailable"
    assert is_core_degraded()


def test_client_trips_breaker_on_503_without_error_kind(hook_env, fake_core):
    """Older cores send no error_kind: every 503 still backs off."""
    fake_core.respond("POST", INGEST, 503, {"success": False, "retryable": True, "error": "x"})
    post_json(INGEST, {"observations": []})
    assert is_core_degraded()


def test_client_treats_timed_out_prompt_context_as_delivered(hook_env, fake_core):
    fake_core.respond("POST", PROMPT, 200, {"success": True, "matches": [], "timed_out": True})
    result = post_json(PROMPT, {"prompt": "hi"})
    assert result["success"] is True
    assert result["error_kind"] == ""
    assert not is_core_degraded()


# ── End to end: real http_app on uvicorn, real hook client ──────────────


def _free_port() -> int:
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


@needs_http_app
def test_slow_healthy_prompt_does_not_black_out_the_next_one(tmp_path, monkeypatch):
    """The reviewer's repro: a 2.1s prompt-context on a healthy DB."""
    import uvicorn

    import http_app
    import tools.hook_endpoints as hook_endpoints

    monkeypatch.setenv("JARVIS_HOME", str(tmp_path / ".jarvis"))
    monkeypatch.setattr(http_app, "authenticate", lambda scope: ("tester", ""))
    monkeypatch.setattr(http_app, "_db_available", lambda: True)
    calls = []
    release = threading.Event()

    def slow_then_fast(prompt):
        calls.append(prompt)
        if len(calls) == 1:
            release.wait(2.1)
        return {"success": True, "enabled": True, "matches": [{"id": "m1", "content": "x"}]}

    monkeypatch.setattr(hook_endpoints, "get_prompt_context", slow_then_fast)

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

        t0 = time.monotonic()
        first = post_json(PROMPT, {"prompt": "what did we decide about the pool timeout?"})
        first_elapsed = time.monotonic() - t0
        assert first["success"] is True, first
        assert first["data"]["timed_out"] is True
        assert first_elapsed < 2.5
        assert not is_core_degraded()

        second = post_json(PROMPT, {"prompt": "another question"})
        assert second["success"] is True and second["skipped"] is False
        assert second["data"]["matches"] == [{"id": "m1", "content": "x"}]
        assert len(calls) == 2  # the second prompt reached core
    finally:
        release.set()
        srv.should_exit = True
        thread.join(timeout=5)
        http_app._shutdown_hook_executor()
