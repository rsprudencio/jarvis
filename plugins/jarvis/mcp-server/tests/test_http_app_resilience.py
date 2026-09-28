"""Outage-resilience tests for the core HTTP transport (http_app.py).

Regression guards for the 2026-09-24 outage: blocking DB calls ran on the
event loop, so every 30s pool wait froze /health and every hook; a client
hanging up mid-body spun the loop at 100% CPU; the access log dumped every
request header. The ASGI app is driven in-process with raw scope/receive/send
(plus one real uvicorn h11 server on an ephemeral localhost port).
"""

import asyncio
import importlib
import json
import logging
import socket
import threading
import time
import urllib.request

import pytest

try:
    import mcp.server.streamable_http_manager  # noqa: F401

    _HAS_STREAMABLE_HTTP = True
except Exception:
    _HAS_STREAMABLE_HTTP = False

pytestmark = pytest.mark.skipif(
    not _HAS_STREAMABLE_HTTP,
    reason="Streamable HTTP module only available in Docker environment",
)

_RECOVERING = {
    "status": "recovering",
    "error": "the database system is in recovery mode",
    "checked_at": 1.0,
    "free_bytes": 123,
}


# ── Raw ASGI driver ─────────────────────────────────────────────────────


def _scope(method: str, path: str, headers: dict | None = None) -> dict:
    return {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": method,
        "scheme": "http",
        "path": path,
        "raw_path": path.encode(),
        "query_string": b"",
        "headers": [
            (k.lower().encode("latin-1"), v.encode("latin-1"))
            for k, v in (headers or {}).items()
        ],
        "client": ("127.0.0.1", 50000),
        "server": ("127.0.0.1", 8741),
    }


async def _call(app, method, path, body=None, headers=None, messages=None):
    """Run one request through the ASGI app; return (status, headers, json)."""
    if messages is None:
        raw = json.dumps(body).encode() if body is not None else b""
        messages = [{"type": "http.request", "body": raw, "more_body": False}]
    inbox = list(messages)
    sent = []

    async def receive():
        if inbox:
            return inbox.pop(0)
        await asyncio.Event().wait()  # live connection: nothing more arrives

    async def send(message):
        sent.append(message)

    await app(_scope(method, path, headers), receive, send)
    starts = [m for m in sent if m["type"] == "http.response.start"]
    if not starts:
        return None, {}, None
    payload = b"".join(m.get("body", b"") for m in sent if m["type"] == "http.response.body")
    resp_headers = {bytes(k).decode(): bytes(v).decode() for k, v in starts[0]["headers"]}
    return starts[0]["status"], resp_headers, json.loads(payload) if payload else None


async def _timed(coro):
    t0 = time.perf_counter()
    result = await coro
    return time.perf_counter() - t0, result


# ── Fixtures ────────────────────────────────────────────────────────────


@pytest.fixture
def release():
    """Event that unblocks the fake hook functions (so no thread outlives a test)."""
    ev = threading.Event()
    yield ev
    ev.set()


@pytest.fixture
def mod(monkeypatch):
    import http_app

    monkeypatch.setattr(http_app, "authenticate", lambda scope: ("tester", ""))
    monkeypatch.setattr("tools.schema.get_db_status", lambda: dict(_RECOVERING), raising=False)

    def _probe_must_not_run_inline():
        raise AssertionError("request path must never probe the DB")

    monkeypatch.setattr("tools.schema.probe_db_status", _probe_must_not_run_inline, raising=False)
    yield http_app
    http_app._shutdown_hook_executor()


def _patch_hooks(monkeypatch, fn):
    import tools.hook_endpoints as hook_endpoints

    monkeypatch.setattr(hook_endpoints, "get_prompt_context", fn)
    monkeypatch.setattr(hook_endpoints, "get_auto_extract_context", fn)
    monkeypatch.setattr(hook_endpoints, "ingest_auto_extract", fn)


# ── /health contract ────────────────────────────────────────────────────


def test_health_reports_cached_db_status_and_stays_ok(mod):
    """Top-level status stays "ok" (entrypoint/compose/statusline gate on it);
    DB state rides in ``postgres`` from the cache, never an inline probe."""
    status, _, data = asyncio.run(_call(mod.app, "GET", "/health"))
    assert status == 200
    assert data["status"] == "ok"
    assert data["server"] == "jarvis-core"
    assert data["postgres"] == _RECOVERING


def test_health_error_text_is_sanitized(mod, monkeypatch):
    leaky = dict(_RECOVERING, error="connect to postgresql://jarvis:hunter2@db:5432/jarvis failed")
    monkeypatch.setattr("tools.schema.get_db_status", lambda: leaky, raising=False)
    _, _, data = asyncio.run(_call(mod.app, "GET", "/health"))
    assert "hunter2" not in json.dumps(data)
    assert data["postgres"]["status"] == "recovering"


def test_health_degrades_to_unknown_when_status_unavailable(mod, monkeypatch):
    def boom():
        raise RuntimeError("status cache broken")

    monkeypatch.setattr("tools.schema.get_db_status", boom, raising=False)
    status, _, data = asyncio.run(_call(mod.app, "GET", "/health"))
    assert status == 200
    assert data["status"] == "ok"
    assert data["postgres"]["status"] == "unknown"


# ── Blocking work stays off the event loop ──────────────────────────────


def test_health_answers_fast_while_hooks_block(mod, monkeypatch, release):
    """With more blocking hook requests in flight than hook workers, /health
    still answers in < 100ms and every hook gets a 503 before the 2.5s client
    deadline — the outage signature was /health waiting out 30-240s freezes."""

    def blocked(*args, **kwargs):
        release.wait(5)
        return {"success": True}

    _patch_hooks(monkeypatch, blocked)
    monkeypatch.setattr(mod, "_db_available", lambda: True)

    async def scenario():
        hooks = [
            asyncio.create_task(_timed(_call(mod.app, "POST", path, body)))
            for path, body in [
                ("/hook/auto-extract/ingest", {"observations": []}),
                ("/hook/auto-extract/ingest", {"observations": []}),
                ("/hook/prompt-context", {"prompt": "hello"}),
                ("/hook/prompt-context", {"prompt": "hello"}),
                ("/hook/auto-extract/context", {"workstream_limit": 5}),
                ("/hook/auto-extract/context", {"workstream_limit": 5}),
            ]
        ]
        await asyncio.sleep(0.2)  # executor now saturated (6 requests > 4 workers)
        health = []
        for _ in range(5):
            health.append(await _timed(_call(mod.app, "GET", "/health")))
            await asyncio.sleep(0.05)
        return health, await asyncio.gather(*hooks)

    health, hooks = asyncio.run(scenario())

    for elapsed, (status, _, data) in health:
        assert status == 200 and data["status"] == "ok"
        assert elapsed < 0.1, f"/health took {elapsed:.3f}s while hooks blocked"
    # The breaker is closed here, so the deadline miss is a slow request, not
    # a DB outage (see test_hook_deadline_classification.py for both cases).
    paths = ["ingest", "ingest", "prompt", "prompt", "context", "context"]
    for kind, (elapsed, (status, headers, data)) in zip(paths, hooks):
        if kind == "prompt":
            assert status == 200
            assert data["success"] is True and data["matches"] == []
            assert data["timed_out"] is True
        else:
            assert status == 503
            assert headers.get("retry-after") == "10"
            assert data == {
                "success": False,
                "retryable": True,
                "error_kind": "deadline",
                "error": "request timed out after 2s",
            }
        assert elapsed < 2.5, f"hook answered after {elapsed:.2f}s (client gives up at 2.5s)"


def test_hook_runs_in_worker_thread_with_request_context(mod, monkeypatch):
    """Hooks run on the hook executor (not the loop thread) and still see the
    request's contextvars (current_user drives per-user isolation)."""
    from jarvis_common.auth import current_user

    seen = {}

    def fake_prompt_context(prompt):
        seen["thread"] = threading.current_thread().name
        seen["user"] = current_user.get()
        return {"success": True, "matches": []}

    _patch_hooks(monkeypatch, fake_prompt_context)
    status, _, data = asyncio.run(_call(mod.app, "POST", "/hook/prompt-context", {"prompt": "hi"}))
    assert status == 200 and data["success"] is True
    assert seen["thread"].startswith("hook-db")
    assert seen["user"] == "tester"


def test_retryable_result_maps_to_503(mod, monkeypatch):
    """ingest's "not stored, try later" result must not look delivered: a 200
    would make the hook client drop the payload instead of re-queueing it."""
    result = {"success": False, "retryable": True, "error": "database unavailable: recovery mode"}
    _patch_hooks(monkeypatch, lambda *a, **k: dict(result))
    status, headers, data = asyncio.run(
        _call(mod.app, "POST", "/hook/auto-extract/ingest", {"observations": []})
    )
    assert status == 503
    assert headers.get("retry-after") == "10"
    assert data == result


def test_degraded_but_successful_result_stays_200(mod, monkeypatch):
    _patch_hooks(monkeypatch, lambda *a, **k: {"success": True, "matches": [], "degraded": True})
    status, _, data = asyncio.run(_call(mod.app, "POST", "/hook/prompt-context", {"prompt": "hi"}))
    assert status == 200
    assert data["degraded"] is True


@pytest.mark.parametrize("path, body", [
    ("/hook/prompt-context", {"prompt": "hi"}),
    ("/hook/auto-extract/context", {"workstream_limit": 5}),
    ("/hook/auto-extract/ingest", {"observations": []}),
])
def test_db_unavailable_exceptions_map_to_503(mod, monkeypatch, path, body):
    import psycopg_pool
    from tools.schema import DatabaseUnavailable

    for exc in (
        psycopg_pool.PoolTimeout("couldn't get a connection after 1.50 sec"),
        psycopg_pool.TooManyRequests("the pool has already 32 requests waiting"),
        DatabaseUnavailable("the database system is in recovery mode"),
    ):
        def raiser(*args, _exc=exc, **kwargs):
            raise _exc

        _patch_hooks(monkeypatch, raiser)
        status, headers, data = asyncio.run(_call(mod.app, "POST", path, body))
        assert status == 503, type(exc).__name__
        assert headers.get("retry-after") == "10"
        assert data["success"] is False and data["retryable"] is True
        assert data["error"].startswith("database unavailable: ")
        assert not data["error"].startswith("database unavailable: database unavailable")


def test_delivery_ack_db_unavailable_maps_to_503(mod, monkeypatch):
    import psycopg_pool
    import tools.retrieval_telemetry as telemetry

    def raiser(trace_id, payload):
        raise psycopg_pool.PoolTimeout("couldn't get a connection after 1.50 sec")

    monkeypatch.setattr(telemetry, "acknowledge_delivery", raiser)
    status, headers, data = asyncio.run(_call(
        mod.app, "PUT", "/telemetry/retrieval/11111111-1111-1111-1111-111111111111/delivery",
        {"delivered_count": 1},
    ))
    assert status == 503
    assert headers.get("retry-after") == "10"
    assert data["retryable"] is True


def test_unexpected_error_text_never_leaks_credentials(mod, monkeypatch):
    def raiser(*args, **kwargs):
        raise RuntimeError("connect to postgresql://jarvis:hunter2@db:5432/jarvis failed")

    _patch_hooks(monkeypatch, raiser)
    status, _, data = asyncio.run(_call(mod.app, "POST", "/hook/prompt-context", {"prompt": "hi"}))
    assert status == 500
    assert data["success"] is False
    assert "hunter2" not in json.dumps(data)


def test_telemetry_runs_off_loop_and_times_out_to_503(mod, monkeypatch, release):
    monkeypatch.setattr(mod, "_TELEMETRY_DEADLINE_SECONDS", 0.3)

    def blocked():
        release.wait(5)
        return {}

    monkeypatch.setattr(mod, "_collect_telemetry", blocked)

    async def scenario():
        tele = asyncio.create_task(_timed(_call(mod.app, "GET", "/telemetry")))
        await asyncio.sleep(0.05)
        health = await _timed(_call(mod.app, "GET", "/health"))
        return health, await tele

    (h_elapsed, (h_status, _, _)), (t_elapsed, (t_status, headers, _)) = asyncio.run(scenario())
    assert h_status == 200 and h_elapsed < 0.1
    assert t_status == 503 and headers.get("retry-after") == "10"
    assert t_elapsed < 1.5


# ── Request body handling ───────────────────────────────────────────────


def test_body_over_cap_returns_413_without_running_hook(mod, monkeypatch):
    called = []
    _patch_hooks(monkeypatch, lambda *a, **k: called.append(1) or {"success": True})
    chunk = b"x" * (256 * 1024)
    messages = [{"type": "http.request", "body": chunk, "more_body": True} for _ in range(4)]
    messages.append({"type": "http.request", "body": b"x", "more_body": False})
    status, _, data = asyncio.run(
        _call(mod.app, "POST", "/hook/auto-extract/ingest", messages=messages)
    )
    assert status == 413
    assert data["success"] is False
    assert called == []


def test_body_at_cap_is_accepted(mod, monkeypatch):
    _patch_hooks(monkeypatch, lambda *a, **k: {"success": True})
    prompt = "p" * (mod.MAX_REQUEST_BODY_BYTES - 100)
    status, _, _ = asyncio.run(_call(mod.app, "POST", "/hook/prompt-context", {"prompt": prompt}))
    assert status == 200


def test_disconnect_mid_body_returns_without_spinning(mod, monkeypatch):
    """uvicorn answers receive() with http.disconnect *without suspending* once
    the client is gone; looping on it pinned the loop at 100% CPU forever."""
    called = []
    _patch_hooks(monkeypatch, lambda *a, **k: called.append(1) or {"success": True})
    calls = {"n": 0}
    sent = []

    async def receive():
        calls["n"] += 1
        if calls["n"] > 50:
            raise AssertionError("receive() polled in a loop after http.disconnect")
        if calls["n"] == 1:
            return {"type": "http.request", "body": b'{"prompt":', "more_body": True}
        return {"type": "http.disconnect"}  # returned immediately, like uvicorn

    async def send(message):
        sent.append(message)

    async def scenario():
        await asyncio.wait_for(
            mod.app(_scope("POST", "/hook/prompt-context"), receive, send), 1.0
        )
        return await _call(mod.app, "GET", "/health")

    status, _, _ = asyncio.run(scenario())
    assert sent == [], "no response may be sent to a disconnected client"
    assert called == []
    assert calls["n"] == 2
    assert status == 200


def test_partial_body_disconnect_real_uvicorn_h11(mod):
    """Same regression through a real uvicorn (h11, as deployed): a short body
    then a hang-up must leave /health answering."""
    import uvicorn

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(
        mod.app, http="h11", lifespan="off", log_config=None, access_log=False,
    ))
    thread = threading.Thread(target=server.run, kwargs={"sockets": [sock]}, daemon=True)
    thread.start()
    try:
        deadline = time.monotonic() + 5
        while not server.started:
            assert time.monotonic() < deadline, "uvicorn did not start"
            time.sleep(0.02)

        client = socket.create_connection(("127.0.0.1", port), timeout=2)
        client.sendall(
            b"POST /hook/prompt-context HTTP/1.1\r\nHost: x\r\n"
            b"Content-Type: application/json\r\nContent-Length: 100\r\n\r\n"
            b'{"prompt":'
        )
        time.sleep(0.2)
        client.close()
        time.sleep(0.2)

        for _ in range(3):
            t0 = time.monotonic()
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as resp:
                assert resp.status == 200
                assert json.loads(resp.read())["status"] == "ok"
            assert time.monotonic() - t0 < 1.0
    finally:
        server.should_exit = True
        thread.join(timeout=5)
        sock.close()


# ── Access log redaction ────────────────────────────────────────────────


_SECRET_HEADERS = {
    "Authorization": "Bearer SECRET-bearer-token",
    "X-Jarvis-Internal-Token": "SECRET-internal-token",
    "Cookie": "session=SECRET-cookie",
    "Proxy-Authorization": "Basic SECRET-proxy",
    "X-Api-Key": "SECRET-api-key",
    "Mcp-Session-Id": "SECRET-session",
}


def test_access_log_only_logs_allowlisted_headers(mod, monkeypatch, caplog):
    _patch_hooks(monkeypatch, lambda *a, **k: {"success": True})
    caplog.set_level(logging.DEBUG, logger="jarvis-core")
    headers = {
        **_SECRET_HEADERS,
        "User-Agent": "resilience-test/1.0",
        "Content-Type": "application/json",
    }
    asyncio.run(_call(mod.app, "POST", "/hook/prompt-context", {"prompt": "hi"}, headers=headers))
    asyncio.run(_call(mod.app, "GET", "/health", headers=headers))

    assert "SECRET" not in caplog.text
    access = [r for r in caplog.records if r.getMessage().startswith("[ACCESS]")]
    hook_line = next(r for r in access if "/hook/prompt-context" in r.getMessage())
    assert hook_line.levelno == logging.INFO
    assert "resilience-test/1.0" in hook_line.getMessage()
    assert "application/json" in hook_line.getMessage()


def test_health_access_line_is_debug_only(mod, caplog):
    caplog.set_level(logging.DEBUG, logger="jarvis-core")
    asyncio.run(_call(mod.app, "GET", "/health"))
    health_lines = [
        r for r in caplog.records if r.getMessage().startswith("[ACCESS] GET /health")
    ]
    assert health_lines and all(r.levelno == logging.DEBUG for r in health_lines)


def test_sensitive_header_is_never_logged_even_if_allowlisted(mod, monkeypatch):
    monkeypatch.setattr(mod, "_ACCESS_LOG_HEADERS", mod._ACCESS_LOG_HEADERS | {"x-auth-token"})
    scope = _scope("GET", "/x", {"X-Auth-Token": "SECRET", "Accept": "*/*"})
    assert mod._loggable_headers(scope) == {"accept": "*/*"}


# ── Lifespan ────────────────────────────────────────────────────────────


def test_lifespan_runs_probe_loop_and_shuts_down_executors(monkeypatch):
    """The DB status probe is a registered background task (started in
    lifespan, cancelled on shutdown); both executors are shut down."""
    import server as server_mod

    monkeypatch.setattr("tools.embedding.warm_embedding_service", lambda: 0.0)
    monkeypatch.setattr("tools.schema.ensure_schema", lambda: None)
    monkeypatch.setattr("tools.schema.check_model_consistency", lambda: None)
    monkeypatch.setattr("tools.schema_registry.rebuild_registry", lambda: None)
    monkeypatch.setattr(server_mod, "DB_STATUS_PROBE_INTERVAL_SECONDS", 0.01)
    probes = []
    monkeypatch.setattr(
        "tools.schema.probe_db_status",
        lambda: probes.append(threading.current_thread().name) or dict(_RECOVERING),
        raising=False,
    )
    monkeypatch.setattr(
        server_mod, "get_background_tasks", lambda: [server_mod.db_status_probe_loop()]
    )
    import http_app

    mod = importlib.reload(http_app)
    sent = []

    async def drive():
        inbox = asyncio.Queue()

        async def receive():
            return await inbox.get()

        async def send(message):
            sent.append(message["type"])

        await inbox.put({"type": "lifespan.startup"})
        life = asyncio.create_task(mod.app({"type": "lifespan"}, receive, send))
        deadline = time.monotonic() + 3
        while len(probes) < 2:
            assert time.monotonic() < deadline, "probe loop never ran"
            await asyncio.sleep(0.01)
        mod._get_hook_executor()
        server_mod._get_tool_executor()
        await inbox.put({"type": "lifespan.shutdown"})
        await asyncio.wait_for(life, 3)
        count = len(probes)
        await asyncio.sleep(0.1)
        return count

    count_at_shutdown = asyncio.run(drive())
    assert sent == ["lifespan.startup.complete", "lifespan.shutdown.complete"]
    assert len(probes) == count_at_shutdown, "probe loop kept running after shutdown"
    assert all(name != threading.main_thread().name for name in probes)
    assert mod._hook_executor is None
    assert server_mod._tool_executor is None
