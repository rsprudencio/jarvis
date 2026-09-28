"""Tests for the HTTP transport wrapper (http_app.py).

Uses Starlette's TestClient which works with any ASGI callable,
including our raw ASGI app (not just Starlette apps).

These tests require the MCP Streamable HTTP SDK module, which is only
available in the Docker environment. They are skipped locally.
"""

import importlib

import pytest

from starlette.testclient import TestClient

try:
    import mcp.server.streamable_http_manager  # noqa: F401

    _HAS_STREAMABLE_HTTP = True
except Exception:
    _HAS_STREAMABLE_HTTP = False

pytestmark = pytest.mark.skipif(
    not _HAS_STREAMABLE_HTTP,
    reason="Streamable HTTP module only available in Docker environment",
)


@pytest.fixture
def client(monkeypatch):
    """Create a test client for the raw ASGI app.

    We reload the module per test because StreamableHTTPSessionManager
    can only run() once per instance. Embedding initialization is covered
    separately; HTTP transport tests must not load the production ONNX model.
    """
    monkeypatch.setattr(
        "tools.embedding.warm_embedding_service",
        lambda: 0.0,
    )
    monkeypatch.setattr("tools.schema.ensure_schema", lambda: None)
    monkeypatch.setattr("tools.schema.check_model_consistency", lambda: None)
    monkeypatch.setattr("tools.schema_registry.rebuild_registry", lambda: None)
    monkeypatch.setattr("server.get_background_tasks", lambda: [])
    import http_app as mod

    importlib.reload(mod)
    with TestClient(mod.app, raise_server_exceptions=False) as c:
        yield c


def test_app_creates_successfully():
    """The ASGI app should import without errors."""
    from http_app import app

    assert callable(app)


def _reloaded_app(monkeypatch, *, warm=lambda: 0.0, consistency=lambda: None):
    monkeypatch.setattr("tools.embedding.warm_embedding_service", warm)
    monkeypatch.setattr("tools.schema.ensure_schema", lambda: None)
    monkeypatch.setattr("tools.schema.check_model_consistency", consistency)
    monkeypatch.setattr("tools.schema_registry.rebuild_registry", lambda: None)
    monkeypatch.setattr("server.get_background_tasks", lambda: [])
    import http_app as mod

    importlib.reload(mod)
    return mod


def test_startup_serves_degraded_when_embedding_warmup_fails(monkeypatch):
    """A down model host must not zombie or kill the server at boot.

    Runtime retrieval fails open when the host service dies, so startup must
    behave the same: log loudly, complete startup, and serve MCP traffic.
    """
    def raise_warmup():
        raise ConnectionError("model host unreachable")

    mod = _reloaded_app(monkeypatch, warm=raise_warmup)
    with TestClient(mod.app, raise_server_exceptions=False) as c:
        assert c.get("/health").status_code == 200


def test_model_mismatch_aborts_startup_instead_of_serving(monkeypatch):
    """Mixed embedding spaces must abort startup via lifespan.startup.failed.

    Raising (even SystemExit) from the lifespan handler is swallowed by
    uvicorn's lifespan="auto" and leaves a zombie server; the ASGI
    startup.failed message is the only reliable abort signal.
    """
    from tools.schema import ModelMismatchError

    def raise_mismatch():
        raise ModelMismatchError("database has 'old' but config specifies 'new'")

    mod = _reloaded_app(monkeypatch, consistency=raise_mismatch)
    with pytest.raises(ModelMismatchError):
        with TestClient(mod.app, raise_server_exceptions=False):
            pass


def test_model_mismatch_emits_lifespan_startup_failed_message(monkeypatch):
    """The load-bearing part of the abort is the ASGI startup.failed message —
    uvicorn (lifespan="auto") ignores a bare raise from the handler and keeps
    serving. Drive the lifespan protocol directly and assert the message is
    sent; the raise-only variant would pass the TestClient test above but
    reintroduce the zombie."""
    import asyncio

    from tools.schema import ModelMismatchError

    def raise_mismatch():
        raise ModelMismatchError("database has 'old' but config specifies 'new'")

    mod = _reloaded_app(monkeypatch, consistency=raise_mismatch)

    sent = []
    inbox = [{"type": "lifespan.startup"}]

    async def receive():
        return inbox.pop(0)

    async def send(message):
        sent.append(message)

    async def drive():
        scope = {"type": "lifespan", "asgi": {"version": "3.0"}}
        with pytest.raises(ModelMismatchError):
            await mod.app(scope, receive, send)

    asyncio.run(drive())

    assert any(m.get("type") == "lifespan.startup.failed" for m in sent), (
        "handler raised without sending lifespan.startup.failed — uvicorn "
        "would keep serving as a zombie"
    )
    assert not any(m.get("type") == "lifespan.startup.complete" for m in sent)


def test_health_endpoint(client):
    """GET /health should return minimal liveness response — no DB, no secrets."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["server"] == "jarvis-core"
    assert "version" in data
    # The only DB detail is the background probe's cached verdict (never an
    # inline query); the statusline and launcher read postgres.status.
    assert set(data["postgres"]) >= {"status", "error", "checked_at", "free_bytes"}
    # Health must NOT contain operational details (those live on /telemetry)
    assert "sync" not in data
    assert "auth" not in data


def test_not_found(client):
    """Unknown paths should return 404."""
    response = client.get("/unknown")
    assert response.status_code == 404
    assert response.json()["error"] == "Not found"


def test_mcp_endpoint_accepts_post(client):
    """POST /mcp should accept JSON-RPC requests (initialize handshake)."""
    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2025-03-26",
            "capabilities": {},
            "clientInfo": {"name": "test", "version": "0.1"},
        },
    }
    response = client.post(
        "/mcp",
        json=payload,
        headers={"Accept": "application/json, text/event-stream"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["jsonrpc"] == "2.0"
    assert data["result"]["serverInfo"]["name"] == "core"


def test_mcp_no_trailing_slash_redirect(client):
    """POST /mcp should NOT redirect to /mcp/ (the raw ASGI fix)."""
    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2025-03-26",
            "capabilities": {},
            "clientInfo": {"name": "test", "version": "0.1"},
        },
    }
    response = client.post(
        "/mcp",
        json=payload,
        headers={"Accept": "application/json, text/event-stream"},
        follow_redirects=False,
    )
    # Should be 200, NOT 307 (trailing slash redirect)
    assert response.status_code == 200


def test_hook_prompt_context_success(client, monkeypatch):
    """POST /hook/prompt-context returns endpoint payload."""
    import tools.hook_endpoints as hook_endpoints

    monkeypatch.setattr(
        hook_endpoints,
        "get_prompt_context",
        lambda prompt: {
            "success": True,
            "enabled": True,
            "debug": False,
            "matches": [{"id": "notes/a.md", "relevance": 0.8}],
            "query_ms": 5,
            "total_searched": 10,
            "budget_used": {"local": 10, "vault": 20, "remote": 0, "total": 8000},
            "todoist_prompt_alerts": {"enabled": False, "max_per_category": 3},
        },
    )
    response = client.post("/hook/prompt-context", json={"prompt": "hello"})
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["matches"][0]["id"] == "notes/a.md"


def test_hook_prompt_context_malformed_body(client):
    """Malformed JSON body returns 400."""
    response = client.post(
        "/hook/prompt-context",
        content="{not-json",
        headers={"content-type": "application/json"},
    )
    assert response.status_code == 400
    assert response.json()["success"] is False


def test_retrieval_delivery_ack(client, monkeypatch):
    """Hook delivery acknowledgement is additive and best-effort."""
    import tools.retrieval_telemetry as telemetry

    seen = {}

    def fake_ack(trace_id, payload):
        seen.update({"trace_id": trace_id, "payload": payload})
        return True

    monkeypatch.setattr(telemetry, "acknowledge_delivery", fake_ack)
    response = client.put(
        "/telemetry/retrieval/11111111-1111-1111-1111-111111111111/delivery",
        json={"delivered_count": 1, "delivered_candidate_keys": ["abc"]},
    )
    assert response.status_code == 200
    assert response.json()["success"] is True
    assert seen["trace_id"] == "11111111-1111-1111-1111-111111111111"
    assert seen["payload"]["delivered_candidate_keys"] == ["abc"]


def test_hook_auto_extract_context_success(client, monkeypatch):
    """POST /hook/auto-extract/context returns config + workstreams."""
    import tools.hook_endpoints as hook_endpoints

    monkeypatch.setattr(
        hook_endpoints,
        "get_auto_extract_context",
        lambda workstream_limit: {
            "success": True,
            "auto_extract": {
                "mode": "background",
                "min_turn_chars": 200,
                "max_transcript_lines": 500,
                "max_observations": 3,
                "dedup_threshold": 0.95,
                "debug": False,
            },
            "worklog": {"enabled": True, "dedup_threshold": 0.7},
            "known_workstreams": ["Jarvis Plugin"],
        },
    )
    response = client.post("/hook/auto-extract/context", json={"workstream_limit": 15})
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert "Jarvis Plugin" in data["known_workstreams"]


def test_hook_auto_extract_ingest_statuses(client, monkeypatch):
    """POST /hook/auto-extract/ingest returns stored/duplicate/error statuses."""
    import tools.hook_endpoints as hook_endpoints

    monkeypatch.setattr(
        hook_endpoints,
        "ingest_auto_extract",
        lambda payload: {
            "success": True,
            "observations": [
                {"status": "stored", "id": "obs::1", "error": ""},
                {"status": "duplicate", "id": "", "error": ""},
                {"status": "error", "id": "", "error": "write failed"},
            ],
            "worklog": {"status": "stored", "id": "worklog::1", "error": ""},
        },
    )

    response = client.post(
        "/hook/auto-extract/ingest",
        json={"observations": [], "worklog": None, "context": {}, "dedup": {}},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["observations"][0]["status"] == "stored"
    assert data["observations"][1]["status"] == "duplicate"
    assert data["observations"][2]["status"] == "error"


def test_hook_auto_extract_ingest_bad_shape(client):
    """Invalid payload shape returns 400."""
    response = client.post(
        "/hook/auto-extract/ingest",
        json={"observations": "wrong-shape"},
    )
    assert response.status_code == 400
    assert response.json()["success"] is False
