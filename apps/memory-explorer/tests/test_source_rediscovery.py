"""Lazy source re-discovery.

An Explorer started while Postgres was down probed nothing, cached an empty
source list forever, and answered 400 "Unknown source: 'local'" until it was
restarted. Discovery now re-runs (in a thread, one run at a time) when 'local'
is missing or a requested source is unknown, and refreshes a stale cache in
the background without delaying the request.
"""

from __future__ import annotations

import asyncio
import json
import socket
import sys
import time
from pathlib import Path
from unittest.mock import MagicMock

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import app as app_module  # noqa: E402

_LOCAL = {
    "id": "local", "label": "Local Memories", "type": "local",
    "schema": "local", "table": "memories", "has_retrieval_count": True,
    "has_status": True, "capabilities": ["text", "metadata"],
    "metadata_filters": ["category", "scope", "project", "source", "status"],
}
_OBSIDIAN = {
    "id": "obsidian", "label": "Obsidian Vault", "type": "local",
    "schema": "obsidian", "table": "documents", "has_retrieval_count": False,
    "has_status": False, "capabilities": ["text", "metadata"],
    "metadata_filters": ["vault_type", "directory"],
}
_SEARCH = {"source": "local", "mode": "text", "query": "", "page_size": 100}


def _closed_port_dsn() -> str:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return f"postgresql://jarvis:pw@127.0.0.1:{port}/jarvis"


def _mock_pool() -> MagicMock:
    """A healthy pool: every query returns COUNT 0 and no rows."""
    pool = MagicMock()
    conn = MagicMock()
    cur = MagicMock()
    cur.fetchone.return_value = (0,)
    cur.fetchall.return_value = []
    conn.execute = MagicMock(return_value=cur)
    pool.connection.return_value.__enter__ = MagicMock(return_value=conn)
    pool.connection.return_value.__exit__ = MagicMock(return_value=False)
    return pool


@pytest.fixture(autouse=True)
def _isolate(tmp_path, monkeypatch):
    monkeypatch.setenv("JARVIS_HOME", str(tmp_path))
    monkeypatch.setenv("POSTGRES_URL", _closed_port_dsn())
    (tmp_path / "config.json").write_text(json.dumps({"memory": {}}))
    from jarvis_common.config import clear_config_cache
    clear_config_cache()
    monkeypatch.setattr(app_module, "_local_pool", _mock_pool())
    monkeypatch.setattr(app_module, "_sources_lock", asyncio.Lock())
    monkeypatch.setattr(app_module, "_discovery_runs", 0)
    monkeypatch.setattr(app_module, "_refresh_task", None)
    monkeypatch.setattr(app_module, "_forced_discovery_at", None)
    monkeypatch.setattr(app_module, "_db_status", {
        "status": "unknown", "error": None, "checked_at": None, "free_bytes": None})
    yield
    clear_config_cache()


def _started(monkeypatch, sources: dict, age: float) -> None:
    """State after lifespan discovered `sources` `age` seconds ago."""
    monkeypatch.setattr(app_module, "_sources", dict(sources))
    monkeypatch.setattr(app_module, "_sources_at", time.monotonic() - age)


def _discover_returning(monkeypatch, *results: dict, delay: float = 0.0) -> list:
    """Replace discovery; successive calls return successive results."""
    calls = []

    def fake(pool):
        calls.append(pool)
        if delay:
            time.sleep(delay)
        return dict(results[min(len(calls), len(results)) - 1])

    monkeypatch.setattr(app_module, "_discover_sources", fake)
    return calls


def _run(coro_fn):
    async def main():
        transport = httpx.ASGITransport(app=app_module.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://t", timeout=10) as c:
            return await coro_fn(c)

    return asyncio.run(main())


def test_started_while_db_down_recovers_without_restart(monkeypatch):
    _started(monkeypatch, {}, age=1)  # PG was down at startup: nothing found
    calls = _discover_returning(monkeypatch, {"local": _LOCAL, "obsidian": _OBSIDIAN})

    async def scenario(c):
        search = await c.post("/api/search", json=_SEARCH)
        sources = await c.get("/api/sources")
        return search, sources

    search, sources = _run(scenario)
    assert search.status_code == 200, search.text
    assert search.json()["total"] == 0
    assert [s["id"] for s in sources.json()] == ["local", "obsidian"]
    assert len(calls) == 1  # fresh result reused, not re-probed per request


def test_unknown_source_while_db_down_is_503_with_cause(monkeypatch):
    _started(monkeypatch, {}, age=1)
    calls = _discover_returning(monkeypatch, {})
    monkeypatch.setattr(app_module, "_probe_db_status", lambda: {
        "status": "recovering", "error": "the database system is in recovery mode",
        "checked_at": time.time(), "free_bytes": None})

    r = _run(lambda c: c.post("/api/search", json=_SEARCH))
    assert r.status_code == 503
    assert r.headers["retry-after"] == "10"
    assert r.json()["detail"] == "Database unavailable: the database system is in recovery mode"
    assert len(calls) == 1


def test_unknown_source_on_healthy_db_is_still_400(monkeypatch):
    _started(monkeypatch, {"local": _LOCAL}, age=1)
    calls = _discover_returning(monkeypatch, {"local": _LOCAL})

    r = _run(lambda c: c.post("/api/search", json=dict(_SEARCH, source="nope")))
    assert r.status_code == 400
    assert "Unknown source" in r.json()["detail"]
    # "nope" is no id discovery could ever produce: answered without one
    # (test_forced_rediscovery_limits.py covers ids that could appear).
    assert calls == []


def test_unknown_content_and_delete_sources_rediscover(monkeypatch):
    _started(monkeypatch, {}, age=1)
    calls = _discover_returning(monkeypatch, {"local": _LOCAL})
    monkeypatch.setattr(app_module, "_fetch_content_sync", lambda src, item_id: None)
    monkeypatch.setattr(app_module, "_delete_sync", lambda item_id: {"deleted": True, "id": item_id})

    async def scenario(c):
        content = await c.get("/api/content?source=local&id=obs::1")
        delete = await c.delete("/api/memories/obs::1?source=local")
        return content, delete

    content, delete = _run(scenario)
    assert content.status_code == 404  # the source resolved; the row did not
    assert content.json()["detail"] == "Item not found"
    assert delete.status_code == 200, delete.text
    assert len(calls) == 1


def test_concurrent_requests_share_one_discovery(monkeypatch):
    _started(monkeypatch, {}, age=1)
    calls = _discover_returning(monkeypatch, {"local": _LOCAL}, delay=0.3)

    async def scenario(c):
        return await asyncio.gather(*(c.get("/api/sources") for _ in range(5)))

    responses = _run(scenario)
    assert all([s["id"] for s in r.json()] == ["local"] for r in responses)
    assert len(calls) == 1


def test_stale_cache_refreshes_in_background(monkeypatch):
    """The healthy search path never waits for a refresh."""
    _started(monkeypatch, {"local": _LOCAL}, age=120)
    calls = _discover_returning(monkeypatch, {"local": _LOCAL, "obsidian": _OBSIDIAN}, delay=1.0)

    async def scenario(c):
        t = time.perf_counter()
        r = await c.post("/api/search", json=_SEARCH)
        elapsed = time.perf_counter() - t
        await app_module._refresh_task
        return r, elapsed

    r, elapsed = _run(scenario)
    assert r.status_code == 200
    assert elapsed < 0.5
    assert len(calls) == 1
    assert set(app_module._sources) == {"local", "obsidian"}


def test_refresh_during_outage_keeps_last_good_sources(monkeypatch):
    _started(monkeypatch, {"local": _LOCAL, "obsidian": _OBSIDIAN}, age=120)
    calls = _discover_returning(monkeypatch, {})

    async def scenario(c):
        first = await c.get("/api/sources")
        await app_module._refresh_task
        second = await c.get("/api/sources")
        return first, second

    first, second = _run(scenario)
    assert [s["id"] for s in second.json()] == ["local", "obsidian"]
    assert len(calls) == 1  # the failed attempt still resets the 60s clock


def test_no_rediscovery_before_lifespan(monkeypatch):
    """Import-time/test use without lifespan keeps the old static behavior."""
    monkeypatch.setattr(app_module, "_sources", {})
    monkeypatch.setattr(app_module, "_sources_at", None)
    calls = _discover_returning(monkeypatch, {"local": _LOCAL})

    r = _run(lambda c: c.post("/api/search", json=_SEARCH))
    assert r.status_code == 400
    assert calls == []


def test_discovery_probes_use_a_short_checkout_timeout():
    pool = _mock_pool()
    found = app_module._discover_sources(pool)
    assert set(found) == {"local", "obsidian"}
    timeouts = [c.kwargs.get("timeout") for c in pool.connection.call_args_list]
    assert timeouts and all(t == 3.0 for t in timeouts)
