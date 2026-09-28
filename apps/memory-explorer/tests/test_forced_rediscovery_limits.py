"""Forced source re-discovery is bounded (review SEC-2).

Any request naming an unknown source used to run a full discovery — local
probes plus a connection to every configured remote (30s pool waits when one
is down) — and any unauthenticated GET, even a cross-site one against
127.0.0.1:8750, could ask for it. Now only ids discovery could produce force
a run, at most once per interval; everything else is answered immediately.
"""

from __future__ import annotations

import asyncio
import json
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
_TEAM = {
    "id": "remote:team", "label": "Remote: team", "type": "remote",
    "remote_name": "team", "schema": "team", "available": True,
    "capabilities": ["text", "metadata"],
    "metadata_filters": ["category", "scope", "project", "source", "status"],
}
_REMOTES = {
    "team": {"enabled": True},
    "old": {"enabled": False},
    "_template": {"enabled": True},
}


@pytest.fixture(autouse=True)
def _isolate(tmp_path, monkeypatch):
    monkeypatch.setenv("JARVIS_HOME", str(tmp_path))
    (tmp_path / "config.json").write_text(json.dumps({"memory": {}}))
    from jarvis_common.config import clear_config_cache
    clear_config_cache()
    monkeypatch.setattr(app_module, "_local_pool", MagicMock())
    monkeypatch.setattr(app_module, "_sources_lock", asyncio.Lock())
    monkeypatch.setattr(app_module, "_discovery_runs", 0)
    monkeypatch.setattr(app_module, "_refresh_task", None)
    monkeypatch.setattr(app_module, "_forced_discovery_at", None)
    monkeypatch.setattr(app_module, "_db_status", {
        "status": "unknown", "error": None, "checked_at": None, "free_bytes": None})
    monkeypatch.setattr(app_module, "get_sync_config", lambda: {"remotes": dict(_REMOTES)})
    monkeypatch.setattr(app_module, "_fetch_content_sync", lambda src, item_id: None)
    yield
    clear_config_cache()


def _started(monkeypatch, sources: dict) -> None:
    monkeypatch.setattr(app_module, "_sources", dict(sources))
    monkeypatch.setattr(app_module, "_sources_at", time.monotonic() - 1)


def _discover_returning(monkeypatch, result: dict) -> list:
    calls = []

    def fake(pool):
        calls.append(pool)
        return dict(result)

    monkeypatch.setattr(app_module, "_discover_sources", fake)
    return calls


def _run(coro_fn):
    async def main():
        transport = httpx.ASGITransport(app=app_module.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://t", timeout=10) as c:
            return await coro_fn(c)

    return asyncio.run(main())


def test_bogus_sources_never_trigger_discovery(monkeypatch):
    """The reviewer's driver: 10 cross-site-style GETs with made-up sources."""
    _started(monkeypatch, {"local": _LOCAL})
    calls = _discover_returning(monkeypatch, {"local": _LOCAL})

    async def scenario(c):
        return [await c.get(f"/api/content?source=bogus{i}&id=x") for i in range(10)]

    responses = _run(scenario)
    assert [r.status_code for r in responses] == [404] * 10
    assert calls == []


@pytest.mark.parametrize("source", ["remote:old", "remote:_template", "remote:unconfigured"])
def test_remotes_that_discovery_would_skip_do_not_trigger_it(monkeypatch, source):
    _started(monkeypatch, {"local": _LOCAL})
    calls = _discover_returning(monkeypatch, {"local": _LOCAL})
    r = _run(lambda c: c.get(f"/api/content?source={source}&id=x"))
    assert r.status_code == 404
    assert calls == []


def test_newly_configured_remote_is_discovered_once_per_interval(monkeypatch):
    _started(monkeypatch, {"local": _LOCAL})
    calls = _discover_returning(monkeypatch, {"local": _LOCAL})  # remote still unreachable

    async def scenario(c):
        return [await c.get("/api/content?source=remote:team&id=x") for _ in range(5)]

    responses = _run(scenario)
    assert [r.status_code for r in responses] == [404] * 5
    assert len(calls) == 1  # rate-limited, not one discovery per request

    # After the interval the next request may try again — and finds it.
    monkeypatch.setattr(
        app_module, "_forced_discovery_at",
        time.monotonic() - app_module._FORCED_DISCOVERY_INTERVAL - 1,
    )
    calls = _discover_returning(monkeypatch, {"local": _LOCAL, "remote:team": _TEAM})
    r = _run(lambda c: c.get("/api/content?source=remote:team&id=x"))
    assert r.json()["detail"] == "Item not found"  # the source resolved
    assert len(calls) == 1


def test_missing_local_forces_at_most_one_discovery_per_interval(monkeypatch):
    _started(monkeypatch, {})
    calls = _discover_returning(monkeypatch, {})
    monkeypatch.setattr(app_module, "_probe_db_status", lambda: {
        "status": "recovering", "error": "the database system is in recovery mode",
        "checked_at": time.time(), "free_bytes": None})

    async def scenario(c):
        return [await c.get("/api/content?source=local&id=x") for _ in range(6)]

    responses = _run(scenario)
    assert [r.status_code for r in responses] == [503] * 6
    assert len(calls) == 1


def test_unknown_source_error_uses_the_fresh_cached_status(monkeypatch):
    """No direct PostgreSQL connect per request while 'local' is missing."""
    _started(monkeypatch, {})
    _discover_returning(monkeypatch, {})
    monkeypatch.setattr(app_module, "_db_status", {
        "status": "disk_full", "error": "disk full (No space left on device)",
        "checked_at": time.time(), "free_bytes": 0})

    def must_not_probe():
        raise AssertionError("fresh cached status must be used")

    monkeypatch.setattr(app_module, "_probe_db_status", must_not_probe)
    r = _run(lambda c: c.get("/api/content?source=local&id=x"))
    assert r.status_code == 503
    assert r.json()["detail"] == "Database unavailable: disk full (No space left on device)"


def test_stale_cached_status_is_reprobed(monkeypatch):
    _started(monkeypatch, {})
    _discover_returning(monkeypatch, {})
    monkeypatch.setattr(app_module, "_db_status", {
        "status": "ok", "error": None, "checked_at": time.time() - 3600, "free_bytes": None})
    probes = []

    def probe():
        probes.append(1)
        return {"status": "unreachable", "error": "connection refused",
                "checked_at": time.time(), "free_bytes": None}

    monkeypatch.setattr(app_module, "_probe_db_status", probe)
    r = _run(lambda c: c.get("/api/content?source=local&id=x"))
    assert r.status_code == 503
    assert probes == [1]
