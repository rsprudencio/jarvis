"""Explorer DB calls are bounded when PostgreSQL freezes (review F2).

A Postgres that accepts connections but never answers blocks the request's
thread with no timeout of its own: statement_timeout is enforced by the very
server that froze, and the pool's check_connection waits forever on a pooled
connection. The request used to hang past the browser's patience; now it is
answered 504 at the deadline while the thread is abandoned.
"""

from __future__ import annotations

import asyncio
import json
import sys
import threading
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
_SEARCH = {"source": "local", "mode": "text", "query": "", "page_size": 100}


@pytest.fixture
def frozen(tmp_path, monkeypatch):
    """Every DB call blocks until released; the deadline is 0.2s."""
    monkeypatch.setenv("JARVIS_HOME", str(tmp_path))
    (tmp_path / "config.json").write_text(json.dumps({"memory": {}}))
    from jarvis_common.config import clear_config_cache
    clear_config_cache()
    release = threading.Event()

    def hang(*args, **kwargs):
        release.wait(5)
        return None

    monkeypatch.setattr(app_module, "_DB_CALL_DEADLINE", 0.2)
    monkeypatch.setattr(app_module, "_local_pool", MagicMock())
    monkeypatch.setattr(app_module, "_sources", {"local": _LOCAL})
    monkeypatch.setattr(app_module, "_sources_at", None)
    for name in ("_search_sync", "_fetch_content_sync", "_count_sync", "_admin_sync"):
        monkeypatch.setattr(app_module, name, hang)
    import tools.retrieval_telemetry as telemetry

    monkeypatch.setattr(telemetry, "get_summary", hang)
    yield
    release.set()
    clear_config_cache()


def _run(coro_fn):
    async def main():
        transport = httpx.ASGITransport(app=app_module.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://t", timeout=10) as c:
            return await coro_fn(c)

    return asyncio.run(main())


async def _timed(coro):
    t0 = time.perf_counter()
    r = await coro
    return r, time.perf_counter() - t0


def test_frozen_db_requests_answer_504_at_the_deadline(frozen):
    async def scenario(c):
        return await asyncio.gather(
            _timed(c.post("/api/search", json=_SEARCH)),
            _timed(c.get("/api/content?source=local&id=obs::1")),
            _timed(c.get("/api/admin")),
            _timed(c.get("/api/retrieval/summary")),
        )

    for r, elapsed in _run(scenario):
        assert r.status_code == 504, r.text
        assert r.json()["detail"] == "query timed out"
        assert elapsed < 2.0


def test_frozen_db_stats_report_per_source_timeout(frozen):
    r = _run(lambda c: c.get("/api/stats"))
    assert r.status_code == 200
    assert r.json() == {"local": {"count": None, "error": "database did not answer within 0.2s"}}


def test_health_answers_while_db_calls_hang(frozen):
    async def scenario(c):
        hanging = asyncio.create_task(c.post("/api/search", json=_SEARCH))
        await asyncio.sleep(0.05)
        health, elapsed = await _timed(c.get("/health"))
        await hanging
        return health, elapsed

    health, elapsed = _run(scenario)
    assert health.status_code == 200
    assert elapsed < 0.1


def test_healthy_calls_are_not_affected(tmp_path, monkeypatch):
    monkeypatch.setattr(app_module, "_sources", {"local": _LOCAL})
    monkeypatch.setattr(app_module, "_sources_at", None)
    monkeypatch.setattr(app_module, "_fetch_content_sync", lambda src, item_id: {"id": item_id})
    r = _run(lambda c: c.get("/api/content?source=local&id=obs::1"))
    assert r.status_code == 200
    assert r.json()["id"] == "obs::1"
