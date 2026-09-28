"""MCP tool calls are bounded when PostgreSQL freezes (review F2).

A PostgreSQL that accepts connections but never answers blocks a tool's
thread with no timeout of its own (the pool's check_connection and every
query wait forever). Calls hung for the whole freeze, and once all four tool
workers were stuck every later call queued behind them — even with the
breaker open. Now each call has a deadline, and while the breaker is open a
call that could only queue behind stuck workers is refused at once.
"""

from __future__ import annotations

import asyncio
import json
import threading
import time

import pytest

import server


@pytest.fixture
def release():
    ev = threading.Event()
    yield ev
    ev.set()


@pytest.fixture(autouse=True)
def _fresh_tool_executor():
    yield
    server.shutdown_tool_executor()


def _payload(result) -> dict:
    return json.loads(result[0].text)


def _blocking(release):
    def handler(args):
        release.wait(5)
        return {"success": True}

    return handler


def _wait_for_idle(timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while server._tool_calls_running and time.monotonic() < deadline:
        time.sleep(0.01)


def test_tool_call_answers_at_its_deadline(monkeypatch, release):
    monkeypatch.setattr(server, "TOOL_DEADLINE_SECONDS", 0.2)
    monkeypatch.setattr(server, "db_available", lambda: True)
    monkeypatch.setitem(server._HANDLERS, "jarvis_retrieve", _blocking(release))

    t0 = time.perf_counter()
    payload = _payload(asyncio.run(server.call_tool("jarvis_retrieve", {"id": "obs::1"})))
    elapsed = time.perf_counter() - t0

    assert elapsed < 1.0
    assert payload["success"] is False
    assert payload["retryable"] is True
    assert payload["error_kind"] == "deadline"
    assert "jarvis_retrieve did not finish within 0.2s" in payload["error"]
    release.set()
    _wait_for_idle()
    assert server._tool_calls_running == {}


def test_full_reindex_has_no_deadline(monkeypatch):
    monkeypatch.setattr(server, "TOOL_DEADLINE_SECONDS", 0.1)

    def slow_reindex(args):
        time.sleep(0.4)
        return {"success": True, "indexed": 3}

    monkeypatch.setitem(server._HANDLERS, "jarvis_index_vault", slow_reindex)
    payload = _payload(asyncio.run(server.call_tool("jarvis_index_vault", {})))
    assert payload == {"success": True, "indexed": 3}


def test_handler_timeout_error_is_not_mistaken_for_the_deadline(monkeypatch):
    def raises_timeout(args):
        raise TimeoutError("model host timed out")

    monkeypatch.setitem(server._HANDLERS, "jarvis_retrieve", raises_timeout)
    payload = _payload(asyncio.run(server.call_tool("jarvis_retrieve", {})))
    assert payload == {"success": False, "error": "model host timed out"}


def test_wedged_workers_with_breaker_open_fail_fast(monkeypatch, release):
    """The reviewer's case: 4 calls stuck on a frozen DB, breaker open."""
    monkeypatch.setattr(server, "TOOL_DEADLINE_SECONDS", 5.0)
    monkeypatch.setattr(server, "_WEDGED_AFTER_SECONDS", 0.05)
    monkeypatch.setitem(server._HANDLERS, "jarvis_retrieve", _blocking(release))
    monkeypatch.setattr(server, "db_available", lambda: False)
    monkeypatch.setattr(server, "db_unavailable_reason", lambda: "PostgreSQL did not answer")

    async def scenario():
        stuck = [
            asyncio.create_task(server.call_tool("jarvis_retrieve", {"id": f"obs::{i}"}))
            for i in range(server._TOOL_WORKERS)
        ]
        await asyncio.sleep(0.1)
        t0 = time.perf_counter()
        refused = await server.call_tool("jarvis_store", {"type": "learning", "content": "x"})
        elapsed = time.perf_counter() - t0
        release.set()
        await asyncio.gather(*stuck)
        return refused, elapsed

    refused, elapsed = asyncio.run(scenario())
    payload = _payload(refused)
    assert elapsed < 0.1
    assert payload["success"] is False
    assert payload["retryable"] is True
    assert payload["error_kind"] == "db_unavailable"
    assert payload["error"].startswith("database unavailable: PostgreSQL did not answer")
    _wait_for_idle()
    assert server._tool_calls_running == {}


def test_short_burst_during_outage_is_not_refused(monkeypatch, release):
    """Workers busy only briefly (failing fast, or file fallbacks) are not wedged."""
    monkeypatch.setattr(server, "_WEDGED_AFTER_SECONDS", 30.0)
    monkeypatch.setitem(server._HANDLERS, "jarvis_retrieve", _blocking(release))
    monkeypatch.setitem(server._HANDLERS, "jarvis_collection_stats", lambda args: {"success": True})
    monkeypatch.setattr(server, "db_available", lambda: False)

    async def scenario():
        busy = [
            asyncio.create_task(server.call_tool("jarvis_retrieve", {}))
            for _ in range(server._TOOL_WORKERS)
        ]
        await asyncio.sleep(0.1)
        queued = asyncio.create_task(server.call_tool("jarvis_collection_stats", {}))
        await asyncio.sleep(0.05)
        release.set()
        await asyncio.gather(*busy)
        return await asyncio.wait_for(queued, 2)

    assert _payload(asyncio.run(scenario())) == {"success": True}


def test_busy_workers_with_healthy_db_just_queue(monkeypatch, release):
    """A long reindex holding workers is contention, not an outage."""
    monkeypatch.setitem(server._HANDLERS, "jarvis_retrieve", _blocking(release))
    monkeypatch.setitem(server._HANDLERS, "jarvis_collection_stats", lambda args: {"success": True})
    monkeypatch.setattr(server, "db_available", lambda: True)

    async def scenario():
        busy = [
            asyncio.create_task(server.call_tool("jarvis_retrieve", {}))
            for _ in range(server._TOOL_WORKERS)
        ]
        await asyncio.sleep(0.1)
        queued = asyncio.create_task(server.call_tool("jarvis_collection_stats", {}))
        await asyncio.sleep(0.1)
        assert not queued.done()
        release.set()
        await asyncio.gather(*busy)
        return await asyncio.wait_for(queued, 2)

    assert _payload(asyncio.run(scenario())) == {"success": True}


def test_non_db_tools_are_never_refused_for_a_db_outage(monkeypatch, release):
    monkeypatch.setattr(server, "_WEDGED_AFTER_SECONDS", 0.0)
    monkeypatch.setitem(server._HANDLERS, "jarvis_retrieve", _blocking(release))
    monkeypatch.setitem(
        server._HANDLERS, "jarvis_read_vault_file", lambda args: {"success": True, "content": "hi"}
    )
    monkeypatch.setattr(server, "db_available", lambda: False)

    async def scenario():
        busy = [
            asyncio.create_task(server.call_tool("jarvis_retrieve", {}))
            for _ in range(server._TOOL_WORKERS)
        ]
        await asyncio.sleep(0.1)
        read = asyncio.create_task(server.call_tool("jarvis_read_vault_file", {"relative_path": "a.md"}))
        await asyncio.sleep(0.05)
        release.set()
        await asyncio.gather(*busy)
        return await asyncio.wait_for(read, 2)

    assert _payload(asyncio.run(scenario())) == {"success": True, "content": "hi"}


def test_in_flight_count_survives_cancelled_queued_calls(monkeypatch, release):
    """A queued call cancelled at its deadline never runs; nothing leaks."""
    monkeypatch.setattr(server, "TOOL_DEADLINE_SECONDS", 0.2)
    monkeypatch.setattr(server, "db_available", lambda: True)
    monkeypatch.setitem(server._HANDLERS, "jarvis_retrieve", _blocking(release))

    async def scenario():
        return await asyncio.gather(*(
            server.call_tool("jarvis_retrieve", {}) for _ in range(server._TOOL_WORKERS + 3)
        ))

    results = [_payload(r) for r in asyncio.run(scenario())]
    assert all(r["error_kind"] == "deadline" for r in results)
    release.set()
    _wait_for_idle()
    assert server._tool_calls_running == {}
