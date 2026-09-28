"""server.py must never run blocking work on the event loop.

A sync tool handler (DB / embedding / filesystem) or list_tools' obsidian
health check executed inline freezes /health and every hook for the call's
duration — the same bug class that turned the 2026-09-24 Postgres outage
into 30-240s server-wide freezes.
"""

import asyncio
import json
import logging
import threading
import time

import pytest

import server


@pytest.fixture
def release():
    """Unblocks the fake handlers so no worker thread outlives a test."""
    ev = threading.Event()
    yield ev
    ev.set()


@pytest.fixture(autouse=True)
def _fresh_tool_executor():
    yield
    server.shutdown_tool_executor()


async def _loop_lag(duration: float = 0.05) -> float:
    """How late a short sleep wakes up — ~0 when the loop is free."""
    t0 = time.perf_counter()
    await asyncio.sleep(duration)
    return time.perf_counter() - t0 - duration


def _payload(result) -> dict:
    return json.loads(result[0].text)


def test_call_tool_runs_sync_handler_off_the_loop(monkeypatch, release):
    seen = {}

    def blocking_handler(args):
        seen["thread"] = threading.current_thread().name
        release.wait(5)
        return {"success": True, "stats": 1}

    monkeypatch.setitem(server._HANDLERS, "jarvis_collection_stats", blocking_handler)

    async def scenario():
        task = asyncio.create_task(server.call_tool("jarvis_collection_stats", {}))
        lag = await _loop_lag()
        assert not task.done()
        release.set()
        return lag, await asyncio.wait_for(task, 2)

    lag, result = asyncio.run(scenario())
    assert lag < 0.1, f"event loop stalled {lag:.3f}s behind a sync tool handler"
    assert _payload(result) == {"success": True, "stats": 1}
    assert seen["thread"].startswith("mcp-tool")


def test_call_tool_still_awaits_async_handlers(monkeypatch):
    async def coro():
        await asyncio.sleep(0)
        return {"success": True, "async": True}

    monkeypatch.setitem(server._HANDLERS, "jarvis_collection_stats", lambda args: coro())
    result = asyncio.run(server.call_tool("jarvis_collection_stats", {}))
    assert _payload(result) == {"success": True, "async": True}


def test_call_tool_propagates_request_context(monkeypatch):
    """current_user drives per-user isolation; it must survive the thread hop."""
    from jarvis_common.auth import current_user

    monkeypatch.setitem(
        server._HANDLERS, "jarvis_collection_stats", lambda args: {"user": current_user.get()}
    )

    async def scenario():
        token = current_user.set("alice")
        try:
            return await server.call_tool("jarvis_collection_stats", {})
        finally:
            current_user.reset(token)

    assert _payload(asyncio.run(scenario())) == {"user": "alice"}


def test_call_tool_logs_argument_keys_never_values(monkeypatch, caplog):
    monkeypatch.setitem(server._HANDLERS, "jarvis_store", lambda args: {"success": True})
    caplog.set_level(logging.INFO, logger="jarvis-core")
    asyncio.run(server.call_tool(
        "jarvis_store",
        {"content": "SECRET-memory-body sk-live-abc123", "type": "observation"},
    ))
    assert "SECRET-memory-body" not in caplog.text
    assert "sk-live-abc123" not in caplog.text
    tool_line = next(r.getMessage() for r in caplog.records if r.getMessage().startswith("Tool:"))
    assert "jarvis_store" in tool_line
    assert "'content'" in tool_line and "'type'" in tool_line


def test_call_tool_error_text_never_leaks_credentials(monkeypatch):
    def raiser(args):
        raise RuntimeError("connect to postgresql://jarvis:hunter2@db:5432/jarvis failed")

    monkeypatch.setitem(server._HANDLERS, "jarvis_collection_stats", raiser)
    data = _payload(asyncio.run(server.call_tool("jarvis_collection_stats", {})))
    assert data["success"] is False
    assert "hunter2" not in data["error"]


def test_index_vault_calls_are_serialized(monkeypatch):
    """Tool calls now run concurrently; two full reindexes must not overlap."""
    state = {"active": 0, "max": 0}
    lock = threading.Lock()

    def fake_index_vault(**kwargs):
        with lock:
            state["active"] += 1
            state["max"] = max(state["max"], state["active"])
        time.sleep(0.05)
        with lock:
            state["active"] -= 1
        return {"success": True}

    monkeypatch.setattr(server, "index_vault", fake_index_vault)

    async def scenario():
        return await asyncio.gather(
            *(server.call_tool("jarvis_index_vault", {"force": True}) for _ in range(3))
        )

    results = asyncio.run(scenario())
    assert all(_payload(r)["success"] for r in results)
    assert state["max"] == 1


def test_list_tools_health_check_does_not_block_loop(monkeypatch, release):
    def hung_urlopen(*args, **kwargs):
        release.wait(5)
        raise OSError("obsidian unreachable")

    monkeypatch.setattr(server.urllib.request, "urlopen", hung_urlopen)
    monkeypatch.setattr(server, "_obsidian_cache", {"available": None, "checked_at": 0.0})

    async def scenario():
        task = asyncio.create_task(server.list_tools())
        lag = await _loop_lag()
        assert not task.done()
        release.set()
        return lag, await asyncio.wait_for(task, 2)

    lag, tools = asyncio.run(scenario())
    assert lag < 0.1, f"event loop stalled {lag:.3f}s behind list_tools' urlopen"
    names = {t.name for t in tools}
    assert names.isdisjoint(server._PKM_TOOLS)
    assert "jarvis_store" in names


def test_list_tools_serves_fresh_cache_without_network(monkeypatch):
    def no_network(*args, **kwargs):
        raise AssertionError("fresh cache must not trigger a health check")

    monkeypatch.setattr(server.urllib.request, "urlopen", no_network)
    monkeypatch.setattr(server, "_obsidian_cache", {"available": True, "checked_at": time.time()})
    tools = asyncio.run(server.list_tools())
    assert {t.name for t in tools} >= server._PKM_TOOLS


def test_db_status_probe_loop_refreshes_in_worker_thread(monkeypatch):
    calls = []

    def fake_probe():
        calls.append(threading.current_thread() is threading.main_thread())
        if len(calls) == 1:
            raise RuntimeError("probe blew up")  # the loop must survive this
        return {"status": "ok"}

    monkeypatch.setattr("tools.schema.probe_db_status", fake_probe, raising=False)
    monkeypatch.setattr(server, "DB_STATUS_PROBE_INTERVAL_SECONDS", 0.01)

    async def scenario():
        task = asyncio.create_task(server.db_status_probe_loop())
        deadline = time.monotonic() + 2
        while len(calls) < 3:
            assert time.monotonic() < deadline, "probe loop stopped refreshing"
            await asyncio.sleep(0.01)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(scenario())
    assert calls and not any(calls), "probe ran on the event-loop thread"


def test_background_registry_includes_db_status_probe():
    tasks = server.get_background_tasks()
    try:
        names = [t.cr_code.co_qualname for t in tasks]
        assert "db_status_probe_loop" in names
        assert "pattern_detection_loop" in names[0]
    finally:
        for t in tasks:
            t.close()
