"""Outage mapping for routes without their own handler, and the probe-loop cap.

The Retrieval tab (/api/retrieval/*) runs on core's pool: during an outage it
raised core's DatabaseUnavailable (or a raw OperationalError) straight into a
bare "Internal Server Error". The app-level handlers turn that into the same
503 + cause + Retry-After the search routes give. The background DB probe is
bounded by connect_timeout only up to the connect, so a server that accepts
and never answers must not wedge /health's cached status or pile up threads.
"""

from __future__ import annotations

import asyncio
import sys
import threading
import time
import types
from pathlib import Path

import psycopg
import psycopg_pool
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import app as app_module  # noqa: E402

_RECOVERING = {
    "status": "recovering",
    "error": "the database system is in recovery mode",
    "checked_at": 1.0,
    "free_bytes": None,
}


@pytest.fixture
def recovering_probe(monkeypatch):
    monkeypatch.setattr(app_module, "_probe_db_status", lambda: dict(_RECOVERING))


@pytest.fixture
def fake_telemetry(monkeypatch):
    """A stand-in tools.retrieval_telemetry whose get_summary raises ``exc``."""
    mod = types.ModuleType("tools.retrieval_telemetry")
    monkeypatch.setitem(sys.modules, "tools.retrieval_telemetry", mod)

    def install(exc):
        def get_summary(days):
            raise exc
        mod.get_summary = get_summary

    return install


def test_core_database_unavailable_on_retrieval_route_is_503(
    monkeypatch, recovering_probe, fake_telemetry
):
    class DatabaseUnavailable(RuntimeError):
        pass

    fake_schema = types.ModuleType("tools.schema")
    fake_schema.DatabaseUnavailable = DatabaseUnavailable
    monkeypatch.setitem(sys.modules, "tools.schema", fake_schema)
    fake_telemetry(DatabaseUnavailable("the database system is in recovery mode (circuit open)"))

    r = TestClient(app_module.app).get("/api/retrieval/summary")
    assert r.status_code == 503, r.text
    assert r.headers["retry-after"] == "10"
    assert r.json()["detail"] == "Database unavailable: the database system is in recovery mode"


def test_pool_timeout_on_retrieval_route_is_503(monkeypatch, recovering_probe, fake_telemetry):
    fake_telemetry(psycopg_pool.PoolTimeout("couldn't get a connection after 10.00 sec"))
    r = TestClient(app_module.app).get("/api/retrieval/summary")
    assert r.status_code == 503
    assert "recovery mode" in r.json()["detail"]
    assert "10.00 sec" not in r.text


def test_lost_connection_on_retrieval_route_hides_the_server_address(monkeypatch, fake_telemetry):
    monkeypatch.setattr(app_module, "_probe_db_status", lambda: pytest.fail("no probe needed"))
    fake_telemetry(psycopg.OperationalError(
        'connection failed: connection to server at "10.1.2.3", port 5432 failed: '
        "server closed the connection unexpectedly"
    ))
    r = TestClient(app_module.app).get("/api/retrieval/summary")
    assert r.status_code == 503
    assert "server closed the connection unexpectedly" in r.json()["detail"]
    assert "10.1.2.3" not in r.text and "5432" not in r.text


def test_query_canceled_on_retrieval_route_is_504(fake_telemetry):
    fake_telemetry(psycopg.errors.QueryCanceled("canceling statement due to statement timeout"))
    r = TestClient(app_module.app).get("/api/retrieval/summary")
    assert r.status_code == 504
    assert r.json() == {"detail": "query timed out"}


def test_other_runtime_errors_still_surface_as_500(fake_telemetry):
    fake_telemetry(RuntimeError("unrelated bug"))
    r = TestClient(app_module.app, raise_server_exceptions=False).get("/api/retrieval/summary")
    assert r.status_code == 500
    assert "unrelated bug" not in r.text


def test_retrieval_tab_reads_errors_through_api_error():
    """A 503 body must reach the tab's esc()-rendered error, not renderRetrieval."""
    page = TestClient(app_module.app).get("/").text
    assert "fetch('/api/retrieval/summary').then(r=>r.ok?r.json():apiError(r))" in page
    assert "fetch('/api/retrieval/events'+query).then(r=>r.ok?r.json():apiError(r))" in page


def test_stuck_probe_marks_unreachable_without_piling_up_threads(monkeypatch):
    release = threading.Event()
    started = []

    def hung_probe():
        started.append(time.monotonic())
        release.wait(5)
        return dict(_RECOVERING)

    monkeypatch.setattr(app_module, "_probe_db_status", hung_probe)
    monkeypatch.setattr(app_module, "_DB_PROBE_WAIT", 0.1)
    monkeypatch.setattr(app_module, "_DB_PROBE_INTERVAL", 0.02)
    monkeypatch.setattr(app_module, "_db_status", {
        "status": "ok", "error": None, "checked_at": 1.0, "free_bytes": 5,
    })

    async def scenario():
        task = asyncio.create_task(app_module._db_status_loop())
        try:
            deadline = time.monotonic() + 3
            while app_module._db_status["status"] != "unreachable":
                assert time.monotonic() < deadline, "stuck probe never reported"
                await asyncio.sleep(0.01)
            await asyncio.sleep(0.4)  # several more loop iterations
            assert len(started) == 1, "a new probe started while one was stuck"
            assert "status probe" in app_module._db_status["error"]
            assert app_module._db_status["free_bytes"] == 5
        finally:
            release.set()
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

    asyncio.run(scenario())
