"""Cross-owner seams of the outage fixes: /health's DB error text and the probe loop.

/health is unauthenticated and serves get_db_status()["error"], and hook 503
bodies carry the same reason, so the text must name the cause ("the database
system is in recovery mode") without the server's host and port. The
background probe's connect_timeout stops bounding it once connected, so a
server that accepts and then never answers must neither wedge the cached
status nor start a new stuck thread every interval.
"""

from __future__ import annotations

import asyncio
import json
import threading
import time

import pytest

import server
import tools.schema as schema
from tests.fake_pg_server import FakeRecoveringPostgres


@pytest.fixture(autouse=True)
def clean_breaker():
    schema.reset_breaker()
    yield
    schema.reset_breaker()


@pytest.mark.parametrize(
    "raw, expected",
    [
        (
            'connection failed: connection to server at "127.0.0.1", port 5432 failed: '
            "FATAL:  the database system is in recovery mode",
            "the database system is in recovery mode",
        ),
        (
            'connection failed: connection to server at "localhost" (::1), port 5432 failed: '
            "Connection refused\n\tIs the server running on that host and accepting TCP/IP connections?",
            "Connection refused Is the server running on that host and accepting TCP/IP connections?",
        ),
        (
            'connection to server on socket "/tmp/.s.PGSQL.5432" failed: No such file or directory',
            "No such file or directory",
        ),
    ],
)
def test_safe_db_error_drops_the_connection_target(raw, expected):
    assert schema.safe_db_error(raw) == expected


def test_probe_error_served_by_health_has_no_host_or_port(monkeypatch):
    import http_app
    import tools.config as config

    with FakeRecoveringPostgres() as fake:
        monkeypatch.setattr(config, "get_postgres_config", lambda: {"url": fake.url()})
        schema.probe_db_status()
        port = str(fake.port)

    sent = []

    async def send(message):
        sent.append(message)

    asyncio.run(http_app.health_response({"type": "http"}, None, send))
    body = json.loads(sent[1]["body"])
    assert body["status"] == "ok"
    assert body["postgres"]["status"] == "recovering"
    assert body["postgres"]["error"] == "the database system is in recovery mode"
    assert "127.0.0.1" not in sent[1]["body"].decode()
    assert port not in sent[1]["body"].decode()
    assert "hunter2" not in sent[1]["body"].decode()


def test_breaker_reason_in_ingest_error_has_no_host(monkeypatch):
    from tools.hook_endpoints import ingest_auto_extract

    with FakeRecoveringPostgres() as fake:
        monkeypatch.setattr("tools.config.get_postgres_config", lambda: {"url": fake.url()})
        schema.probe_db_status()  # trips the breaker with the probe's reason
    result = ingest_auto_extract({"observations": [{"content": "x" * 300}]})
    assert result["retryable"] is True
    assert result["error"] == "database unavailable: the database system is in recovery mode"


def test_stuck_probe_is_capped_and_never_duplicated(monkeypatch):
    release = threading.Event()
    started = []

    def hung_probe():
        started.append(threading.current_thread().name)
        release.wait(5)
        return schema.get_db_status()

    monkeypatch.setattr("tools.schema.probe_db_status", hung_probe)
    monkeypatch.setattr(server, "DB_STATUS_PROBE_TIMEOUT_SECONDS", 0.1)
    monkeypatch.setattr(server, "DB_STATUS_PROBE_INTERVAL_SECONDS", 0.02)

    async def scenario():
        task = asyncio.create_task(server.db_status_probe_loop())
        try:
            deadline = time.monotonic() + 3
            while schema.get_db_status()["status"] != "unreachable":
                assert time.monotonic() < deadline, "stuck probe never reported"
                await asyncio.sleep(0.01)
            lag_start = time.perf_counter()
            await asyncio.sleep(0.4)  # several more loop iterations
            assert time.perf_counter() - lag_start < 0.6, "loop was blocked"
            assert len(started) == 1, "a second probe started while one was stuck"
            assert "did not answer the status probe" in schema.get_db_status()["error"]
        finally:
            release.set()
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

    asyncio.run(scenario())
    assert started and started[0] != "MainThread"


def test_probe_resumes_after_a_stuck_probe_returns(monkeypatch):
    release = threading.Event()
    calls = []

    def probe():
        calls.append(time.monotonic())
        if len(calls) == 1:
            release.wait(5)
        return {"status": "ok"}

    monkeypatch.setattr("tools.schema.probe_db_status", probe)
    monkeypatch.setattr(server, "DB_STATUS_PROBE_TIMEOUT_SECONDS", 0.05)
    monkeypatch.setattr(server, "DB_STATUS_PROBE_INTERVAL_SECONDS", 0.01)

    async def scenario():
        task = asyncio.create_task(server.db_status_probe_loop())
        try:
            await asyncio.sleep(0.2)
            assert len(calls) == 1
            release.set()
            deadline = time.monotonic() + 3
            while len(calls) < 3:
                assert time.monotonic() < deadline, "probing did not resume"
                await asyncio.sleep(0.01)
        finally:
            release.set()
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

    asyncio.run(scenario())
