"""Explorer behavior while PostgreSQL is unavailable (2026-09-24/25 outage).

The Docker VM disk filled, Postgres crash-looped in recovery mode for ~15h,
and every Explorer search spun 30s and then printed psycopg_pool's raw
"couldn't get a connection after 30.00 sec" as an HTTP 500. These tests drive
the real app (real pool, real handlers) against a fake Postgres that rejects
every login with FATAL 57P03, plus mocked pools for the error-mapping paths.
"""

from __future__ import annotations

import asyncio
import json
import socket
import socketserver
import struct
import sys
import threading
import time
import types
from pathlib import Path
from unittest.mock import MagicMock

import httpx
import psycopg
import psycopg_pool
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import app as app_module  # noqa: E402

_PASSWORD = "s3cret-pw-do-not-leak"
_UNKNOWN_STATUS = {"status": "unknown", "error": None, "checked_at": None, "free_bytes": None}

_LOCAL = {
    "id": "local", "label": "Local Memories", "type": "local",
    "schema": "local", "table": "memories", "has_retrieval_count": True,
    "has_status": True, "capabilities": ["text", "metadata", "semantic"],
    "metadata_filters": ["category", "scope", "project", "source", "status"],
}
_OBSIDIAN = {
    "id": "obsidian", "label": "Obsidian Vault", "type": "local",
    "schema": "obsidian", "table": "documents", "has_retrieval_count": False,
    "has_status": False, "capabilities": ["text", "metadata", "semantic"],
    "metadata_filters": ["vault_type", "directory"],
}


# ── Fake Postgres ─────────────────────────────────────────────────────────

class _RecoveryModePG(socketserver.BaseRequestHandler):
    """Rejects startup the way a postmaster in crash recovery does."""

    def handle(self):
        s = self.request
        try:
            while True:
                hdr = s.recv(8)
                if len(hdr) < 8:
                    return
                length, code = struct.unpack("!ii", hdr)
                if code in (80877103, 80877104):  # SSLRequest / GSSENCRequest
                    s.sendall(b"N")
                    continue
                rest = length - 8
                while rest > 0:
                    chunk = s.recv(rest)
                    if not chunk:
                        return
                    rest -= len(chunk)
                fields = (
                    b"SFATAL\0VFATAL\0C57P03\0"
                    b"Mthe database system is in recovery mode\0\0"
                )
                s.sendall(b"E" + struct.pack("!i", len(fields) + 4) + fields)
                return
        except OSError:
            return


@pytest.fixture
def fake_pg():
    srv = socketserver.ThreadingTCPServer(("127.0.0.1", 0), _RecoveryModePG)
    srv.daemon_threads = True
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    try:
        yield srv.server_address[1]
    finally:
        srv.shutdown()
        srv.server_close()


@pytest.fixture(autouse=True)
def _isolate(tmp_path, monkeypatch):
    """Isolate config, and restore every module global the app mutates."""
    monkeypatch.setenv("JARVIS_HOME", str(tmp_path))
    monkeypatch.setenv("PGDATA", str(tmp_path))
    # Never reach a real local Postgres by accident: a closed port by default.
    monkeypatch.setenv("POSTGRES_URL", _dsn(_free_port()))
    (tmp_path / "config.json").write_text(json.dumps({"memory": {}}))
    from jarvis_common.config import clear_config_cache
    clear_config_cache()
    monkeypatch.setattr(app_module, "_db_status", dict(_UNKNOWN_STATUS))
    monkeypatch.setattr(app_module, "_last_disk_full_at", None)
    monkeypatch.setattr(app_module, "_sources_at", None)
    yield
    clear_config_cache()


def _dsn(port: int) -> str:
    return f"postgresql://jarvis:{_PASSWORD}@127.0.0.1:{port}/jarvis"


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _assert_no_conninfo(text: str, port: int) -> None:
    assert _PASSWORD not in text
    assert "127.0.0.1" not in text
    assert str(port) not in text


# ── Real pool against a recovery-mode Postgres ────────────────────────────

def test_search_fails_fast_with_real_cause(fake_pg, monkeypatch):
    """Empty and non-empty searches alike: 503 in < 7s naming recovery mode."""
    monkeypatch.setenv("POSTGRES_URL", _dsn(fake_pg))
    pool = app_module._make_local_pool()
    monkeypatch.setattr(app_module, "_local_pool", pool)
    monkeypatch.setattr(app_module, "_sources", {"local": _LOCAL, "obsidian": _OBSIDIAN})

    empty = {"source": "local", "mode": "text", "query": "", "filters": {},
             "page": 0, "page_size": 100, "sort_by": "date_desc"}
    bodies = {
        "local/text/empty": empty,
        "local/text/non-empty": dict(empty, query="docker"),
        "local/metadata": dict(empty, mode="metadata", filters={"category": "decision"}),
        "obsidian/text/empty": dict(empty, source="obsidian"),
    }

    async def drive():
        transport = httpx.ASGITransport(app=app_module.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://t", timeout=30) as c:
            health_latency: list[float] = []
            done = asyncio.Event()

            async def poll_health():
                while not done.is_set():
                    t = time.perf_counter()
                    assert (await c.get("/health")).status_code == 200
                    health_latency.append(time.perf_counter() - t)
                    await asyncio.sleep(0.2)

            async def search(body):
                t = time.perf_counter()
                r = await c.post("/api/search", json=body)
                return r, time.perf_counter() - t

            poller = asyncio.create_task(poll_health())
            results = await asyncio.gather(*(search(b) for b in bodies.values()))
            done.set()
            await poller
            return results, health_latency

    try:
        results, health_latency = asyncio.run(drive())
    finally:
        pool.close(timeout=1)

    for label, (r, elapsed) in zip(bodies, results):
        assert r.status_code == 503, (label, r.text)
        assert elapsed < 7, (label, elapsed)
        assert r.headers.get("retry-after") == "10", label
        detail = r.json()["detail"]
        assert detail.startswith("Database unavailable: "), detail
        assert "recovery mode" in detail, detail
        assert "couldn't get a connection" not in detail
        _assert_no_conninfo(r.text, fake_pg)
    # The waits run in executor threads: the event loop stays responsive.
    assert health_latency and max(health_latency) < 0.5


def test_probe_reports_recovering(fake_pg, monkeypatch):
    monkeypatch.setenv("POSTGRES_URL", _dsn(fake_pg))
    t = time.perf_counter()
    st = app_module._probe_db_status()
    assert time.perf_counter() - t < 3
    assert st["status"] == "recovering"
    assert "recovery mode" in st["error"]
    _assert_no_conninfo(st["error"], fake_pg)
    assert isinstance(st["checked_at"], float)
    assert isinstance(st["free_bytes"], int)  # PGDATA points at tmp_path
    assert app_module._db_status == st

    r = TestClient(app_module.app).get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"  # entrypoint/compose depend on it
    assert body["postgres"]["status"] == "recovering"


def test_probe_reports_unreachable(monkeypatch):
    port = _free_port()
    monkeypatch.setenv("POSTGRES_URL", _dsn(port))
    st = app_module._probe_db_status()
    assert st["status"] == "unreachable"
    assert "refused" in st["error"].lower()
    _assert_no_conninfo(st["error"], port)


class _FakeConn:
    def __init__(self, in_recovery: bool):
        self.in_recovery = in_recovery

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, query):
        assert query == "SELECT pg_is_in_recovery()"
        cur = MagicMock()
        cur.fetchone.return_value = (self.in_recovery,)
        return cur


def test_probe_reports_ok_and_disk_full(monkeypatch):
    seen = {}

    def fake_connect(url, connect_timeout):
        seen["connect_timeout"] = connect_timeout
        return _FakeConn(in_recovery=False)

    monkeypatch.setattr(psycopg, "connect", fake_connect)
    st = app_module._probe_db_status()
    assert st["status"] == "ok" and st["error"] is None
    assert seen["connect_timeout"] == 2

    monkeypatch.setattr(app_module, "_pgdata_free_bytes", lambda: 10 * 1024 * 1024)
    st = app_module._probe_db_status()
    assert st["status"] == "disk_full"
    assert st["free_bytes"] == 10 * 1024 * 1024
    assert "10 MiB" in st["error"]


def test_probe_reports_recent_disk_full_error(monkeypatch):
    monkeypatch.setattr(psycopg, "connect", lambda url, connect_timeout: _FakeConn(False))
    monkeypatch.setattr(app_module, "_pgdata_free_bytes", lambda: None)
    app_module._note_db_error(psycopg.errors.DiskFull("could not extend file: No space left on device"))
    st = app_module._probe_db_status()
    assert st["status"] == "disk_full"
    assert st["free_bytes"] is None


def test_probe_reports_in_recovery_server(monkeypatch):
    monkeypatch.setattr(psycopg, "connect", lambda url, connect_timeout: _FakeConn(True))
    assert app_module._probe_db_status()["status"] == "recovering"


def test_pool_fails_fast_and_checks_connections(monkeypatch):
    real = psycopg_pool.ConnectionPool
    captured = {}

    class SpyPool(real):
        def __init__(self, *args, **kwargs):
            captured.update(kwargs)
            kwargs["open"] = False
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(psycopg_pool, "ConnectionPool", SpyPool)
    app_module._make_local_pool().close()
    assert captured["timeout"] == 5.0
    assert captured["check"] is real.check_connection


# ── Error mapping with mocked pools ───────────────────────────────────────

def _pool_raising(exc: Exception):
    pool = MagicMock()
    conn = MagicMock()
    pool.connection.return_value.__enter__ = MagicMock(return_value=conn)
    pool.connection.return_value.__exit__ = MagicMock(return_value=False)
    conn.execute = MagicMock(side_effect=exc)
    conn.cursor.return_value.__enter__ = MagicMock(side_effect=exc)
    conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
    return pool


def _probe_must_not_run():
    raise AssertionError("the direct DB probe must not run for this error")


@pytest.fixture
def recovering_probe(monkeypatch):
    calls = []

    def fake():
        calls.append(1)
        return {"status": "recovering", "error": "the database system is in recovery mode",
                "checked_at": time.time(), "free_bytes": None}

    monkeypatch.setattr(app_module, "_probe_db_status", fake)
    return calls


def _client(monkeypatch, pool) -> TestClient:
    monkeypatch.setattr(app_module, "_local_pool", pool)
    monkeypatch.setattr(app_module, "_sources", {"local": _LOCAL, "obsidian": _OBSIDIAN})
    return TestClient(app_module.app)


def test_query_canceled_is_504_not_503(monkeypatch):
    """statement_timeout's QueryCanceled is an OperationalError subclass."""
    exc = psycopg.errors.QueryCanceled("canceling statement due to statement timeout")
    monkeypatch.setattr(app_module, "_probe_db_status", _probe_must_not_run)
    tc = _client(monkeypatch, _pool_raising(exc))
    for query in ("", "slow ilike"):
        r = tc.post("/api/search", json={"source": "local", "mode": "text", "query": query})
        assert r.status_code == 504, r.text
        assert r.json()["detail"] == "query timed out"
    r = tc.get("/api/content?source=local&id=x")
    assert r.status_code == 504


def test_lost_connection_is_503_with_first_line_reason(monkeypatch):
    exc = psycopg.OperationalError(
        "consuming input failed: server closed the connection unexpectedly\n"
        "\tThis probably means the server terminated abnormally"
    )
    monkeypatch.setattr(app_module, "_probe_db_status", _probe_must_not_run)
    tc = _client(monkeypatch, _pool_raising(exc))
    r = tc.post("/api/search", json={"source": "obsidian", "mode": "text", "query": ""})
    assert r.status_code == 503
    assert r.headers["retry-after"] == "10"
    assert r.json()["detail"] == (
        "Database unavailable: consuming input failed: server closed the connection unexpectedly"
    )


def test_pool_timeout_on_content_and_delete_is_503(monkeypatch, recovering_probe):
    pool = MagicMock()
    pool.connection.side_effect = psycopg_pool.PoolTimeout("couldn't get a connection after 5.00 sec")
    tc = _client(monkeypatch, pool)
    for r in (
        tc.get("/api/content?source=local&id=obs::1"),
        tc.delete("/api/memories/obs::1?source=local"),
    ):
        assert r.status_code == 503, r.text
        assert r.headers["retry-after"] == "10"
        assert r.json()["detail"] == "Database unavailable: the database system is in recovery mode"
    assert len(recovering_probe) == 2


def test_pool_timeout_with_healthy_server_says_exhausted(monkeypatch):
    monkeypatch.setattr(app_module, "_probe_db_status", lambda: {
        "status": "ok", "error": None, "checked_at": time.time(), "free_bytes": None})
    pool = MagicMock()
    pool.connection.side_effect = psycopg_pool.PoolTimeout("couldn't get a connection after 5.00 sec")
    r = _client(monkeypatch, pool).post("/api/search", json={"source": "local", "mode": "text"})
    assert r.status_code == 503
    assert r.json()["detail"] == "Database unavailable: connection pool exhausted"


def test_admin_pool_timeout_is_503(monkeypatch, recovering_probe):
    def boom():
        raise psycopg_pool.PoolTimeout("couldn't get a connection after 5.00 sec")

    monkeypatch.setattr(app_module, "_admin_sync", boom)
    r = TestClient(app_module.app).get("/api/admin")
    assert r.status_code == 503
    assert "recovery mode" in r.json()["detail"]


def test_core_breaker_error_is_503(monkeypatch, recovering_probe):
    """Semantic mode runs on core's pool, whose breaker raises DatabaseUnavailable."""
    class DatabaseUnavailable(RuntimeError):
        pass

    fake_schema = types.ModuleType("tools.schema")
    fake_schema.DatabaseUnavailable = DatabaseUnavailable
    monkeypatch.setitem(sys.modules, "tools.schema", fake_schema)

    def raise_breaker(src, req):
        raise DatabaseUnavailable("database unavailable (circuit open)")

    monkeypatch.setattr(app_module, "_search_sync", raise_breaker)
    r = _client(monkeypatch, MagicMock()).post(
        "/api/search", json={"source": "obsidian", "mode": "text", "query": "x"})
    assert r.status_code == 503
    assert "recovery mode" in r.json()["detail"]


def test_unrelated_errors_stay_500(monkeypatch):
    monkeypatch.setattr(app_module, "_probe_db_status", _probe_must_not_run)
    tc = _client(monkeypatch, _pool_raising(psycopg.errors.UndefinedColumn('column "x" does not exist')))
    r = tc.post("/api/search", json={"source": "local", "mode": "text", "query": ""})
    assert r.status_code == 500
    assert "does not exist" in r.json()["detail"]


def test_stats_counts_run_concurrently(monkeypatch):
    def slow_count(src):
        time.sleep(0.4)
        return 7

    monkeypatch.setattr(app_module, "_count_sync", slow_count)
    tc = _client(monkeypatch, MagicMock())
    t = time.perf_counter()
    r = tc.get("/api/stats")
    assert time.perf_counter() - t < 0.75
    assert r.json() == {"local": {"count": 7}, "obsidian": {"count": 7}}


def test_err_reason_strips_conninfo():
    e = psycopg.OperationalError(
        'connection failed: connection to server at "10.0.0.5", port 6543 failed: '
        "FATAL:  the database system is starting up"
    )
    assert app_module._db_err_reason(e) == "the database system is starting up"
    e = psycopg.OperationalError(
        'connection is bad: connection to server on socket "/tmp/.s.PGSQL.5432" failed: '
        "No such file or directory\n\tIs the server running locally?"
    )
    assert app_module._db_err_reason(e) == "No such file or directory"
    e = psycopg.OperationalError("invalid dsn postgresql://u:pw@h/db")
    assert app_module._db_err_reason(e) == "Database error (credentials redacted)"


# ── Lifespan: background probe feeds /health ──────────────────────────────

def test_lifespan_runs_and_cancels_db_probe(monkeypatch):
    for name in ("_local_pool", "_sources", "_sources_at", "_sources_lock",
                 "_db_probe_task", "_refresh_task"):
        monkeypatch.setattr(app_module, name, getattr(app_module, name))
    monkeypatch.setattr(app_module, "_make_local_pool", lambda: MagicMock())
    monkeypatch.setattr(app_module, "_discover_sources", lambda pool: {"local": _LOCAL})

    def fake_probe():
        app_module._db_status = {"status": "ok", "error": None,
                                 "checked_at": time.time(), "free_bytes": 123}
        return app_module._db_status

    monkeypatch.setattr(app_module, "_probe_db_status", fake_probe)
    with TestClient(app_module.app) as tc:
        deadline = time.monotonic() + 3
        body = tc.get("/health").json()
        while body["postgres"]["status"] == "unknown" and time.monotonic() < deadline:
            time.sleep(0.05)
            body = tc.get("/health").json()
        assert body["status"] == "ok"
        assert body["sources"] == ["local"]
        assert body["postgres"]["status"] == "ok"
        assert body["postgres"]["free_bytes"] == 123
        task = app_module._db_probe_task
    assert task.done()


# ── Fast fail from the probe's verdict ────────────────────────────────────

def _pool_must_not_be_used():
    pool = MagicMock()
    pool.connection.side_effect = AssertionError("the pool must not be waited on")
    return pool


def _unreachable_verdict(monkeypatch, error="Connection refused", status="unreachable"):
    monkeypatch.setattr(app_module, "_db_unreachable_at", time.monotonic())
    monkeypatch.setattr(app_module, "_db_status", {
        "status": status, "error": error, "checked_at": time.time(), "free_bytes": None})


def test_known_outage_answers_503_without_waiting_on_the_pool(monkeypatch):
    """Every Explorer request during the disk-full rehearsal waited out the
    5s pool timeout before its 503 while core hooks answered in 2-4ms."""
    _unreachable_verdict(monkeypatch, "disk full (No space left on device); Connection refused",
                         status="disk_full")
    monkeypatch.setattr(app_module, "_probe_db_status", _probe_must_not_run)
    import tools.retrieval_telemetry as telemetry
    monkeypatch.setattr(telemetry, "get_summary", lambda days: pytest.fail("reached the DB"))
    tc = _client(monkeypatch, _pool_must_not_be_used())
    expected = "Database unavailable: disk full (No space left on device); Connection refused"
    for method, url, body in (
        ("post", "/api/search", {"source": "local", "mode": "text", "query": ""}),
        ("post", "/api/search", {"source": "obsidian", "mode": "metadata", "filters": {}}),
        ("get", "/api/content?source=local&id=obs::1", None),
        ("delete", "/api/memories/obs::1?source=local", None),
        ("get", "/api/retrieval/summary", None),
    ):
        t = time.perf_counter()
        r = tc.request(method, url, json=body)
        assert time.perf_counter() - t < 1, url
        assert r.status_code == 503, (url, r.text)
        assert r.headers["retry-after"] == "10", url
        assert r.json()["detail"] == expected, url
    stats = tc.get("/api/stats").json()
    assert stats["local"]["count"] is None and "Connection refused" in stats["local"]["error"]


def test_semantic_search_on_core_pool_fails_fast_too(monkeypatch):
    _unreachable_verdict(monkeypatch)
    monkeypatch.setattr(app_module, "get_embedding_config", lambda: {"backend": "host"})
    import tools.query as query
    monkeypatch.setattr(query, "semantic_candidate_search",
                        lambda *a, **k: pytest.fail("reached core's pool"))
    r = _client(monkeypatch, _pool_must_not_be_used()).post(
        "/api/search", json={"source": "local", "mode": "semantic", "query": "docker"})
    assert r.status_code == 503, r.text
    assert r.json()["detail"] == "Database unavailable: Connection refused"


def test_stale_unreachable_verdict_is_ignored(monkeypatch):
    """A verdict older than one probe cycle no longer blocks: a recovered
    database is used again even if the probe has not run since."""
    _unreachable_verdict(monkeypatch)
    monkeypatch.setattr(
        app_module, "_db_unreachable_at", time.monotonic() - app_module._FAST_FAIL_WINDOW - 1)
    monkeypatch.setattr(app_module, "_fetch_content_sync", lambda src, item_id: {"id": item_id})
    r = _client(monkeypatch, MagicMock()).get("/api/content?source=local&id=obs::1")
    assert r.status_code == 200, r.text


def test_probe_sets_and_clears_the_unreachable_verdict(monkeypatch):
    monkeypatch.setenv("POSTGRES_URL", _dsn(_free_port()))
    app_module._probe_db_status()
    assert app_module._db_unreachable_at is not None
    monkeypatch.setattr(psycopg, "connect", lambda url, connect_timeout: _FakeConn(False))
    app_module._probe_db_status()
    assert app_module._db_unreachable_at is None


def test_second_request_of_an_outage_is_fast_against_real_pool(fake_pg, monkeypatch):
    """Real pool + recovery-mode server: the first request pays the pool
    timeout and learns the cause; the next ones answer at once."""
    monkeypatch.setenv("POSTGRES_URL", _dsn(fake_pg))
    monkeypatch.setattr(app_module, "_POOL_TIMEOUT", 1.0)
    pool = app_module._make_local_pool()
    monkeypatch.setattr(app_module, "_local_pool", pool)
    monkeypatch.setattr(app_module, "_sources", {"local": _LOCAL, "obsidian": _OBSIDIAN})
    body = {"source": "local", "mode": "text", "query": ""}
    try:
        tc = TestClient(app_module.app)
        first = tc.post("/api/search", json=body)
        t = time.perf_counter()
        second = tc.post("/api/search", json=dict(body, source="obsidian"))
        elapsed = time.perf_counter() - t
    finally:
        pool.close(timeout=1)
    for r in (first, second):
        assert r.status_code == 503, r.text
        assert "recovery mode" in r.json()["detail"]
        _assert_no_conninfo(r.text, fake_pg)
    assert elapsed < 0.5


def test_background_probe_stall_counts_as_unreachable(monkeypatch):
    """A frozen server (accepts, never answers): the loop's cap marks it."""
    monkeypatch.setattr(app_module, "_DB_PROBE_WAIT", 0.05)
    monkeypatch.setattr(app_module, "_DB_PROBE_INTERVAL", 0.01)
    release = threading.Event()
    monkeypatch.setattr(app_module, "_probe_db_status", lambda: release.wait(2))

    async def run_once():
        task = asyncio.create_task(app_module._db_status_loop())
        await asyncio.sleep(0.2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    try:
        asyncio.run(run_once())
    finally:
        release.set()
    assert app_module._db_unreachable_at is not None
    assert app_module._db_status["status"] == "unreachable"


# ── Disk-full status recovers ─────────────────────────────────────────────

def test_disk_full_clears_once_space_is_back_and_errors_stop(monkeypatch):
    monkeypatch.setattr(psycopg, "connect", lambda url, connect_timeout: _FakeConn(False))
    monkeypatch.setattr(app_module, "_pgdata_free_bytes", lambda: 4 * 1024 ** 3)
    app_module._note_db_error(psycopg.errors.DiskFull("could not extend file: No space left on device"))
    assert app_module._probe_db_status()["status"] == "disk_full"  # no flapping

    monkeypatch.setattr(
        app_module, "_last_disk_full_at", time.monotonic() - app_module._DISK_FULL_CLEAR - 1)
    st = app_module._probe_db_status()
    assert st["status"] == "ok" and st["error"] is None
    assert app_module._last_disk_full_at is None


def test_disk_full_is_kept_while_the_server_is_down(monkeypatch):
    port = _free_port()
    monkeypatch.setenv("POSTGRES_URL", _dsn(port))
    monkeypatch.setattr(app_module, "_pgdata_free_bytes", lambda: 4 * 1024 ** 3)
    monkeypatch.setattr(
        app_module, "_last_disk_full_at", time.monotonic() - app_module._DISK_FULL_CLEAR - 1)
    st = app_module._probe_db_status()
    assert st["status"] == "disk_full"
    assert st["error"].startswith("disk full (No space left on device); ")
    assert "refused" in st["error"].lower()
    _assert_no_conninfo(st["error"], port)
