"""Outage resilience of core's PostgreSQL access (tools/schema.py).

Covers the fail-fast pool, the circuit breaker, the cached health probe and
disk-full reporting added after the 2026-09 disk-full outage, where every
checkout waited 30s against a database stuck in recovery mode.
"""

from __future__ import annotations

import contextlib
import logging
import threading
import time
from types import SimpleNamespace

import psycopg
import psycopg_pool
import pytest

import tools.schema as schema
from tests.fake_pg_server import FakeRecoveringPostgres


@pytest.fixture(autouse=True)
def clean_breaker():
    schema.reset_breaker()
    yield
    schema.reset_breaker()


@pytest.fixture
def clock(monkeypatch):
    """Controllable monotonic clock for the breaker (time.monotonic in schema)."""
    state = {"now": 10_000.0}
    fake_time = SimpleNamespace(
        monotonic=lambda: state["now"], time=time.time, sleep=time.sleep
    )
    monkeypatch.setattr(schema, "time", fake_time)
    return state


def _unopened_pool() -> schema._JarvisPool:
    return schema._JarvisPool(conninfo="", open=False)


def _fake_getconn(monkeypatch, behaviour):
    """Replace ConnectionPool.getconn (the super() call) with ``behaviour``."""
    calls = []

    def getconn(self, timeout=None):
        calls.append(timeout)
        return behaviour(timeout)

    monkeypatch.setattr(psycopg_pool.ConnectionPool, "getconn", getconn)
    return calls


def _raise(exc):
    def behaviour(timeout):
        raise exc
    return behaviour


# ── Breaker ───────────────────────────────────────────────────────────


def test_pool_timeout_after_connect_error_trips_breaker(monkeypatch, clock):
    schema._record_connect_error(psycopg.OperationalError(
        "connection failed: FATAL:  the database system is in recovery mode"
    ))
    calls = _fake_getconn(
        monkeypatch, _raise(psycopg_pool.PoolTimeout("couldn't get a connection after 1.50 sec"))
    )
    pool = _unopened_pool()

    with pytest.raises(schema.DatabaseUnavailable) as first:
        pool.getconn()
    assert "recovery mode" in str(first.value)
    assert schema.db_available() is False

    # Open breaker: fail immediately, never touch the pool again.
    with pytest.raises(schema.DatabaseUnavailable) as second:
        pool.getconn()
    assert "circuit open" in str(second.value)
    assert len(calls) == 1


def test_pool_timeout_without_connect_errors_is_contention_not_outage(monkeypatch):
    """A saturated but healthy pool fails the call without tripping everyone."""
    _fake_getconn(
        monkeypatch, _raise(psycopg_pool.PoolTimeout("couldn't get a connection after 1.50 sec"))
    )
    with pytest.raises(schema.DatabaseUnavailable) as err:
        _unopened_pool().getconn()
    assert "pool busy" in str(err.value)
    assert schema.db_available() is True


def test_too_many_requests_fails_the_call_without_tripping_breaker(monkeypatch):
    # TooManyRequests is an OperationalError but NOT a PoolTimeout. It means
    # max_waiting callers are queued on a healthy database: contention, which
    # must not push every other caller into fail-fast (review finding CR-5).
    assert not issubclass(psycopg_pool.TooManyRequests, psycopg_pool.PoolTimeout)
    _fake_getconn(
        monkeypatch, _raise(psycopg_pool.TooManyRequests("32 requests waiting"))
    )
    with pytest.raises(schema.DatabaseUnavailable, match="connection pool busy"):
        _unopened_pool().getconn()
    assert schema.db_available() is True


def test_operational_error_on_checkout_trips_breaker(monkeypatch):
    _fake_getconn(monkeypatch, _raise(psycopg.OperationalError("server closed the connection")))
    with pytest.raises(schema.DatabaseUnavailable):
        _unopened_pool().getconn()
    assert schema.db_available() is False


def test_pool_closed_passes_through_without_tripping(monkeypatch):
    _fake_getconn(monkeypatch, _raise(psycopg_pool.PoolClosed("pool is closed")))
    with pytest.raises(psycopg_pool.PoolClosed):
        _unopened_pool().getconn()
    assert schema.db_available() is True


def test_half_open_admits_one_trial_and_success_closes(monkeypatch, clock):
    schema._trip_breaker("the database system is in recovery mode")
    sentinel = object()
    calls = _fake_getconn(monkeypatch, lambda timeout: sentinel)
    pool = _unopened_pool()

    with pytest.raises(schema.DatabaseUnavailable):
        pool.getconn()
    assert calls == []

    clock["now"] += schema._BREAKER_OPEN_SECONDS + 0.1
    assert schema.db_available() is True  # half-open

    # The trial claims the window: a concurrent caller keeps failing fast.
    admitted = threading.Event()
    release = threading.Event()

    def slow_trial(timeout):
        admitted.set()
        release.wait(2)
        return sentinel

    monkeypatch.setattr(
        psycopg_pool.ConnectionPool, "getconn",
        lambda self, timeout=None: slow_trial(timeout),
    )
    result = {}
    trial = threading.Thread(target=lambda: result.setdefault("conn", pool.getconn()))
    trial.start()
    assert admitted.wait(2)
    with pytest.raises(schema.DatabaseUnavailable):
        pool.getconn()
    release.set()
    trial.join(2)

    assert result["conn"] is sentinel
    assert schema.db_available() is True
    assert schema.db_unavailable_reason() == "PostgreSQL is unreachable"


def test_half_open_trial_failure_reopens(monkeypatch, clock):
    schema._trip_breaker("first failure")
    clock["now"] += schema._BREAKER_OPEN_SECONDS + 0.1
    schema._record_connect_error(psycopg.OperationalError("still in recovery mode"))
    _fake_getconn(monkeypatch, _raise(psycopg_pool.PoolTimeout("timeout")))

    with pytest.raises(schema.DatabaseUnavailable):
        _unopened_pool().getconn()
    assert schema.db_available() is False
    assert "still in recovery mode" in schema.db_unavailable_reason()


def test_successful_checkout_clears_connect_error_and_breaker(monkeypatch, clock):
    schema._record_connect_error(psycopg.OperationalError("refused"))
    schema._trip_breaker("refused")
    clock["now"] += schema._BREAKER_OPEN_SECONDS + 0.1
    _fake_getconn(monkeypatch, lambda timeout: object())

    _unopened_pool().getconn()
    assert schema.db_available() is True
    assert schema._last_connect_error is None


def test_checkout_timeout_context_bounds_getconn(monkeypatch):
    calls = _fake_getconn(monkeypatch, lambda timeout: object())
    pool = _unopened_pool()

    pool.getconn()
    with schema.checkout_timeout():
        pool.getconn()
        pool.getconn(timeout=7.0)  # an explicit timeout still wins
        with schema.checkout_timeout(None):
            pool.getconn()
    pool.getconn()

    assert calls == [None, schema.HOOK_CONN_TIMEOUT, 7.0, None, None]
    assert schema.HOOK_CONN_TIMEOUT == 1.5


def test_checkout_timeout_follows_into_threads_via_copy_context():
    import contextvars
    from concurrent.futures import ThreadPoolExecutor

    with schema.checkout_timeout(0.75):
        ctx = contextvars.copy_context()
    with ThreadPoolExecutor(1) as executor:
        seen = executor.submit(ctx.run, schema._checkout_timeout_var.get).result()
    assert seen == 0.75
    assert schema._checkout_timeout_var.get() is None


def test_broken_connection_mid_query_trips_breaker(monkeypatch):
    broken = SimpleNamespace(broken=True)

    @contextlib.contextmanager
    def connection(self, timeout=None):
        yield broken

    monkeypatch.setattr(psycopg_pool.ConnectionPool, "connection", connection)
    with pytest.raises(psycopg.OperationalError):
        with _unopened_pool().connection():
            raise psycopg.OperationalError("server closed the connection unexpectedly")
    assert schema.db_available() is False


def test_statement_error_on_healthy_connection_does_not_trip(monkeypatch):
    healthy = SimpleNamespace(broken=False)

    @contextlib.contextmanager
    def connection(self, timeout=None):
        yield healthy

    monkeypatch.setattr(psycopg_pool.ConnectionPool, "connection", connection)
    with pytest.raises(psycopg.errors.CheckViolation):
        with _unopened_pool().connection():
            raise psycopg.errors.CheckViolation("violates check constraint")
    assert schema.db_available() is True


def test_reconnect_failed_callback_trips_breaker():
    schema._record_connect_error(psycopg.OperationalError("in recovery mode"))
    schema._on_reconnect_failed(None)
    assert schema.db_available() is False
    assert "recovery mode" in schema.db_unavailable_reason()


# ── Pool construction ─────────────────────────────────────────────────


def _patch_config(monkeypatch, url="postgresql://u:p@localhost:5432/db", memory=None):
    import tools.config as config

    monkeypatch.setattr(config, "get_postgres_config", lambda: {"url": url})
    monkeypatch.setattr(config, "get_embedding_config", lambda: {"dimensions": 384})
    monkeypatch.setattr(config, "get_memory_config", lambda: dict(memory or {}))


def test_pool_is_created_fail_fast(monkeypatch):
    _patch_config(monkeypatch, memory={"pool_timeout_seconds": 4})
    created = []

    class FakePool:
        def __init__(self, **kwargs):
            created.append(kwargs)

        def close(self):
            pass

    monkeypatch.setattr(schema, "_JarvisPool", FakePool)
    monkeypatch.setattr(schema, "_pool", None)
    monkeypatch.setattr(schema, "_pool_cache_key", None)

    schema._get_pool()
    kwargs = created[0]
    assert kwargs["timeout"] == 4.0
    assert kwargs["max_waiting"] == 32
    assert kwargs["reconnect_timeout"] == 60.0
    assert kwargs["check"] is psycopg_pool.ConnectionPool.check_connection
    assert kwargs["reconnect_failed"] is schema._on_reconnect_failed
    assert kwargs["kwargs"]["connect_timeout"] == 3
    assert kwargs["kwargs"]["keepalives"] == 1
    assert kwargs["kwargs"]["keepalives_idle"] == 30
    assert kwargs["connection_class"] is schema._TrackedConnection


def test_pool_timeout_defaults_to_ten_seconds(monkeypatch):
    _patch_config(monkeypatch, memory={"pool_timeout_seconds": "nonsense"})
    assert schema._pool_timeout_seconds() == 10.0
    _patch_config(monkeypatch, memory={})
    assert schema._pool_timeout_seconds() == 10.0


def test_get_pool_creates_one_pool_under_concurrency(monkeypatch):
    _patch_config(monkeypatch)
    created = []

    class SlowPool:
        def __init__(self, **kwargs):
            time.sleep(0.05)  # widen the check-then-create window
            created.append(self)

        def close(self):
            pass

    monkeypatch.setattr(schema, "_JarvisPool", SlowPool)
    monkeypatch.setattr(schema, "_pool", None)
    monkeypatch.setattr(schema, "_pool_cache_key", None)

    barrier = threading.Barrier(8)
    results = []

    def worker():
        barrier.wait()
        results.append(schema._get_pool())

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(5)

    assert len(created) == 1
    assert all(pool is created[0] for pool in results)


# ── Connection tracking, probe, status ────────────────────────────────


def test_tracked_connection_records_real_recovery_mode_error():
    with FakeRecoveringPostgres() as fake:
        started = time.monotonic()
        with pytest.raises(psycopg.OperationalError):
            schema._TrackedConnection.connect(fake.url(), connect_timeout=2)
        assert time.monotonic() - started < 2.0
    assert "recovery mode" in schema._last_connect_error
    assert "hunter2-secret" not in schema._last_connect_error


def test_probe_reports_recovering_against_fake_pg(monkeypatch):
    with FakeRecoveringPostgres() as fake:
        _patch_config(monkeypatch, url=fake.url())
        started = time.monotonic()
        status = schema.probe_db_status()
        elapsed = time.monotonic() - started

    assert elapsed < 2.5
    assert status["status"] == "recovering"
    assert "recovery mode" in status["error"]
    assert "hunter2-secret" not in status["error"]
    assert status["checked_at"] is not None
    assert schema.db_available() is False
    assert schema.get_db_status() == status


def test_probe_reports_unreachable_when_nothing_listens(monkeypatch):
    import socket

    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()  # nothing listens on this port now
    _patch_config(monkeypatch, url=f"postgresql://u:pw@127.0.0.1:{port}/db")

    status = schema.probe_db_status()
    assert status["status"] == "unreachable"
    assert schema.db_available() is False


class _FakeProbeConnection:
    def __init__(self, in_recovery=False):
        self.in_recovery = in_recovery

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, sql):
        assert sql == "SELECT pg_is_in_recovery()"
        return SimpleNamespace(fetchone=lambda: (self.in_recovery,))


def test_probe_ok_closes_breaker_and_reports_free_space(monkeypatch, tmp_path):
    _patch_config(monkeypatch)
    schema._trip_breaker("was down")
    captured = {}

    def fake_connect(url, **kwargs):
        captured.update(kwargs)
        return _FakeProbeConnection()

    monkeypatch.setattr(schema.psycopg, "connect", fake_connect)
    monkeypatch.setenv("PGDATA", str(tmp_path))

    status = schema.probe_db_status()
    assert status["status"] == "ok"
    assert status["error"] is None
    assert isinstance(status["free_bytes"], int) and status["free_bytes"] > 0
    assert captured["connect_timeout"] == 2
    assert schema.db_available() is True


def test_probe_reports_disk_full_on_low_free_space(monkeypatch, tmp_path):
    _patch_config(monkeypatch)
    monkeypatch.setattr(schema.psycopg, "connect", lambda url, **kw: _FakeProbeConnection())
    monkeypatch.setenv("PGDATA", str(tmp_path))
    monkeypatch.setattr(
        schema.os, "statvfs", lambda path: SimpleNamespace(f_bavail=10, f_frsize=4096)
    )

    status = schema.probe_db_status()
    assert status["status"] == "disk_full"
    assert status["free_bytes"] == 40960
    assert "MiB free" in status["error"]
    assert schema.db_available() is True  # reachable; reads still work


def test_probe_skips_statvfs_for_remote_database(monkeypatch, tmp_path):
    _patch_config(monkeypatch, url="postgresql://u:p@db.internal.example:5432/jarvis")
    monkeypatch.setattr(schema.psycopg, "connect", lambda url, **kw: _FakeProbeConnection())
    monkeypatch.setenv("PGDATA", str(tmp_path))
    assert schema.probe_db_status()["free_bytes"] is None


def test_probe_never_raises(monkeypatch):
    import tools.config as config

    def broken_config():
        raise RuntimeError("config unreadable")

    monkeypatch.setattr(config, "get_postgres_config", broken_config)
    status = schema.probe_db_status()
    assert status["status"] == "unreachable"


def test_get_db_status_is_unknown_until_probed_and_returns_copies():
    status = schema.get_db_status()
    assert status == {"status": "unknown", "error": None, "checked_at": None, "free_bytes": None}
    status["status"] = "mutated"
    assert schema.get_db_status()["status"] == "unknown"


def test_disk_full_is_critical_rate_limited_and_feeds_status(monkeypatch, caplog):
    _patch_config(monkeypatch)
    exc = psycopg.errors.DiskFull(
        'could not extend file "base/16384/19825": No space left on device'
    )
    with caplog.at_level(logging.CRITICAL, logger="jarvis-core"):
        schema.note_db_error(exc)
        schema.note_db_error(exc)
    critical = [r for r in caplog.records if r.levelno == logging.CRITICAL]
    assert len(critical) == 1
    assert "DISK FULL" in critical[0].getMessage()
    assert schema.get_db_status()["status"] == "disk_full"

    # A probe within the 5-minute window keeps reporting it even if connect works.
    monkeypatch.setattr(schema.psycopg, "connect", lambda url, **kw: _FakeProbeConnection())
    assert schema.probe_db_status()["status"] == "disk_full"


_DISK_FULL_EXC = psycopg.errors.DiskFull(
    'could not extend file "base/16384/19825": No space left on device'
)


def _plenty_of_space(monkeypatch, tmp_path):
    monkeypatch.setenv("PGDATA", str(tmp_path))
    monkeypatch.setattr(  # 4 GiB free
        schema.os, "statvfs", lambda path: SimpleNamespace(f_bavail=1 << 20, f_frsize=4096)
    )


def test_disk_full_clears_once_space_is_back_and_errors_stop(monkeypatch, tmp_path, clock, caplog):
    """B1 rehearsal: space freed and writes succeeding again, yet /health said
    disk_full for the whole 5-minute window (statusline red, launcher telling
    the user to free disk space that was already free)."""
    _patch_config(monkeypatch)
    _plenty_of_space(monkeypatch, tmp_path)
    monkeypatch.setattr(schema.psycopg, "connect", lambda url, **kw: _FakeProbeConnection())
    schema.note_db_error(_DISK_FULL_EXC)

    clock["now"] += 30  # still inside the quiet period: no flapping to ok
    assert schema.probe_db_status()["status"] == "disk_full"

    clock["now"] += schema._DISK_FULL_CLEAR_SECONDS
    with caplog.at_level(logging.WARNING, logger="jarvis-core"):
        status = schema.probe_db_status()
    assert status["status"] == "ok"
    assert status["error"] is None
    assert any("disk-full condition cleared" in r.getMessage() for r in caplog.records)
    # Cleared for good, not just for one probe.
    clock["now"] += 1
    assert schema.probe_db_status()["status"] == "ok"


def test_disk_full_stays_while_errors_keep_coming(monkeypatch, tmp_path, clock):
    _patch_config(monkeypatch)
    _plenty_of_space(monkeypatch, tmp_path)
    monkeypatch.setattr(schema.psycopg, "connect", lambda url, **kw: _FakeProbeConnection())
    schema.note_db_error(_DISK_FULL_EXC)
    clock["now"] += schema._DISK_FULL_CLEAR_SECONDS - 10
    schema.note_db_error(_DISK_FULL_EXC)  # the volume is still refusing writes
    clock["now"] += 20
    assert schema.probe_db_status()["status"] == "disk_full"


def test_disk_full_is_kept_while_the_server_is_down(monkeypatch, tmp_path, clock):
    """After a WAL PANIC the postmaster is gone: the probe cannot connect, and
    'disk_full; Connection refused' explains that better than 'unreachable'."""
    _patch_config(monkeypatch)
    _plenty_of_space(monkeypatch, tmp_path)

    def refused(url, **kw):
        raise psycopg.OperationalError("connection failed: Connection refused")

    monkeypatch.setattr(schema.psycopg, "connect", refused)
    schema.note_db_error(_DISK_FULL_EXC)
    clock["now"] += schema._DISK_FULL_CLEAR_SECONDS + 30
    status = schema.probe_db_status()
    assert status["status"] == "disk_full"
    assert "Connection refused" in status["error"]


def test_remote_disk_full_clears_only_after_a_committed_write(monkeypatch, clock):
    """No free-space reading for a remote database: a write that committed
    after the disk-full error is the evidence instead."""
    _patch_config(monkeypatch, url="postgresql://u:p@db.internal.example:5432/jarvis")
    monkeypatch.setattr(schema.psycopg, "connect", lambda url, **kw: _FakeProbeConnection())
    schema.note_db_write_ok()  # before the error: proves nothing
    clock["now"] += 1
    schema.note_db_error(_DISK_FULL_EXC)

    clock["now"] += schema._DISK_FULL_CLEAR_SECONDS + 1
    assert schema.probe_db_status()["status"] == "disk_full"

    schema.note_db_write_ok()
    clock["now"] += 1
    assert schema.probe_db_status()["status"] == "ok"


def test_committed_writes_are_recorded(monkeypatch, clock):
    class _Cursor:
        rowcount = 2
        description = None

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def execute(self, sql, params=None):
            pass

        def executemany(self, sql, params_list):
            pass

    class _Conn:
        def cursor(self):
            return _Cursor()

        def commit(self):
            pass

    @contextlib.contextmanager
    def checkout(pool, conn_timeout):
        yield _Conn()

    monkeypatch.setattr(schema, "_get_pool", lambda: object())
    monkeypatch.setattr(schema, "_checkout", checkout)
    assert schema._last_write_ok_at is None
    schema.execute_write("UPDATE local.meta SET value = value")
    assert schema._last_write_ok_at == clock["now"]
    clock["now"] += 5
    assert schema.execute_batch("UPDATE local.meta SET value = %s", [("x",), ("y",)]) == 2
    assert schema._last_write_ok_at == clock["now"]


def test_disk_full_body_error_is_noted_by_pool(monkeypatch, caplog):
    healthy = SimpleNamespace(broken=False)

    @contextlib.contextmanager
    def connection(self, timeout=None):
        yield healthy

    monkeypatch.setattr(psycopg_pool.ConnectionPool, "connection", connection)
    with caplog.at_level(logging.CRITICAL, logger="jarvis-core"):
        with pytest.raises(psycopg.errors.DiskFull):
            with _unopened_pool().connection():
                raise psycopg.errors.DiskFull("No space left on device")
    assert any("DISK FULL" in r.getMessage() for r in caplog.records)
    assert schema.get_db_status()["status"] == "disk_full"


# ── Classification and sanitizing ─────────────────────────────────────


@pytest.mark.parametrize(
    "exc, expected",
    [
        (schema.DatabaseUnavailable("down"), True),
        (psycopg_pool.PoolTimeout("timeout"), True),
        (psycopg_pool.TooManyRequests("busy"), True),
        (psycopg.OperationalError("server closed the connection unexpectedly"), True),
        (psycopg.errors.DiskFull("No space left on device"), True),
        (psycopg.errors.CannotConnectNow("in recovery mode"), True),
        (psycopg.errors.AdminShutdown("terminating connection"), True),
        (psycopg.errors.CheckViolation("violates check constraint"), False),
        (psycopg.errors.ProgramLimitExceeded("index row size exceeds maximum"), False),
        (ValueError("bad input"), False),
    ],
)
def test_is_db_unavailable_error(exc, expected):
    assert schema.is_db_unavailable_error(exc) is expected


def test_safe_db_error_redacts_credentials():
    msg = schema.safe_db_error(
        "connection failed: could not connect to postgresql://jarvis:s3cr3t@h:5432/db\n"
        "   password=hunter2 host=h"
    )
    assert "s3cr3t" not in msg and "hunter2" not in msg
    assert "postgresql://jarvis:***@h:5432/db" in msg
    assert "password=***" in msg
    assert "\n" not in msg
    assert not msg.startswith("connection failed")
    assert len(schema.safe_db_error("x" * 5000)) == 300


def test_reset_pool_resets_breaker():
    schema._trip_breaker("down")
    schema.reset_pool()
    assert schema.db_available() is True
    assert schema.get_db_status()["status"] == "unknown"
