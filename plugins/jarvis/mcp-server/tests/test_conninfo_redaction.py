"""Connection-string passwords never reach clients or logs (review SEC-1, SEC-4).

libpq's conninfo parser quotes the offending component verbatim. With an
unencoded '%' or a space in the password that component is the password, and
the outage work routed connect errors into the unauthenticated /health, the
hook bodies (injected into every prompt), MCP errors and the log. A key/value
DSN has no '@', so the pool-creation log line printed it whole.
"""

from __future__ import annotations

import asyncio
import json
import logging

import psycopg
import pytest

import tools.schema as schema

LEAKY_URLS = {
    "percent": ("postgresql://jarvis:Gen%Pass-LEAKME@127.0.0.1:1/jarvis", "Gen%Pass-LEAKME"),
    "space": ("postgresql://jarvis:ab cSECRETX@127.0.0.1:1/jarvis", "cSECRETX"),
    "trailing-percent": ("postgresql://jarvis:SECRETXX%@127.0.0.1:1/jarvis", "SECRETXX"),
    "kv-space": ("host=127.0.0.1 port=1 user=jarvis password=KV SECRETKV dbname=jarvis", "SECRETKV"),
}


@pytest.fixture(autouse=True)
def clean_state(monkeypatch):
    schema.reset_breaker()
    monkeypatch.setattr(schema, "_known_secrets", set())
    yield
    schema.reset_pool()
    schema.reset_breaker()


def _connect_error(url: str) -> Exception:
    try:
        psycopg.connect(url, connect_timeout=1)
    except Exception as exc:
        return exc
    raise AssertionError("connect unexpectedly succeeded")


def _patch_config(monkeypatch, url: str) -> None:
    import tools.config as config

    monkeypatch.setattr(config, "get_postgres_config", lambda: {"url": url})
    monkeypatch.setattr(config, "get_embedding_config", lambda: {"dimensions": 384})
    monkeypatch.setattr(config, "get_memory_config", lambda: {"pool_timeout_seconds": 0.5})


# ── safe_db_error ───────────────────────────────────────────────────────


@pytest.mark.parametrize("case", sorted(LEAKY_URLS))
def test_conninfo_parse_errors_collapse_to_a_fixed_message(case):
    url, secret = LEAKY_URLS[case]
    exc = _connect_error(url)
    assert secret in str(exc)  # libpq really does quote it
    assert schema.safe_db_error(exc) == schema.INVALID_CONNINFO_MESSAGE


def test_ordinary_db_errors_are_unchanged():
    msg = schema.safe_db_error(
        'connection failed: connection to server at "127.0.0.1", port 5432 failed: '
        "FATAL:  the database system is in recovery mode"
    )
    assert msg == "the database system is in recovery mode"


def test_registered_password_is_scrubbed_raw_and_decoded():
    schema.register_conninfo_secret("postgresql://jarvis:s3cr%40t-word@db:5432/jarvis")
    out = schema.safe_db_error("weird driver text quoting s3cr%40t-word and s3cr@t-word")
    assert "s3cr" not in out
    assert out.count("***") == 2


def test_key_value_password_is_registered():
    schema.register_conninfo_secret("host=db user=jarvis password='kv secret 1' dbname=jarvis")
    assert "kv secret 1" not in schema.safe_db_error("echo: kv secret 1")


def test_password_equal_to_user_or_short_is_not_registered():
    """Scrubbing "jarvis" would mangle every message naming the user or DB."""
    schema.register_conninfo_secret("postgresql://jarvis:jarvis@db/jarvis")
    schema.register_conninfo_secret("postgresql://jarvis:pw@db/jarvis")
    assert schema._known_secrets == set()
    assert schema.safe_db_error('database "jarvis" does not exist') == 'database "jarvis" does not exist'


# ── display_conninfo / pool-creation log ────────────────────────────────


def test_display_conninfo_never_shows_the_password():
    assert schema.display_conninfo(
        "host=127.0.0.1 port=1 user=jarvis password=KVSECRET123 dbname=jarvis"
    ) == "127.0.0.1:1/jarvis"
    assert schema.display_conninfo("postgresql://u:secret@db:5433/mem?sslmode=disable") == "db:5433/mem"
    assert schema.display_conninfo(LEAKY_URLS["percent"][0]) == "(unparseable connection string)"


def test_pool_creation_log_redacts_key_value_dsn(monkeypatch, caplog):
    url = "host=127.0.0.1 port=1 user=jarvis password=KVSECRET123 dbname=jarvis"
    _patch_config(monkeypatch, url)

    class FakePool:
        def __init__(self, **kwargs):
            pass

        def close(self):
            pass

    monkeypatch.setattr(schema, "_JarvisPool", FakePool)
    monkeypatch.setattr(schema, "_pool", None)
    monkeypatch.setattr(schema, "_pool_cache_key", None)
    with caplog.at_level(logging.INFO, logger="jarvis-core"):
        schema._get_pool()
    assert "KVSECRET123" not in caplog.text
    assert "PostgreSQL connection pool created for 127.0.0.1:1/jarvis" in caplog.text


# ── End to end: a leaky DSN through the pool, probe, /health and hooks ──


def test_leaky_dsn_never_reaches_status_breaker_or_logs(monkeypatch, caplog):
    url, secret = LEAKY_URLS["percent"]
    _patch_config(monkeypatch, url)
    caplog.set_level(logging.DEBUG)

    status = schema.probe_db_status()
    assert status["status"] == "unreachable"
    assert status["error"] == schema.INVALID_CONNINFO_MESSAGE
    assert schema.db_available() is False

    schema.reset_breaker()
    pool = schema._get_pool()
    with pytest.raises(schema.DatabaseUnavailable) as exc:
        with pool.connection(timeout=0.5):
            pass
    assert secret not in str(exc.value)
    assert secret not in schema.db_unavailable_reason()
    assert secret not in caplog.text, caplog.text


def test_leaky_dsn_never_reaches_health_or_hook_bodies(monkeypatch):
    pytest.importorskip("mcp.server.streamable_http_manager")
    import http_app

    url, secret = LEAKY_URLS["space"]
    _patch_config(monkeypatch, url)
    monkeypatch.setattr(http_app, "authenticate", lambda scope: ("tester", ""))
    schema.probe_db_status()  # the background probe's verdict, cached

    async def call(method, path, body=None):
        raw = json.dumps(body).encode() if body is not None else b""
        inbox = [{"type": "http.request", "body": raw, "more_body": False}]
        sent = []

        async def receive():
            if inbox:
                return inbox.pop(0)
            await asyncio.Event().wait()

        async def send(message):
            sent.append(message)

        scope = {
            "type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1",
            "method": method, "scheme": "http", "path": path, "raw_path": path.encode(),
            "query_string": b"", "headers": [], "client": ("127.0.0.1", 1),
            "server": ("127.0.0.1", 8741),
        }
        await http_app.app(scope, receive, send)
        return b"".join(m.get("body", b"") for m in sent if m["type"] == "http.response.body")

    async def main():
        return await asyncio.wait_for(asyncio.gather(
            call("GET", "/health"),
            call("POST", "/hook/prompt-context", {"prompt": "what did we decide?"}),
            call("POST", "/hook/auto-extract/ingest", {"observations": [{"content": "x"}]}),
        ), 10)

    try:
        bodies = asyncio.run(main())
    finally:
        http_app._shutdown_hook_executor()
    for body in bodies:
        assert secret.encode() not in body
    assert json.loads(bodies[0])["postgres"]["error"] == schema.INVALID_CONNINFO_MESSAGE
