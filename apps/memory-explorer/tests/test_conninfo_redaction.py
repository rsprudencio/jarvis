"""The Explorer never echoes a connection-string password (review SEC-1).

libpq's conninfo parser quotes the offending component verbatim. With an
unencoded '%' or a space in the password that component is the password, and
the Explorer's unauthenticated /health and its 503 details carried it.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from pathlib import Path

import httpx
import psycopg
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import app as app_module  # noqa: E402
import tools.schema as core_schema  # noqa: E402

_CASES = {
    "percent": ("postgresql://jarvis:Gen%Pass-LEAKME@127.0.0.1:1/jarvis", "Gen%Pass-LEAKME"),
    "space": ("postgresql://jarvis:ab cSECRETX@127.0.0.1:1/jarvis", "cSECRETX"),
}


@pytest.fixture(autouse=True)
def _isolate(tmp_path, monkeypatch):
    monkeypatch.setenv("JARVIS_HOME", str(tmp_path))
    monkeypatch.setenv("PGDATA", str(tmp_path))
    (tmp_path / "config.json").write_text(json.dumps({"memory": {}}))
    from jarvis_common.config import clear_config_cache
    clear_config_cache()
    monkeypatch.setattr(core_schema, "_known_secrets", set())
    monkeypatch.setattr(app_module, "_db_status", {
        "status": "unknown", "error": None, "checked_at": None, "free_bytes": None})
    monkeypatch.setattr(app_module, "_last_disk_full_at", None)
    yield
    clear_config_cache()


def _connect_error(url: str) -> Exception:
    try:
        psycopg.connect(url, connect_timeout=1)
    except Exception as exc:
        return exc
    raise AssertionError("connect unexpectedly succeeded")


@pytest.mark.parametrize("case", sorted(_CASES))
def test_safe_err_and_reason_hide_conninfo_parse_errors(case):
    url, secret = _CASES[case]
    exc = _connect_error(url)
    assert secret in str(exc)
    assert app_module._safe_err(exc) == core_schema.INVALID_CONNINFO_MESSAGE
    assert app_module._db_err_reason(exc) == core_schema.INVALID_CONNINFO_MESSAGE


@pytest.mark.parametrize("case", sorted(_CASES))
def test_probe_and_health_never_carry_the_password(case, monkeypatch, caplog):
    url, secret = _CASES[case]
    monkeypatch.setenv("POSTGRES_URL", url)
    caplog.set_level(logging.DEBUG)
    status = app_module._probe_db_status()
    assert status["status"] == "unreachable"
    assert status["error"] == core_schema.INVALID_CONNINFO_MESSAGE

    async def main():
        transport = httpx.ASGITransport(app=app_module.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://t", timeout=10) as c:
            return await c.get("/health")

    health = asyncio.run(main())
    assert health.status_code == 200
    assert secret not in health.text
    assert secret not in caplog.text


def test_registered_password_is_scrubbed_from_other_errors(monkeypatch):
    monkeypatch.setenv("POSTGRES_URL", "postgresql://jarvis:Sup3rSecret@127.0.0.1:1/jarvis")
    app_module._probe_db_status()  # registers the configured password
    out = app_module._safe_err(RuntimeError("driver said Sup3rSecret somewhere"))
    assert "Sup3rSecret" not in out
