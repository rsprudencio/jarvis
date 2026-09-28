"""Contract: the statusline parses what the real /health handler returns.

The statusline's DB indicator was dead code for months because its tests
mocked a /health shape the server no longer produced (de72931). This drives
the real http_app.health_response() and feeds its body to the statusline.
"""

import asyncio
import json
import os
import sys
from unittest import mock

import pytest

try:
    import http_app

    _HAS_HTTP_APP = True
except Exception:  # pragma: no cover - SDK missing outside the image
    _HAS_HTTP_APP = False

STATUSLINE_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "statusline")
sys.path.insert(0, STATUSLINE_DIR)

import statusline  # noqa: E402

pytestmark = pytest.mark.skipif(not _HAS_HTTP_APP, reason="http_app not importable")


def _health_body(monkeypatch, db_status: dict) -> str:
    monkeypatch.setattr("tools.schema.get_db_status", lambda: dict(db_status))
    sent = []

    async def send(message):
        sent.append(message)

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    scope = {"type": "http", "method": "GET", "path": "/health", "headers": []}
    asyncio.run(http_app.health_response(scope, receive, send))
    start = next(m for m in sent if m["type"] == "http.response.start")
    assert start["status"] == 200
    return b"".join(m.get("body", b"") for m in sent if m["type"] == "http.response.body").decode()


def _render(tmp_path, body: str) -> str:
    with mock.patch.object(statusline, "CACHE_DIR", tmp_path), \
         mock.patch.object(statusline, "_git_info", return_value={"branch": "", "dirty": False}), \
         mock.patch.object(statusline, "_account_name", return_value=""), \
         mock.patch.object(statusline.subprocess, "run",
                           return_value=mock.Mock(returncode=0, stdout=body)):
        return statusline.generate({"model": "test", "cwd": "/tmp/x"})


@pytest.mark.parametrize("status,expected", [
    ("recovering", "DB: recovering"),
    ("unreachable", "DB: unreachable"),
    ("disk_full", "DB: disk full (1.0M free)"),
    ("unknown", "DB: unknown"),
])
def test_not_ok_db_status_reaches_statusline(status, expected, monkeypatch, tmp_path):
    body = _health_body(monkeypatch, {
        "status": status,
        "error": "the database system is in recovery mode",
        "checked_at": 1790000000.0,
        "free_bytes": 1024 * 1024,
    })
    data = json.loads(body)
    assert data["status"] == "ok"
    assert data["postgres"]["status"] == status

    output = _render(tmp_path, body)
    assert "JARVIS" in output
    assert expected in output


def test_ok_db_status_renders_plain_branding(monkeypatch, tmp_path):
    body = _health_body(monkeypatch, {
        "status": "ok", "error": None, "checked_at": 1790000000.0, "free_bytes": 10 ** 11,
    })
    output = _render(tmp_path, body)
    assert "JARVIS" in output
    assert "DB:" not in output
