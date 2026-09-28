"""Tests for hooks-handlers/hook_http_client.py."""

import json
import os
import sys
from pathlib import Path
from urllib import error

# Add hooks-handlers to import path
HOOKS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "hooks-handlers")
sys.path.insert(0, HOOKS_DIR)

import hook_http_client
import pytest


@pytest.fixture(autouse=True)
def _isolated_state_dir(tmp_path: Path, monkeypatch):
    """post_json reads/writes the core-degraded marker; keep it off ~/.jarvis."""
    monkeypatch.setenv("JARVIS_HOME", str(tmp_path / ".jarvis"))


def test_resolve_base_url_from_mcp_json(tmp_path: Path):
    """core.url ending in /mcp is normalized to server base URL."""
    mcp_json = tmp_path / ".mcp.json"
    mcp_json.write_text(json.dumps({"core": {"type": "http", "url": "http://localhost:8741/mcp"}}))

    base_url = hook_http_client.resolve_base_url(mcp_json)
    assert base_url == "http://localhost:8741"


def test_resolve_base_url_fallback_when_missing(tmp_path: Path):
    """Missing config falls back to localhost default."""
    missing = tmp_path / "missing.json"
    base_url = hook_http_client.resolve_base_url(missing)
    assert base_url == hook_http_client.DEFAULT_BASE_URL


def test_post_json_timeout_error_normalization(monkeypatch):
    """Transport errors return normalized failure contract."""
    def _raise(*args, **kwargs):
        raise error.URLError("timed out")

    monkeypatch.setattr(hook_http_client.request, "urlopen", _raise)
    result = hook_http_client.post_json(
        "/hook/prompt-context",
        {"prompt": "hello"},
        mcp_json_path="/nonexistent/path.json",
    )

    assert result["success"] is False
    assert result["data"] is None
    assert "timed out" in result["error"]


def test_post_json_successful_decode(monkeypatch):
    """Valid JSON response returns success with decoded dict."""
    class FakeResponse:
        def __init__(self, payload: bytes):
            self._payload = payload

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return self._payload

    def _fake_urlopen(req, timeout):
        payload = json.dumps({"success": True, "matches": []}).encode("utf-8")
        return FakeResponse(payload)

    monkeypatch.setattr(hook_http_client.request, "urlopen", _fake_urlopen)
    monkeypatch.setattr(
        hook_http_client,
        "resolve_base_url",
        lambda mcp_json_path=None: "http://localhost:8741",
    )

    result = hook_http_client.post_json("/hook/prompt-context", {"prompt": "test"})
    assert result["success"] is True
    assert result["error"] == ""
    assert result["data"]["matches"] == []
