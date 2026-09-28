"""Hook HTTP client: delivery classification and the core-degraded breaker.

Runs the real hooks-handlers/hook_http_client.py against a local fake core
(127.0.0.1, ephemeral port) with JARVIS_HOME in tmp_path.
"""

import json
import os
import socket
import stat
import time

import pytest

from tests.fake_core_server import fake_core, hook_env  # noqa: F401 (fixtures)

import hook_http_client
from hook_http_client import (
    core_degraded_marker_path,
    core_degraded_reason,
    is_core_degraded,
    mark_core_degraded,
    post_json,
    put_json,
)

INGEST = "/hook/auto-extract/ingest"


def _closed_port() -> int:
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


class TestClassification:
    def test_success(self, hook_env, fake_core):
        fake_core.respond("POST", INGEST, 200, {"success": True, "observations": []})
        result = post_json(INGEST, {"observations": []})
        assert result["success"] is True
        assert result["http_status"] == 200
        assert result["retryable"] is False
        assert result["permanent"] is False
        assert result["skipped"] is False
        assert not core_degraded_marker_path().exists()

    def test_503_retryable_body_is_not_delivered(self, hook_env, fake_core):
        fake_core.respond(
            "POST", INGEST, 503,
            {"success": False, "retryable": True, "error": "database unavailable (timeout)"},
        )
        result = post_json(INGEST, {"observations": []})
        assert result["success"] is False
        assert result["retryable"] is True
        assert result["permanent"] is False
        assert result["http_status"] == 503
        assert result["error"] == "http_error:503:database unavailable (timeout)"
        assert "database unavailable" in core_degraded_reason()

    def test_200_retryable_body_is_not_delivered(self, hook_env, fake_core):
        """retryable=true means not delivered even without success:false."""
        fake_core.respond("POST", INGEST, 200, {"retryable": True, "error": "database unavailable"})
        result = post_json(INGEST, {"observations": []})
        assert result["success"] is False
        assert result["retryable"] is True
        assert result["data"] == {"retryable": True, "error": "database unavailable"}
        assert is_core_degraded()

    def test_200_success_false_is_a_plain_failure(self, hook_env, fake_core):
        fake_core.respond("POST", INGEST, 200, {"success": False, "error": "write failed"})
        result = post_json(INGEST, {"observations": []})
        assert result["success"] is False
        assert result["retryable"] is False
        assert result["permanent"] is False
        assert not is_core_degraded()

    @pytest.mark.parametrize("status", [400, 413, 422])
    def test_payload_rejection_is_permanent(self, status, hook_env, fake_core):
        fake_core.respond("POST", INGEST, status, {"success": False, "error": "'observations' must be a list"})
        result = post_json(INGEST, {"observations": "bad"})
        assert result["success"] is False
        assert result["permanent"] is True
        assert result["retryable"] is False
        assert result["error"].startswith(f"http_error:{status}:")
        assert not is_core_degraded()

    def test_500_is_neither_permanent_nor_degraded(self, hook_env, fake_core):
        fake_core.respond("POST", INGEST, 500, {"success": False, "error": "boom"})
        result = post_json(INGEST, {})
        assert result["success"] is False
        assert result["permanent"] is False
        assert result["retryable"] is False
        assert not is_core_degraded()

    @pytest.mark.parametrize("status", [502, 504])
    def test_gateway_errors_are_retryable(self, status, hook_env, fake_core):
        fake_core.respond("POST", INGEST, status, b"")
        result = post_json(INGEST, {})
        assert result["retryable"] is True
        assert is_core_degraded()

    def test_timeout_is_retryable_and_trips_breaker(self, hook_env, fake_core):
        fake_core.respond("POST", INGEST, 200, {"success": True}, delay=1.0)
        start = time.monotonic()
        result = post_json(INGEST, {}, timeout_seconds=0.2)
        assert time.monotonic() - start < 0.9
        assert result["success"] is False
        assert result["retryable"] is True
        assert result["http_status"] is None
        assert is_core_degraded()

    def test_connection_refused_is_retryable_and_trips_breaker(self, hook_env, monkeypatch):
        port = _closed_port()
        monkeypatch.setattr(
            hook_http_client, "resolve_base_url",
            lambda mcp_json_path=None: f"http://127.0.0.1:{port}",
        )
        result = post_json(INGEST, {}, timeout_seconds=1.0)
        assert result["success"] is False
        assert result["retryable"] is True
        assert is_core_degraded()

    def test_degraded_success_body_trips_breaker(self, hook_env, fake_core):
        fake_core.respond(
            "POST", "/hook/prompt-context", 200,
            {"success": True, "enabled": True, "degraded": True, "matches": []},
        )
        result = post_json("/hook/prompt-context", {"prompt": "hello there"})
        assert result["success"] is True
        assert core_degraded_reason() == "database unavailable"

    def test_trip_breaker_false_leaves_marker_alone(self, hook_env, fake_core):
        fake_core.respond("PUT", "/telemetry/retrieval/t/delivery", 200, {"success": True}, delay=1.0)
        result = put_json("/telemetry/retrieval/t/delivery", {}, timeout_seconds=0.2, trip_breaker=False)
        assert result["retryable"] is True
        assert not core_degraded_marker_path().exists()


class TestBreaker:
    def test_fresh_marker_skips_network(self, hook_env, fake_core):
        fake_core.respond("POST", INGEST, 200, {"success": True})
        mark_core_degraded("timed out")
        start = time.monotonic()
        result = post_json(INGEST, {"observations": []})
        assert time.monotonic() - start < 0.1
        assert fake_core.requests == []
        assert result["success"] is False
        assert result["skipped"] is True
        assert result["retryable"] is True
        assert result["error"] == "core degraded: timed out"

    def test_breaker_applies_to_put(self, hook_env, fake_core):
        mark_core_degraded("timed out")
        result = put_json("/telemetry/retrieval/t/delivery", {}, trip_breaker=False)
        assert result["skipped"] is True
        assert fake_core.requests == []

    def test_stale_marker_sends_request(self, hook_env, fake_core):
        fake_core.respond("POST", INGEST, 200, {"success": True})
        mark_core_degraded("timed out")
        past = time.time() - hook_http_client.CORE_DEGRADED_TTL_SECONDS - 1
        os.utime(core_degraded_marker_path(), (past, past))
        result = post_json(INGEST, {})
        assert result["success"] is True
        assert fake_core.paths() == [INGEST]

    def test_marker_far_in_future_is_stale(self, hook_env, fake_core):
        """A clock jump cannot silence the hooks beyond the TTL."""
        fake_core.respond("POST", INGEST, 200, {"success": True})
        mark_core_degraded("timed out")
        future = time.time() + 3600
        os.utime(core_degraded_marker_path(), (future, future))
        assert not is_core_degraded()
        assert post_json(INGEST, {})["success"] is True

    def test_ttl_is_sixty_seconds(self):
        assert hook_http_client.CORE_DEGRADED_TTL_SECONDS == 60.0

    def test_one_failure_silences_following_hooks(self, hook_env, fake_core):
        """The first 503 trips the breaker; the next calls never reach core."""
        fake_core.respond("POST", INGEST, 503, {"success": False, "retryable": True, "error": "database unavailable"})
        fake_core.respond("POST", "/hook/prompt-context", 200, {"success": True})
        assert post_json(INGEST, {})["retryable"] is True
        assert post_json("/hook/prompt-context", {"prompt": "x"})["skipped"] is True
        assert post_json(INGEST, {})["skipped"] is True
        assert fake_core.paths() == [INGEST]


class TestMarkerFile:
    def test_atomic_private_single_line(self, hook_env):
        mark_core_degraded("line one\nline two " + "x" * 500)
        path = core_degraded_marker_path()
        assert stat.S_IMODE(path.stat().st_mode) == 0o600
        data = json.loads(path.read_text())
        assert "\n" not in data["reason"]
        assert len(data["reason"]) == 200
        assert data["reason"].startswith("line one line two")
        assert [p.name for p in path.parent.iterdir()] == ["core_degraded"]

    def test_write_failure_never_raises(self, tmp_path, monkeypatch):
        blocker = tmp_path / "not-a-dir"
        blocker.write_text("")
        monkeypatch.setenv("JARVIS_HOME", str(blocker))
        mark_core_degraded("timed out")
        assert not is_core_degraded()

    def test_unreadable_marker_still_counts(self, hook_env):
        path = core_degraded_marker_path()
        path.parent.mkdir(parents=True)
        path.write_text("not json")
        assert core_degraded_reason() == "core unavailable"
