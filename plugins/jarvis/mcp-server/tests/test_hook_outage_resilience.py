"""Hook clients during a core/DB outage: ingest queue backoff, cap, dedupe,
permanent drops, and the core-degraded short-circuit in both hooks.

Queue-logic tests stub extract_observation.post_json; the end-to-end tests
run the real HTTP client against a local fake core (127.0.0.1, ephemeral
port). JARVIS_HOME always points at tmp_path.
"""

import io
import json
import sys
import time
from pathlib import Path

import pytest

from tests.fake_core_server import fake_core, hook_env  # noqa: F401 (fixtures)

import context_enrichment
import extract_observation
from extract_observation import (
    enqueue_ingest_payload,
    ingest_backoff_seconds,
    replay_ingest_queue,
    read_watermark,
)
from hook_http_client import core_degraded_marker_path, is_core_degraded, mark_core_degraded

INGEST = "/hook/auto-extract/ingest"
CONTEXT = "/hook/auto-extract/context"
PROMPT = "/hook/prompt-context"

RETRYABLE_503 = {"success": False, "retryable": True, "error": "database unavailable: the database system is in recovery mode"}


def _payload(*event_ids, content="obs"):
    return {
        "observations": [
            {"content": f"{content}-{eid}", "ingest_event_id": eid} for eid in event_ids
        ],
        "worklog": None,
        "context": {},
    }


def _queue_path(jarvis_home: Path) -> Path:
    return jarvis_home / "state" / "auto_extract_ingest_queue.jsonl"


def _queue(jarvis_home: Path) -> list[dict]:
    path = _queue_path(jarvis_home)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _ts(value: str) -> float:
    return extract_observation._parse_queue_time(value)


def _make_due(jarvis_home: Path):
    """Move every entry's next_attempt_at into the past."""
    entries = _queue(jarvis_home)
    for entry in entries:
        entry["next_attempt_at"] = "2000-01-01T00:00:00Z"
    _queue_path(jarvis_home).write_text("".join(json.dumps(e) + "\n" for e in entries))


@pytest.fixture
def jarvis_home(tmp_path, monkeypatch):
    home = tmp_path / ".jarvis"
    monkeypatch.setenv("JARVIS_HOME", str(home))
    return home


def _stub_post(monkeypatch, responses):
    """Stub post_json with a list of responses (last one repeats); record calls."""
    calls = []

    def fake(endpoint_path, payload, timeout_seconds=2.5, mcp_json_path=None):
        calls.append((endpoint_path, payload))
        return responses[min(len(calls) - 1, len(responses) - 1)]

    monkeypatch.setattr(extract_observation, "post_json", fake)
    return calls


FAIL = {"success": False, "data": None, "error": "timed out", "retryable": True}
OK = {"success": True, "data": {"success": True}, "error": ""}
PERMANENT = {"success": False, "data": None, "error": "http_error:400:bad", "permanent": True}
SKIPPED = {"success": False, "data": None, "error": "core degraded: timed out", "retryable": True, "skipped": True}


# --- Backoff scheduling ---


class TestBackoff:
    def test_schedule_doubles_from_30s_to_30min_cap(self):
        assert [ingest_backoff_seconds(n) for n in range(9)] == [
            0, 30, 60, 120, 240, 480, 960, 1800, 1800,
        ]
        assert ingest_backoff_seconds(10_000) == 1800

    def test_enqueue_records_attempts_and_next_attempt(self, jarvis_home):
        now = time.time()
        enqueue_ingest_payload(_payload("a"), attempts=1, last_error="timed out")
        enqueue_ingest_payload(_payload("b"))
        first, second = _queue(jarvis_home)
        assert first["attempts"] == 1
        assert abs(_ts(first["next_attempt_at"]) - (now + 30)) <= 2
        assert first["last_error"] == "timed out"
        assert second["attempts"] == 0
        assert abs(_ts(second["next_attempt_at"]) - now) <= 2
        assert "enqueued_at" in first and first["payload"] == _payload("a")

    def test_failed_replay_reschedules_head_only(self, jarvis_home, monkeypatch):
        enqueue_ingest_payload(_payload("a"))
        enqueue_ingest_payload(_payload("b"))
        calls = _stub_post(monkeypatch, [FAIL])
        now = time.time()

        assert replay_ingest_queue() == (0, True)
        assert len(calls) == 1
        head, tail = _queue(jarvis_home)
        assert head["attempts"] == 1
        assert abs(_ts(head["next_attempt_at"]) - (now + 30)) <= 2
        assert head["last_error"] == "timed out"
        assert tail["attempts"] == 0

    def test_backoff_grows_across_consecutive_failures(self, jarvis_home, monkeypatch):
        enqueue_ingest_payload(_payload("a"))
        _stub_post(monkeypatch, [FAIL])
        for expected_attempts, expected_delay in [(1, 30), (2, 60), (3, 120), (4, 240)]:
            _make_due(jarvis_home)
            now = time.time()
            replay_ingest_queue()
            (entry,) = _queue(jarvis_home)
            assert entry["attempts"] == expected_attempts
            assert abs(_ts(entry["next_attempt_at"]) - (now + expected_delay)) <= 2

    def test_entries_not_yet_due_are_skipped(self, jarvis_home, monkeypatch):
        """A backed-off head does not block due entries behind it."""
        enqueue_ingest_payload(_payload("a"), attempts=3)  # due in 120s
        enqueue_ingest_payload(_payload("b"))
        calls = _stub_post(monkeypatch, [OK])

        assert replay_ingest_queue() == (1, False)
        assert [c[1] for c in calls] == [_payload("b")]
        assert [e["payload"] for e in _queue(jarvis_home)] == [_payload("a")]

    def test_nothing_due_sends_nothing(self, jarvis_home, monkeypatch):
        enqueue_ingest_payload(_payload("a"), attempts=1)
        calls = _stub_post(monkeypatch, [OK])
        assert replay_ingest_queue() == (0, False)
        assert calls == []
        assert len(_queue(jarvis_home)) == 1

    def test_legacy_entry_without_schedule_is_due(self, jarvis_home, monkeypatch):
        path = _queue_path(jarvis_home)
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps({"enqueued_at": "2026-09-24T17:05:06Z", "payload": _payload("old")}) + "\n")
        calls = _stub_post(monkeypatch, [OK])
        assert replay_ingest_queue() == (1, False)
        assert len(calls) == 1
        assert _queue(jarvis_home) == []

    def test_skipped_request_stops_without_counting_attempt(self, jarvis_home, monkeypatch):
        enqueue_ingest_payload(_payload("a"))
        _stub_post(monkeypatch, [SKIPPED])
        assert replay_ingest_queue() == (0, True)
        (entry,) = _queue(jarvis_home)
        assert entry["attempts"] == 0

    def test_batch_limit_bounds_requests(self, jarvis_home, monkeypatch):
        for i in range(5):
            enqueue_ingest_payload(_payload(f"e{i}"))
        calls = _stub_post(monkeypatch, [OK])
        assert replay_ingest_queue(batch_limit=2) == (2, False)
        assert len(calls) == 2
        assert len(_queue(jarvis_home)) == 3

    def test_replay_keeps_entries_enqueued_concurrently(self, jarvis_home, monkeypatch):
        """Replay posts outside the lock and merges, so a parallel Stop hook's
        enqueue is not overwritten."""
        enqueue_ingest_payload(_payload("a"))

        def fake(endpoint_path, payload, timeout_seconds=2.5, mcp_json_path=None):
            enqueue_ingest_payload(_payload("late"))
            return OK

        monkeypatch.setattr(extract_observation, "post_json", fake)
        assert replay_ingest_queue() == (1, False)
        assert [e["payload"] for e in _queue(jarvis_home)] == [_payload("late")]


# --- Permanent rejection ---


class TestPermanentDrop:
    def test_400_is_dropped_and_replay_continues(self, jarvis_home, monkeypatch, capsys):
        enqueue_ingest_payload(_payload("poison"))
        enqueue_ingest_payload(_payload("good"))
        calls = _stub_post(monkeypatch, [PERMANENT, OK])

        assert replay_ingest_queue() == (1, False)
        assert len(calls) == 2
        assert _queue(jarvis_home) == []
        assert "WARNING: dropped queued ingest payload" in capsys.readouterr().err


# --- Cap and dedupe ---


class TestCapAndDedupe:
    def test_default_cap_is_500(self):
        assert extract_observation.INGEST_QUEUE_MAX_ENTRIES == 500

    def test_cap_drops_oldest_with_warning(self, jarvis_home, monkeypatch, capsys):
        monkeypatch.setattr(extract_observation, "INGEST_QUEUE_MAX_ENTRIES", 3)
        for i in range(5):
            enqueue_ingest_payload(_payload(f"e{i}"))
        assert [e["payload"] for e in _queue(jarvis_home)] == [
            _payload("e2"), _payload("e3"), _payload("e4"),
        ]
        err = capsys.readouterr().err
        assert err.count("WARNING: ingest queue over 3 entries, dropped 1 oldest") == 2

    def test_identical_payload_is_not_queued_twice(self, jarvis_home):
        assert enqueue_ingest_payload(_payload("a", "b")) is True
        assert enqueue_ingest_payload(_payload("a", "b")) is False
        assert len(_queue(jarvis_home)) == 1

    def test_dedupe_is_by_ingest_event_id_not_content(self, jarvis_home):
        assert enqueue_ingest_payload(_payload("a", "b", content="first")) is True
        # Same ids, different wording: the server dedupes by id anyway.
        assert enqueue_ingest_payload(_payload("b", content="reworded")) is False
        # A new id is new work.
        assert enqueue_ingest_payload(_payload("b", "c")) is True
        assert len(_queue(jarvis_home)) == 2

    def test_worklog_ids_count(self, jarvis_home):
        payload = {"observations": [], "worklog": {"task_summary": "x", "ingest_event_id": "wl-1"}}
        assert enqueue_ingest_payload(payload) is True
        assert enqueue_ingest_payload(dict(payload)) is False

    def test_payload_without_ids_dedupes_by_content(self, jarvis_home):
        payload = {"observations": [{"content": "no id"}], "worklog": None}
        assert enqueue_ingest_payload(payload) is True
        assert enqueue_ingest_payload(json.loads(json.dumps(payload))) is False
        assert enqueue_ingest_payload({"observations": [{"content": "other"}]}) is True

    def test_queue_rewrite_is_atomic(self, jarvis_home):
        enqueue_ingest_payload(_payload("a"))
        leftovers = [p.name for p in (jarvis_home / "state").iterdir() if p.suffix == ".tmp"]
        assert leftovers == []


# --- Real HTTP client against a fake core ---


class TestRealTransport:
    def test_503_retryable_is_requeued_and_trips_breaker(self, hook_env, fake_core):
        fake_core.respond("POST", INGEST, 503, RETRYABLE_503)
        enqueue_ingest_payload(_payload("a"))
        assert replay_ingest_queue() == (0, True)
        (entry,) = _queue(hook_env)
        assert entry["attempts"] == 1
        assert "recovery mode" in entry["last_error"]
        assert is_core_degraded()

    def test_200_retryable_body_is_requeued(self, hook_env, fake_core):
        fake_core.respond("POST", INGEST, 200, {"retryable": True, "error": "database unavailable"})
        enqueue_ingest_payload(_payload("a"))
        assert replay_ingest_queue() == (0, True)
        assert _queue(hook_env)[0]["attempts"] == 1

    def test_400_is_dropped(self, hook_env, fake_core):
        fake_core.respond("POST", INGEST, 400, {"success": False, "error": "'observations' must be a list"})
        enqueue_ingest_payload(_payload("a"))
        assert replay_ingest_queue() == (0, False)
        assert _queue(hook_env) == []
        assert not is_core_degraded()


# --- extract_observation.main() end to end ---


def _transcript(tmp_path: Path) -> str:
    path = tmp_path / "transcript.jsonl"
    path.write_text(
        json.dumps({"type": "user", "message": {"content": [{"type": "text", "text": "Please harden the hook clients against a database outage."}]}})
        + "\n"
        + json.dumps({"type": "assistant", "message": {"content": [{"type": "text", "text": "Added per-entry exponential backoff, a queue cap with ingest_event_id dedupe, and a shared core-degraded marker so every hook in every session skips the network while core is down."}], "usage": {"input_tokens": 10, "output_tokens": 10}}})
        + "\n"
    )
    return str(path)


CONTEXT_OK = {
    "success": True,
    "auto_extract": {"min_turn_chars": 10, "max_observations": 3, "debug": False},
    "worklog": {"enabled": True, "dedup_threshold": 0.7},
    "known_workstreams": [],
}


@pytest.fixture
def extract_main(tmp_path, monkeypatch, hook_env):
    monkeypatch.setattr(extract_observation, "WATERMARK_DIR", tmp_path / "sessions")
    monkeypatch.delenv("JARVIS_HOOK_INPUT", raising=False)
    monkeypatch.setattr(
        extract_observation,
        "call_haiku",
        lambda prompt, mode="background": (
            {
                "observations": [{"content": "User wants hooks to back off during outages.", "importance_score": 0.6, "tags": ["hooks"], "scope": "project"}],
                "worklog": {"task_summary": "Hardening hook clients", "workstream": "Jarvis", "activity_type": "coding", "tags": []},
            },
            100, 50, "API",
        ),
    )
    transcript = _transcript(tmp_path)

    def run(session="sess-1"):
        monkeypatch.setattr(sys, "argv", ["extract_observation.py", "background", transcript, session, str(tmp_path), "main"])
        try:
            extract_observation.main()
        except SystemExit as exc:
            assert exc.code in (0, None)

    return run


class TestExtractMain:
    def test_fresh_marker_queues_without_network(self, extract_main, hook_env, fake_core):
        enqueue_ingest_payload(_payload("queued-earlier"))
        mark_core_degraded("timed out")

        extract_main()

        assert fake_core.requests == []
        queue = _queue(hook_env)
        assert len(queue) == 2
        assert queue[0]["attempts"] == 0  # replay was skipped, not attempted
        assert queue[1]["attempts"] == 0  # new payload was never sent
        assert queue[1]["last_error"].startswith("core degraded")
        assert read_watermark("sess-1") >= 1

    def test_ingest_503_requeues_with_backoff(self, extract_main, hook_env, fake_core):
        fake_core.respond("POST", CONTEXT, 200, CONTEXT_OK)
        fake_core.respond("POST", INGEST, 503, RETRYABLE_503)
        now = time.time()

        extract_main()

        assert fake_core.paths() == [CONTEXT, INGEST]
        (entry,) = _queue(hook_env)
        assert entry["attempts"] == 1
        assert abs(_ts(entry["next_attempt_at"]) - (now + 30)) <= 3
        assert is_core_degraded()

    def test_context_503_short_circuits_ingest(self, extract_main, hook_env, fake_core):
        fake_core.respond("POST", CONTEXT, 503, RETRYABLE_503)
        fake_core.respond("POST", INGEST, 200, {"success": True})

        extract_main()

        assert fake_core.paths() == [CONTEXT]
        (entry,) = _queue(hook_env)
        assert entry["attempts"] == 0

    def test_degraded_context_body_short_circuits_ingest(self, extract_main, hook_env, fake_core):
        fake_core.respond("POST", CONTEXT, 200, {**CONTEXT_OK, "degraded": True})
        fake_core.respond("POST", INGEST, 200, {"success": True})

        extract_main()

        assert fake_core.paths() == [CONTEXT]
        assert len(_queue(hook_env)) == 1

    def test_failed_replay_skips_context_and_ingest(self, extract_main, hook_env, fake_core):
        fake_core.respond("POST", INGEST, 503, RETRYABLE_503)
        fake_core.respond("POST", CONTEXT, 200, CONTEXT_OK)
        enqueue_ingest_payload(_payload("queued-earlier"))

        extract_main()

        assert fake_core.paths() == [INGEST]  # the one replay, nothing else
        head, new = _queue(hook_env)
        assert head["attempts"] == 1
        assert new["attempts"] == 0

    def test_ingest_400_is_dropped_not_queued(self, extract_main, hook_env, fake_core, capsys):
        fake_core.respond("POST", CONTEXT, 200, CONTEXT_OK)
        fake_core.respond("POST", INGEST, 400, {"success": False, "error": "'context' must be an object"})

        extract_main()

        assert _queue(hook_env) == []
        assert "WARNING: ingest rejected payload, dropped" in capsys.readouterr().err
        assert not is_core_degraded()

    def test_healthy_core_delivers_and_drains_queue(self, extract_main, hook_env, fake_core):
        fake_core.respond("POST", CONTEXT, 200, CONTEXT_OK)
        fake_core.respond("POST", INGEST, 200, {"success": True, "observations": [{"status": "stored", "id": "obs-1"}]})
        enqueue_ingest_payload(_payload("queued-earlier"))

        extract_main()

        assert fake_core.paths() == [INGEST, CONTEXT, INGEST]
        assert _queue(hook_env) == []
        assert not core_degraded_marker_path().exists()


# --- context_enrichment.main() end to end ---


@pytest.fixture
def enrichment_main(tmp_path, monkeypatch, hook_env):
    import precompact_dedup

    monkeypatch.setattr(precompact_dedup, "STATE_DIR", tmp_path / "sessions")
    monkeypatch.setattr(context_enrichment, "TELEMETRY_FILE", tmp_path / "telemetry.jsonl")
    monkeypatch.setattr(context_enrichment, "DEBUG_LOG_FILE", tmp_path / "debug.log")

    def run(capsys, prompt="How should the hooks behave while Postgres is recovering?"):
        monkeypatch.setattr(sys, "argv", ["context_enrichment.py", prompt])
        monkeypatch.setattr(sys, "stdin", io.StringIO(""))
        with pytest.raises(SystemExit) as exc:
            context_enrichment.main()
        assert exc.value.code == 0
        return capsys.readouterr().out

    return run


PROMPT_OK = {
    "success": True,
    "enabled": True,
    "debug": False,
    "matches": [],
    "query_ms": 3,
    "budget_used": {"local": 0, "vault": 0, "remote": 0},
    "todoist_prompt_alerts": {"enabled": False, "max_per_category": 3},
}


class TestContextEnrichment:
    def test_fresh_marker_fails_fast_with_warning(self, enrichment_main, fake_core, capsys):
        fake_core.respond("POST", PROMPT, 200, PROMPT_OK)
        mark_core_degraded("timed out")
        start = time.monotonic()
        out = enrichment_main(capsys)
        assert time.monotonic() - start < 0.5
        assert fake_core.requests == []
        assert '<jarvis-warning type="memory-unavailable">' in out
        assert "core degraded: timed out" in out

    def test_503_warns_and_silences_next_prompt(self, enrichment_main, fake_core, capsys):
        fake_core.respond("POST", PROMPT, 503, RETRYABLE_503)
        first = enrichment_main(capsys)
        second = enrichment_main(capsys)
        assert fake_core.paths() == [PROMPT]
        assert "memory-unavailable" in first and "memory-unavailable" in second
        assert is_core_degraded()

    def test_degraded_body_warns(self, enrichment_main, fake_core, capsys):
        fake_core.respond("POST", PROMPT, 200, {**PROMPT_OK, "degraded": True})
        out = enrichment_main(capsys)
        assert '<jarvis-warning type="memory-unavailable">' in out
        assert "database unavailable" in out
        assert is_core_degraded()

    def test_warning_escapes_server_text(self, enrichment_main, fake_core, capsys):
        fake_core.respond("POST", PROMPT, 200, {**PROMPT_OK, "degraded": True, "error": "<b>down</b>"})
        out = enrichment_main(capsys)
        assert "&lt;b&gt;down&lt;/b&gt;" in out
        assert "<b>" not in out

    def test_healthy_empty_result_stays_silent(self, enrichment_main, fake_core, capsys):
        fake_core.respond("POST", PROMPT, 200, PROMPT_OK)
        assert enrichment_main(capsys) == ""
        assert not core_degraded_marker_path().exists()

    def test_slow_delivery_ack_does_not_trip_breaker(self, enrichment_main, fake_core, capsys):
        trace = "11111111-1111-1111-1111-111111111111"
        fake_core.respond(
            "POST", PROMPT, 200,
            {**PROMPT_OK, "trace_id": trace, "matches": [{"content": "memory", "id": "a.md", "relevance": 0.9, "type": "note", "candidate_key": "k"}]},
        )
        fake_core.respond("PUT", f"/telemetry/retrieval/{trace}/delivery", 200, {"success": True}, delay=1.0)
        out = enrichment_main(capsys)
        assert "relevant-vault-memories" in out
        assert not core_degraded_marker_path().exists()
