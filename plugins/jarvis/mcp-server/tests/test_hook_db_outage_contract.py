"""Hook-path behaviour while PostgreSQL is unavailable.

The 2026-09 outage showed two failure modes this file guards against:
- one ingest kept going after the first DB failure (8 x 30s pool waits);
- once failures are fast, ``success: True`` with per-item errors would make the
  hook client drop the payload as delivered.
Plus the retrieval janitor fixes that shipped with it.
"""

from __future__ import annotations

import asyncio
import logging
from types import SimpleNamespace
from unittest.mock import MagicMock

import psycopg
import pytest

import tools.hook_endpoints as hook_endpoints
import tools.retrieval_telemetry as telemetry
import tools.schema as schema


RECOVERY = "the database system is in recovery mode"


@pytest.fixture(autouse=True)
def clean_breaker():
    schema.reset_breaker()
    yield
    schema.reset_breaker()


def _payload(n_observations=2, worklog=True):
    payload = {
        "observations": [
            {"content": f"observation {i}", "ingest_event_id": f"obs:{i}"}
            for i in range(n_observations)
        ],
        "worklog": None,
        "context": {"session_id": "s-1"},
        "dedup": {"observation_threshold": 0.95, "worklog_threshold": 0.6},
    }
    if worklog:
        payload["worklog"] = {"task_summary": "Harden the pool", "ingest_event_id": "wl:1"}
    return payload


def _no_duplicates(monkeypatch):
    monkeypatch.setattr(
        hook_endpoints, "query_vault", lambda **kw: {"success": True, "results": []}
    )
    monkeypatch.setattr(
        hook_endpoints, "content_list", lambda **kw: {"success": True, "documents": []}
    )


# ── ingest_auto_extract ───────────────────────────────────────────────


def test_ingest_refuses_immediately_while_breaker_open(monkeypatch):
    schema._trip_breaker(RECOVERY)
    query = MagicMock()
    write = MagicMock()
    monkeypatch.setattr(hook_endpoints, "query_vault", query)
    monkeypatch.setattr(hook_endpoints, "content_write", write)

    result = hook_endpoints.ingest_auto_extract(_payload())

    assert result["success"] is False
    assert result["retryable"] is True
    assert result["error"] == f"database unavailable: {RECOVERY}"
    query.assert_not_called()
    write.assert_not_called()


def test_ingest_stops_at_first_dedup_failure(monkeypatch):
    query = MagicMock(return_value={
        "success": False, "retryable": True,
        "error": f"Database unavailable: {RECOVERY}",
    })
    write = MagicMock()
    content_list = MagicMock()
    monkeypatch.setattr(hook_endpoints, "query_vault", query)
    monkeypatch.setattr(hook_endpoints, "content_write", write)
    monkeypatch.setattr(hook_endpoints, "content_list", content_list)

    result = hook_endpoints.ingest_auto_extract(_payload(n_observations=3))

    assert result == {
        "success": False, "retryable": True,
        "error": f"database unavailable: {RECOVERY}",
        "observations": [], "worklog": None,
    }
    assert query.call_count == 1  # no second observation, no worklog
    write.assert_not_called()
    content_list.assert_not_called()


def test_ingest_stops_at_first_retryable_write_failure(monkeypatch):
    _no_duplicates(monkeypatch)
    calls = []

    def write(**kwargs):
        calls.append(kwargs["content"])
        if len(calls) == 1:
            return {"success": True, "id": "obs::1"}
        return {"success": False, "retryable": True, "error_kind": "db_unavailable",
                "error": "couldn't get a connection after 1.50 sec"}

    monkeypatch.setattr(hook_endpoints, "content_write", write)

    result = hook_endpoints.ingest_auto_extract(_payload(n_observations=3))

    assert result["success"] is False and result["retryable"] is True
    assert result["error"] == "database unavailable: couldn't get a connection after 1.50 sec"
    assert calls == ["observation 0", "observation 1"]  # stopped; no obs 2, no worklog
    assert result["observations"] == [{"status": "stored", "id": "obs::1", "error": ""}]


def test_ingest_model_host_outage_is_retryable(monkeypatch):
    _no_duplicates(monkeypatch)
    monkeypatch.setattr(hook_endpoints, "content_write", lambda **kw: {
        "success": False, "retryable": True, "error_kind": "model_host_unavailable",
        "error": "Model host request failed: timed out",
    })

    result = hook_endpoints.ingest_auto_extract(_payload(n_observations=1))

    assert result["retryable"] is True
    assert result["error"] == "embedding service unavailable: Model host request failed: timed out"


def test_ingest_worklog_dedup_failure_is_retryable(monkeypatch):
    monkeypatch.setattr(
        hook_endpoints, "query_vault", lambda **kw: {"success": True, "results": []}
    )
    monkeypatch.setattr(hook_endpoints, "content_list", lambda **kw: {
        "success": False, "retryable": True, "error": RECOVERY,
    })
    writes = []
    monkeypatch.setattr(
        hook_endpoints, "content_write",
        lambda **kw: writes.append(kw["content_type"]) or {"success": True, "id": "obs::1"},
    )

    result = hook_endpoints.ingest_auto_extract(_payload(n_observations=1))

    assert result["retryable"] is True
    assert writes == ["observation"]  # worklog never written blind


def test_ingest_permanent_failures_stay_per_item(monkeypatch):
    """Secret-scan / validation rejections must not jam the replay queue."""
    _no_duplicates(monkeypatch)
    results = iter([
        {"success": False, "error": "Secret detected in content", "detections": []},
        {"success": True, "id": "obs::2"},
        {"success": True, "id": "worklog::1"},
    ])
    monkeypatch.setattr(hook_endpoints, "content_write", lambda **kw: next(results))

    result = hook_endpoints.ingest_auto_extract(_payload(n_observations=2))

    assert result["success"] is True
    assert "retryable" not in result
    assert result["observations"][0] == {
        "status": "error", "id": "", "error": "Secret detected in content",
    }
    assert result["observations"][1]["status"] == "stored"
    assert result["worklog"]["status"] == "stored"


def test_ingest_db_work_runs_under_hook_checkout_timeout(monkeypatch):
    seen = []
    monkeypatch.setattr(hook_endpoints, "query_vault", lambda **kw: (
        seen.append(schema._checkout_timeout_var.get()) or {"success": True, "results": []}
    ))
    monkeypatch.setattr(hook_endpoints, "content_list", lambda **kw: (
        seen.append(schema._checkout_timeout_var.get()) or {"success": True, "documents": []}
    ))
    monkeypatch.setattr(hook_endpoints, "content_write", lambda **kw: (
        seen.append(schema._checkout_timeout_var.get()) or {"success": True, "id": "x"}
    ))

    hook_endpoints.ingest_auto_extract(_payload(n_observations=1))

    assert seen == [schema.HOOK_CONN_TIMEOUT] * 4
    assert schema._checkout_timeout_var.get() is None


def test_duplicate_check_propagates_db_unavailable(monkeypatch):
    monkeypatch.setattr(hook_endpoints, "query_vault", lambda **kw: {
        "success": False, "retryable": True, "error": f"Database unavailable: {RECOVERY}",
    })
    with pytest.raises(schema.DatabaseUnavailable, match="recovery mode"):
        hook_endpoints._is_duplicate_observation("content", 0.95)


def test_duplicate_check_non_db_failure_still_means_not_duplicate(monkeypatch):
    monkeypatch.setattr(hook_endpoints, "query_vault", lambda **kw: {
        "success": False, "error": "Query failed: Unknown schemas",
    })
    assert hook_endpoints._is_duplicate_observation("content", 0.95) is False


# ── context endpoints ─────────────────────────────────────────────────


def _enrichment_enabled(monkeypatch):
    monkeypatch.setattr(
        hook_endpoints, "get_context_enrichment_config",
        lambda: {"enabled": True, "budget": 8000, "threshold": 0.85},
    )
    monkeypatch.setattr(
        hook_endpoints, "get_todoist_prompt_alerts_config",
        lambda: {"enabled": False, "max_per_category": 3},
    )


def test_prompt_context_short_circuits_while_breaker_open(monkeypatch):
    _enrichment_enabled(monkeypatch)
    schema._trip_breaker(RECOVERY)
    semantic = MagicMock()
    monkeypatch.setattr(hook_endpoints, "semantic_context", semantic)

    result = hook_endpoints.get_prompt_context("what did we decide about the pool")

    semantic.assert_not_called()
    assert result["success"] is True
    assert result["degraded"] is True
    assert result["error"] == f"database unavailable: {RECOVERY}"
    assert result["matches"] == []
    assert result["budget_used"] == {"core": 0, "vault": 0, "total": 8000}
    assert "todoist_prompt_alerts" in result


def test_prompt_context_marks_degraded_search(monkeypatch):
    _enrichment_enabled(monkeypatch)
    monkeypatch.setattr(hook_endpoints, "semantic_context", lambda **kw: {
        "matches": [], "query_ms": 0, "total_searched": 0,
        "degraded": True, "error": "connection pool busy",
    })
    result = hook_endpoints.get_prompt_context("hello there, jarvis")
    assert result["degraded"] is True
    assert result["error"] == "database unavailable: connection pool busy"


def test_prompt_context_healthy_has_no_degraded_flag_and_short_timeout(monkeypatch):
    _enrichment_enabled(monkeypatch)
    seen = []

    def semantic(**kw):
        seen.append(schema._checkout_timeout_var.get())
        return {"matches": [{"id": "a"}], "query_ms": 3, "total_searched": 9}

    monkeypatch.setattr(hook_endpoints, "semantic_context", semantic)
    result = hook_endpoints.get_prompt_context("hello there, jarvis")
    assert "degraded" not in result
    assert result["matches"] == [{"id": "a"}]
    assert seen == [schema.HOOK_CONN_TIMEOUT]


def test_auto_extract_context_short_circuits_while_breaker_open(monkeypatch):
    schema._trip_breaker(RECOVERY)
    content_list = MagicMock()
    monkeypatch.setattr(hook_endpoints, "content_list", content_list)

    result = hook_endpoints.get_auto_extract_context()

    content_list.assert_not_called()
    assert result["success"] is True
    assert result["degraded"] is True
    assert result["known_workstreams"] == []
    assert "auto_extract" in result and "worklog" in result


def test_auto_extract_context_degraded_on_db_failure(monkeypatch):
    monkeypatch.setattr(hook_endpoints, "content_list", lambda **kw: {
        "success": False, "retryable": True, "error": "connection pool busy",
    })
    result = hook_endpoints.get_auto_extract_context()
    assert result["degraded"] is True
    assert result["known_workstreams"] == []


# ── content / query structured failures ───────────────────────────────


class _DownPool:
    def connection(self, timeout=None):
        raise schema.DatabaseUnavailable(RECOVERY)


def test_content_write_flags_db_outage_retryable(mock_config, monkeypatch):
    from tools.content import content_write

    monkeypatch.setattr(schema, "_get_pool", lambda: _DownPool())
    result = content_write(content="a fresh insight", content_type="observation")
    assert result == {"success": False, "error": RECOVERY,
                      "retryable": True, "error_kind": "db_unavailable"}


def test_content_write_flags_model_host_outage_retryable(mock_config, monkeypatch):
    import tools.document_index as document_index
    from tools.content import content_write
    from tools.model_host_client import ModelHostError

    def unavailable(*args, **kwargs):
        raise ModelHostError("Model host request failed: timed out")

    monkeypatch.setattr(document_index, "prepare_document", unavailable)
    result = content_write(content="a fresh insight", content_type="observation")
    assert result["retryable"] is True
    assert result["error_kind"] == "model_host_unavailable"


def test_content_write_permanent_db_error_is_not_retryable(mock_config, monkeypatch):
    from tools.content import content_write

    class _ConstraintPool:
        def connection(self, timeout=None):
            raise psycopg.errors.CheckViolation("violates check constraint")

    monkeypatch.setattr(schema, "_get_pool", lambda: _ConstraintPool())
    result = content_write(content="a fresh insight", content_type="observation")
    assert result["success"] is False
    assert "retryable" not in result


def test_content_list_flags_db_outage_retryable(mock_config, monkeypatch):
    import tools.content as content

    def down(*args, **kwargs):
        raise schema.DatabaseUnavailable(RECOVERY)

    monkeypatch.setattr(content, "execute_query", down)
    result = content.content_list(content_type="worklog")
    assert result == {"success": False, "error": RECOVERY, "retryable": True}


def test_query_vault_flags_db_outage_retryable(mock_config, monkeypatch):
    import tools.query as query

    def down(*args, **kwargs):
        raise schema.DatabaseUnavailable(RECOVERY)

    monkeypatch.setattr(query, "execute_query", down)
    result = query.query_vault("anything")
    assert result == {"success": False, "retryable": True,
                      "error": f"Database unavailable: {RECOVERY}"}


def test_query_vault_other_failures_are_not_retryable(mock_config, monkeypatch):
    import tools.query as query

    def broken(*args, **kwargs):
        raise psycopg.errors.UndefinedTable('relation "local.memories" does not exist')

    monkeypatch.setattr(query, "execute_query", broken)
    result = query.query_vault("anything")
    assert result["success"] is False
    assert "retryable" not in result


def test_semantic_context_marks_db_outage_degraded(mock_config, monkeypatch):
    import tools.query as query

    def down(*args, **kwargs):
        raise schema.DatabaseUnavailable(RECOVERY)

    monkeypatch.setattr(query, "execute_query", down)
    result = query.semantic_context("anything at all")
    assert result["matches"] == []
    assert result["degraded"] is True
    assert result["error"] == RECOVERY


def _default_schemas_only(monkeypatch):
    """Empty registry → the built-in local + obsidian fallback, nothing remote."""
    import tools.schema_registry as schema_registry

    monkeypatch.setattr(schema_registry, "get_searchable_schemas", lambda *a, **k: [])


def test_cross_schema_search_propagates_db_unavailable(mock_config, monkeypatch):
    import tools.query as query

    def down(*args, **kwargs):
        raise schema.DatabaseUnavailable(RECOVERY)

    obsidian = MagicMock(return_value=[])
    _default_schemas_only(monkeypatch)
    monkeypatch.setattr(query, "_query_local_schema", down)
    monkeypatch.setattr(query, "_query_obsidian_schema", obsidian)
    with pytest.raises(schema.DatabaseUnavailable):
        query._cross_schema_search([0.0] * 384, 10)
    obsidian.assert_not_called()  # no second wait on a pool that just failed


def test_cross_schema_search_still_isolates_ordinary_schema_errors(mock_config, monkeypatch):
    import tools.query as query

    def broken(*args, **kwargs):
        raise RuntimeError("one schema broke")

    _default_schemas_only(monkeypatch)
    monkeypatch.setattr(query, "_query_local_schema", broken)
    monkeypatch.setattr(query, "_query_obsidian_schema", lambda *a, **k: [{"distance": 0.1}])
    assert query._cross_schema_search([0.0] * 384, 10) == [{"distance": 0.1}]


# ── retrieval telemetry ───────────────────────────────────────────────


def test_record_event_reports_disk_full_at_critical(monkeypatch, caplog):
    class _FullPool:
        def connection(self, timeout=None):
            raise psycopg.errors.DiskFull(
                'could not extend file "base/16384/19825": No space left on device'
            )

    monkeypatch.setattr(schema, "_pool", _FullPool())
    monkeypatch.setattr(telemetry, "_config", lambda: {"enabled": True})
    with caplog.at_level(logging.DEBUG, logger="jarvis-core"):
        trace = telemetry.record_event(
            purpose="context_injection", query="q", candidates=[], funnel={},
            latency={}, outcome="empty", shadow_eligible=False,
        )
    assert trace is None
    assert any(
        r.levelno == logging.CRITICAL and "DISK FULL" in r.getMessage()
        for r in caplog.records
    )
    assert schema.get_db_status()["status"] == "disk_full"


def test_acknowledge_delivery_uses_hook_checkout_timeout(monkeypatch):
    seen = []

    class _RecordingPool:
        def connection(self, timeout=None):
            seen.append(schema._checkout_timeout_var.get())
            raise schema.DatabaseUnavailable(RECOVERY)

    monkeypatch.setattr(schema, "_get_pool", lambda: _RecordingPool())
    assert telemetry.acknowledge_delivery(
        "00000000-0000-0000-0000-000000000000", {"delivered_candidate_keys": []}
    ) is False
    assert seen == [schema.HOOK_CONN_TIMEOUT]


class _StopLoop(Exception):
    pass


def _drive_janitor(
    monkeypatch, *, start, step, sleeps, cleanup, shadow=lambda: False, db_status="ok"
):
    """Run retrieval_telemetry_loop for ``sleeps`` iterations on a fake clock."""
    clock = {"now": float(start)}
    # Pinned: another test's leftover probe state must not defer the janitor.
    monkeypatch.setattr(schema, "get_db_status", lambda: {
        "status": db_status, "error": None, "checked_at": None, "free_bytes": None})
    real_sleep = asyncio.sleep
    count = {"sleeps": 0}

    async def fake_sleep(seconds):
        count["sleeps"] += 1
        if count["sleeps"] >= sleeps:
            raise _StopLoop
        clock["now"] += step
        await real_sleep(0)

    monkeypatch.setattr(telemetry, "time", SimpleNamespace(monotonic=lambda: clock["now"]))
    monkeypatch.setattr(telemetry, "asyncio", SimpleNamespace(
        sleep=fake_sleep, to_thread=asyncio.to_thread,
        CancelledError=asyncio.CancelledError,
    ))
    monkeypatch.setattr(telemetry, "process_one_shadow_job", shadow)
    monkeypatch.setattr(telemetry, "requeue_failed_shadow_jobs", lambda *a, **k: 0)
    monkeypatch.setattr(telemetry, "_config", lambda: {
        "shadow": {"poll_seconds": 0.25, "max_jobs_per_second": 100},
    })

    calls = []

    def recording_cleanup():
        calls.append(clock["now"])
        return cleanup()

    monkeypatch.setattr(telemetry, "cleanup_expired", recording_cleanup)
    with pytest.raises(_StopLoop):
        asyncio.run(telemetry.retrieval_telemetry_loop())
    return calls


def test_janitor_runs_on_first_iteration_even_with_short_uptime(monkeypatch):
    # time.monotonic() is time since the VM booted: 1h of uptime used to
    # mean no retention for another 23h (and forever with frequent restarts).
    calls = _drive_janitor(monkeypatch, start=3600, step=1, sleeps=1, cleanup=lambda: 0)
    assert calls == [3600]


def test_janitor_runs_even_when_shadow_job_fails(monkeypatch):
    def failing_shadow():
        raise RuntimeError("model host down")

    calls = _drive_janitor(
        monkeypatch, start=3600, step=1, sleeps=1, cleanup=lambda: 0, shadow=failing_shadow
    )
    assert calls == [3600]


def test_janitor_retries_failed_cleanup_after_five_minutes(monkeypatch):
    outcomes = iter([RuntimeError("database unavailable"), 0])

    def flaky_cleanup():
        outcome = next(outcomes)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    calls = _drive_janitor(
        monkeypatch, start=3600, step=150, sleeps=4, cleanup=flaky_cleanup
    )
    # 3600 fails; 3750 and 3900 are inside the 300s retry delay; 4050 retries.
    assert calls == [3600, 4050]


def test_janitor_waits_a_day_after_success(monkeypatch):
    calls = _drive_janitor(
        monkeypatch, start=3600, step=50_000, sleeps=3, cleanup=lambda: 0
    )
    assert calls == [3600, 103_600]


def test_janitor_logs_what_it_deleted(monkeypatch, caplog):
    """A successful run used to log nothing: retention not running at all
    went unnoticed for two months."""
    with caplog.at_level(logging.INFO, logger=telemetry.logger.name):
        _drive_janitor(monkeypatch, start=3600, step=1, sleeps=1, cleanup=lambda: 72)
    assert any(
        "Retention cleanup deleted 72 expired retrieval event(s)" in r.getMessage()
        for r in caplog.records
    )


def test_janitor_continues_a_capped_backlog_in_five_minutes(monkeypatch):
    monkeypatch.setattr(telemetry, "_CLEANUP_BATCH_SIZE", 2)
    monkeypatch.setattr(telemetry, "_CLEANUP_MAX_BATCHES", 3)
    outcomes = iter([6, 6, 1, 0])
    calls = _drive_janitor(
        monkeypatch, start=3600, step=150, sleeps=8, cleanup=lambda: next(outcomes)
    )
    # 3600 and 4050 hit the cap of 6 and come back 300s later (not a day);
    # 4500 finishes the backlog, after which the next run is a day away.
    assert calls == [3600, 4050, 4500]


def test_janitor_defers_while_the_volume_is_full(monkeypatch, caplog):
    with caplog.at_level(logging.INFO, logger=telemetry.logger.name):
        calls = _drive_janitor(
            monkeypatch, start=3600, step=150, sleeps=3, cleanup=lambda: 0,
            db_status="disk_full",
        )
    assert calls == []
    assert any("deferred" in r.getMessage() for r in caplog.records)
