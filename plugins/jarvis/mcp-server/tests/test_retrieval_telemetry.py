"""Unit contracts for retrieval telemetry and score diagnostics."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict


def test_candidate_trace_cannot_hold_document_bodies():
    from tools.retrieval_telemetry import CandidateTrace

    fields = asdict(CandidateTrace(schema_name="local", doc_id="obs::1"))
    assert "document" not in fields
    assert "content" not in fields
    assert fields["candidate_key"]


def test_retrieval_telemetry_defaults_are_shadow_only(mock_config):
    from tools.config import get_retrieval_telemetry_config

    cfg = get_retrieval_telemetry_config()
    assert cfg["enabled"] is True
    assert cfg["retention_days"] == 30
    assert cfg["candidate_detail_limit"] == 100
    assert cfg["shadow"]["enabled"] is True
    assert "threshold" not in cfg  # no live retrieval policy lives here


def test_simulator_is_read_only_and_reports_label_readiness(monkeypatch):
    import tools.schema as schema
    from tools.retrieval_telemetry import simulate_policy

    rows = [
        {"event_id": "a", "candidate_key": "1", "similarity": 0.90,
         "raw_bge_logit": 1.2, "request_label": "useful", "candidate_label": "relevant"},
        {"event_id": "b", "candidate_key": "2", "similarity": 0.70,
         "raw_bge_logit": -4.0, "request_label": "noisy", "candidate_label": "irrelevant"},
    ]
    monkeypatch.setattr(schema, "execute_query", lambda *a, **k: rows)
    result = simulate_policy({"policy": "coarse+bge", "cosine_threshold": 0.85, "bge_logit_threshold": -2.5})
    assert result["selected_count"] == 1
    assert result["precision"] == 1.0
    assert result["recall"] == 1.0
    assert result["positive_request_recall"] == 1.0
    assert result["negative_rejection_rate"] == 1.0
    assert result["recommendation_ready"] is False
    assert result["config_snippet"] == {"cosine_threshold": 0.85, "bge_logit_threshold": -2.5}


def test_simulator_cosine_or_bge_rehearses_production_gate(monkeypatch):
    """The 'cosine-or-bge' policy mirrors production's recall-additive gate:
    keep = cosine-pass OR logit-pass. Evaluability requires a similarity (the
    cosine clause always applies); a missing logit only forgoes the rescue —
    it does NOT censor the row (unlike bge-only)."""
    import tools.schema as schema
    from tools.retrieval_telemetry import simulate_policy

    rows = [
        # cosine passes → kept regardless of a poor logit.
        {"event_id": "a", "candidate_key": "1", "similarity": 0.90,
         "raw_bge_logit": -9.0, "request_label": "useful", "candidate_label": "relevant"},
        # cosine fails, logit rescues → kept.
        {"event_id": "b", "candidate_key": "2", "similarity": 0.50,
         "raw_bge_logit": -1.0, "request_label": "useful", "candidate_label": "relevant"},
        # cosine fails, logit fails → dropped.
        {"event_id": "c", "candidate_key": "3", "similarity": 0.50,
         "raw_bge_logit": -9.0, "request_label": "noisy", "candidate_label": "irrelevant"},
        # cosine fails, NO logit → cosine decides (fail), still evaluable.
        {"event_id": "d", "candidate_key": "4", "similarity": 0.50,
         "raw_bge_logit": None, "request_label": "noisy", "candidate_label": "irrelevant"},
    ]
    monkeypatch.setattr(schema, "execute_query", lambda *a, **k: rows)

    result = simulate_policy({
        "policy": "cosine-or-bge", "cosine_threshold": 0.85,
        "bge_logit_threshold": -4.0,
    })
    # Similarity present on all four → all evaluable (the censoring difference).
    assert result["scored_count"] == 4
    assert result["unscored_count"] == 0
    assert result["selected_count"] == 2  # a (cosine) + b (logit rescue)
    assert result["precision"] == 1.0
    assert result["recall"] == 1.0

    # The same missing-logit row IS censored under bge-only.
    bge = simulate_policy({"policy": "bge-only", "bge_logit_threshold": -4.0})
    assert bge["unscored_count"] == 1
    assert bge["scored_count"] == 3


def test_structured_host_rerank_preserves_raw_logits(monkeypatch):
    import tools.reranking as reranking

    class Client:
        def rerank(self, query, documents):
            return [-2.0, 3.0]

    monkeypatch.setattr(reranking, "_get_host_client", lambda config: Client())
    result = reranking.rerank_detailed(
        "query", ["a", "b"], [0.8, 0.7],
        {"backend": "host", "model": "bge", "alpha": 0.7, "max_latency_ms": 5000},
    )
    assert result.applied is True
    assert result.raw_logits == [-2.0, 3.0]
    assert 0 < result.sigmoid_scores[0] < result.sigmoid_scores[1] < 1
    assert result.backend == "host"
    assert result.model == "bge"


def test_record_event_is_fail_open(monkeypatch):
    import tools.retrieval_telemetry as telemetry

    monkeypatch.setattr(telemetry, "telemetry_enabled", lambda: True)
    monkeypatch.setattr(telemetry, "_config", lambda: {
        "enabled": True, "retention_days": 30, "candidate_detail_limit": 100,
        "store_user_prompts": True, "shadow": {"enabled": False},
    })
    monkeypatch.setattr("tools.schema._get_pool", lambda: (_ for _ in ()).throw(RuntimeError("down")))
    assert telemetry.record_event(
        purpose="context_injection", query="hello", candidates=[], funnel={},
        latency={}, outcome="empty",
    ) is None


def test_schema_has_cascade_and_no_candidate_body_column():
    from tools.schema import RETRIEVAL_TELEMETRY_SCHEMA_SQL

    candidate_ddl = RETRIEVAL_TELEMETRY_SCHEMA_SQL.split(
        "CREATE TABLE IF NOT EXISTS local.retrieval_candidates", 1
    )[1].split("CREATE TABLE IF NOT EXISTS local.retrieval_feedback", 1)[0]
    assert "ON DELETE CASCADE" in candidate_ddl
    assert " document " not in candidate_ddl.lower()
    assert " content " not in candidate_ddl.lower()


def test_loop_runs_shadow_job_off_the_event_loop(monkeypatch):
    """process_one_shadow_job blocks on psycopg + host HTTP calls; the async
    loop must offload it (asyncio.to_thread) so /hook/prompt-context and MCP
    traffic never queue behind a shadow rerank."""
    import asyncio
    import threading

    import tools.retrieval_telemetry as telemetry

    seen: dict[str, threading.Thread] = {}

    def fake_job() -> bool:
        seen["thread"] = threading.current_thread()
        return False

    monkeypatch.setattr(telemetry, "process_one_shadow_job", fake_job)
    monkeypatch.setattr(telemetry, "cleanup_expired", lambda: 0)
    # The janitor now runs on the first iteration; keep it off the real DB.
    monkeypatch.setattr(telemetry, "requeue_failed_shadow_jobs", lambda: 0)
    monkeypatch.setattr(telemetry, "_config", lambda: {
        "shadow": {"poll_seconds": 0.25, "max_jobs_per_second": 100},
    })

    async def run_one_iteration():
        task = asyncio.ensure_future(telemetry.retrieval_telemetry_loop())
        for _ in range(200):
            if "thread" in seen:
                break
            await asyncio.sleep(0.01)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    asyncio.run(run_one_iteration())

    assert "thread" in seen, "shadow job never ran"
    assert seen["thread"] is not threading.main_thread(), (
        "shadow job executed on the event loop thread"
    )


def test_simulator_excludes_unscored_candidates_from_bge_metrics(monkeypatch):
    """Candidates the shadow scorer never reached (NULL logit) must not count
    as threshold rejections — that deflates bge-policy recall with data that
    never measured BGE at all."""
    import tools.schema as schema
    from tools.retrieval_telemetry import simulate_policy

    rows = [
        # Scored, relevant, passes the gate.
        {"event_id": "a", "candidate_key": "1", "similarity": 0.90,
         "raw_bge_logit": 1.2, "request_label": "useful", "candidate_label": "relevant"},
        # Labeled relevant but never shadow-scored: censored, not a miss.
        {"event_id": "b", "candidate_key": "2", "similarity": 0.88,
         "raw_bge_logit": None, "request_label": "useful", "candidate_label": "relevant"},
    ]
    monkeypatch.setattr(schema, "execute_query", lambda *a, **k: rows)

    result = simulate_policy({"policy": "bge-only", "bge_logit_threshold": -2.5})
    assert result["false_negative"] == 0
    assert result["recall"] == 1.0
    assert result["scored_count"] == 1
    assert result["unscored_count"] == 1
    assert result["labeled_unscored_count"] == 1
    # Event b has no evaluable candidate under this policy at all.
    assert result["requests_excluded_unscored"] == 1

    # The same unscored row IS evaluable under cosine-only.
    result = simulate_policy({"policy": "cosine-only", "cosine_threshold": 0.85})
    assert result["scored_count"] == 2
    assert result["true_positive"] == 2


def test_rebuild_live_query_windows_matches_live_tokenizer_split(monkeypatch):
    """The shadow scorer must split the prompt exactly as live retrieval did
    (service tokenizer), or refuse — byte-fallback windows have different
    boundaries and would score candidates against the wrong text."""
    import tools.embedding as embedding
    from tools.retrieval_telemetry import _rebuild_live_query_windows

    class FakeService:
        def tokenize(self, text, with_pieces=True):
            # Few large pieces: tokenized split keeps the text in ONE window
            # while the byte fallback would split it in two.
            chunk = 100
            return [
                {"piece": text[i:i + chunk]} for i in range(0, len(text), chunk)
            ]

    long_text = "x" * 3000  # >2048 bytes, <2048 fake tokens

    monkeypatch.setattr(embedding, "get_embedding_service", lambda: FakeService())
    windows, error = _rebuild_live_query_windows(long_text, max_index=0)
    assert error == ""
    assert windows == [long_text]  # tokenizer path, not byte fallback

    # Index beyond the rebuilt windows means the reconstruction diverged.
    windows, error = _rebuild_live_query_windows(long_text, max_index=3)
    assert windows == []
    assert "exceeds rebuilt window count" in error


def test_rebuild_live_query_windows_refuses_untokenizable_long_prompts(monkeypatch):
    import tools.embedding as embedding
    from tools.retrieval_telemetry import _rebuild_live_query_windows

    class NoTokenizer:
        pass

    monkeypatch.setattr(embedding, "get_embedding_service", lambda: NoTokenizer())

    # Short prompts are one window under any splitter — always safe.
    windows, error = _rebuild_live_query_windows("short prompt", max_index=0)
    assert windows == ["short prompt"]
    assert error == ""

    # Long prompts would byte-fallback to different boundaries — refuse.
    windows, error = _rebuild_live_query_windows("y" * 3000, max_index=1)
    assert windows == []
    assert "tokenizer unavailable" in error


def test_rebuild_live_query_windows_refuses_when_tokenizer_raises(monkeypatch):
    """split_text_windows silently byte-falls back when tokenize raises (host
    outage) — different boundaries than live. The rebuild must refuse, not
    score candidates against byte-window slices marked 'complete'."""
    import tools.embedding as embedding
    from tools.retrieval_telemetry import _rebuild_live_query_windows

    class FailingTokenizer:
        def tokenize(self, text, with_pieces=True):
            raise ConnectionError("host tokenizer timeout")

    monkeypatch.setattr(embedding, "get_embedding_service", lambda: FailingTokenizer())

    windows, error = _rebuild_live_query_windows("z" * 3000, max_index=0)
    assert windows == []
    assert "tokenizer failed" in error

    # Short prompts never tokenize (single window under any splitter) — safe.
    windows, error = _rebuild_live_query_windows("short", max_index=0)
    assert windows == ["short"]
    assert error == ""


def test_candidate_row_cut_never_drops_returned_candidates():
    """ANN overfetch can trace more rows than candidate_detail_limit; delivery
    marking, shadow scoring, and label export all join on the returned rows,
    so they must win the cut."""
    from tools.retrieval_telemetry import CandidateTrace, _select_candidate_rows

    rejected = [
        CandidateTrace(schema_name="obsidian", doc_id=f"doc-{i}", vector_rank=i)
        for i in range(1, 120)
    ]
    deep_returned = CandidateTrace(
        schema_name="local", doc_id="memory-deep", vector_rank=130, returned=True
    )

    kept = _select_candidate_rows([*rejected, deep_returned], limit=100)
    assert len(kept) == 100
    assert any(c.doc_id == "memory-deep" for c in kept)

    # Under the limit nothing is reordered or dropped.
    kept = _select_candidate_rows(rejected[:5], limit=100)
    assert [c.doc_id for c in kept] == [f"doc-{i}" for i in range(1, 6)]


def test_shadow_backoff_grows_and_caps():
    """Retrying every poll cycle burned all attempts inside one host outage;
    backoff must span long enough to outlive a model reload."""
    from tools.retrieval_telemetry import _shadow_backoff_seconds

    assert [_shadow_backoff_seconds({}, n) for n in (1, 2, 3)] == [30, 120, 480]
    # Three attempts span ~10 minutes with shipped defaults.
    assert sum(_shadow_backoff_seconds({}, n) for n in (1, 2)) >= 120
    # Cap honored, config overridable through the shadow section.
    assert _shadow_backoff_seconds({}, 9) == 900
    assert _shadow_backoff_seconds({"retry_base_seconds": 5, "retry_backoff_factor": 2}, 3) == 20
    assert _shadow_backoff_seconds({"retry_max_seconds": 60}, 5) == 60


def test_cleanup_query_exempts_labeled_events():
    """Labels CASCADE from their event, so retention would silently destroy
    hand-labeled ground truth. The delete must exclude labeled events."""
    import tools.schema as schema
    from tools.retrieval_telemetry import cleanup_expired

    captured = {}

    def fake_execute_write(sql, params=None, *, returning=False, conn_timeout=None):
        captured["sql"] = sql
        return {"count": 0}

    original = schema.execute_write
    schema.execute_write = fake_execute_write
    try:
        cleanup_expired()
    finally:
        schema.execute_write = original

    sql = " ".join(captured["sql"].split())
    assert "DELETE FROM local.retrieval_events" in sql
    assert "NOT EXISTS" in sql
    assert "local.retrieval_feedback" in sql
    assert "local.retrieval_candidate_feedback" in sql


def test_cleanup_deletes_in_bounded_batches(monkeypatch):
    """One unbounded DELETE of a 5.7k-event backlog wrote ~117 MiB of WAL in
    under a second; each transaction is now capped, and so is each run."""
    import tools.retrieval_telemetry as telemetry
    import tools.schema as schema

    monkeypatch.setattr(telemetry, "_CLEANUP_BATCH_SIZE", 3)
    monkeypatch.setattr(telemetry, "_CLEANUP_MAX_BATCHES", 4)
    backlog = {"left": 7}
    calls = []

    def fake_execute_write(sql, params=None, *, returning=False, conn_timeout=None):
        calls.append(params)
        assert returning is True
        assert "LIMIT %s" in sql
        n = min(params[0], backlog["left"])
        backlog["left"] -= n
        return {"count": n}

    monkeypatch.setattr(schema, "execute_write", fake_execute_write)
    assert telemetry.cleanup_expired() == 7
    assert calls == [(3,), (3,), (3,)]  # 3 + 3 + 1: the short batch ends it

    backlog["left"] = 100
    calls.clear()
    assert telemetry.cleanup_expired() == 12  # capped at 4 batches of 3
    assert len(calls) == 4
    assert telemetry._cleanup_backlog_left(12) is True
    assert telemetry._cleanup_backlog_left(11) is False


def test_shadow_claim_query_respects_next_attempt_gate():
    """The claim query must skip events still inside their backoff window."""
    import inspect

    import tools.retrieval_telemetry as telemetry

    source = inspect.getsource(telemetry.process_one_shadow_job)
    normalized = " ".join(source.split())
    assert "shadow_next_attempt_at IS NULL" in normalized
    assert "shadow_next_attempt_at <= now()" in normalized


def test_shadow_scoring_widens_budget_beyond_live_hook_limits():
    """The live reranker budget (1.5s) protects the user-facing hook. Shadow
    scoring is a background job; inheriting that budget systematically censored
    long multi-window prompts (measured: 121/121 short prompts complete vs 6/13
    long prompts timing out)."""
    import json
    from pathlib import Path

    from tools.retrieval_telemetry import _shadow_rerank_config

    live = {"backend": "host", "model": "bge", "alpha": 0.7,
            "host_timeout_ms": 1500, "max_latency_ms": 1500}
    merged = _shadow_rerank_config(live, {})

    assert merged["host_timeout_ms"] == 15000     # shadow default wins
    assert merged["max_latency_ms"] == 60000
    assert merged["model"] == "bge"               # identity untouched
    assert live["host_timeout_ms"] == 1500        # live config not mutated

    # Never narrows an already-generous live budget.
    generous = _shadow_rerank_config(
        {"host_timeout_ms": 90000, "max_latency_ms": 90000}, {}
    )
    assert generous["host_timeout_ms"] == 90000
    assert generous["max_latency_ms"] == 90000

    # Configurable through the shadow section, and None is tolerated.
    tuned = _shadow_rerank_config(
        {"host_timeout_ms": None, "max_latency_ms": None},
        {"host_timeout_ms": 20000, "max_latency_ms": 120000},
    )
    assert (tuned["host_timeout_ms"], tuned["max_latency_ms"]) == (20000, 120000)

    # Shipped defaults must actually carry generous shadow budgets.
    defaults_path = (
        Path(__file__).resolve().parents[2] / "defaults" / "config.json"
    )
    shipped = json.loads(defaults_path.read_text())[
        "memory"]["retrieval_telemetry"]["shadow"]
    assert shipped["host_timeout_ms"] >= 10000
    assert shipped["max_latency_ms"] >= 30000


# ── Augmentation-era separation for Phase-2 calibration ───────────────
#
# Mechanical-era and summary-era BGE logits are drawn from DIFFERENT rerank
# input spaces — the same chunk measured −8.16 with the mechanical prefix and
# +0.03 with its LLM summary. A sweep across the pooled set selects a threshold
# correct for neither era, and the operator could not notice because the export
# dropped config_snapshot, the only column carrying the era.


def _era_rows():
    return [
        {"event_id": "m1", "candidate_key": "1", "similarity": 0.90,
         "raw_bge_logit": -8.16, "request_label": "useful",
         "candidate_label": "relevant", "contextual_augmentation": "mechanical"},
        {"event_id": "m2", "candidate_key": "1", "similarity": 0.50,
         "raw_bge_logit": -9.0, "request_label": "noisy",
         "candidate_label": "irrelevant", "contextual_augmentation": "mechanical"},
        {"event_id": "s1", "candidate_key": "1", "similarity": 0.90,
         "raw_bge_logit": 0.03, "request_label": "useful",
         "candidate_label": "relevant", "contextual_augmentation": "summary"},
        {"event_id": "u1", "candidate_key": "1", "similarity": 0.60,
         "raw_bge_logit": -5.0, "request_label": "noisy",
         "candidate_label": "irrelevant", "contextual_augmentation": "unstamped"},
    ]


def test_simulator_reports_the_era_census_and_flags_mixing(monkeypatch):
    import tools.schema as schema
    from tools.retrieval_telemetry import simulate_policy

    monkeypatch.setattr(schema, "execute_query", lambda *a, **k: _era_rows())
    result = simulate_policy({"policy": "bge-only", "bge_logit_threshold": -4.0})
    assert result["contextual_augmentation"] == "all"
    assert result["augmentation_eras"] == {
        "mechanical": 2, "summary": 1, "unstamped": 1,
    }
    assert result["augmentation_eras_mixed"] is True
    assert result["candidate_count"] == 4


def test_simulator_filters_to_one_era(monkeypatch):
    """The fix: a threshold can be swept inside ONE embedding-space era."""
    import tools.schema as schema
    from tools.retrieval_telemetry import simulate_policy

    monkeypatch.setattr(schema, "execute_query", lambda *a, **k: _era_rows())

    summary_only = simulate_policy({
        "policy": "bge-only", "bge_logit_threshold": -4.0,
        "contextual_augmentation": "summary",
    })
    assert summary_only["contextual_augmentation"] == "summary"
    assert summary_only["candidate_count"] == 1
    assert summary_only["selected_count"] == 1  # +0.03 clears −4.0
    assert summary_only["augmentation_eras_mixed"] is False
    # The census still describes the WHOLE pool, so mixing stays visible.
    assert summary_only["augmentation_eras"]["mechanical"] == 2

    mechanical_only = simulate_policy({
        "policy": "bge-only", "bge_logit_threshold": -4.0,
        "contextual_augmentation": "mechanical",
    })
    assert mechanical_only["candidate_count"] == 2
    assert mechanical_only["selected_count"] == 0  # −8.16 and −9.0 both fail
    # Same threshold, opposite verdict — which is the whole point.
    assert summary_only["selected_count"] != mechanical_only["selected_count"]


def test_simulator_rejects_an_unknown_era(monkeypatch):
    import pytest as _pytest
    import tools.schema as schema
    from tools.retrieval_telemetry import simulate_policy

    monkeypatch.setattr(schema, "execute_query", lambda *a, **k: _era_rows())
    with _pytest.raises(ValueError):
        simulate_policy({"policy": "bge-only", "contextual_augmentation": "bogus"})


def test_simulator_treats_missing_era_as_unstamped(monkeypatch):
    """Rows from before any stamping must be bucketed, not dropped."""
    import tools.schema as schema
    from tools.retrieval_telemetry import simulate_policy

    rows = [{"event_id": "a", "candidate_key": "1", "similarity": 0.9,
             "raw_bge_logit": 1.0, "request_label": None,
             "candidate_label": None}]
    monkeypatch.setattr(schema, "execute_query", lambda *a, **k: rows)
    result = simulate_policy({"policy": "cosine-only"})
    assert result["augmentation_eras"] == {"unstamped": 1}
    assert simulate_policy({
        "policy": "cosine-only", "contextual_augmentation": "unstamped",
    })["candidate_count"] == 1


def test_simulator_query_joins_the_events_table_for_the_era(monkeypatch):
    import tools.schema as schema
    from tools.retrieval_telemetry import simulate_policy

    seen = {}

    def capture(sql, params=None, **kwargs):
        seen["sql"] = " ".join(sql.split()).lower()
        return []

    monkeypatch.setattr(schema, "execute_query", capture)
    simulate_policy({"policy": "cosine-only"})
    assert "local.retrieval_events" in seen["sql"]
    assert "contextual_augmentation" in seen["sql"]
    # Legacy boolean events must migrate, not vanish.
    assert "contextual_embeddings" in seen["sql"]


def test_export_carries_the_augmentation_era(monkeypatch):
    import tools.schema as schema
    from tools.retrieval_telemetry import export_labeled_events

    seen = {}

    def capture(sql, params=None, **kwargs):
        seen["sql"] = " ".join(sql.split()).lower()
        return [{"trace_id": "a", "contextual_augmentation": "summary"}]

    monkeypatch.setattr(schema, "execute_query", capture)
    rows = export_labeled_events()
    assert "contextual_augmentation" in seen["sql"]
    assert rows[0]["contextual_augmentation"] == "summary"


def test_known_eras_are_declared():
    from tools.retrieval_telemetry import AUGMENTATION_ERAS

    assert set(AUGMENTATION_ERAS) == {"none", "mechanical", "summary", "unstamped"}
