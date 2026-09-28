"""Real PostgreSQL contracts for retrieval telemetry."""

import psycopg
import time


def test_retrieval_event_roundtrip_and_no_candidate_body(e2e_config):
    from tools.retrieval_telemetry import CandidateTrace, get_event, get_summary, record_event
    from tools.schema import _get_pool

    _get_pool()  # retrieval paths have already established this in production
    trace_id = record_event(
        purpose="context_injection",
        query="full user prompt for calibration",
        candidates=[CandidateTrace(
            schema_name="local", doc_id="obs::telemetry", vector_rank=1,
            similarity=0.91, terminal_reason="selected", returned=True,
        )],
        funnel={"ann_unique": 1, "budget_selected": 1},
        latency={"total_ms": 3.2}, outcome="results",
        shadow_eligible=False,
    )
    assert trace_id
    event = get_event(trace_id)
    assert event["query_text"] == "full user prompt for calibration"
    assert event["candidates"][0]["doc_id"] == "obs::telemetry"
    assert "document" not in event["candidates"][0]
    assert "content" not in event["candidates"][0]
    summary = get_summary(7)
    assert summary["requests"] == 1
    assert summary["funnel"]["candidates"] == 1
    assert summary["schemas"][0]["schema"] == "local"
    assert "cosine" in summary["histograms"]


def test_feedback_cascades_with_event(e2e_config):
    from tools.retrieval_telemetry import (
        CandidateTrace, put_candidate_feedback, put_event_feedback, record_event,
    )
    from tools.schema import _get_pool

    _get_pool()
    candidate = CandidateTrace(schema_name="local", doc_id="obs::cascade", vector_rank=1)
    trace_id = record_event(
        purpose="explicit_recall", query="cascade", candidates=[candidate],
        funnel={}, latency={}, outcome="results", shadow_eligible=False,
    )
    put_event_feedback(trace_id, {"verdict": "useful", "expected_missing_ids": []})
    put_candidate_feedback(trace_id, candidate.candidate_key, {"verdict": "relevant"})

    with psycopg.connect(e2e_config["db_url"]) as conn:
        conn.execute("DELETE FROM local.retrieval_events WHERE id = %s::uuid", (trace_id,))
        remaining = conn.execute(
            """SELECT
                 (SELECT count(*) FROM local.retrieval_candidates WHERE event_id = %s::uuid),
                 (SELECT count(*) FROM local.retrieval_feedback WHERE event_id = %s::uuid),
                 (SELECT count(*) FROM local.retrieval_candidate_feedback WHERE event_id = %s::uuid)""",
            (trace_id, trace_id, trace_id),
        ).fetchone()
    assert remaining == (0, 0, 0)


def test_schema_initialization_is_idempotent_with_telemetry(e2e_config):
    from tools.schema import ensure_schema

    ensure_schema()
    ensure_schema()


def test_telemetry_write_p95_under_ten_ms(e2e_config, monkeypatch):
    import tools.retrieval_telemetry as telemetry
    from tools.schema import _get_pool

    _get_pool()
    monkeypatch.setattr(telemetry, "_config", lambda: {
        "enabled": True, "retention_days": 30, "store_user_prompts": True,
        "candidate_detail_limit": 100, "shadow": {"enabled": False},
    })
    timings = []
    for index in range(30):
        started = time.perf_counter()
        assert telemetry.record_event(
            purpose="performance_probe", query=f"probe {index}",
            candidates=[telemetry.CandidateTrace(
                schema_name="local", doc_id=f"obs::probe-{index}",
                vector_rank=1, similarity=0.9,
            )],
            funnel={"ann_unique": 1}, latency={"total_ms": 1},
            outcome="results", shadow_eligible=False,
        )
        timings.append((time.perf_counter() - started) * 1000)
    p95 = sorted(timings)[int(len(timings) * 0.95) - 1]
    assert p95 < 10.0, f"telemetry p95 was {p95:.2f}ms"


def test_shadow_job_scores_pending_candidate(e2e_config, monkeypatch):
    import tools.config as config
    import tools.reranking as reranking
    import tools.retrieval_telemetry as telemetry
    from tools.schema import _get_pool

    vector = "[" + ",".join(["0.01"] * 384) + "]"
    with psycopg.connect(e2e_config["db_url"]) as conn:
        conn.execute(
            """INSERT INTO local.memories
               (id, document, embedding, category, scope, source)
               VALUES ('obs::shadow', 'host model shadow document', %s::halfvec,
                       'observation', 'global', 'test')""",
            (vector,),
        )
    shadow_cfg = {
        "enabled": True, "retention_days": 30, "store_user_prompts": True,
        "candidate_detail_limit": 100,
        "shadow": {"enabled": True, "candidate_count": 20, "delay_seconds": 0,
                   "poll_seconds": 0.1, "max_attempts": 3, "max_jobs_per_second": 10},
    }
    rerank_cfg = {"backend": "host", "model": "bge-test", "alpha": 0.7}
    monkeypatch.setattr(telemetry, "_config", lambda: shadow_cfg)
    monkeypatch.setattr(config, "get_reranking_config", lambda: rerank_cfg)
    monkeypatch.setattr(
        reranking, "rerank_raw",
        lambda query, docs, cfg: {"logits": [2.0], "probabilities": [0.880797], "latency_ms": 1.0},
    )
    _get_pool()
    trace_id = telemetry.record_event(
        purpose="context_injection", query="shadow query",
        candidates=[telemetry.CandidateTrace(
            schema_name="local", doc_id="obs::shadow", chunk_index=0,
            vector_rank=1, similarity=0.7, terminal_reason="cosine_rejected",
        )],
        funnel={"ann_unique": 1}, latency={"total_ms": 2}, outcome="empty",
        model_snapshot={"reranker_model": "bge-test"},
    )
    started = time.perf_counter()
    assert telemetry.process_one_shadow_job() is True
    elapsed = time.perf_counter() - started
    event = telemetry.get_event(trace_id)
    assert event["shadow_status"] == "complete"
    assert event["candidates"][0]["raw_bge_logit"] == 2.0
    assert elapsed < 60


def test_event_documents_resolve_bodies_on_demand(e2e_config):
    """Telemetry stores only locators; the UI resolves bodies from source
    tables at inspection time — memories, chunk windows, and remote mirrors
    without a local body."""
    from tools.retrieval_telemetry import CandidateTrace, get_event_documents, record_event
    from tools.schema import _get_pool

    pool = _get_pool()
    with pool.connection() as conn:
        conn.execute(
            """INSERT INTO local.memories (id, document, embedding)
               VALUES ('obs::doc-body', 'the full memory body', %s::halfvec)
               ON CONFLICT (id) DO UPDATE SET document = EXCLUDED.document""",
            ("[" + ",".join(["0"] * 384) + "]",),
        )
        conn.commit()

    local_hit = CandidateTrace(schema_name="local", doc_id="obs::doc-body", vector_rank=1, returned=True)
    remote_miss = CandidateTrace(schema_name="remote_team", doc_id="obs::elsewhere", vector_rank=2)
    trace_id = record_event(
        purpose="context_injection", query="doc bodies", candidates=[local_hit, remote_miss],
        funnel={}, latency={}, outcome="results", shadow_eligible=False,
    )

    docs = get_event_documents(trace_id, preview_chars=10)
    assert docs[local_hit.candidate_key]["found"] is True
    assert docs[local_hit.candidate_key]["text"] == "the full m"
    assert docs[local_hit.candidate_key]["truncated"] is True
    assert docs[local_hit.candidate_key]["size"] == len("the full memory body")
    assert docs[remote_miss.candidate_key]["found"] is False

    full = get_event_documents(trace_id, candidate_key=local_hit.candidate_key)
    assert full[local_hit.candidate_key]["text"] == "the full memory body"
    assert full[local_hit.candidate_key]["truncated"] is False


def test_labeled_events_survive_retention_expiry(e2e_config):
    """Labels CASCADE from their event, so a 30-day TTL would hard-delete
    hand-labeled ground truth on a rolling window. Labeled traces are exempt;
    unlabeled expired traces still go."""
    from tools.retrieval_telemetry import (
        CandidateTrace, cleanup_expired, get_event, put_candidate_feedback,
        put_event_feedback, record_event,
    )
    from tools.schema import _get_pool

    pool = _get_pool()
    candidate = CandidateTrace(schema_name="local", doc_id="obs::retained", vector_rank=1)
    labeled = record_event(
        purpose="context_injection", query="labeled trace", candidates=[candidate],
        funnel={}, latency={}, outcome="results", shadow_eligible=False,
    )
    candidate_only = CandidateTrace(schema_name="local", doc_id="obs::cand", vector_rank=1)
    candidate_labeled = record_event(
        purpose="context_injection", query="candidate-labeled trace",
        candidates=[candidate_only], funnel={}, latency={}, outcome="results",
        shadow_eligible=False,
    )
    unlabeled = record_event(
        purpose="context_injection", query="unlabeled trace", candidates=[],
        funnel={}, latency={}, outcome="empty", shadow_eligible=False,
    )

    put_event_feedback(labeled, {"verdict": "useful", "expected_missing_ids": []})
    put_candidate_feedback(candidate_labeled, candidate_only.candidate_key,
                           {"verdict": "relevant"})

    # Secondary guard: labeling pushes expiry out in the data itself.
    with pool.connection() as conn:
        rows = conn.execute(
            "SELECT id::text, expires_at > now() + interval '300 days' "
            "FROM local.retrieval_events WHERE id = ANY(%s::uuid[])",
            ([labeled, candidate_labeled],),
        ).fetchall()
    assert rows and all(extended for _, extended in rows), (
        "labeling must extend expires_at"
    )

    # Force all three past their retention horizon (defeats the secondary
    # guard on purpose, so the cleanup-query exemption is what is under test).
    with pool.connection() as conn:
        conn.execute(
            "UPDATE local.retrieval_events SET expires_at = now() - interval '1 day' "
            "WHERE id = ANY(%s::uuid[])",
            ([labeled, candidate_labeled, unlabeled],),
        )
        conn.commit()

    deleted = cleanup_expired()

    assert deleted >= 1
    assert get_event(labeled) is not None, "event-labeled trace was deleted"
    assert get_event(candidate_labeled) is not None, "candidate-labeled trace was deleted"
    assert get_event(unlabeled) is None, "unlabeled expired trace should be pruned"


def test_shadow_failure_backs_off_and_can_be_requeued(e2e_config, monkeypatch):
    """A transient model-host outage must not burn every attempt in one poll
    cycle, and terminally-failed events must be recoverable."""
    import tools.config as config
    import tools.reranking as reranking
    import tools.retrieval_telemetry as telemetry
    from tools.schema import _get_pool

    vector = "[" + ",".join(["0.01"] * 384) + "]"
    pool = _get_pool()
    with pool.connection() as conn:
        conn.execute(
            """INSERT INTO local.memories (id, document, embedding, category, scope, source)
               VALUES ('obs::backoff', 'backoff document', %s::halfvec,
                       'observation', 'global', 'test')
               ON CONFLICT (id) DO NOTHING""",
            (vector,),
        )
        conn.commit()

    shadow_cfg = {
        "enabled": True, "retention_days": 30, "store_user_prompts": True,
        "candidate_detail_limit": 100,
        "shadow": {"enabled": True, "candidate_count": 20, "delay_seconds": 0,
                   "poll_seconds": 0.1, "max_attempts": 2, "max_jobs_per_second": 10,
                   "retry_base_seconds": 300},
    }
    monkeypatch.setattr(telemetry, "_config", lambda: shadow_cfg)
    monkeypatch.setattr(config, "get_reranking_config",
                        lambda: {"backend": "host", "model": "bge-test", "alpha": 0.7})
    # Host is down for every attempt.
    def boom(*args, **kwargs):
        raise RuntimeError("model host request failed: timed out")

    monkeypatch.setattr(reranking, "rerank_raw", boom)

    trace_id = telemetry.record_event(
        purpose="context_injection", query="backoff probe",
        candidates=[telemetry.CandidateTrace(
            schema_name="local", doc_id="obs::backoff", chunk_index=0,
            vector_rank=1, similarity=0.7, terminal_reason="cosine_rejected",
        )],
        funnel={}, latency={}, outcome="empty",
        model_snapshot={"reranker_model": "bge-test"},
    )

    # Attempt 1 fails → pending with a future next-attempt gate.
    assert telemetry.process_one_shadow_job() is True
    with pool.connection() as conn:
        status, next_at = conn.execute(
            "SELECT shadow_status, shadow_next_attempt_at > now() "
            "FROM local.retrieval_events WHERE id = %s::uuid", (trace_id,),
        ).fetchone()
    assert status == "pending"
    assert next_at is True, "failed attempt must schedule a future retry"

    # The backoff gate makes the claim query skip it — attempts are not burned.
    assert telemetry.process_one_shadow_job() is False
    with pool.connection() as conn:
        attempts = conn.execute(
            "SELECT shadow_attempts FROM local.retrieval_events WHERE id = %s::uuid",
            (trace_id,),
        ).fetchone()[0]
    assert attempts == 1, "backoff window must not consume another attempt"

    # Force the window open; the second (terminal) failure marks it failed.
    with pool.connection() as conn:
        conn.execute(
            "UPDATE local.retrieval_events SET shadow_next_attempt_at = now() - interval '1 second' "
            "WHERE id = %s::uuid", (trace_id,),
        )
        conn.commit()
    assert telemetry.process_one_shadow_job() is True
    with pool.connection() as conn:
        status = conn.execute(
            "SELECT shadow_status FROM local.retrieval_events WHERE id = %s::uuid",
            (trace_id,),
        ).fetchone()[0]
    assert status == "failed"

    # Recovery: requeue resets attempts so the event can be scored later.
    assert telemetry.requeue_failed_shadow_jobs() >= 1
    with pool.connection() as conn:
        status, attempts, next_at = conn.execute(
            "SELECT shadow_status, shadow_attempts, shadow_next_attempt_at "
            "FROM local.retrieval_events WHERE id = %s::uuid", (trace_id,),
        ).fetchone()
    assert (status, attempts, next_at) == ("pending", 0, None)


def test_retention_deletes_in_capped_batches_and_keeps_labels(e2e_config, monkeypatch):
    """Retention runs as bounded DELETE transactions with a per-run cap (an
    unbounded one wrote ~117 MiB of WAL at once); candidates still cascade and
    labeled traces are still exempt."""
    import tools.retrieval_telemetry as telemetry
    from tools.retrieval_telemetry import (
        CandidateTrace, cleanup_expired, get_event, put_event_feedback, record_event,
    )
    from tools.schema import _get_pool

    monkeypatch.setattr(telemetry, "_CLEANUP_BATCH_SIZE", 3)
    monkeypatch.setattr(telemetry, "_CLEANUP_MAX_BATCHES", 2)
    pool = _get_pool()  # record_event only writes once the pool exists

    def trace(i: int) -> str:
        return record_event(
            purpose="context_injection", query=f"expired trace {i}",
            candidates=[CandidateTrace(schema_name="local", doc_id=f"obs::{i}", vector_rank=1)],
            funnel={}, latency={}, outcome="results", shadow_eligible=False,
        )

    expired = [trace(i) for i in range(7)]
    labeled = trace(99)
    fresh = trace(100)
    assert all(expired) and labeled and fresh
    put_event_feedback(labeled, {"verdict": "useful", "expected_missing_ids": []})
    with pool.connection() as conn:
        conn.execute(
            "UPDATE local.retrieval_events SET expires_at = now() - interval '1 day' "
            "WHERE id = ANY(%s::uuid[])",
            (expired + [labeled],),
        )
        conn.commit()

    assert cleanup_expired() == 6  # two batches of three, then the run stops
    assert telemetry._cleanup_backlog_left(6) is True
    assert cleanup_expired() == 1
    assert cleanup_expired() == 0

    assert all(get_event(e) is None for e in expired)
    assert get_event(labeled) is not None
    assert get_event(fresh) is not None
    with pool.connection() as conn:
        orphans = conn.execute(
            "SELECT count(*) FROM local.retrieval_candidates c "
            "WHERE NOT EXISTS (SELECT 1 FROM local.retrieval_events e WHERE e.id = c.event_id)"
        ).fetchone()[0]
        remaining = conn.execute(
            "SELECT count(*) FROM local.retrieval_candidates WHERE event_id = ANY(%s::uuid[])",
            (expired,),
        ).fetchone()[0]
    assert orphans == 0 and remaining == 0
