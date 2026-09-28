"""Durable, content-safe observability for semantic retrieval.

The live retrieval path calls :func:`record_event` best-effort. Every public
function is fail-open so telemetry can never affect what Jarvis retrieves.
Candidate bodies are deliberately absent from both the schema and API shapes.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from psycopg.types.json import Jsonb

logger = logging.getLogger("jarvis-core")


@dataclass
class CandidateTrace:
    schema_name: str
    doc_id: str
    parent_id: Optional[str] = None
    parent_file: Optional[str] = None
    chunk_index: Optional[int] = None
    query_window_index: int = 0
    vector_rank: Optional[int] = None
    final_rank: Optional[int] = None
    similarity: Optional[float] = None
    pre_score: Optional[float] = None
    raw_bge_logit: Optional[float] = None
    bge_probability: Optional[float] = None
    blended_score: Optional[float] = None
    display_cost: Optional[int] = None
    terminal_reason: Optional[str] = None
    returned: bool = False
    delivered: bool = False
    channel: str = "semantic"
    candidate_key: str = ""

    def __post_init__(self) -> None:
        if not self.candidate_key:
            raw = f"{self.schema_name}\0{self.doc_id}\0{self.query_window_index}"
            self.candidate_key = hashlib.sha256(raw.encode()).hexdigest()[:24]


def _config() -> dict:
    from .config import get_retrieval_telemetry_config

    return get_retrieval_telemetry_config()


def telemetry_enabled() -> bool:
    try:
        return bool(_config().get("enabled", True))
    except Exception:
        return False


def _json(value: Any) -> Jsonb:
    return Jsonb(value or {})


def _query_identity(query: str) -> tuple[str, int]:
    encoded = query.encode("utf-8", errors="replace")
    return hashlib.sha256(encoded).hexdigest(), len(query)


def _select_candidate_rows(candidates: list, limit: int) -> list[CandidateTrace]:
    """Bound per-event candidate rows without dropping returned candidates.

    ANN overfetch can trace more rows than candidate_detail_limit. A returned
    candidate must always keep its row — delivery marking, shadow scoring, and
    label export all join on it — so returned rows win the cut and the
    remaining budget goes to the best-ranked rejected rows.
    """
    normalized = [
        item if isinstance(item, CandidateTrace) else CandidateTrace(**item)
        for item in candidates
    ]
    if len(normalized) <= limit:
        return normalized
    returned = [c for c in normalized if c.returned]
    # Rows the reranker actually judged (lexical rescues/rejections carry no
    # vector_rank and would otherwise be truncated out of the trace, making
    # gate decisions invisible in the UI).
    scored = [c for c in normalized if not c.returned and c.raw_bge_logit is not None]
    others = [c for c in normalized if not c.returned and c.raw_bge_logit is None]
    return (returned + scored + others)[:limit]


def record_event(
    *,
    purpose: str,
    query: str,
    candidates: list[CandidateTrace | dict],
    funnel: dict,
    latency: dict,
    outcome: str,
    pipeline: str = "semantic",
    user_name: Optional[str] = None,
    user_facing: bool = True,
    query_ref: Optional[str] = None,
    query_window_count: int = 1,
    model_snapshot: Optional[dict] = None,
    config_snapshot: Optional[dict] = None,
    status: str = "complete",
    shadow_eligible: bool = True,
) -> Optional[str]:
    """Persist one retrieval event and its candidate score trail.

    Returns the trace UUID, or ``None`` when disabled/unavailable. No document
    body is accepted by this API, making accidental body persistence harder.
    """
    if not telemetry_enabled():
        return None
    try:
        from . import schema as schema_module

        # Retrieval already owns a live pool. Do not create/wait for a new one
        # merely to write telemetry during startup, teardown, or degradation.
        if schema_module._pool is None:
            return None

        cfg = _config()
        retention = max(1, int(cfg.get("retention_days", 30)))
        limit = max(0, min(1000, int(cfg.get("candidate_detail_limit", 100))))
        shadow_cfg = cfg.get("shadow") or {}
        shadow_requested = shadow_eligible and shadow_cfg.get("enabled", True) and candidates
        if shadow_requested:
            try:
                from .config import get_reranking_config

                shadow_status = "pending" if get_reranking_config().get("backend") == "host" else "skipped"
            except Exception:
                shadow_status = "skipped"
        else:
            shadow_status = "disabled"
        event_id = str(uuid.uuid4())
        query_sha, query_length = _query_identity(query)
        store_prompt = user_facing and bool(cfg.get("store_user_prompts", True))
        expires_at = datetime.now(timezone.utc) + timedelta(days=retention)
        normalized = _select_candidate_rows(candidates, limit)

        pool = schema_module._pool
        with pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """INSERT INTO local.retrieval_events
                       (id, expires_at, user_name, purpose, pipeline, status, outcome,
                        query_text, query_sha256, query_ref, query_length,
                        query_window_count, model_snapshot, config_snapshot,
                        funnel, latency, shadow_status)
                       VALUES (%s::uuid, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                               %s, %s, %s, %s, %s, %s, %s)""",
                    (
                        event_id, expires_at, user_name, purpose, pipeline, status,
                        outcome, query if store_prompt else None, query_sha, query_ref,
                        query_length, query_window_count, _json(model_snapshot),
                        _json(config_snapshot), _json(funnel), _json(latency),
                        shadow_status,
                    ),
                )
                if normalized:
                    cur.executemany(
                        """INSERT INTO local.retrieval_candidates
                           (event_id, candidate_key, schema_name, doc_id, parent_id,
                            parent_file, chunk_index, query_window_index, vector_rank,
                            final_rank, similarity, pre_score, raw_bge_logit,
                            bge_probability, blended_score, display_cost,
                            terminal_reason, returned, delivered, channel)
                           VALUES (%s::uuid, %s, %s, %s, %s, %s, %s, %s, %s,
                                   %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
                        [
                            (
                                event_id, c.candidate_key, c.schema_name, c.doc_id,
                                c.parent_id, c.parent_file, c.chunk_index,
                                c.query_window_index, c.vector_rank, c.final_rank,
                                c.similarity, c.pre_score, c.raw_bge_logit,
                                c.bge_probability, c.blended_score, c.display_cost,
                                c.terminal_reason, c.returned, c.delivered, c.channel,
                            )
                            for c in normalized
                        ],
                    )
            conn.commit()
        return event_id
    except Exception as exc:
        # This is the most frequent writer, so it is usually the first to hit
        # ENOSPC; note_db_error logs that at CRITICAL instead of a DEBUG line.
        from .schema import note_db_error

        note_db_error(exc)
        logger.debug("Retrieval telemetry write skipped: %s", exc)
        return None


def acknowledge_delivery(trace_id: str, payload: dict) -> bool:
    """Record what the hook actually emitted after session-level dedup.

    Hook path: the checkout is bounded by HOOK_CONN_TIMEOUT, and an open
    circuit breaker makes it fail immediately.
    """
    from .schema import HOOK_CONN_TIMEOUT, checkout_timeout

    with checkout_timeout(HOOK_CONN_TIMEOUT):
        return _acknowledge_delivery(trace_id, payload)


def _acknowledge_delivery(trace_id: str, payload: dict) -> bool:
    try:
        from .schema import _get_pool

        delivered = [str(key) for key in payload.get("delivered_candidate_keys", [])]
        safe_payload = {
            "status": str(payload.get("status", "complete")),
            "returned_count": int(payload.get("returned_count", 0)),
            "delivered_count": int(payload.get("delivered_count", len(delivered))),
            "suppressed_count": int(payload.get("suppressed_count", 0)),
            "output_chars": int(payload.get("output_chars", 0)),
            "acknowledged_at": datetime.now(timezone.utc).isoformat(),
        }
        pool = _get_pool()
        with pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE local.retrieval_events SET delivery = %s WHERE id = %s::uuid",
                    (_json(safe_payload), trace_id),
                )
                event_updated = cur.rowcount
                if delivered:
                    cur.execute(
                        """UPDATE local.retrieval_candidates SET delivered = true
                           WHERE event_id = %s::uuid AND candidate_key = ANY(%s)""",
                        (trace_id, delivered),
                    )
            conn.commit()
        return event_updated > 0
    except Exception as exc:
        from .schema import note_db_error

        note_db_error(exc)
        logger.debug("Retrieval delivery acknowledgement skipped: %s", exc)
        return False


def get_summary(days: int = 7) -> dict:
    """Return compact aggregate telemetry for health/UI consumers."""
    from .schema import execute_query

    days = max(1, min(int(days), 90))
    row = execute_query(
        """SELECT count(*) AS requests,
                  count(*) FILTER (WHERE outcome = 'empty') AS zero_results,
                  count(*) FILTER (WHERE shadow_status = 'pending') AS shadow_pending,
                  count(*) FILTER (WHERE shadow_status = 'failed') AS shadow_failed,
                  percentile_cont(0.5) WITHIN GROUP
                    (ORDER BY NULLIF(latency->>'total_ms','')::double precision) AS p50_ms,
                  percentile_cont(0.95) WITHIN GROUP
                    (ORDER BY NULLIF(latency->>'total_ms','')::double precision) AS p95_ms
           FROM local.retrieval_events
           WHERE created_at >= now() - (%s * interval '1 day')""",
        (days,), fetch="one",
    ) or {}
    purposes = execute_query(
        """SELECT purpose, count(*) AS requests,
                  count(*) FILTER (WHERE outcome = 'empty') AS zero_results
           FROM local.retrieval_events
           WHERE created_at >= now() - (%s * interval '1 day')
           GROUP BY purpose ORDER BY requests DESC""",
        (days,),
    )
    candidate_totals = execute_query(
        """SELECT count(*) AS candidates,
                  count(*) FILTER (WHERE terminal_reason = 'cosine_rejected') AS cosine_rejected,
                  count(*) FILTER (WHERE terminal_reason = 'logit_rejected') AS logit_rejected,
                  count(*) FILTER (WHERE terminal_reason = 'sensitive') AS sensitive_rejected,
                  count(*) FILTER (WHERE terminal_reason = 'parent_dedup') AS parent_dedup,
                  count(*) FILTER (WHERE terminal_reason = 'semantic_duplicate') AS semantic_duplicates,
                  count(*) FILTER (WHERE terminal_reason = 'candidate_cap') AS candidate_cap,
                  count(*) FILTER (WHERE terminal_reason = 'result_cap') AS result_cap,
                  count(*) FILTER (WHERE terminal_reason = 'budget_rejected') AS budget_rejected,
                  count(*) FILTER (WHERE channel = 'lexical') AS lexical_channel,
                  count(*) FILTER (WHERE channel = 'both') AS both_channel,
                  count(*) FILTER (WHERE returned) AS returned,
                  count(*) FILTER (WHERE delivered) AS delivered,
                  count(*) FILTER (WHERE raw_bge_logit IS NOT NULL) AS bge_scored
           FROM local.retrieval_candidates c
           JOIN local.retrieval_events e ON e.id = c.event_id
           WHERE e.created_at >= now() - (%s * interval '1 day')""",
        (days,), fetch="one",
    ) or {}
    delivery = execute_query(
        """SELECT count(*) FILTER (WHERE delivery ? 'acknowledged_at') AS acknowledged,
                  COALESCE(sum(COALESCE(NULLIF(delivery->>'returned_count','')::integer, 0)), 0) AS returned,
                  COALESCE(sum(COALESCE(NULLIF(delivery->>'delivered_count','')::integer, 0)), 0) AS delivered,
                  COALESCE(sum(COALESCE(NULLIF(delivery->>'suppressed_count','')::integer, 0)), 0) AS suppressed
           FROM local.retrieval_events
           WHERE created_at >= now() - (%s * interval '1 day')""",
        (days,), fetch="one",
    ) or {}
    shadow_rows = execute_query(
        """SELECT shadow_status AS status, count(*) AS count
           FROM local.retrieval_events
           WHERE created_at >= now() - (%s * interval '1 day')
           GROUP BY shadow_status ORDER BY shadow_status""",
        (days,),
    )
    schemas = execute_query(
        """SELECT c.schema_name AS schema, count(*) AS candidates,
                  count(*) FILTER (WHERE c.returned) AS returned
           FROM local.retrieval_candidates c
           JOIN local.retrieval_events e ON e.id = c.event_id
           WHERE e.created_at >= now() - (%s * interval '1 day')
           GROUP BY c.schema_name ORDER BY candidates DESC""",
        (days,),
    )
    recent_contracts = execute_query(
        """SELECT model_snapshot, config_snapshot, funnel
           FROM local.retrieval_events
           WHERE created_at >= now() - (%s * interval '1 day')
           ORDER BY created_at DESC LIMIT 100""",
        (days,),
    )
    models = []
    thresholds = []
    for contract in recent_contracts:
        model = contract.get("model_snapshot") or {}
        threshold = (contract.get("config_snapshot") or {}).get("cosine_threshold")
        if model and model not in models:
            models.append(model)
        if threshold is not None and threshold not in thresholds:
            thresholds.append(threshold)
    cosine_histogram = execute_query(
        """SELECT round((floor(c.similarity * 10) / 10)::numeric, 1)::float AS bucket,
                  count(*) AS count
           FROM local.retrieval_candidates c
           JOIN local.retrieval_events e ON e.id = c.event_id
           WHERE e.created_at >= now() - (%s * interval '1 day')
             AND c.similarity IS NOT NULL
           GROUP BY bucket ORDER BY bucket""",
        (days,),
    )
    logit_histogram = execute_query(
        """SELECT floor(c.raw_bge_logit)::integer AS bucket, count(*) AS count
           FROM local.retrieval_candidates c
           JOIN local.retrieval_events e ON e.id = c.event_id
           WHERE e.created_at >= now() - (%s * interval '1 day')
             AND c.raw_bge_logit IS NOT NULL
           GROUP BY bucket ORDER BY bucket""",
        (days,),
    )
    return {
        **row, "days": days, "purposes": purposes,
        "funnel": candidate_totals,
        "delivery": delivery,
        "reranking": {
            "bge_scored": candidate_totals.get("bge_scored", 0),
            "live_applied": sum(
                1 for event in recent_contracts
                if (event.get("funnel") or {}).get("live_reranker_applied")
            ),
        },
        "shadow": {item["status"]: item["count"] for item in shadow_rows},
        "schemas": schemas,
        "models": models,
        "thresholds": thresholds,
        "histograms": {"cosine": cosine_histogram, "raw_bge_logit": logit_histogram},
    }


def list_events(*, limit: int = 50, offset: int = 0, purpose: str = "", outcome: str = "") -> list[dict]:
    from .schema import execute_query

    conditions = ["true"]
    params: list[Any] = []
    if purpose:
        conditions.append("purpose = %s")
        params.append(purpose)
    if outcome:
        conditions.append("outcome = %s")
        params.append(outcome)
    params.extend([max(1, min(limit, 200)), max(0, offset)])
    return execute_query(
        f"""SELECT id::text, created_at, purpose, pipeline, status, outcome,
                    query_text, query_sha256, query_length, query_window_count,
                    funnel, latency, delivery, shadow_status, shadow_attempts,
                    shadow_error
             FROM local.retrieval_events WHERE {' AND '.join(conditions)}
             ORDER BY created_at DESC LIMIT %s OFFSET %s""",
        tuple(params),
    )


def get_event(event_id: str) -> Optional[dict]:
    from .schema import execute_query

    event = execute_query(
        """SELECT id::text, created_at, expires_at, user_name, purpose, pipeline,
                  status, outcome, query_text, query_sha256, query_ref, query_length,
                  query_window_count, model_snapshot, config_snapshot, funnel,
                  latency, delivery, shadow_status, shadow_attempts, shadow_error
           FROM local.retrieval_events WHERE id = %s::uuid""",
        (event_id,), fetch="one",
    )
    if not event:
        return None
    event["candidates"] = execute_query(
        """SELECT candidate_key, schema_name, doc_id, parent_id, parent_file,
                  chunk_index, query_window_index, vector_rank, final_rank,
                  similarity, pre_score, raw_bge_logit, bge_probability,
                  blended_score, display_cost, terminal_reason, returned,
                  delivered, channel
           FROM local.retrieval_candidates WHERE event_id = %s::uuid
           ORDER BY vector_rank NULLS LAST, candidate_key""",
        (event_id,),
    )
    event["feedback"] = execute_query(
        "SELECT verdict, expected_missing_ids, note, user_name, updated_at FROM local.retrieval_feedback WHERE event_id = %s::uuid",
        (event_id,), fetch="one",
    )
    feedback = execute_query(
        "SELECT candidate_key, verdict, note, user_name, updated_at FROM local.retrieval_candidate_feedback WHERE event_id = %s::uuid",
        (event_id,),
    )
    by_key = {row["candidate_key"]: row for row in feedback}
    for candidate in event["candidates"]:
        candidate["feedback"] = by_key.get(candidate["candidate_key"])
    return event


def get_event_documents(
    event_id: str,
    preview_chars: int = 240,
    candidate_key: Optional[str] = None,
) -> dict:
    """Resolve candidate locators to source text for UI inspection.

    Telemetry never stores document bodies (content-safe by design), so this
    reads them from the live source tables on demand — the same resolution the
    shadow scorer uses. Keyed by candidate_key; remote mirrors without a local
    body return found=False. preview_chars=0 (or a single candidate_key)
    returns the full text.
    """
    from .schema import _get_pool

    pool = _get_pool()
    out: dict[str, dict] = {}
    full = bool(candidate_key) or preview_chars <= 0
    with pool.connection() as conn:
        params: list[Any] = [event_id]
        key_filter = ""
        if candidate_key:
            key_filter = " AND candidate_key = %s"
            params.append(candidate_key)
        refs = conn.execute(
            f"""SELECT candidate_key, schema_name, doc_id, chunk_index
               FROM local.retrieval_candidates
               WHERE event_id = %s::uuid{key_filter}
               ORDER BY vector_rank NULLS LAST""",
            params,
        ).fetchall()
        for key, schema_name, doc_id, chunk_index in refs:
            row = None
            if schema_name == "local":
                row = conn.execute(
                    """SELECT COALESCE(
                           (SELECT document FROM local.memory_chunks
                            WHERE parent_id = %s AND chunk_index = %s),
                           (SELECT document FROM local.memories WHERE id = %s))""",
                    (doc_id, chunk_index, doc_id),
                ).fetchone()
            elif schema_name == "obsidian":
                row = conn.execute(
                    "SELECT document FROM obsidian.documents WHERE id = %s", (doc_id,)
                ).fetchone()
            document = row[0] if row and row[0] is not None else None
            if document is None:
                out[key] = {"found": False, "size": 0, "text": None, "truncated": False}
            else:
                text = document if full else document[:preview_chars]
                out[key] = {
                    "found": True,
                    "size": len(document),
                    "text": text,
                    "truncated": len(text) < len(document),
                }
    return out


def _protect_labeled_event(event_id: str) -> None:
    """Push a labeled event's expiry far out (secondary guard).

    ``cleanup_expired`` already exempts labeled events; this makes the
    protection visible in the data and survives any future code path that
    deletes purely by ``expires_at``. Losing labels is silent and
    unrecoverable, so it gets two independent guards.
    """
    try:
        from .schema import execute_write

        execute_write(
            """UPDATE local.retrieval_events
                  SET expires_at = GREATEST(expires_at, now() + interval '365 days')
                WHERE id = %s::uuid""",
            (event_id,),
        )
    except Exception as exc:  # never fail a label write over retention
        logger.warning("Could not extend retention for labeled event %s: %s", event_id, exc)


def put_event_feedback(event_id: str, payload: dict, user_name: Optional[str] = None) -> bool:
    verdict = str(payload.get("verdict", ""))
    if verdict not in {"useful", "mixed", "noisy", "missed", "unsure"}:
        raise ValueError("invalid retrieval feedback verdict")
    from .schema import execute_write

    execute_write(
        """INSERT INTO local.retrieval_feedback
           (event_id, verdict, expected_missing_ids, note, user_name)
           VALUES (%s::uuid, %s, %s, %s, %s)
           ON CONFLICT (event_id) DO UPDATE SET verdict = EXCLUDED.verdict,
             expected_missing_ids = EXCLUDED.expected_missing_ids,
             note = EXCLUDED.note, user_name = EXCLUDED.user_name,
             updated_at = now()""",
        (event_id, verdict, _json(payload.get("expected_missing_ids", [])),
         payload.get("note"), user_name),
    )
    _protect_labeled_event(event_id)
    return True


def put_candidate_feedback(event_id: str, candidate_key: str, payload: dict, user_name: Optional[str] = None) -> bool:
    verdict = str(payload.get("verdict", ""))
    if verdict not in {"relevant", "irrelevant", "unsure"}:
        raise ValueError("invalid candidate feedback verdict")
    from .schema import execute_write

    execute_write(
        """INSERT INTO local.retrieval_candidate_feedback
           (event_id, candidate_key, verdict, note, user_name)
           VALUES (%s::uuid, %s, %s, %s, %s)
           ON CONFLICT (event_id, candidate_key) DO UPDATE SET
             verdict = EXCLUDED.verdict, note = EXCLUDED.note,
             user_name = EXCLUDED.user_name, updated_at = now()""",
        (event_id, candidate_key, verdict, payload.get("note"), user_name),
    )
    _protect_labeled_event(event_id)
    return True


_AUGMENTATION_ERA_SQL = """\
COALESCE(
  e.config_snapshot->>'contextual_augmentation',
  CASE
    WHEN e.config_snapshot->>'contextual_embeddings' = 'true' THEN 'mechanical'
    WHEN e.config_snapshot->>'contextual_embeddings' = 'false' THEN 'none'
  END,
  'unstamped')"""

# The recorded augmentation eras a caller may filter calibration data to.
# 'unstamped' covers events from before any augmentation marker existed.
AUGMENTATION_ERAS = ("none", "mechanical", "summary", "unstamped")


def export_labeled_events() -> list[dict]:
    """Labeled events for offline threshold calibration.

    Carries ``contextual_augmentation`` per event: mechanical-era and
    summary-era logits are drawn from DIFFERENT embedding/rerank input spaces
    (the same chunk measured −8.16 mechanical and +0.03 with its summary), so an
    export that cannot separate them cannot be used to pick a threshold. It used
    to drop ``config_snapshot`` entirely — the only column carrying the era.
    """
    from .schema import execute_query

    return execute_query(
        f"""SELECT e.id::text AS trace_id, e.created_at, e.purpose, e.query_text,
                  e.query_sha256, f.verdict, f.expected_missing_ids, f.note,
                  {_AUGMENTATION_ERA_SQL} AS contextual_augmentation,
                  COALESCE(jsonb_agg(jsonb_build_object(
                    'candidate_key', c.candidate_key, 'schema', c.schema_name,
                    'doc_id', c.doc_id, 'similarity', c.similarity,
                    'raw_bge_logit', c.raw_bge_logit,
                    'bge_probability', c.bge_probability,
                    'blended_score', c.blended_score,
                    'terminal_reason', c.terminal_reason,
                    'label', cf.verdict) ORDER BY c.vector_rank)
                    FILTER (WHERE c.candidate_key IS NOT NULL), '[]'::jsonb) AS candidates
           FROM local.retrieval_events e
           JOIN local.retrieval_feedback f ON f.event_id = e.id
           LEFT JOIN local.retrieval_candidates c ON c.event_id = e.id
           LEFT JOIN local.retrieval_candidate_feedback cf
             ON cf.event_id = c.event_id AND cf.candidate_key = c.candidate_key
           GROUP BY e.id, f.event_id ORDER BY e.created_at DESC"""
    )


def simulate_policy(payload: dict) -> dict:
    """Evaluate thresholds against stored scores; never mutates live config.

    ``contextual_augmentation`` restricts the pool to ONE augmentation era
    ('none' | 'mechanical' | 'summary' | 'unstamped'). Without it, a Phase-2
    sweep pools logits produced from incommensurable rerank inputs — the same
    chunk scores −8.16 with the mechanical prefix and +0.03 with its summary — so
    the selected threshold is a compromise correct for neither era. The response
    always reports ``augmentation_eras`` (candidate counts per era in the
    unfiltered pool) so mixing is visible even when no filter is passed.
    """
    policy = str(payload.get("policy", "cosine-only"))
    if policy not in {"cosine-only", "bge-only", "coarse+bge", "cosine-or-bge"}:
        raise ValueError("invalid policy")
    era = str(payload.get("contextual_augmentation") or "").strip().lower()
    if era and era not in AUGMENTATION_ERAS:
        raise ValueError("invalid contextual_augmentation filter")
    cosine = float(payload.get("cosine_threshold", 0.85))
    bge = float(payload.get("bge_logit_threshold", -2.5))
    from .schema import execute_query

    rows = execute_query(
        f"""SELECT c.event_id::text, c.candidate_key, c.similarity,
                  c.raw_bge_logit, f.verdict AS request_label,
                  cf.verdict AS candidate_label,
                  {_AUGMENTATION_ERA_SQL} AS contextual_augmentation
           FROM local.retrieval_candidates c
           LEFT JOIN local.retrieval_events e ON e.id = c.event_id
           LEFT JOIN local.retrieval_feedback f ON f.event_id = c.event_id
           LEFT JOIN local.retrieval_candidate_feedback cf
             ON cf.event_id = c.event_id AND cf.candidate_key = c.candidate_key"""
    )
    # Era census over the FULL pool, before any filter — the number an operator
    # needs to notice that their calibration set spans two spaces.
    eras: dict[str, int] = {}
    for row in rows:
        key = str(row.get("contextual_augmentation") or "unstamped")
        eras[key] = eras.get(key, 0) + 1
    if era:
        rows = [
            row for row in rows
            if str(row.get("contextual_augmentation") or "unstamped") == era
        ]
    # BGE logits only exist for candidates the shadow scorer reached (top-N,
    # local/obsidian bodies, host backend active, job succeeded). Treating a
    # missing score as "rejected by the threshold" would count unscored
    # relevant candidates as false negatives and make bge policies look
    # arbitrarily bad — evaluate each policy only on candidates it can
    # actually score, and report the censored remainder explicitly.
    def _evaluable(row: dict) -> bool:
        # cosine-or-bge mirrors production's recall-additive gate: it requires a
        # similarity (the cosine clause always applies) and treats BGE as an
        # optional rescue — the logit is only consulted when present, so a
        # missing logit does NOT make the row unevaluable (unlike bge-only).
        needs_cosine = policy in ("cosine-only", "coarse+bge", "cosine-or-bge")
        needs_bge = policy in ("bge-only", "coarse+bge")
        if needs_cosine and row["similarity"] is None:
            return False
        if needs_bge and row["raw_bge_logit"] is None:
            return False
        return True

    def _key(row: dict) -> tuple:
        return (row["event_id"], row["candidate_key"])

    scored = [r for r in rows if _evaluable(r)]
    unscored = [r for r in rows if not _evaluable(r)]
    selected = []
    selected_keys: set[tuple] = set()
    for row in scored:
        cos_ok = row["similarity"] is not None and float(row["similarity"]) >= cosine
        bge_ok = row["raw_bge_logit"] is not None and float(row["raw_bge_logit"]) >= bge
        if policy == "cosine-only":
            keep = cos_ok
        elif policy == "bge-only":
            keep = bge_ok
        elif policy == "coarse+bge":
            keep = cos_ok and bge_ok
        else:  # cosine-or-bge — production's recall-additive gate
            keep = cos_ok or bge_ok
        if keep:
            selected.append(row)
            selected_keys.add(_key(row))
    labeled = [r for r in scored if r.get("candidate_label") in ("relevant", "irrelevant")]
    labeled_unscored = sum(
        1 for r in unscored if r.get("candidate_label") in ("relevant", "irrelevant")
    )
    tp = sum(1 for r in labeled if _key(r) in selected_keys and r["candidate_label"] == "relevant")
    fp = sum(1 for r in labeled if _key(r) in selected_keys and r["candidate_label"] == "irrelevant")
    fn = sum(1 for r in labeled if _key(r) not in selected_keys and r["candidate_label"] == "relevant")
    # Request-level metrics carry the same censoring: an event whose
    # candidates were never scored can never be "selected" under a bge
    # policy, so only events with at least one evaluable candidate count.
    evaluable_events = {r["event_id"] for r in scored}
    pos_all = {r["event_id"] for r in rows if r.get("request_label") in ("useful", "mixed", "missed")}
    neg_all = {r["event_id"] for r in rows if r.get("request_label") == "noisy"}
    positive_ids = pos_all & evaluable_events
    negative_ids = neg_all & evaluable_events
    selected_ids = {r["event_id"] for r in selected}
    pos_requests = len(positive_ids)
    neg_requests = len(negative_ids)
    enough = pos_requests >= 20 and neg_requests >= 20
    return {
        "policy": policy,
        "contextual_augmentation": era or "all",
        "augmentation_eras": eras,
        "augmentation_eras_mixed": len(eras) > 1 and not era,
        "candidate_count": len(rows), "selected_count": len(selected),
        "scored_count": len(scored), "unscored_count": len(unscored),
        "labeled_count": len(labeled), "labeled_unscored_count": labeled_unscored,
        "true_positive": tp, "false_positive": fp,
        "false_negative": fn, "precision": tp / (tp + fp) if tp + fp else None,
        "recall": tp / (tp + fn) if tp + fn else None,
        "positive_requests": pos_requests, "negative_requests": neg_requests,
        "requests_excluded_unscored": len((pos_all | neg_all) - evaluable_events),
        "positive_request_recall": len(positive_ids & selected_ids) / pos_requests if pos_requests else None,
        "negative_rejection_rate": len(negative_ids - selected_ids) / neg_requests if neg_requests else None,
        "recommendation_ready": enough,
        "config_snippet": {"cosine_threshold": cosine, "bge_logit_threshold": bge},
    }


def _summary_cache_drifted(pool, event_id: str, event_created_at) -> bool:
    """Whether any vault candidate's cached summary is newer than the event.

    One indexed query against ``obsidian.document_context.generated_at``. True
    means the text ``_fetch_candidate_documents`` would rebuild cannot be the
    text the live reranker scored, so the event must be skipped rather than
    mislabeled.

    Fail-open (returns False) on any error, including a pre-migration database
    with no ``document_context`` table: failing closed would censor every event
    forever on a transient DB problem, and before this feature existed there was
    no drift to detect.
    """
    if event_created_at is None:
        return False
    try:
        with pool.connection() as conn:
            row = conn.execute(
                """SELECT 1
                     FROM obsidian.document_context dc
                    WHERE dc.generated_at > %s
                      AND dc.parent_file IN (
                            SELECT parent_file FROM local.retrieval_candidates
                             WHERE event_id = %s::uuid
                               AND schema_name = 'obsidian'
                               AND parent_file IS NOT NULL)
                    LIMIT 1""",
                (event_created_at, event_id),
            ).fetchone()
        return row is not None
    except Exception as exc:
        logger.debug("Summary cache drift check unavailable for %s: %s", event_id, exc)
        return False


def _fetch_candidate_documents(conn, event_id: str, limit: int) -> tuple[list[dict], int]:
    """Resolve telemetry locators back to source text only for scoring.

    The shadow reranker MUST score exactly the text the live reranker scored, so
    obsidian fragments are augmented with the same document-context prefix used
    at index/query time (see tools/chunk_context.py). Omitting it here would make
    shadow logits diverge from live logits and corrupt the calibration dataset.
    Local memories are never augmented (their embed/rerank path isn't either).
    """
    from .chunk_context import augment_vault_row
    from .config import get_contextual_augmentation_mode
    from .context_summary import fetch_document_summaries

    contextual_mode = get_contextual_augmentation_mode()
    refs = conn.execute(
        """SELECT candidate_key, schema_name, doc_id, chunk_index,
                  query_window_index, raw_bge_logit
           FROM local.retrieval_candidates
           WHERE event_id = %s::uuid
           ORDER BY vector_rank NULLS LAST LIMIT %s""",
        (event_id, limit),
    ).fetchall()
    # Pass 1: resolve bodies (vault rows are held aside, un-augmented, so their
    # summaries can be fetched in ONE query instead of one per candidate).
    resolved: list[tuple] = []  # (key, window_index, document | None, vault_row | None)
    missing = 0
    for key, schema_name, doc_id, chunk_index, window_index, raw_logit in refs:
        if raw_logit is not None:
            continue
        missing += 1
        if schema_name == "local":
            row = conn.execute(
                """SELECT COALESCE(
                       (SELECT document FROM local.memory_chunks
                        WHERE parent_id = %s AND chunk_index = %s),
                       (SELECT document FROM local.memories WHERE id = %s))""",
                (doc_id, chunk_index, doc_id),
            ).fetchone()
            if row and row[0] is not None:
                resolved.append((key, window_index, row[0], None))
        elif schema_name == "obsidian":
            row = conn.execute(
                """SELECT document, title, chunk_heading, chunk_total, parent_file
                   FROM obsidian.documents WHERE id = %s""",
                (doc_id,),
            ).fetchone()
            if row and row[0] is not None:
                resolved.append((key, window_index, None, row))
        # else: remote mirrors may not have a stable local body — partial.

    # Pass 2: one batched summary lookup, then augment through the shared entry
    # point so the resolved text is byte-identical to what the live reranker saw.
    summaries = fetch_document_summaries(
        [row[4] or "" for _key, _win, doc, row in resolved if row is not None],
        conn=conn,
    )
    out = []
    for key, window_index, document, row in resolved:
        if row is not None:
            document = augment_vault_row(
                row[0],
                parent_file=row[4] or "",
                title=row[1] or "",
                chunk_heading=row[2] or "",
                chunk_total=row[3],
                mode=contextual_mode,
                summary=summaries.get(row[4] or ""),
            )
        out.append({
            "candidate_key": key,
            "document": document,
            "query_window_index": window_index,
        })
    return out, missing


def _shadow_backoff_seconds(shadow: dict, used_attempts: int) -> int:
    """Exponential delay before the next shadow attempt.

    Defaults give 30s → 120s → 480s, so three attempts span ~10 minutes — long
    enough to outlive a model reload instead of burning the whole retry budget
    inside one poll cycle.
    """
    base = max(1, int(shadow.get("retry_base_seconds", 30)))
    factor = max(1, int(shadow.get("retry_backoff_factor", 4)))
    cap = max(base, int(shadow.get("retry_max_seconds", 900)))
    return min(cap, base * (factor ** max(0, used_attempts - 1)))


def requeue_failed_shadow_jobs(max_age_days: int = 7) -> int:
    """Return terminally-failed shadow events to the queue.

    A transient model-host outage can exhaust an event's attempts; its
    candidates then never receive logits and silently drop out of the
    calibration corpus. This resets attempts so the (now backed-off) worker
    can score them, bounded to events still inside their retention window.
    """
    from .schema import execute_query

    row = execute_query(
        """WITH requeued AS (
               UPDATE local.retrieval_events
                  SET shadow_status = 'pending', shadow_attempts = 0,
                      shadow_error = NULL, shadow_finished_at = NULL,
                      shadow_next_attempt_at = NULL
                WHERE shadow_status = 'failed'
                  AND expires_at > now()
                  AND created_at >= now() - (%s * interval '1 day')
               RETURNING 1
           ) SELECT count(*) AS count FROM requeued""",
        (max(1, int(max_age_days)),), fetch="one",
    )
    return int((row or {}).get("count", 0))


def _shadow_rerank_config(live_config: dict, shadow: dict) -> dict:
    """Widen the reranker budget for background shadow scoring.

    Latency budgets in the reranking config exist to protect the USER-FACING
    path (the injection hook has ~2.5s total). Shadow scoring is a background
    job nobody waits for, and inheriting the tight live budget systematically
    censors long multi-window prompts: measured 121/121 short prompts complete
    vs 6/13 long prompts (>4k chars) failing with "model host request timed
    out" — precisely the events the calibration corpus most needs. Trade
    latency for completeness, never narrowing an already-generous live value.
    """
    merged = dict(live_config or {})
    merged["host_timeout_ms"] = max(
        int(merged.get("host_timeout_ms") or 1500),
        int(shadow.get("host_timeout_ms", 15000)),
    )
    merged["max_latency_ms"] = max(
        int(merged.get("max_latency_ms") or 1500),
        int(shadow.get("max_latency_ms", 60000)),
    )
    return merged


def _mark_shadow_skipped(pool, event_id: str, reason: str) -> None:
    with pool.connection() as conn:
        conn.execute(
            """UPDATE local.retrieval_events SET shadow_status = 'skipped',
                      shadow_error = %s, shadow_finished_at = now()
               WHERE id = %s::uuid""",
            (reason, event_id),
        )
        conn.commit()


def _rebuild_live_query_windows(query_text: str, max_index: int) -> tuple[list[str], str]:
    """Rebuild the base query windows exactly as live retrieval built them.

    Live retrieval splits with the embedding service's tokenizer
    (query._prepare_query_windows); candidates store an index into that base
    window list. Rebuilding with different boundaries would score candidates
    against the wrong slice of the prompt and silently corrupt calibration
    data, so return ([], reason) whenever the reconstruction can't be trusted.
    """
    from .embedding import get_embedding_service
    from .query import _QUERY_WINDOW_OVERLAP, _QUERY_WINDOW_TOKENS
    from .text_windows import split_text_windows

    tokenizer = getattr(get_embedding_service(), "tokenize", None)
    if tokenizer is None and len(query_text.encode("utf-8")) > _QUERY_WINDOW_TOKENS:
        return [], "tokenizer unavailable to rebuild multi-window query"

    # split_text_windows silently byte-falls back when the tokenizer RAISES
    # (host outage, non-host backend); byte boundaries differ from the live
    # tokenizer split, so a tokenizer failure must surface as a refusal here,
    # never as differently-sliced windows scored as if they were the originals.
    tokenizer_failure: list[Exception] = []
    strict_tokenize = None
    if tokenizer is not None:
        def strict_tokenize(text, **kwargs):
            try:
                return tokenizer(text, **kwargs)
            except Exception as exc:
                tokenizer_failure.append(exc)
                raise

    windows = split_text_windows(
        query_text,
        max_tokens=_QUERY_WINDOW_TOKENS,
        overlap_tokens=_QUERY_WINDOW_OVERLAP,
        tokenize=strict_tokenize,
    ) or [query_text]
    if tokenizer_failure:
        return [], (
            f"tokenizer failed while rebuilding query windows: "
            f"{tokenizer_failure[0]}"
        )
    if max_index >= len(windows):
        return [], (
            f"stored query_window_index {max_index} exceeds rebuilt "
            f"window count {len(windows)}"
        )
    return windows, ""


def process_one_shadow_job() -> bool:
    """Claim and score one pending event. Safe across multiple workers."""
    if not telemetry_enabled():
        return False
    cfg = _config()
    shadow = cfg.get("shadow") or {}
    if not shadow.get("enabled", True):
        return False
    from .config import get_reranking_config
    from .reranking import rerank_raw
    from .schema import _get_pool

    pool = _get_pool()
    event = None
    with pool.connection() as conn:
        with conn.transaction():
            conn.execute(
                """UPDATE local.retrieval_events SET shadow_status = 'pending',
                          shadow_started_at = NULL
                   WHERE shadow_status = 'running'
                     AND shadow_started_at < now() - interval '5 minutes'"""
            )
            event = conn.execute(
                """SELECT id::text, query_text, query_ref, query_window_count,
                          model_snapshot, shadow_attempts, config_snapshot,
                          created_at
                   FROM local.retrieval_events
                   WHERE shadow_status = 'pending'
                     AND created_at <= now() - (%s * interval '1 second')
                     AND (shadow_next_attempt_at IS NULL
                          OR shadow_next_attempt_at <= now())
                   ORDER BY created_at FOR UPDATE SKIP LOCKED LIMIT 1""",
                (max(0, float(shadow.get("delay_seconds", 2))),),
            ).fetchone()
            if event:
                conn.execute(
                    """UPDATE local.retrieval_events SET shadow_status = 'running',
                              shadow_started_at = now(), shadow_attempts = shadow_attempts + 1
                       WHERE id = %s::uuid""", (event[0],)
                )
    if not event:
        return False
    (
        event_id, query_text, query_ref, _, model_snapshot, attempts,
        config_snapshot, event_created_at,
    ) = event
    max_attempts = max(1, int(shadow.get("max_attempts", 3)))
    try:
        rerank_cfg = _shadow_rerank_config(get_reranking_config(), shadow)
        expected_model = (model_snapshot or {}).get("reranker_model")
        if expected_model and expected_model != rerank_cfg.get("model"):
            _mark_shadow_skipped(
                pool, event_id, "reranker model identity changed since retrieval"
            )
            return True
        # Window boundaries depend on the EMBEDDING tokenizer, so an embedding
        # model swap invalidates stored query_window_index values even when
        # the reranker is unchanged.
        expected_embedding = (model_snapshot or {}).get("embedding_model")
        if expected_embedding:
            from .config import get_embedding_config
            from .embedding import get_embedding_model_identity

            if expected_embedding != get_embedding_model_identity(get_embedding_config()):
                _mark_shadow_skipped(
                    pool, event_id, "embedding model identity changed since retrieval"
                )
                return True
        # Rerank input text depends on the chunk-context augmentation MODE
        # ('none' | 'mechanical' | 'summary'); scoring an old event under a
        # different mode would contaminate the calibration dataset. Events
        # recorded before the mode was stamped fall back to their legacy boolean
        # (True == the mechanical prefix, which is all that existed then);
        # events recorded before ANY stamping (both keys missing) are scored
        # as-is.
        event_mode = (config_snapshot or {}).get("contextual_augmentation")
        if event_mode is None:
            event_mode = (config_snapshot or {}).get("contextual_embeddings")
        if event_mode is not None:
            from .chunk_context import normalize_augmentation_mode
            from .config import get_contextual_augmentation_mode

            if normalize_augmentation_mode(event_mode) != get_contextual_augmentation_mode():
                _mark_shadow_skipped(
                    pool, event_id,
                    "chunk-context augmentation mode changed since retrieval",
                )
                return True
        # The mode guard above is necessary but NOT sufficient: the augmented
        # rerank text is a function of the mode AND the live summary cache, and
        # the cache is not stamped into the event. A summary generated between
        # retrieval and shadow scoring passes the mode guard ('summary' both
        # sides) and then produces a logit for text the live reranker never saw
        # — silently populating the calibration corpus with unreachable scores
        # (the mandate chunk would record +0.03 where the live path produced
        # −8.16). Cheapest sound test: did any candidate's summary appear or
        # change after this event was recorded?
        if _summary_cache_drifted(pool, event_id, event_created_at):
            _mark_shadow_skipped(
                pool, event_id,
                "document summary cache changed since retrieval",
            )
            return True
        with pool.connection() as conn:
            docs, missing_count = _fetch_candidate_documents(
                conn, event_id, max(1, int(shadow.get("candidate_count", 20)))
            )
        if not query_text:
            _mark_shadow_skipped(
                pool, event_id,
                f"shadow query unavailable ({query_ref or 'prompt storage disabled'})",
            )
            return True
        max_index = max(
            (max(0, int(doc["query_window_index"] or 0)) for doc in docs),
            default=0,
        )
        windows, window_error = _rebuild_live_query_windows(query_text, max_index)
        if window_error:
            _mark_shadow_skipped(pool, event_id, window_error)
            return True
        grouped: dict[str, list[dict]] = {}
        for doc in docs:
            index = max(0, int(doc["query_window_index"] or 0))
            grouped.setdefault(windows[index], []).append(doc)
        scores: dict[str, tuple[float, float]] = {}
        total_ms = 0.0
        for query_window, group in grouped.items():
            raw = rerank_raw(query_window, [item["document"] for item in group], rerank_cfg)
            total_ms += float(raw.get("latency_ms", 0))
            for item, logit, probability in zip(group, raw["logits"], raw["probabilities"]):
                scores[item["candidate_key"]] = (float(logit), float(probability))
        with pool.connection() as conn:
            with conn.cursor() as cur:
                for key, (logit, probability) in scores.items():
                    cur.execute(
                        """UPDATE local.retrieval_candidates
                           SET raw_bge_logit = %s, bge_probability = %s
                           WHERE event_id = %s::uuid AND candidate_key = %s""",
                        (logit, probability, event_id, key),
                    )
                cur.execute(
                    """UPDATE local.retrieval_events
                       SET shadow_status = %s, shadow_finished_at = now(),
                           latency = latency || %s
                       WHERE id = %s::uuid""",
                    ("complete" if len(scores) == missing_count else "partial",
                     _json({"shadow_rerank_ms": round(total_ms, 2)}), event_id),
                )
            conn.commit()
        return True
    except Exception as exc:
        used_attempts = int(attempts or 0) + 1
        terminal = used_attempts >= max_attempts
        # Exponential backoff between attempts. The model host takes tens of
        # seconds to reload a model, so retrying every poll interval just burns
        # max_attempts on a single outage and censors the event forever.
        backoff_seconds = 0 if terminal else _shadow_backoff_seconds(shadow, used_attempts)
        with pool.connection() as conn:
            conn.execute(
                """UPDATE local.retrieval_events SET shadow_status = %s,
                          shadow_error = %s,
                          shadow_finished_at = CASE WHEN %s THEN now() ELSE NULL END,
                          shadow_next_attempt_at = CASE WHEN %s THEN NULL
                              ELSE now() + (%s * interval '1 second') END
                   WHERE id = %s::uuid""",
                ("failed" if terminal else "pending", str(exc)[:1000], terminal,
                 terminal, backoff_seconds, event_id),
            )
            conn.commit()
        logger.debug(
            "Shadow reranking deferred for %s (attempt %d/%d, retry in %ss): %s",
            event_id, used_attempts, max_attempts, backoff_seconds, exc,
        )
        return True


_CLEANUP_INTERVAL_SECONDS = 86400
_CLEANUP_RETRY_SECONDS = 300
# Retention deletes in bounded transactions: each event takes ~50 candidate
# rows with it, and one unbounded DELETE of a large backlog wrote ~117 MiB of
# WAL in under a second (rehearsal: 5.7k events / 295k candidates). A run
# stops after _CLEANUP_MAX_BATCHES; any remaining backlog continues
# _CLEANUP_RETRY_SECONDS later, so its WAL is spread across checkpoints.
_CLEANUP_BATCH_SIZE = 200
_CLEANUP_MAX_BATCHES = 10

_CLEANUP_BATCH_SQL = """
    WITH doomed AS (
        SELECT e.id FROM local.retrieval_events e
         WHERE e.expires_at < now()
           AND NOT EXISTS (
               SELECT 1 FROM local.retrieval_feedback f
                WHERE f.event_id = e.id)
           AND NOT EXISTS (
               SELECT 1 FROM local.retrieval_candidate_feedback cf
                WHERE cf.event_id = e.id)
         ORDER BY e.expires_at
         LIMIT %s
    ), deleted AS (
        DELETE FROM local.retrieval_events e
         USING doomed d
         WHERE e.id = d.id
        RETURNING 1
    ) SELECT count(*) AS count FROM deleted"""


def cleanup_expired() -> int:
    """Delete expired traces, but NEVER human-labeled ones.

    Feedback rows CASCADE from their event, so retention would hard-delete
    hand-labeled ground truth on a rolling window — the scarcest asset the
    calibration work has. Labeled events are exempt from expiry entirely; the
    raw trace (candidate scores) IS the training data and must outlive the
    telemetry retention window.

    Deletes at most _CLEANUP_BATCH_SIZE events per transaction and
    _CLEANUP_MAX_BATCHES batches per call; returns the number deleted.
    """
    from .schema import execute_write

    batch_size = max(1, int(_CLEANUP_BATCH_SIZE))
    total = 0
    for _ in range(max(1, int(_CLEANUP_MAX_BATCHES))):
        row = execute_write(_CLEANUP_BATCH_SQL, (batch_size,), returning=True)
        deleted = int((row or {}).get("count", 0))
        total += deleted
        if deleted < batch_size:
            break
    return total


def _cleanup_backlog_left(deleted: int) -> bool:
    """True when the last cleanup_expired() hit its per-run cap."""
    return deleted >= max(1, int(_CLEANUP_BATCH_SIZE)) * max(1, int(_CLEANUP_MAX_BATCHES))


def _database_disk_full() -> bool:
    from .schema import get_db_status

    return get_db_status().get("status") == "disk_full"


async def retrieval_telemetry_loop() -> None:
    """Background shadow scorer and retention janitor.

    The job body is synchronous (psycopg round-trips plus a blocking host
    rerank HTTP call up to host_timeout_ms) — run it in a worker thread so a
    shadow job can never stall the event loop that serves /hook/prompt-context
    and MCP traffic.
    """
    # None = run the janitor on the first iteration. A 0.0 sentinel compared
    # against time.monotonic() (time since the VM booted) meant retention never
    # ran while uptime stayed under 24h.
    last_cleanup: float | None = None
    while True:
        try:
            cfg = _config()
            shadow = cfg.get("shadow") or {}
            poll = max(0.25, float(shadow.get("poll_seconds", 2)))
            rate = max(0.01, float(shadow.get("max_jobs_per_second", 1)))
            poll = max(poll, 1.0 / rate)
            await asyncio.to_thread(process_one_shadow_job)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.debug("Retrieval telemetry worker idle: %s", exc)
            poll = 5.0
        # Retention has its own error handling: a failing shadow job must not
        # starve cleanup (the telemetry tables are the fastest-growing data).
        now = time.monotonic()
        if last_cleanup is None or now - last_cleanup > _CLEANUP_INTERVAL_SECONDS:
            try:
                if _database_disk_full():
                    # DELETEs write WAL and give no space back before VACUUM:
                    # on a full volume they can only make things worse.
                    logger.info(
                        "Retention cleanup deferred %ds: PostgreSQL volume is full",
                        _CLEANUP_RETRY_SECONDS,
                    )
                    last_cleanup = now - _CLEANUP_INTERVAL_SECONDS + _CLEANUP_RETRY_SECONDS
                else:
                    deleted = await asyncio.to_thread(cleanup_expired)
                    if deleted:
                        logger.info(
                            "Retention cleanup deleted %d expired retrieval event(s)", deleted
                        )
                    # Recover events whose attempts were exhausted by a transient
                    # model-host outage; bounded by age + retention, and attempts
                    # now carry real backoff between them.
                    requeued = await asyncio.to_thread(requeue_failed_shadow_jobs)
                    if requeued:
                        logger.info("Requeued %d failed shadow scoring job(s)", requeued)
                    last_cleanup = time.monotonic()
                    if _cleanup_backlog_left(deleted):
                        # More expired than one run deletes: continue soon.
                        last_cleanup += _CLEANUP_RETRY_SECONDS - _CLEANUP_INTERVAL_SECONDS
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning(
                    "Retrieval telemetry retention cleanup failed, retrying in %ds: %s",
                    _CLEANUP_RETRY_SECONDS, exc,
                )
                # Due again in _CLEANUP_RETRY_SECONDS, not in a day.
                last_cleanup = now - _CLEANUP_INTERVAL_SECONDS + _CLEANUP_RETRY_SECONDS
        await asyncio.sleep(poll)
