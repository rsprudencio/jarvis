"""Internal HTTP endpoint services for hook handlers.

These are intentionally thin wrappers around existing tool internals so the
hook scripts can call stable local HTTP endpoints without importing tools.*
directly.

Outage contract: every DB checkout here is bounded by HOOK_CONN_TIMEOUT. While
PostgreSQL is known-down (circuit breaker open) the context endpoints answer
immediately with their empty shape plus ``degraded: True``, and ingest answers
``success: False, retryable: True`` so the hook client keeps the payload queued
instead of treating a partial failure as delivered.
"""

from __future__ import annotations

import os
from typing import Any

from .config import (
    get_auto_extract_config,
    get_context_enrichment_config,
    get_todoist_prompt_alerts_config,
    get_worklog_config,
)
from .query import query_vault, semantic_context, _parse_schemas
from .content import content_list, content_write
from .schema import (
    HOOK_CONN_TIMEOUT,
    DatabaseUnavailable,
    checkout_timeout,
    db_available,
    db_unavailable_reason,
)

_DEDUP_JACCARD_THRESHOLD = 0.5
_DEDUP_RELEVANCE_THRESHOLD = 0.95


def _clamp_importance(value: Any, default: float = 0.5) -> float:
    """Parse and clamp importance score to [0.0, 1.0]."""
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = default
    return max(0.0, min(1.0, parsed))


def _safe_int(value: Any, default: int) -> int:
    """Best-effort integer parsing with fallback."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _normalize_tags(value: Any) -> list[str]:
    """Normalize tags to a list of non-empty strings."""
    if not isinstance(value, list):
        return []
    return [str(tag).strip() for tag in value if str(tag).strip()]


def _safe_str(value: Any) -> str:
    """Convert nullable values to stripped string."""
    if value is None:
        return ""
    return str(value).strip()


def _jaccard_similarity(text_a: str, text_b: str) -> float:
    """Compute simple word-overlap Jaccard similarity."""
    words_a = set(text_a.lower().split())
    words_b = set(text_b.lower().split())
    if not words_a and not words_b:
        return 0.0
    intersection = words_a & words_b
    union = words_a | words_b
    return len(intersection) / len(union) if union else 0.0


def _strip_db_prefix(error: Any) -> str:
    """Drop a leading "Database unavailable:" so re-wrapping doesn't stutter."""
    text = _safe_str(error)
    prefix = "database unavailable:"
    if text.lower().startswith(prefix):
        text = text[len(prefix):].strip()
    return text or "PostgreSQL is unreachable"


def _db_unavailable_error(reason: Any) -> str:
    return f"database unavailable: {_strip_db_prefix(reason)}"


def _retryable_failure(error: str, observations: list | None = None) -> dict:
    """Ingest could not finish for a transient reason: the caller must retry.

    Observations handled before the failure are included for debugging.
    Replaying the whole payload is safe: content_write deduplicates by
    ingest_event_id and observation dedup matches anything already stored.
    """
    return {
        "success": False,
        "retryable": True,
        "error": error,
        "observations": observations or [],
        "worklog": None,
    }


def _transient_write_error(write_result: dict) -> str | None:
    """Client-facing error when a write failed for a retryable reason."""
    if not write_result.get("retryable"):
        return None
    if write_result.get("error_kind") == "model_host_unavailable":
        return f"embedding service unavailable: {_safe_str(write_result.get('error'))}"
    return _db_unavailable_error(write_result.get("error"))


def _is_duplicate_observation(content: str, threshold: float) -> bool:
    """Embedding-similarity dedup for observations.

    Raises DatabaseUnavailable when the lookup failed because PostgreSQL is
    down: "unknown" must not be read as "not a duplicate" and written anyway.

    No cross-encoder reranking: the gate below reads raw cosine similarity,
    and a host rerank (up to host_timeout_ms, 1.5s) per observation pushed
    healthy ingests past the 2s hook deadline.
    """
    result = query_vault(
        query=content,
        n_results=5,
        filter={"type": "observation"},
        purpose="duplicate_detection",
        telemetry_user_facing=False,
        telemetry_query_ref="auto_extract_candidate",
        rerank=False,
    )
    if not result.get("success") and result.get("retryable"):
        raise DatabaseUnavailable(_strip_db_prefix(result.get("error")))
    if not result.get("success") or not result.get("results"):
        return False

    # Duplicate detection is a similarity question — gate on raw similarity,
    # not the importance-boosted relevance score (an important observation is
    # not more likely to be a duplicate). query_vault orders by relevance, so
    # the top hit is not necessarily the nearest-by-similarity observation
    # (importance nudges and staleness penalties can push a true near-duplicate
    # below an unrelated-but-fresh one) — take the max similarity across the
    # candidate window. Fall back to relevance for entries without similarity.
    best = max(
        float(r.get("similarity", r.get("relevance", 0.0)))
        for r in result["results"]
    )
    return best >= threshold


def _is_duplicate_worklog(task_summary: str, session_id: str, threshold: float) -> bool:
    """Cross-session Jaccard dedup for worklogs.

    Checks the last 20 worklogs globally (not session-scoped) so that
    repeated work on the same project across sessions doesn't generate
    redundant entries.  The session_id parameter is accepted for API
    compatibility but no longer used for filtering.
    """
    _ = session_id  # kept for call-site compat
    result = content_list(
        content_type="worklog",
        limit=20,
        sort_by="created_at_desc",
    )
    if not result.get("success") and result.get("retryable"):
        raise DatabaseUnavailable(_strip_db_prefix(result.get("error")))
    if not result.get("success") or not result.get("documents"):
        return False

    for doc in result["documents"]:
        existing = _safe_str(doc.get("content"))
        if existing and _jaccard_similarity(task_summary, existing) >= threshold:
            return True
    return False


def _extract_workstreams(limit: int) -> list[str]:
    """Load known workstream names from recent worklog entries."""
    result = content_list(
        content_type="worklog",
        limit=limit,
        sort_by="created_at_desc",
        include_content=False,
    )
    if not result.get("success") and result.get("retryable"):
        raise DatabaseUnavailable(_strip_db_prefix(result.get("error")))
    if not result.get("success") or not result.get("documents"):
        return []

    workstreams = set()
    for doc in result["documents"]:
        metadata = doc.get("metadata", {})
        ws = _safe_str(metadata.get("workstream"))
        if ws and ws != "misc":
            workstreams.add(ws)
    return sorted(workstreams)


def _build_common_metadata(context: dict, include_project: bool = False) -> dict:
    """Build shared metadata fields for observation/worklog writes."""
    meta: dict[str, str] = {}

    project_path = _safe_str(context.get("project_path"))
    if project_path:
        meta["project_path"] = project_path
        if include_project:
            meta["project"] = os.path.basename(project_path)

    git_branch = _safe_str(context.get("git_branch"))
    if git_branch:
        meta["git_branch"] = git_branch

    relevant_files = context.get("relevant_files")
    if isinstance(relevant_files, list):
        cleaned = [str(path).strip() for path in relevant_files if str(path).strip()]
        if cleaned:
            meta["relevant_files"] = ",".join(cleaned)

    file_mtimes = context.get("file_mtimes")
    if isinstance(file_mtimes, dict) and file_mtimes:
        meta["file_mtimes"] = file_mtimes

    session_id = _safe_str(context.get("session_id"))
    if session_id:
        meta["session_id"] = session_id

    transcript_line = context.get("transcript_line")
    if transcript_line is not None and transcript_line != "":
        try:
            parsed_line = int(transcript_line)
        except (TypeError, ValueError):
            parsed_line = None
        if parsed_line is not None and parsed_line >= 0:
            meta["transcript_line"] = str(parsed_line)

    return meta


def get_prompt_context(prompt: str) -> dict:
    """Return per-prompt semantic context and todoist alert config."""
    config = get_context_enrichment_config()
    todoist_cfg = get_todoist_prompt_alerts_config()

    budget = _safe_int(config.get("budget"), 8000)
    threshold = float(config.get("threshold", 0.85))
    enabled = bool(config.get("enabled", True))
    debug = bool(config.get("debug", False))
    max_results = max(1, min(100, _safe_int(config.get("max_results"), 20)))

    result = {
        "success": True,
        "enabled": enabled,
        "debug": debug,
        "matches": [],
        "query_ms": 0,
        "total_searched": 0,
        "semantic_duplicates_suppressed": 0,
        "budget_used": {"core": 0, "vault": 0, "total": budget},
        "todoist_prompt_alerts": {
            "enabled": bool(todoist_cfg.get("enabled", False)),
            "max_per_category": _safe_int(todoist_cfg.get("max_per_category"), 3),
        },
    }

    if not enabled or not _safe_str(prompt):
        return result

    # PostgreSQL known-down: answer now. An empty injection is the degradation
    # the hook already handles; waiting on the pool is not.
    if not db_available():
        result.update(degraded=True, error=_db_unavailable_error(db_unavailable_reason()))
        return result

    # Default to all schemas for session injection (local + obsidian + discovered remotes).
    schemas_str = config.get("schemas", "all")
    with checkout_timeout(HOOK_CONN_TIMEOUT):
        search = semantic_context(
            query=prompt,
            threshold=threshold,
            budget=budget,
            skip_retrieval_increment=False,
            schemas=_parse_schemas(schemas_str),
            max_results=max_results,
        )
    # Degraded only when nothing usable came back: matches found before a
    # late failure (retrieval-count bump, telemetry) are still worth injecting.
    if search.get("degraded") or (not search.get("matches") and not db_available()):
        reason = search.get("error") or db_unavailable_reason()
        result.update(degraded=True, error=_db_unavailable_error(reason))
        return result
    result.update(
        {
            "matches": search.get("matches", []),
            "query_ms": search.get("query_ms", 0),
            "total_searched": search.get("total_searched", 0),
            "semantic_duplicates_suppressed": search.get(
                "semantic_duplicates_suppressed", 0
            ),
            "budget_used": search.get(
                "budget_used",
                {"core": 0, "vault": 0, "total": budget},
            ),
            "trace_id": search.get("trace_id"),
        }
    )
    return result


def get_auto_extract_context(workstream_limit: int = 30) -> dict:
    """Return extraction/worklog config and known workstreams."""
    auto_extract = get_auto_extract_config()
    worklog = get_worklog_config()
    limit = max(1, min(200, _safe_int(workstream_limit, 30)))

    known_workstreams = []
    degraded_error = None
    if bool(worklog.get("enabled", True)):
        if not db_available():
            degraded_error = _db_unavailable_error(db_unavailable_reason())
        else:
            try:
                with checkout_timeout(HOOK_CONN_TIMEOUT):
                    known_workstreams = _extract_workstreams(limit)
            except DatabaseUnavailable as exc:
                degraded_error = _db_unavailable_error(exc)

    response = {
        "success": True,
        "auto_extract": {
            "mode": auto_extract.get("mode", "background"),
            "min_turn_chars": _safe_int(auto_extract.get("min_turn_chars"), 200),
            "max_transcript_lines": _safe_int(
                auto_extract.get("max_transcript_lines"), 500
            ),
            "max_observations": _safe_int(auto_extract.get("max_observations"), 3),
            "dedup_threshold": float(
                auto_extract.get("dedup_threshold", _DEDUP_RELEVANCE_THRESHOLD)
            ),
            "debug": bool(auto_extract.get("debug", False)),
        },
        "worklog": {
            "enabled": bool(worklog.get("enabled", True)),
            "dedup_threshold": float(
                worklog.get("dedup_threshold", _DEDUP_JACCARD_THRESHOLD)
            ),
        },
        "known_workstreams": known_workstreams,
    }
    if degraded_error:
        response.update(degraded=True, error=degraded_error)
    return response


def ingest_auto_extract(payload: dict) -> dict:
    """Persist extracted observations/worklog with dedup checks.

    Returns ``success: False, retryable: True`` as soon as a write or dedup
    lookup fails because PostgreSQL (or the model host) is unavailable, and
    makes no further attempts. ``success: True`` with per-item ``error``
    entries is reserved for permanent failures (validation, secret scan).
    """
    # Breaker open: refuse before any work so the payload stays queued.
    if not db_available():
        return _retryable_failure(_db_unavailable_error(db_unavailable_reason()))
    with checkout_timeout(HOOK_CONN_TIMEOUT):
        return _ingest_auto_extract(payload)


def _ingest_auto_extract(payload: dict) -> dict:
    observations_payload = payload.get("observations", [])
    if not isinstance(observations_payload, list):
        observations_payload = []

    worklog_payload = payload.get("worklog")
    if worklog_payload is not None and not isinstance(worklog_payload, dict):
        worklog_payload = None

    context = payload.get("context", {})
    if not isinstance(context, dict):
        context = {}

    dedup_cfg = payload.get("dedup", {})
    if not isinstance(dedup_cfg, dict):
        dedup_cfg = {}

    obs_threshold = float(
        dedup_cfg.get("observation_threshold", _DEDUP_RELEVANCE_THRESHOLD)
    )
    worklog_threshold = float(
        dedup_cfg.get("worklog_threshold", _DEDUP_JACCARD_THRESHOLD)
    )

    observation_results = []
    for raw in observations_payload:
        if not isinstance(raw, dict):
            observation_results.append({"status": "error", "id": "", "error": "invalid"})
            continue

        content = _safe_str(raw.get("content"))
        if not content:
            observation_results.append(
                {"status": "error", "id": "", "error": "content is required"}
            )
            continue

        try:
            is_duplicate = _is_duplicate_observation(content, obs_threshold)
        except DatabaseUnavailable as exc:
            return _retryable_failure(_db_unavailable_error(exc), observation_results)
        if is_duplicate:
            observation_results.append({"status": "duplicate", "id": "", "error": ""})
            continue

        metadata = _build_common_metadata(context, include_project=False)
        scope = _safe_str(raw.get("scope"))
        if scope in ("project", "global"):
            metadata["scope"] = scope
            # When scope='project', ensure project name is set so the
            # chk_scope_project DB constraint is satisfied.
            if scope == "project":
                project_name = _safe_str(raw.get("project"))
                if not project_name:
                    project_path = metadata.get("project_path", "")
                    project_name = os.path.basename(project_path) if project_path else ""
                if project_name:
                    metadata["project"] = project_name

        ingest_event_id = _safe_str(raw.get("ingest_event_id"))
        if ingest_event_id:
            metadata["ingest_event_id"] = ingest_event_id

        write_result = content_write(
            content=content,
            content_type="observation",
            importance_score=_clamp_importance(raw.get("importance_score"), default=0.5),
            source="auto-extract:stop-hook",
            tags=_normalize_tags(raw.get("tags")),
            extra_metadata=metadata or None,
            skip_secret_scan=False,
        )

        transient_error = _transient_write_error(write_result)
        if transient_error:
            return _retryable_failure(transient_error, observation_results)

        if write_result.get("success"):
            status = "duplicate" if write_result.get("deduplicated") else "stored"
            observation_results.append(
                {"status": status, "id": write_result.get("id", ""), "error": ""}
            )
        else:
            observation_results.append(
                {
                    "status": "error",
                    "id": "",
                    "error": _safe_str(write_result.get("error")) or "write failed",
                }
            )

    worklog_result = {"status": "duplicate", "id": "", "error": ""}
    if isinstance(worklog_payload, dict):
        task_summary = _safe_str(worklog_payload.get("task_summary"))
        if task_summary:
            session_id = _safe_str(context.get("session_id"))
            try:
                is_duplicate = _is_duplicate_worklog(
                    task_summary, session_id, worklog_threshold
                )
            except DatabaseUnavailable as exc:
                return _retryable_failure(_db_unavailable_error(exc), observation_results)
            if is_duplicate:
                worklog_result = {"status": "duplicate", "id": "", "error": ""}
            else:
                metadata = _build_common_metadata(context, include_project=True)

                workstream = _safe_str(worklog_payload.get("workstream")) or "misc"
                activity_type = (
                    _safe_str(worklog_payload.get("activity_type")) or "other"
                )
                metadata["workstream"] = workstream
                metadata["activity_type"] = activity_type

                ingest_event_id = _safe_str(worklog_payload.get("ingest_event_id"))
                if ingest_event_id:
                    metadata["ingest_event_id"] = ingest_event_id

                write_result = content_write(
                    content=task_summary,
                    content_type="worklog",
                    importance_score=0.5,
                    source="auto-extract:stop-hook:worklog",
                    tags=_normalize_tags(worklog_payload.get("tags")),
                    extra_metadata=metadata,
                    skip_secret_scan=False,
                )

                transient_error = _transient_write_error(write_result)
                if transient_error:
                    return _retryable_failure(transient_error, observation_results)

                if write_result.get("success"):
                    status = (
                        "duplicate" if write_result.get("deduplicated") else "stored"
                    )
                    worklog_result = {
                        "status": status,
                        "id": write_result.get("id", ""),
                        "error": "",
                    }
                else:
                    worklog_result = {
                        "status": "error",
                        "id": "",
                        "error": _safe_str(write_result.get("error")) or "write failed",
                    }
        else:
            worklog_result = {"status": "error", "id": "", "error": "task_summary is required"}

    return {
        "success": True,
        "observations": observation_results,
        "worklog": worklog_result,
    }
