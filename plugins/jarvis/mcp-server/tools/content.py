"""Content CRUD operations backed by PostgreSQL + pgvector.

Stores memory content in the local.memories table with proper columns
for classification (category, scope, source, importance_score).

Content types: observation, pattern, summary, relationship, hint,
plan, learning, decision, worklog, memory.
"""

import logging
import threading
import time
from datetime import datetime, timezone
from typing import Optional

from .schema import (
    execute_query, execute_write, is_db_unavailable_error, jsonb_to_metadata,
    metadata_to_jsonb, note_db_write_ok, safe_db_error,
)
from .namespaces import (
    ContentType,
    VALID_CATEGORIES,
    observation_id,
    pattern_id,
    summary_id,
    code_id,
    relationship_id,
    hint_id,
    plan_id,
    learning_id,
    decision_id,
    worklog_id,
    NAMESPACE_OBS,
    NAMESPACE_PATTERN,
    NAMESPACE_SUMMARY,
    NAMESPACE_CODE,
    NAMESPACE_REL,
    NAMESPACE_HINT,
    NAMESPACE_PLAN,
    NAMESPACE_LEARNING,
    NAMESPACE_DECISION,
    NAMESPACE_WORKLOG,
)
from .config import get_memory_config
from .secret_scan import scan_for_secrets

logger = logging.getLogger("jarvis-core")

VALID_CONTENT_TYPES = (
    "observation",
    "pattern",
    "summary",
    "code",
    "relationship",
    "hint",
    "plan",
    "learning",
    "decision",
    "worklog",
)

# Map content_type string to (ContentType enum, NAMESPACE constant, ID generator)
_TYPE_MAP = {
    "observation": (ContentType.OBSERVATION, NAMESPACE_OBS, observation_id),
    "pattern": (ContentType.PATTERN, NAMESPACE_PATTERN, pattern_id),
    "summary": (ContentType.SUMMARY, NAMESPACE_SUMMARY, summary_id),
    "code": (ContentType.CODE, NAMESPACE_CODE, code_id),
    "relationship": (ContentType.RELATIONSHIP, NAMESPACE_REL, relationship_id),
    "hint": (ContentType.HINT, NAMESPACE_HINT, hint_id),
    "plan": (ContentType.PLAN, NAMESPACE_PLAN, plan_id),
    "learning": (ContentType.LEARNING, NAMESPACE_LEARNING, learning_id),
    "decision": (ContentType.DECISION, NAMESPACE_DECISION, decision_id),
    "worklog": (ContentType.WORKLOG, NAMESPACE_WORKLOG, worklog_id),
}

# Types whose ID is ``<namespace>::<epoch ms>`` from the clock rather than a
# caller-given name. Such a row must never be overwritten by another write.
_GENERATED_ID_TYPES = frozenset({"observation", "learning", "worklog"})
_GENERATED_ID_ATTEMPTS = 5
_generated_id_lock = threading.Lock()
_last_generated_ms = 0


def _unique_id_ms() -> int:
    """Epoch milliseconds, strictly increasing within this process.

    Hooks and MCP tools write from worker threads, and the insert used to
    upsert on id: two writes in the same millisecond got the same ID and the
    second silently replaced the first while both reported "stored".
    """
    global _last_generated_ms
    with _generated_id_lock:
        _last_generated_ms = max(int(time.time() * 1000), _last_generated_ms + 1)
        return _last_generated_ms


_INSERT_MEMORY_SQL = """INSERT INTO local.memories
                       (id, document, embedding, category, scope, project,
                        source, importance_score, retrieval_count, status,
                        metadata, created_at, updated_at)
                       VALUES (%s, %s, %s::halfvec, %s, %s, %s,
                               %s, %s, 0.0, 'active',
                               %s::jsonb, %s, %s)"""
# Caller-named IDs (pattern::<name>, ...) are upserts by design.
_UPSERT_MEMORY_SQL = _INSERT_MEMORY_SQL + """
                       ON CONFLICT (id) DO UPDATE SET
                           document = EXCLUDED.document,
                           embedding = EXCLUDED.embedding,
                           category = EXCLUDED.category,
                           scope = EXCLUDED.scope,
                           project = EXCLUDED.project,
                           source = EXCLUDED.source,
                           importance_score = EXCLUDED.importance_score,
                           metadata = EXCLUDED.metadata,
                           updated_at = EXCLUDED.updated_at"""
# Generated IDs: a taken ID (another process, or a clock step back) is
# skipped and a fresh one tried, never overwritten.
_INSERT_NEW_MEMORY_SQL = _INSERT_MEMORY_SQL + """
                       ON CONFLICT (id) DO NOTHING"""

_INGEST_EVENT_LOOKUP_SQL = (
    "SELECT id FROM local.memories WHERE metadata->>'ingest_event_id' = %s LIMIT 1"
)
# Transaction-scoped advisory lock serializing writes of one ingest_event_id.
_INGEST_EVENT_LOCK_SQL = "SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))"
_INGEST_EVENT_LOCK_PREFIX = "jarvis:ingest_event_id:"


def _write_failure(exc: Exception) -> dict:
    """Error result for a failed write, flagging outages as retryable.

    ``retryable`` lets replay pipelines (the auto-extract hook) keep a payload
    that failed because PostgreSQL or the model host was down, instead of
    dropping it like a permanent failure (validation, secret scan, constraint).
    """
    from .model_host_client import ModelHostError

    if is_db_unavailable_error(exc):
        return {"success": False, "error": safe_db_error(exc),
                "retryable": True, "error_kind": "db_unavailable"}
    if isinstance(exc, ModelHostError):
        return {"success": False, "error": str(exc),
                "retryable": True, "error_kind": "model_host_unavailable"}
    return {"success": False, "error": str(exc)}


def content_write(
    content: str,
    content_type: str,
    name: Optional[str] = None,
    importance_score: float = 0.5,
    source: str = "auto-extract",
    tags: Optional[list] = None,
    session_id: Optional[str] = None,
    extra_metadata: Optional[dict] = None,
    skip_secret_scan: bool = False,
) -> dict:
    """Write content to local.memories with proper column values.

    Args:
        content: Document content (markdown)
        content_type: Category (observation, pattern, summary, etc.)
        name: Required for pattern/plan/decision, optional for others
        importance_score: Importance score 0.0-1.0 (default 0.5)
        source: Source of content (default "auto-extract")
        tags: Optional list of tags for categorization
        session_id: Optional session identifier
        extra_metadata: Optional dict of additional metadata key-value pairs
        skip_secret_scan: Skip secret detection (default False)

    Returns:
        Result dict with success, id, content_type, importance_score
    """
    # Validate content type
    if content_type not in VALID_CONTENT_TYPES:
        return {
            "success": False,
            "error": f"Invalid content_type '{content_type}'. "
            f"Valid types: {', '.join(VALID_CONTENT_TYPES)}",
        }

    # Validate name requirement
    if content_type in ("pattern", "plan", "decision") and not name:
        return {
            "success": False,
            "error": f"content_type '{content_type}' requires a name parameter",
        }

    # Validate importance score
    if not 0.0 <= importance_score <= 1.0:
        return {
            "success": False,
            "error": f"importance_score must be between 0.0 and 1.0, got {importance_score}",
        }

    # Secret scan (respects both per-call skip and global config toggle)
    if not skip_secret_scan and get_memory_config().get("secret_detection", True):
        detections = scan_for_secrets(content)
        if detections:
            return {
                "success": False,
                "error": "Secret detected in content",
                "detections": detections,
            }

    ingest_event_id = None
    if extra_metadata and isinstance(extra_metadata, dict):
        raw_event_id = str(extra_metadata.get("ingest_event_id", "")).strip()
        if raw_event_id:
            ingest_event_id = raw_event_id

    # Generate ID
    type_const, namespace, id_gen = _TYPE_MAP[content_type]

    generated_id = content_type in _GENERATED_ID_TYPES
    if generated_id:
        doc_id = id_gen(_unique_id_ms())
    elif content_type == "pattern":
        doc_id = id_gen(name)
    elif content_type == "summary":
        doc_id = id_gen(session_id)
    elif content_type == "code":
        if name and "::" in name:
            file_path, symbol = name.split("::", 1)
            doc_id = id_gen(file_path, symbol)
        else:
            doc_id = id_gen(name or "unknown", "__module__")
    elif content_type == "relationship":
        if name and "::" in name:
            entity_a, entity_b = name.split("::", 1)
            doc_id = id_gen(entity_a, entity_b)
        else:
            return {
                "success": False,
                "error": "relationship type requires name in format 'entity_a::entity_b'",
            }
    elif content_type == "hint":
        if name and "::" in name:
            topic, seq_str = name.split("::", 1)
            doc_id = id_gen(topic, int(seq_str))
        else:
            doc_id = id_gen(name or "general", 0)
    elif content_type == "plan":
        doc_id = id_gen(name)
    elif content_type == "decision":
        doc_id = id_gen(name)
    else:
        return {"success": False, "error": f"Unknown content_type: {content_type}"}

    # Determine scope from extra_metadata
    scope = "global"
    project = None
    if extra_metadata:
        if extra_metadata.get("scope") in ("global", "project"):
            scope = extra_metadata["scope"]
        if extra_metadata.get("project"):
            project = extra_metadata["project"]

    # Guard: scope='project' requires a non-null project name (chk_scope_project)
    if scope == "project" and not project:
        scope = "global"

    # Build remaining metadata (everything NOT in columns)
    metadata = {}
    if tags:
        metadata["tags"] = ",".join(tags)
    if session_id:
        metadata["session_id"] = session_id
    if name:
        metadata["name"] = name
    if extra_metadata:
        # Copy extra_metadata but skip fields that are now columns
        for k, v in extra_metadata.items():
            if k not in ("scope", "project", "source", "importance_score",
                         "retrieval_count", "status", "superseded_by",
                         "type", "tier", "namespace", "promoted",
                         "promoted_at", "original_tier2_id", "category"):
                metadata[k] = v

    # Multi-user attribution
    from jarvis_common.auth import get_current_user

    user = get_current_user()
    if user != "anonymous":
        metadata["user"] = user

    # Write to PostgreSQL
    try:
        from .embedding import get_embedding_service
        from .schema import _get_pool

        def _deduplicated(existing_id: str) -> dict:
            return {
                "success": True,
                "id": existing_id,
                "content_type": content_type,
                "importance_score": importance_score,
                "deduplicated": True,
            }

        # Idempotency for retry/replay pipelines. Cheap early exit before the
        # embedding; re-checked under a lock in the INSERT transaction below.
        if ingest_event_id:
            existing = execute_query(_INGEST_EVENT_LOOKUP_SQL, (ingest_event_id,), fetch="one")
            if existing:
                return _deduplicated(existing["id"])

        # Preserve canonical content while embedding bounded search windows.
        service = get_embedding_service()
        from .document_index import prepare_document, replace_local_chunks

        prepared = prepare_document(content, service)
        embedding = prepared.canonical_embedding

        now = datetime.now(timezone.utc)
        now_ts = now

        # Insert into local.memories with proper columns
        pool = _get_pool()
        with pool.connection() as conn:
            with conn.cursor() as cur:
                if ingest_event_id:
                    # The check above is not atomic with this INSERT: a replay
                    # can overlap the write it replays (the hook answered 503
                    # at its deadline while the worker thread carried on).
                    # Serialize per event id and look again under the lock.
                    cur.execute(
                        _INGEST_EVENT_LOCK_SQL, (_INGEST_EVENT_LOCK_PREFIX + ingest_event_id,)
                    )
                    cur.execute(_INGEST_EVENT_LOOKUP_SQL, (ingest_event_id,))
                    existing_row = cur.fetchone()
                    if existing_row:
                        conn.commit()  # releases the lock
                        return _deduplicated(existing_row[0])
                for _attempt in range(_GENERATED_ID_ATTEMPTS):
                    cur.execute(
                        _INSERT_NEW_MEMORY_SQL if generated_id else _UPSERT_MEMORY_SQL,
                        (
                            doc_id,
                            content,
                            embedding,
                            content_type,
                            scope,
                            project,
                            source,
                            importance_score,
                            metadata_to_jsonb(metadata),
                            now_ts,
                            now_ts,
                        ),
                    )
                    if not generated_id or cur.rowcount != 0:
                        break
                    doc_id = id_gen(_unique_id_ms())  # taken: never overwrite it
                else:
                    raise RuntimeError(
                        f"no free {namespace} id after {_GENERATED_ID_ATTEMPTS} attempts"
                    )
                replace_local_chunks(cur, doc_id, prepared)

                # Transactional outbox: evaluate routing + enqueue sync
                # within the same transaction as the memory INSERT
                try:
                    from .config import get_sync_config
                    from .routing import evaluate_routing
                    from .sync_queue import enqueue_sync
                    from .sync_config import load_routing_rules

                    sync_cfg = get_sync_config()
                    if sync_cfg.get("enabled"):
                        # Include project_path from extra_metadata for path-based routing
                        routing_metadata = dict(metadata)
                        pp = extra_metadata.get("project_path") if extra_metadata else None
                        if isinstance(pp, str) and pp:
                            routing_metadata["project_path"] = pp
                        memory_dict = {
                            "category": content_type,
                            "scope": scope,
                            "project": project,
                            "importance_score": importance_score,
                            "metadata": routing_metadata,
                        }
                        rules = load_routing_rules(sync_cfg)
                        project_groups = sync_cfg.get("project_groups", {})
                        decision = evaluate_routing(
                            memory_dict, rules,
                            sync_cfg.get("strategy", "first-match"),
                            project_groups,
                        )
                        if decision.destinations:
                            enqueue_sync(cur, doc_id, decision.destinations)
                except Exception as e:
                    logger.warning("Sync routing failed for write '%s': %s",
                                   doc_id, e)

                conn.commit()
        note_db_write_ok()

        result = {
            "success": True,
            "id": doc_id,
            "content_type": content_type,
            "importance_score": importance_score,
        }

        # Post-write conflict detection
        try:
            from .conflict import detect_conflicts

            superseded = detect_conflicts(doc_id, content)
            if superseded:
                result["conflicts_resolved"] = len(superseded)
                result["superseded_ids"] = superseded
        except Exception as e:
            logger.debug(f"Conflict detection skipped: {e}")

        return result
    except Exception as e:
        logger.error(f"content_write failed: {e}")
        return _write_failure(e)


def content_read(doc_id: str) -> dict:
    """Read content from local.memories and increment retrieval count.

    Args:
        doc_id: Document ID to read

    Returns:
        Result dict with success, found, id, content, metadata, and column values
    """
    try:
        from .schema import _get_pool

        pool = _get_pool()
        with pool.connection() as conn:
            with conn.cursor() as cur:
                # Atomic read + increment retrieval count
                cur.execute(
                    """UPDATE local.memories
                       SET retrieval_count = retrieval_count + 1,
                           updated_at = now()
                       WHERE id = %s AND status = 'active'
                       RETURNING id, document, category, scope, project,
                                 source, importance_score, retrieval_count,
                                 status, superseded_by, metadata""",
                    (doc_id,),
                )
                row = cur.fetchone()
                conn.commit()

        if not row:
            return {
                "success": True,
                "found": False,
                "id": doc_id,
            }

        metadata = jsonb_to_metadata(row[10])
        return {
            "success": True,
            "found": True,
            "id": row[0],
            "content": row[1],
            "category": row[2],
            "scope": row[3],
            "project": row[4],
            "source": row[5],
            "importance_score": row[6],
            "retrieval_count": row[7],
            "status": row[8],
            "metadata": metadata,
        }
    except Exception as e:
        logger.error(f"content_read failed: {e}")
        return {"success": False, "error": str(e)}


VALID_SORT_OPTIONS = (
    "importance_desc",
    "importance_asc",
    "created_at_desc",
    "created_at_asc",
    "none",
)

# Map sort_by values to SQL ORDER BY clauses (column-based)
_SORT_SQL = {
    "importance_desc": "ORDER BY importance_score DESC NULLS LAST",
    "importance_asc": "ORDER BY importance_score ASC NULLS LAST",
    "created_at_desc": "ORDER BY created_at DESC",
    "created_at_asc": "ORDER BY created_at ASC",
    "none": "",
}


def content_list(
    content_type: Optional[str] = None,
    min_importance: Optional[float] = None,
    source: Optional[str] = None,
    limit: int = 20,
    sort_by: str = "importance_desc",
    session_id: Optional[str] = None,
    include_content: bool = True,
    filter: Optional[dict] = None,
    scope: Optional[str] = None,
    project: Optional[str] = None,
) -> dict:
    """List content from local.memories with column-based filtering.

    Args:
        content_type: Filter by category (observation, pattern, etc.)
        min_importance: Minimum importance score (0.0-1.0)
        source: Filter by source (e.g., "auto-extract")
        limit: Maximum number of results (default 20)
        sort_by: Sort order
        session_id: Filter by session_id
        include_content: Include document text in results (default True)
        filter: Generic metadata filter dict. Any key becomes a JSONB
            equality check (metadata->>'key' = value).
        scope: Filter by scope ('global' or 'project')
        project: Filter by project name (dedicated column)

    Returns:
        Result dict with success, documents, total
    """
    try:
        # Validate sort_by
        if sort_by not in VALID_SORT_OPTIONS:
            return {
                "success": False,
                "error": f"Invalid sort_by '{sort_by}'. "
                f"Valid options: {', '.join(VALID_SORT_OPTIONS)}",
            }

        # Build SQL WHERE clause using columns
        conditions = ["status = 'active'"]
        params = []

        if content_type:
            if content_type not in VALID_CONTENT_TYPES:
                return {
                    "success": False,
                    "error": f"Invalid content_type '{content_type}'. "
                    f"Valid types: {', '.join(VALID_CONTENT_TYPES)}",
                }
            conditions.append("category = %s")
            params.append(content_type)

        if source:
            conditions.append("source = %s")
            params.append(source)

        if min_importance is not None:
            conditions.append("importance_score >= %s")
            params.append(min_importance)

        if scope:
            conditions.append("scope = %s")
            params.append(scope)

        if project:
            conditions.append("project = %s")
            params.append(project)

        # JSONB conditions after column conditions (mock cursor parses
        # column %s params before JSONB %s params)
        if session_id:
            conditions.append("metadata->>'session_id' = %s")
            params.append(session_id)

        # Generic JSONB fallback from filter dict
        if filter:
            # Keys already handled by typed params above
            _KNOWN_LIST_KEYS = {"type", "importance", "tags", "source", "session_id", "scope", "project"}
            for key, val in filter.items():
                if key in _KNOWN_LIST_KEYS or not val:
                    continue
                conditions.append("metadata->>%s = %s")
                params.extend([key, str(val)])

        where_clause = " AND ".join(conditions)
        order_clause = _SORT_SQL.get(sort_by, "")

        # Get total count
        count_sql = f"SELECT count(*) AS cnt FROM local.memories WHERE {where_clause}"
        count_result = execute_query(count_sql, tuple(params), fetch="one")
        total = count_result["cnt"] if count_result else 0

        # Fetch results
        select_cols = "id, category, scope, project, source, importance_score, metadata"
        if include_content:
            select_cols += ", document"
        fetch_sql = f"SELECT {select_cols} FROM local.memories WHERE {where_clause} {order_clause} LIMIT %s"
        rows = execute_query(fetch_sql, tuple(params) + (limit,))

        docs = []
        for row in rows:
            entry = {
                "id": row["id"],
                "category": row["category"],
                "scope": row.get("scope", "global"),
                "source": row.get("source", "auto-extract"),
                "importance_score": row.get("importance_score", 0.5),
                "metadata": jsonb_to_metadata(row["metadata"]),
            }
            if include_content and "document" in row:
                entry["content"] = row["document"]
            docs.append(entry)

        return {
            "success": True,
            "documents": docs,
            "total": total,
            "returned": len(docs),
        }
    except Exception as e:
        logger.error(f"content_list failed: {e}")
        result = {"success": False, "error": str(e)}
        if is_db_unavailable_error(e):
            result.update(error=safe_db_error(e), retryable=True)
        return result


def content_delete(doc_id: str, hard: bool = False) -> dict:
    """Delete content from local.memories.

    Soft delete by default (sets status='deleted', deleted_at=now()).
    Hard delete removes the row entirely.

    Args:
        doc_id: Document ID to delete
        hard: If True, permanently delete. If False, soft delete.

    Returns:
        Result dict with success, id, deleted
    """
    try:
        from .schema import _get_pool

        pool = _get_pool()
        with pool.connection() as conn:
            with conn.cursor() as cur:
                if hard:
                    cur.execute("DELETE FROM local.memories WHERE id = %s", (doc_id,))
                    deleted = cur.rowcount > 0
                else:
                    # Soft delete: also propagate to synced remotes
                    cur.execute(
                        """UPDATE local.memories
                           SET status = 'deleted', deleted_at = now(), updated_at = now()
                           WHERE id = %s
                           RETURNING synced_to""",
                        (doc_id,),
                    )
                    row = cur.fetchone()
                    deleted = row is not None

                    # Enqueue delete sync for remotes that have this memory
                    if deleted and row and row[0]:
                        try:
                            from .sync_queue import enqueue_sync
                            synced_remotes = row[0]
                            if synced_remotes:
                                enqueue_sync(cur, doc_id, synced_remotes)
                        except Exception as e:
                            logger.debug(f"Delete sync propagation skipped: {e}")

                conn.commit()

        if not deleted:
            return {
                "success": True,
                "id": doc_id,
                "deleted": False,
                "reason": "not found",
            }

        return {
            "success": True,
            "id": doc_id,
            "deleted": True,
        }
    except Exception as e:
        logger.error(f"content_delete failed: {e}")
        return {"success": False, "error": str(e)}


def content_upsert(doc_id: str, content: str, metadata: dict) -> dict:
    """Update existing content by ID with re-embedding.

    Unlike content_write which generates new IDs, this updates in-place.
    Extracts column values from metadata for proper storage.

    Args:
        doc_id: Existing document ID
        content: Updated content
        metadata: Updated metadata dict (may contain column values)

    Returns:
        Result dict with success, doc_id, updated
    """
    try:
        from .embedding import get_embedding_service
        from .schema import _get_pool

        # Extract column values from metadata
        category = metadata.pop("type", None) or metadata.pop("category", "observation")
        if category not in VALID_CATEGORIES:
            category = "observation"
        scope = metadata.pop("scope", "global")
        if scope not in ("global", "project"):
            scope = "global"
        project = metadata.pop("project", None)

        # Guard: scope='project' requires a non-null project name (chk_scope_project)
        if scope == "project" and not project:
            scope = "global"
        source = metadata.pop("source", "auto-extract")
        importance_score = 0.5
        raw_imp = metadata.pop("importance_score", None)
        if raw_imp is not None:
            try:
                importance_score = max(0.0, min(1.0, float(raw_imp)))
            except (ValueError, TypeError):
                pass

        # Clean out old tier/namespace fields from metadata
        for old_key in ("tier", "namespace", "promoted", "promoted_at",
                        "original_tier2_id", "retrieval_count", "status",
                        "superseded_by"):
            metadata.pop(old_key, None)

        # Re-embed every bounded window while retaining canonical content.
        service = get_embedding_service()
        from .document_index import prepare_document, replace_local_chunks

        prepared = prepare_document(content, service)
        embedding = prepared.canonical_embedding

        now = datetime.now(timezone.utc)

        pool = _get_pool()
        with pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """INSERT INTO local.memories
                       (id, document, embedding, category, scope, project,
                        source, importance_score, metadata, created_at, updated_at)
                       VALUES (%s, %s, %s::halfvec, %s, %s, %s, %s, %s,
                               %s::jsonb, now(), now())
                       ON CONFLICT (id) DO UPDATE SET
                           document = EXCLUDED.document,
                           embedding = EXCLUDED.embedding,
                           category = EXCLUDED.category,
                           scope = EXCLUDED.scope,
                           project = EXCLUDED.project,
                           source = EXCLUDED.source,
                           importance_score = EXCLUDED.importance_score,
                           metadata = EXCLUDED.metadata,
                           updated_at = EXCLUDED.updated_at""",
                    (
                        doc_id,
                        content,
                        embedding,
                        category,
                        scope,
                        project,
                        source,
                        importance_score,
                        metadata_to_jsonb(metadata),
                    ),
                )
                replace_local_chunks(cur, doc_id, prepared)
                # Transactional outbox: evaluate routing + enqueue sync
                try:
                    from .config import get_sync_config
                    from .routing import evaluate_routing
                    from .sync_queue import enqueue_sync
                    from .sync_config import load_routing_rules

                    sync_cfg = get_sync_config()
                    if sync_cfg.get("enabled"):
                        routing_metadata = dict(metadata) if metadata else {}
                        memory_dict = {
                            "category": category,
                            "scope": scope,
                            "project": project,
                            "importance_score": importance_score,
                            "metadata": routing_metadata,
                        }
                        rules = load_routing_rules(sync_cfg)
                        project_groups = sync_cfg.get("project_groups", {})
                        decision = evaluate_routing(
                            memory_dict, rules,
                            sync_cfg.get("strategy", "first-match"),
                            project_groups,
                        )
                        if decision.destinations:
                            enqueue_sync(cur, doc_id, decision.destinations)
                except Exception as e:
                    logger.warning("Sync routing failed for upsert '%s': %s",
                                   doc_id, e)

                conn.commit()

        return {"success": True, "doc_id": doc_id, "updated": True}
    except Exception as e:
        logger.error(f"content_upsert failed: {e}")
        return {"success": False, "error": str(e)}


# Backward compatibility aliases — will be removed in v3.1
tier2_write = content_write
tier2_read = content_read
tier2_list = content_list
tier2_delete = content_delete
tier2_upsert = content_upsert
