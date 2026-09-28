"""Vault memory querying for semantic search.

Provides query, read, and stats operations across local.memories and
obsidian.documents schemas with pgvector embeddings. Uses per-schema CTEs
with UNION ALL for cross-schema search (DAR F5: preserves HNSW index usage).

All document IDs use namespaced format (vault:: prefix) for type-safe identification.
"""

import logging
import os
import re
import time
from datetime import datetime, timezone
from typing import Optional

import numpy as np

from .schema import (
    DatabaseUnavailable, execute_query, is_db_unavailable_error,
    jsonb_to_metadata, metadata_to_jsonb, safe_db_error,
)
from .paths import get_path, SENSITIVE_PATHS
from .namespaces import parse_id, ALL_TYPES, schema_for_id, SCHEMA_LOCAL, SCHEMA_OBSIDIAN
from .expansion import expand_query as _expand_query
from .config import (
    get_context_enrichment_config, get_contextual_augmentation_mode,
    get_contextual_embeddings_enabled, get_decay_config, get_expansion_config,
    get_lexical_config, get_ranking_config, get_reranking_config,
    get_staleness_config,
)
from .format_support import detect_format
from .ranking import DEFAULT_IMPORTANCE_WEIGHT, compute_unified_score, score_memory
from .staleness import check_staleness, deserialize_mtimes


logger = logging.getLogger(__name__)

_INJECTION_DEDUP_TYPES = {"memory", "observation", "worklog"}
_INJECTION_TYPE_PRIORITY = {"memory": 3, "observation": 2, "worklog": 1}
_QUERY_WINDOW_TOKENS = 2048
_QUERY_WINDOW_OVERLAP = 128


def _rerank_doc_text(entry: dict, mode: str, summaries: Optional[dict] = None) -> str:
    """Return the text to hand the reranker for one candidate entry.

    The cross-encoder must see exactly what the embedder saw, so vault (obsidian)
    fragments are augmented with the same document-context prefix used at index
    time — including the file's cached LLM summary in 'summary' mode (see
    tools/chunk_context.py and tools/context_summary.py). The entry's stored
    ``document`` is left untouched — only this returned copy carries the prefix.
    Local memories are never augmented (their embed path isn't either), so live
    and shadow scores stay consistent per schema.

    ``summaries`` is the batch cache lookup from ``_rerank_summaries`` — one
    query per rerank batch, never per row, and never a generation.
    """
    from .chunk_context import MODE_NONE, augment_vault_row

    document = entry.get("document") or ""
    meta = entry.get("metadata", {}) or {}
    if entry.get("_schema") != "obsidian":
        return document
    parent_file = entry.get("parent_file") or meta.get("parent_file", "")
    return augment_vault_row(
        document,
        parent_file=parent_file,
        title=meta.get("title", ""),
        chunk_heading=meta.get("chunk_heading", ""),
        chunk_total=meta.get("chunk_total", 1),
        mode=mode,
        summary=(summaries or {}).get(parent_file),
    )


def _rerank_summaries(entries: list) -> dict:
    """Batched, read-only summary lookup for one rerank batch.

    Collects the vault fragments' parent files and resolves their cached
    summaries in a single query. Returns ``{}`` outside 'summary' mode or on any
    failure — the rerank text then falls back to the mechanical prefix.
    """
    try:
        from .context_summary import fetch_document_summaries

        parent_files = {
            entry.get("parent_file") or (entry.get("metadata") or {}).get("parent_file", "")
            for entry in entries
            if entry.get("_schema") == "obsidian"
        }
        return fetch_document_summaries([f for f in parent_files if f])
    except Exception:
        return {}


def _record_empty_trace(
    purpose: str,
    query: str,
    *,
    outcome: str = "empty",
    error: str = "",
    user_name: Optional[str] = None,
    user_facing: bool = True,
    query_ref: Optional[str] = None,
) -> Optional[str]:
    """Best-effort trace for exits that occur before candidates exist."""
    try:
        from .config import get_embedding_config
        from .retrieval_telemetry import record_event

        embedding = get_embedding_config()
        return record_event(
            purpose=purpose, query=query, candidates=[],
            funnel={"ann_unique": 0, "returned": 0},
            latency={"total_ms": 0}, outcome=outcome,
            status="failed" if error else "complete", user_name=user_name,
            user_facing=user_facing, query_ref=query_ref,
            model_snapshot={
                "embedding_model": embedding.get("model_id"),
                "embedding_backend": embedding.get("backend"),
            },
            config_snapshot={"error": error[:500]} if error else {},
            shadow_eligible=False,
        )
    except Exception:
        return None


def _prepare_query_windows(query: str, service) -> tuple[list[dict], dict]:
    """Expand and bound every part of an arbitrarily large query."""
    from .text_windows import split_text_windows

    tokenizer = getattr(service, "tokenize", None)
    base_windows = split_text_windows(
        query,
        max_tokens=_QUERY_WINDOW_TOKENS,
        overlap_tokens=_QUERY_WINDOW_OVERLAP,
        tokenize=tokenizer,
    ) or [""]
    expansion_config = get_expansion_config()
    prepared = []
    terms_added = []
    intents = []
    for base_index, base in enumerate(base_windows):
        expansion = _expand_query(base, expansion_config)
        terms_added.extend(expansion.get("terms_added", []))
        if expansion.get("intent"):
            intents.append(expansion["intent"])
        expanded_windows = split_text_windows(
            expansion["expanded"],
            max_tokens=_QUERY_WINDOW_TOKENS,
            overlap_tokens=_QUERY_WINDOW_OVERLAP,
            tokenize=tokenizer,
        ) or [base]
        for search_text in expanded_windows:
            prepared.append(
                {
                    "search_text": search_text,
                    "rerank_text": base,
                    "base_index": base_index,
                }
            )
    # Expansion overlap can produce an identical inference window twice.
    unique = []
    seen = set()
    for window in prepared:
        key = (window["search_text"], window["rerank_text"])
        if key not in seen:
            seen.add(key)
            unique.append(window)
    return unique, {
        "terms_added": list(dict.fromkeys(terms_added)),
        "intent": intents[0] if intents else None,
        "base_window_count": len(base_windows),
        "inference_window_count": len(unique),
    }


def _search_query_windows(
    query: str,
    service,
    fetch_count: int,
    *,
    filter_dict: Optional[dict] = None,
    user: Optional[str] = None,
    schemas: Optional[list[str]] = None,
) -> tuple[list[dict], dict]:
    """Search every bounded query window and keep each row's best match."""
    windows, expansion = _prepare_query_windows(query, service)
    texts = [window["search_text"] for window in windows]
    if len(texts) == 1:
        embeddings = [service.encode(texts[0])]
    else:
        embeddings = service.encode_batch(texts, batch_size=8)
    if len(embeddings) != len(windows):
        raise ValueError(
            f"embedding service returned {len(embeddings)} vectors "
            f"for {len(windows)} query windows"
        )

    best: dict[tuple[str, str], dict] = {}
    for window, embedding in zip(windows, embeddings):
        rows = _cross_schema_search(
            embedding,
            fetch_count,
            filter_dict=filter_dict,
            user=user,
            schemas=schemas,
        )
        for row in rows:
            key = (str(row.get("_schema", "obsidian")), str(row.get("id", "")))
            candidate = dict(row)
            candidate["_query_window"] = window["rerank_text"]
            candidate["_query_window_index"] = window["base_index"]
            if key not in best or float(candidate["distance"]) < float(best[key]["distance"]):
                best[key] = candidate
    return sorted(best.values(), key=lambda row: float(row["distance"])), expansion


def _lexical_base_window(query: str, service) -> str:
    """Return base window 0 of the query — the lexical rows' rerank window.

    Multi-window prompts extract lexical terms from the full prompt but score
    every lexical row against base window 0 (matches the spec: lexical rows use
    base window 0 as their rerank query window).
    """
    from .text_windows import split_text_windows

    tokenizer = getattr(service, "tokenize", None)
    windows = split_text_windows(
        query,
        max_tokens=_QUERY_WINDOW_TOKENS,
        overlap_tokens=_QUERY_WINDOW_OVERLAP,
        tokenize=tokenizer,
    ) or [query]
    return windows[0]


def _augment_rows_with_lexical(
    rows: list[dict],
    query: str,
    service,
    *,
    schemas: Optional[list[str]] = None,
    user: Optional[str] = None,
    filter_dict: Optional[dict] = None,
) -> tuple[list[dict], dict]:
    """Union a statistical (lexical) recall channel into the ANN ``rows``.

    ANN rows are tagged ``_channel='semantic'`` (upgraded to ``'both'`` when
    also surfaced lexically); lexical-only rows are appended with
    ``_channel='lexical'`` carrying their true raw cosine (via ``distance``) so
    they flow through the SAME downstream scoring/dedup/rerank/telemetry path.

    Access control: ``user`` is threaded into the lexical SQL so it honors the
    same per-user isolation the ANN path applies. When ``filter_dict`` is
    non-empty the lexical union is SKIPPED entirely (the lexical SQL does not yet
    honor metadata filters — Phase 2), so it never returns rows that violate the
    caller's filter.

    Fail-open: any error (or a disabled channel, no informative terms, an
    incompatible schema filter) yields the ANN rows unchanged. Returns
    ``(rows, stats)`` where stats has ``lexical_candidates`` (rows fetched),
    ``lexical_added`` (rows appended post-dedup) and ``informative_terms``.
    """
    stats = {"lexical_candidates": 0, "lexical_added": 0, "informative_terms": []}

    # Tag ANN rows before any early return so every row carries a channel.
    for row in rows:
        if "_channel" not in row:
            row["_channel"] = "semantic"

    lexical_config = get_lexical_config()
    if not lexical_config.get("enabled", True):
        return rows, stats

    # Phase-1 limitation: the lexical SQL honors user scope but NOT arbitrary
    # metadata filters. Skip the channel rather than silently return rows that
    # violate the caller's filter (type/directory/project/etc.).
    if filter_dict:
        return rows, stats

    if schemas is None:
        target_schemas = list(_LEXICAL_SCHEMAS)
    else:
        target_schemas = [s for s in schemas if s in _LEXICAL_SCHEMAS]
    if not target_schemas:
        return rows, stats

    try:
        from . import lexical
        from .schema import _get_pool

        pool = _get_pool()
        with pool.connection() as conn:
            lexemes = lexical.extract_query_lexemes(conn, query)
            terms = lexical.informative_terms(
                conn,
                lexemes,
                max_df_ratio=float(lexical_config.get("max_df_ratio", 0.10)),
                max_terms=int(lexical_config.get("max_terms", 8)),
            )
            stats["informative_terms"] = terms
            if not terms:
                return rows, stats

            base0 = _lexical_base_window(query, service)
            query_embedding = service.encode(base0)
            candidate_limit = int(lexical_config.get("candidate_limit", 30))

            # (schema, id) already present via ANN → dedup to 'both'.
            ann_by_key: dict[tuple[str, str], dict] = {}
            for row in rows:
                ann_by_key[(str(row.get("_schema", "obsidian")), str(row.get("id", "")))] = row
            seen_keys = set(ann_by_key)

            appended: list[dict] = []
            for schema in target_schemas:
                lex_rows = lexical.lexical_candidates(
                    conn,
                    terms,
                    schema=schema,
                    limit=candidate_limit,
                    query_embedding=query_embedding,
                    user=user,
                )
                stats["lexical_candidates"] += len(lex_rows)
                for lex in lex_rows:
                    key = (str(lex.get("_schema", schema)), str(lex.get("id", "")))
                    if key in ann_by_key:
                        ann_by_key[key]["_channel"] = "both"
                        continue
                    if key in seen_keys:
                        continue  # already appended (cross-schema dup guard)
                    lex["_channel"] = "lexical"
                    # The channel's native rank: lexical_candidates returns
                    # rows rarest-term-first; the reserved rerank seats honor
                    # this order (see _select_rerank_retained).
                    lex["_lexical_rank"] = len(appended)
                    lex.setdefault("_query_window", base0)
                    lex.setdefault("_query_window_index", 0)
                    appended.append(lex)
                    seen_keys.add(key)

            stats["lexical_added"] = len(appended)
            rows.extend(appended)
    except Exception as exc:
        logger.debug("Lexical channel skipped: %s", exc)

    return rows, stats


def _select_rerank_retained(
    best_per_file: dict,
    candidate_count: int,
    lexical_slots: int,
    result_limit: int = 0,
) -> set:
    """Pick the pre-rerank candidate-cap survivors — passers exempt from failers.

    STRICT RECALL-ADDITIVITY: a cosine-PASSING row (``_cosine_ok`` truthy —
    always true in the gate-free ``query_vault`` path, which applies no cosine
    threshold) is NEVER evicted by a cosine-failing row. The cap ranks passers
    ONLY against each other, retaining the top ``max(candidate_count,
    result_limit)`` of them by pre-score. Flooring at ``result_limit`` (the
    caller's final result cap) guarantees the cap never drops a passer that the
    reranking-disabled path would inject, so enabled-injected ⊇ disabled-injected.

    Cosine-failing rows reach this point only via the LEXICAL recall channel
    (semantic_context drops cosine-failing ANN rows at the threshold; Phase 1).
    Up to ``lexical_slots`` such ``_channel == 'lexical'`` rows get reserved
    seats IN ADDITION to the passer winners, so their low cosine cannot evict
    them before the reranker can earn them a logit rescue. Everything else is
    capped.
    """
    passer_keep = max(int(candidate_count), int(result_limit))
    ranked = sorted(
        best_per_file.items(),
        key=lambda item: float(item[1].get("relevance", 0.0)),
        reverse=True,
    )
    passers = [(key, entry) for key, entry in ranked if entry.get("_cosine_ok", True)]
    retained = {key for key, _ in passers[:passer_keep]}
    if lexical_slots > 0:
        # Reserved seats go by the LEXICAL channel's own native rank (rarity
        # order from informative_terms — stamped as _lexical_rank at union
        # time), NOT by cosine pre-score. Ranking lexical rows by cosine here
        # re-introduces the IDF-blind flooding one stage down: common-term
        # matches carry higher cosine and would take every seat while the
        # df=1 term's rows (the reason this channel exists) never reach BGE.
        lexical_rows = sorted(
            (
                (key, entry) for key, entry in best_per_file.items()
                if key not in retained and entry.get("_channel") == "lexical"
            ),
            key=lambda item: (
                item[1].get("_lexical_rank")
                if isinstance(item[1].get("_lexical_rank"), int)
                else 1_000_000,
                -float(item[1].get("relevance", 0.0)),
            ),
        )
        for key, _entry in lexical_rows[:lexical_slots]:
            retained.add(key)
    return retained


_LEXICAL_SCHEMAS = ("obsidian", "local")


def _semantic_deduplicate_context(
    entries: list[dict],
    embedding_service,
    similarity_threshold: float,
) -> tuple[list[dict], int]:
    """Collapse redundant local memory/observation/worklog candidates.

    Only records from the same project are compared. Durable strategic memories
    win over observations and worklogs even when a transient record has a small
    ranking advantage. Any embedding failure preserves the original candidates.
    """
    eligible = []
    for index, entry in enumerate(entries):
        meta = entry.get("metadata") or {}
        doc_type = str(meta.get("type") or "").lower()
        if entry.get("_schema") != SCHEMA_LOCAL or doc_type not in _INJECTION_DEDUP_TYPES:
            continue
        document = entry.get("document") or ""
        if document.strip():
            eligible.append((index, entry, doc_type, document))

    if len(eligible) < 2:
        return entries, 0

    try:
        vectors = np.asarray(
            embedding_service.encode_batch([item[3] for item in eligible]),
            dtype=np.float32,
        )
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        vectors = vectors / np.clip(norms, a_min=1e-12, a_max=None)
    except Exception as exc:
        logger.warning("Semantic injection dedup skipped: %s", exc)
        return entries, 0

    threshold = min(max(float(similarity_threshold), -1.0), 1.0)
    # Passer-priority: cosine-passing rows are processed (kept) strictly ahead
    # of logit-rescued cosine-failers, so a rescued row can never suppress a
    # near-duplicate passer that today's pipeline injects.
    ordered = sorted(
        range(len(eligible)),
        key=lambda pos: (
            -int(bool(eligible[pos][1].get("_cosine_ok", True))),
            -_INJECTION_TYPE_PRIORITY[eligible[pos][2]],
            -float(eligible[pos][1].get("relevance", 0.0)),
        ),
    )
    kept_positions: list[int] = []
    suppressed_indexes: set[int] = set()

    for pos in ordered:
        index, entry, _, _ = eligible[pos]
        project = str((entry.get("metadata") or {}).get("project") or "")
        duplicate = False
        for kept_pos in kept_positions:
            kept_entry = eligible[kept_pos][1]
            kept_project = str(
                (kept_entry.get("metadata") or {}).get("project") or ""
            )
            if project != kept_project:
                continue
            if float(np.dot(vectors[pos], vectors[kept_pos])) >= threshold:
                duplicate = True
                break
        if duplicate:
            suppressed_indexes.add(index)
        else:
            kept_positions.append(pos)

    if not suppressed_indexes:
        return entries, 0
    return (
        [entry for index, entry in enumerate(entries) if index not in suppressed_indexes],
        len(suppressed_indexes),
    )


def _parse_optional_row_datetime(value) -> Optional[datetime]:
    """Parse an optional datetime from row data without inventing a timestamp."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if isinstance(value, str):
        try:
            dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
            return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
        except (ValueError, TypeError):
            pass
    return None


def _parse_row_datetime(value) -> datetime:
    """Parse a required datetime from row data (missing/invalid → now)."""
    return _parse_optional_row_datetime(value) or datetime.now(timezone.utc)


def _annotate_staleness(raw_entries: list, staleness_config: dict) -> None:
    """In-place annotate raw query entries with staleness information.

    Only processes entries in the obs:: namespace (auto-extracted observations).
    Reads file_mtimes from already-fetched metadata and compares against current
    filesystem state. No additional database operations.

    Skips remote schema entries: their file_mtimes reference paths on the
    remote machine that don't exist locally, so os.stat() always fails,
    producing a permanent false-positive staleness penalty.

    Args:
        raw_entries: List of entry dicts with 'doc_id', 'metadata', 'relevance' keys.
        staleness_config: Config dict with 'penalty' (float) key.
    """
    penalty = staleness_config.get("penalty", 0.15)

    for entry in raw_entries:
        doc_id = entry.get("doc_id", "")
        if not doc_id.startswith("obs::"):
            continue

        # Skip remote entries — file paths are from the remote machine
        schema = entry.get("_schema", "")
        if schema.startswith("remote_"):
            continue

        meta = entry.get("metadata", {})
        mtimes_raw = meta.get("file_mtimes")
        if not mtimes_raw:
            continue

        recorded = deserialize_mtimes(mtimes_raw)
        if not recorded:
            continue

        result = check_staleness(recorded)
        if result["is_stale"]:
            entry["is_stale"] = True
            entry["staleness_info"] = result
            # No floor: scores are unclamped (Layer 4), and flooring at 0.0
            # would let a stale entry outrank fresher negative-scored ones.
            entry["relevance"] -= penalty


def _detect_format_from_entry(entry: dict) -> str:
    """Detect format from a query result entry's parent_file path."""
    parent_file = entry.get("parent_file", "")
    return detect_format(parent_file) if parent_file else "markdown"


def _extract_preview(content: str, max_len: int = 150, fmt: str = "markdown") -> str:
    """Extract a clean preview from document content.

    Strips frontmatter/properties and leading headings, format-aware.
    """
    from .format_support import strip_frontmatter

    stripped = strip_frontmatter(content, fmt)
    # Strip leading headings (both # and * styles)
    if fmt == "org":
        stripped = re.sub(
            r"^\*+\s+.*$", "", stripped, count=1, flags=re.MULTILINE
        ).strip()
        # Strip #+TITLE lines
        stripped = re.sub(
            r"^#\+TITLE:.*$", "", stripped, count=1, flags=re.MULTILINE
        ).strip()
    else:
        stripped = re.sub(
            r"^#+\s+.*$", "", stripped, count=1, flags=re.MULTILINE
        ).strip()
    # Collapse whitespace
    stripped = re.sub(r"\s+", " ", stripped).strip()

    if len(stripped) <= max_len:
        return stripped

    # Truncate at word boundary
    truncated = stripped[:max_len]
    last_space = truncated.rfind(" ")
    if last_space > max_len * 0.5:
        truncated = truncated[:last_space]
    return truncated + "..."


# Keys that map to dedicated columns — skip in generic JSONB loop
_KNOWN_CORE_KEYS = {"type", "importance", "tags", "scope", "project"}
_KNOWN_VAULT_KEYS = {"type", "importance", "tags", "directory"}


def _build_core_filter(
    filter_dict: Optional[dict] = None, user: Optional[str] = None
) -> tuple:
    """Build WHERE conditions for local.memories using columns.

    Known keys (type, importance, tags) map to dedicated columns.
    Any other key falls back to a JSONB equality check on the metadata column.

    Returns:
        Tuple of (conditions_list, params_list) for SQL WHERE clause.
    """
    conditions = ["status = 'active'"]
    params = []

    if not filter_dict:
        filter_dict = {}

    # Multi-user filter (opt-in, stored in JSONB)
    if user and user != "anonymous":
        conditions.append("metadata->>'user' = %s")
        params.append(user)

    if "type" in filter_dict and filter_dict["type"]:
        type_val = filter_dict["type"]
        if type_val in ALL_TYPES:
            # Map to category column
            conditions.append("category = %s")
            params.append(type_val)
        # Non-content types (note, journal, etc.) don't exist in core — skip

    if "importance" in filter_dict and filter_dict["importance"]:
        try:
            imp_val = float(filter_dict["importance"])
            conditions.append("importance_score >= %s")
            params.append(imp_val)
        except (ValueError, TypeError):
            pass

    if "tags" in filter_dict and filter_dict["tags"]:
        tag = filter_dict["tags"].split(",")[0].strip()
        conditions.append("metadata->>'tags' LIKE %s")
        params.append(f"%{tag}%")

    if "scope" in filter_dict and filter_dict["scope"]:
        conditions.append("scope = %s")
        params.append(filter_dict["scope"])

    if "project" in filter_dict and filter_dict["project"]:
        conditions.append("project = %s")
        params.append(filter_dict["project"])

    # Generic JSONB fallback: any unknown key → metadata->>'key' = value
    for key, val in filter_dict.items():
        if key in _KNOWN_CORE_KEYS or not val:
            continue
        conditions.append("metadata->>%s = %s")
        params.extend([key, str(val)])

    return conditions, params


def _build_vault_filter(
    filter_dict: Optional[dict] = None, user: Optional[str] = None
) -> tuple:
    """Build WHERE conditions for obsidian.documents using columns.

    Known keys (type, importance, tags, directory) map to dedicated columns.
    Any other key falls back to a JSONB equality check on the metadata column.

    Returns:
        Tuple of (conditions_list, params_list) for SQL WHERE clause.
    """
    conditions = []
    params = []

    if not filter_dict:
        filter_dict = {}

    # Multi-user filter (opt-in, stored in JSONB)
    if user and user != "anonymous":
        conditions.append("metadata->>'user' = %s")
        params.append(user)

    if "directory" in filter_dict and filter_dict["directory"]:
        # directory is a proper column now
        conditions.append("directory = %s")
        params.append(filter_dict["directory"])

    if "type" in filter_dict and filter_dict["type"]:
        type_val = filter_dict["type"]
        if type_val in ALL_TYPES:
            # Content type filter — vault is always 'vault', so if they filter
            # for a non-vault content type, exclude vault entirely
            if type_val != "vault":
                conditions.append("1 = 0")  # no match
        else:
            # Vault-entry type (note, journal, work, etc.)
            conditions.append("vault_type = %s")
            params.append(type_val)

    if "importance" in filter_dict and filter_dict["importance"]:
        try:
            imp_val = float(filter_dict["importance"])
            conditions.append("importance_score >= %s")
            params.append(imp_val)
        except (ValueError, TypeError):
            pass

    if "tags" in filter_dict and filter_dict["tags"]:
        tag = filter_dict["tags"].split(",")[0].strip()
        conditions.append("metadata->>'tags' LIKE %s")
        params.append(f"%{tag}%")

    # Generic JSONB fallback: any unknown key → metadata->>'key' = value
    for key, val in filter_dict.items():
        if key in _KNOWN_VAULT_KEYS or not val:
            continue
        conditions.append("metadata->>%s = %s")
        params.extend([key, str(val)])

    return conditions, params


def _display_path(doc_id: str) -> str:
    """Strip namespace prefix from ID for display purposes."""
    parsed = parse_id(doc_id)
    return parsed.content_id


def _format_core_result(row: dict) -> dict:
    """Format a local.memories row into a result dict.

    Promotes columns (category, scope, source, importance_score) to
    top-level keys. Remaining metadata stays in 'metadata'.
    """
    meta = jsonb_to_metadata(row.get("metadata", {}))
    # Add promoted column values to metadata for downstream compat
    meta["category"] = row.get("category", "observation")
    meta["scope"] = row.get("scope", "global")
    meta["source"] = row.get("source", "auto-extract")
    meta["importance_score"] = str(row.get("importance_score", 0.5))
    meta["type"] = row.get("category", "observation")
    if row.get("chunk_total") is not None:
        meta["chunk_index"] = str(row.get("chunk_index", 0))
        meta["chunk_total"] = str(row.get("chunk_total", 1))
    return meta


def _format_vault_result(row: dict) -> dict:
    """Format a obsidian.documents row into a result dict.

    Promotes columns (parent_file, directory, vault_type, etc.) to
    metadata dict for downstream compat.
    """
    meta = jsonb_to_metadata(row.get("metadata", {}))
    meta["parent_file"] = row.get("parent_file", "")
    meta["directory"] = row.get("directory", "")
    meta["vault_type"] = row.get("vault_type", "document")
    meta["title"] = row.get("title", "")
    meta["chunk_index"] = str(row.get("chunk_index", 0))
    meta["chunk_total"] = str(row.get("chunk_total", 1))
    meta["chunk_heading"] = row.get("chunk_heading", "")
    meta["importance_score"] = str(row.get("importance_score", 0.5))
    meta["type"] = "vault"
    return meta


def _increment_retrieval_counts(doc_ids: list, increment: float = 1.0) -> None:
    """Batch increment retrieval counts for local.memories documents.

    Best-effort operation: errors are logged but don't block query response.
    Only updates local.memories (vault documents don't track retrieval counts).

    Uses a single SQL UPDATE with column-level increment (much more efficient
    than the old JSONB string extraction pattern).

    Args:
        doc_ids: List of document IDs to increment
        increment: Amount to add to retrieval_count (default 1.0).
                   Use fractional values (e.g. 0.01) for passive surfacing.
    """
    if not doc_ids:
        return

    try:
        from .schema import _get_pool

        # Filter to only core (non-vault) IDs
        core_ids = [doc_id for doc_id in doc_ids
                    if schema_for_id(doc_id) == SCHEMA_LOCAL]
        if not core_ids:
            return

        pool = _get_pool()
        with pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """UPDATE local.memories
                       SET retrieval_count = retrieval_count + %s,
                           updated_at = now()
                       WHERE id = ANY(%s) AND status = 'active'""",
                    (increment, core_ids),
                )
                conn.commit()

    except Exception as e:
        import logging

        logger = logging.getLogger("jarvis-core")
        logger.warning(f"Failed to increment retrieval counts: {e}")


def _parse_schemas(schemas_str: Optional[str]) -> Optional[list[str]]:
    """Parse a schemas string into a list of schema names.

    Args:
        schemas_str: 'all', None, or comma-separated schema names
                     (e.g. 'local,remote_personio').

    Returns:
        None (= search all registered schemas) or a list of schema names.
    """
    if schemas_str is None or schemas_str.strip().lower() == "all":
        return None
    return [s.strip() for s in schemas_str.split(",") if s.strip()]


def _query_local_schema(pool, query_embedding, fetch_count: int,
                        filter_dict: Optional[dict], user: Optional[str]) -> list:
    """Query canonical local rows plus search-only windows with HNSW ordering."""
    core_conditions, core_params = _build_core_filter(filter_dict, user)
    core_where = " AND ".join(f"m.{condition}" for condition in core_conditions)
    with pool.connection() as conn:
        with conn.cursor() as cur:
            # Canonical rows without search chunks retain the original direct
            # HNSW path. Chunked parents are deliberately excluded so their
            # mean compatibility vector cannot compete with precise windows.
            cur.execute(
                f"""SELECT id, document, metadata,
                           category, scope, source, importance_score,
                           retrieval_count, created_at,
                           embedding <=> %s::halfvec AS distance,
                           'local' AS _schema
                    FROM local.memories AS m
                    WHERE {core_where}
                      AND NOT EXISTS (
                          SELECT 1 FROM local.memory_chunks AS mc
                          WHERE mc.parent_id = m.id
                      )
                    ORDER BY embedding <=> %s::halfvec ASC
                    LIMIT %s""",
                tuple([query_embedding] + core_params + [query_embedding, fetch_count]),
            )
            columns = [desc.name for desc in cur.description]
            rows = [dict(zip(columns, row)) for row in cur.fetchall()]

            # Search chunk embeddings while projecting canonical metadata and
            # parent ID. Retrieval/reads therefore remain parent-addressed.
            cur.execute(
                f"""SELECT m.id, mc.document, m.metadata,
                           m.category, m.scope, m.source, m.importance_score,
                           m.retrieval_count, m.created_at,
                           mc.chunk_index, mc.chunk_total,
                           mc.embedding <=> %s::halfvec AS distance,
                           'local' AS _schema
                    FROM local.memory_chunks AS mc
                    JOIN local.memories AS m ON m.id = mc.parent_id
                    WHERE {core_where}
                    ORDER BY mc.embedding <=> %s::halfvec ASC
                    LIMIT %s""",
                tuple([query_embedding] + core_params + [query_embedding, fetch_count]),
            )
            columns = [desc.name for desc in cur.description]
            rows.extend(dict(zip(columns, row)) for row in cur.fetchall())
            rows.sort(key=lambda row: float(row.get("distance", 999)))
            return rows[:fetch_count]


def _query_obsidian_schema(pool, query_embedding, fetch_count: int,
                           filter_dict: Optional[dict], user: Optional[str]) -> list:
    """Query obsidian.documents with HNSW ordering."""
    vault_conditions, vault_params = _build_vault_filter(filter_dict, user)
    vault_where = " AND ".join(vault_conditions) if vault_conditions else None
    vault_where_clause = f"WHERE {vault_where}" if vault_where else ""
    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"""SELECT id, document, metadata,
                           parent_file, directory, vault_type, title,
                           chunk_index, chunk_total, chunk_heading,
                           importance_score,
                           embedding <=> %s::halfvec AS distance,
                           'obsidian' AS _schema
                    FROM obsidian.documents
                    {vault_where_clause}
                    ORDER BY embedding <=> %s::halfvec ASC
                    LIMIT %s""",
                tuple([query_embedding] + vault_params + [query_embedding, fetch_count]),
            )
            columns = [desc.name for desc in cur.description]
            return [dict(zip(columns, row)) for row in cur.fetchall()]


def _query_remote_schema(pool, query_embedding, fetch_count: int,
                         filter_dict: Optional[dict], user: Optional[str],
                         entry) -> list:
    """Query a remote mirror schema (same column structure as local.memories).

    Remote mirror retrieval_count stays frozen after this query — mirrors are
    read-only. This is intentional: remote retrieval patterns should not
    influence local ranking (D12).

    Args:
        entry: SchemaEntry from the schema registry (provides name + table).
    """
    from psycopg import sql as psql
    from .schema_registry import is_valid_pg_identifier

    # Defence-in-depth: validate both identifiers before SQL composition
    if not is_valid_pg_identifier(entry.name):
        import logging as _logging
        _logging.getLogger("jarvis-core").error(
            "Invalid schema name in remote query (skipping): %r", entry.name)
        return []
    if not is_valid_pg_identifier(entry.table):
        import logging as _logging
        _logging.getLogger("jarvis-core").error(
            "Invalid table name in remote query (skipping): %r", entry.table)
        return []

    core_conditions, core_params = _build_core_filter(filter_dict, user)
    core_where = " AND ".join(core_conditions)

    query = psql.SQL(
        "SELECT id, document, metadata, "
        "category, scope, source, importance_score, "
        "retrieval_count, created_at, "
        "embedding <=> %s::halfvec AS distance, "
        "%s AS _schema "
        "FROM {schema}.{table} "
        "WHERE {where} "
        "ORDER BY embedding <=> %s::halfvec ASC "
        "LIMIT %s"
    ).format(
        schema=psql.Identifier(entry.name),
        table=psql.Identifier(entry.table),
        where=psql.SQL(core_where),
    )

    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                query,
                tuple([query_embedding] + core_params + [entry.name, query_embedding, fetch_count]),
            )
            columns = [desc.name for desc in cur.description]
            return [dict(zip(columns, row)) for row in cur.fetchall()]


def _remote_count(entry) -> int:
    """Count active rows in a remote schema using safe identifier composition.

    Args:
        entry: SchemaEntry with validated name and table fields.

    Returns:
        Row count, or 0 on error.
    """
    from psycopg import sql as psql
    from .schema import _get_pool

    query = psql.SQL(
        "SELECT count(*) AS cnt FROM {schema}.{table} WHERE status = 'active'"
    ).format(
        schema=psql.Identifier(entry.name),
        table=psql.Identifier(entry.table),
    )

    pool = _get_pool()
    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query)
            row = cur.fetchone()
            return row[0] if row else 0


def _cross_schema_search(query_embedding, fetch_count: int,
                         filter_dict: Optional[dict] = None,
                         user: Optional[str] = None,
                         schemas: Optional[list[str]] = None) -> list:
    """Execute per-schema search across all registered searchable schemas.

    Registry-driven (N schemas): iterates get_searchable_schemas() instead of
    hardcoding local + obsidian. Each schema query uses its own pool connection
    for per-schema error isolation (D7).

    Args:
        query_embedding: The query embedding vector
        fetch_count: Max results per schema
        filter_dict: Optional metadata filter dict
        user: Optional user filter for multi-user isolation
        schemas: Optional list of schema names to restrict search.
                 None = all registered searchable schemas.

    Returns:
        List of row dicts with _schema column, sorted by distance ascending.

    Raises:
        ValueError: If schemas filter contains unknown schema names (D8).
    """
    from .schema import _get_pool
    from .schema_registry import (
        get_searchable_schemas, SchemaKind, SchemaEntry,
        is_valid_schema_name,
    )
    from .config import get_embedding_config

    # Get all searchable schemas from registry
    all_searchable = get_searchable_schemas()
    if not all_searchable:
        # Defensive fallback: registry not yet initialized
        all_searchable = [
            SchemaEntry(name=SCHEMA_LOCAL, kind=SchemaKind.LOCAL, table="memories"),
            SchemaEntry(name=SCHEMA_OBSIDIAN, kind=SchemaKind.OBSIDIAN, table="documents"),
        ]

    # D8: Validate and filter by caller-specified schemas
    if schemas is not None:
        available_names = {e.name for e in all_searchable}
        unknown = set(schemas) - available_names
        if unknown:
            raise ValueError(
                f"Unknown schemas: {sorted(unknown)}. "
                f"Available: {sorted(available_names)}"
            )
        searchable = [e for e in all_searchable if e.name in schemas]
    else:
        searchable = all_searchable

    # D2: Skip schemas whose embedding_model doesn't match the active model
    try:
        active_model = get_embedding_config().get("model")
    except Exception:
        active_model = None

    compatible = []
    for entry in searchable:
        model = entry.metadata.get("embedding_model")
        if model and active_model and model != active_model:
            import logging as _logging
            _logging.getLogger("jarvis-core").warning(
                "Skipping schema %s: embedding model mismatch (%s != %s)",
                entry.name, model, active_model,
            )
            continue
        compatible.append(entry)

    pool = _get_pool()
    rows: list = []

    # D7: Per-schema try/except — one schema failing does not abort others
    for entry in compatible:
        try:
            if entry.kind == SchemaKind.LOCAL:
                schema_rows = _query_local_schema(pool, query_embedding, fetch_count, filter_dict, user)
            elif entry.kind == SchemaKind.OBSIDIAN:
                schema_rows = _query_obsidian_schema(pool, query_embedding, fetch_count, filter_dict, user)
            elif entry.kind == SchemaKind.REMOTE:
                schema_rows = _query_remote_schema(pool, query_embedding, fetch_count, filter_dict, user, entry)
            else:
                continue
            rows.extend(schema_rows)
        except DatabaseUnavailable:
            # Every schema shares the one pool: when it cannot hand out a
            # connection, skipping would just repeat the wait per schema and
            # disguise an outage as "no results".
            raise
        except Exception as e:
            import logging as _logging
            _logging.getLogger("jarvis-core").error(
                "Schema %s query failed (skipping): %s", entry.name, e
            )
            continue

    # Sort merged results by distance
    rows.sort(key=lambda r: float(r.get("distance", 999)))
    return rows


def semantic_candidate_search(
    query: str,
    *,
    limit: int = 20,
    offset: int = 0,
    schemas: Optional[list[str]] = None,
    user: Optional[str] = None,
    purpose: str = "memory_explorer",
) -> dict:
    """Shared raw semantic recall for diagnostic/search UIs.

    Unlike :func:`query_vault`, this preserves pagination and does not mutate
    retrieval counts. It still uses the exact core query-window and cross-schema
    ANN primitive, and emits one telemetry trace for the operation.
    """
    started = time.perf_counter()
    limit = max(1, min(int(limit), 100))
    offset = max(0, int(offset))
    fetch_count = min(offset + limit, 1000)
    from .embedding import get_embedding_service

    service = get_embedding_service()
    rows, expansion = _search_query_windows(
        query, service, fetch_count, user=user, schemas=schemas
    )
    page = rows[offset:offset + limit]
    results = []
    for row in page:
        document = row.get("document") or ""
        results.append({
            "id": str(row.get("id", "")),
            "document": document,
            "created_at": row.get("created_at"),
            "updated_at": row.get("updated_at"),
            "importance_score": float(row.get("importance_score", 0.5) or 0.5),
            "retrieval_count": float(row.get("retrieval_count", 0) or 0),
            "score": 1.0 - float(row.get("distance", 1.0)),
            "schema": row.get("_schema", "obsidian"),
            "query_window_index": int(row.get("_query_window_index", 0) or 0),
        })
    trace_id = None
    try:
        from .config import get_embedding_config
        from .retrieval_telemetry import CandidateTrace, record_event

        returned_ids = {item["id"] for item in results}
        traces = [
            CandidateTrace(
                schema_name=str(row.get("_schema", "obsidian")),
                doc_id=str(row.get("id", "")),
                parent_id=str(row.get("parent_file") or row.get("id", "")),
                parent_file=str(row.get("parent_file") or ""),
                chunk_index=int(row.get("chunk_index", 0) or 0),
                query_window_index=int(row.get("_query_window_index", 0) or 0),
                vector_rank=index,
                final_rank=(index - offset) if row.get("id") in returned_ids else None,
                similarity=1.0 - float(row.get("distance", 1.0)),
                pre_score=1.0 - float(row.get("distance", 1.0)),
                terminal_reason="selected" if row.get("id") in returned_ids else "result_cap",
                returned=row.get("id") in returned_ids,
            )
            for index, row in enumerate(rows, 1)
        ]
        emb_cfg = get_embedding_config()
        trace_id = record_event(
            purpose=purpose, query=query, candidates=traces,
            funnel={
                "query_windows": expansion.get("inference_window_count", 1),
                "ann_unique": len(rows), "offset": offset,
                "returned": len(results),
            },
            latency={"total_ms": round((time.perf_counter() - started) * 1000, 2)},
            outcome="results" if results else "empty",
            query_window_count=expansion.get("inference_window_count", 1),
            model_snapshot={
                "embedding_model": emb_cfg.get("model_id"),
                "embedding_backend": emb_cfg.get("backend"),
            },
            config_snapshot={"limit": limit, "offset": offset, "schemas": schemas or ["all"]},
        )
    except Exception:
        pass
    return {
        "results": results,
        "total": len(rows) if len(rows) < fetch_count else None,
        "trace_id": trace_id,
        "query_windows": expansion,
    }


def query_vault(
    query: str,
    n_results: int = 5,
    filter: Optional[dict] = None,
    user: Optional[str] = None,
    schemas: Optional[list[str]] = None,
    purpose: str = "explicit_recall",
    telemetry_user_facing: bool = True,
    telemetry_query_ref: Optional[str] = None,
    rerank: bool = True,
) -> dict:
    """Semantic search across vault memory.

    Searches both local.memories and obsidian.documents using per-schema CTEs
    for optimal HNSW index usage.

    Args:
        query: Natural language search query
        n_results: Max results (capped at 20)
        filter: Optional metadata filters (directory, type, importance, tags)
        user: Optional user filter for multi-user isolation
        rerank: False skips cross-encoder reranking even when it is enabled
            (callers that only need raw similarity, e.g. dedup, on a deadline)

    Returns:
        Formatted results dict with titles, paths, excerpts, relevance scores
    """
    retrieval_started = time.perf_counter()
    from .embedding import get_embedding_service
    from .schema import _get_pool

    try:
        # Check total across local + obsidian (required schemas)
        core_count = execute_query(
            "SELECT count(*) AS cnt FROM local.memories WHERE status = 'active'",
            fetch="one",
        )
        vault_count = execute_query(
            "SELECT count(*) AS cnt FROM obsidian.documents", fetch="one"
        )
        total = (core_count["cnt"] if core_count else 0) + (
            vault_count["cnt"] if vault_count else 0
        )
    except Exception as e:
        if is_db_unavailable_error(e):
            # Structured, so callers (hook ingest dedup) can tell "retry
            # later" from "no match" without parsing the message.
            return {"success": False, "error": f"Database unavailable: {safe_db_error(e)}",
                    "retryable": True}
        return {"success": False, "error": f"Database unavailable: {e}"}

    # D4: Add remote schema counts (best-effort — failures don't abort the query)
    from .schema_registry import get_searchable_schemas, SchemaKind, is_valid_pg_identifier
    for _re in get_searchable_schemas(kind=SchemaKind.REMOTE):
        if not is_valid_pg_identifier(_re.name) or not is_valid_pg_identifier(_re.table):
            continue
        try:
            _rc = _remote_count(_re)
            total += _rc
        except Exception:
            pass

    if total == 0:
        response = {
            "success": True,
            "query": query,
            "results": [],
            "total_in_collection": 0,
            "message": "No documents indexed. Ask Jarvis to 'index my vault' or use jarvis_index_vault tool.",
        }
        trace_id = _record_empty_trace(
            purpose, query, user_name=user, user_facing=telemetry_user_facing,
            query_ref=telemetry_query_ref,
        )
        if trace_id:
            response["trace_id"] = trace_id
        return response

    n_results = min(max(1, n_results), 20)

    # Over-fetch to account for chunk deduplication (and reranking if enabled).
    # The window fills by raw vector distance BEFORE per-file chunk dedup, so
    # a multi-chunk document can crowd out others — overfetch_factor (default 5)
    # sizes the window to survive dedup.
    reranking_config = get_reranking_config()
    if not rerank:
        reranking_config = {**reranking_config, "enabled": False}
    if reranking_config.get("enabled", False):
        # candidate_count caps how many POST-dedup survivors reach the
        # reranker (below); it must not shrink this pre-dedup ANN window or
        # chunk-heavy documents crowd everything else out and enabling the
        # reranker REDUCES recall. Mirror semantic_context's wide fetch.
        fetch_count = min(100, total)
    else:
        overfetch = get_ranking_config().get("overfetch_factor", 5)
        fetch_count = min(n_results * max(int(overfetch), 1), 60, total)

    search_started = time.perf_counter()
    try:
        service = get_embedding_service()
        rows, expansion = _search_query_windows(
            query,
            service,
            fetch_count,
            filter_dict=filter,
            user=user,
            schemas=schemas,
        )
    except ValueError as e:
        # D8: Unknown schema names in the filter
        return {"success": False, "error": str(e)}
    except Exception as e:
        if is_db_unavailable_error(e):
            # No error trace: writing it needs the database that just failed.
            return {"success": False, "error": f"Database unavailable: {safe_db_error(e)}",
                    "retryable": True}
        trace_id = _record_empty_trace(
            purpose, query, outcome="error", error=str(e), user_name=user,
            user_facing=telemetry_user_facing, query_ref=telemetry_query_ref,
        )
        response = {"success": False, "error": f"Query failed: {e}"}
        if trace_id:
            response["trace_id"] = trace_id
        return response
    search_ms = (time.perf_counter() - search_started) * 1000

    # Capture the true ANN-unique count BEFORE the lexical union mutates `rows`
    # in place — otherwise funnel.ann_unique would double-count lexical rows.
    ann_unique = len(rows)

    # Union the lexical recall channel into the ANN pool. query_vault stays
    # gate-free (Part D is injection-only) — lexical rows just join the ranking
    # pool and compete by unified score. The lexical channel honors the same
    # per-user scope as the ANN path; when a metadata `filter` is supplied it is
    # skipped entirely (Phase-1 limitation) rather than returning rows that
    # violate the filter.
    rows, lexical_stats = _augment_rows_with_lexical(
        rows, query, service, schemas=schemas, user=user, filter_dict=filter
    )

    # Build raw result entries with relevance scores
    decay_config = get_decay_config()
    ranking_cfg = get_ranking_config()
    use_decay = decay_config.get("enabled", True)

    # D12: Core-like schemas (LOCAL + REMOTE) support blended decay scoring —
    # they share the local.memories column structure.
    from .schema_registry import _core_like_schemas as _get_core_like
    core_like = _get_core_like()

    raw_entries = []
    for row in rows:
        schema = row.get("_schema", "obsidian")

        # Remote mirrors share the local.memories structure; obsidian has its own
        if schema in core_like:
            meta = _format_core_result(row)
        else:
            meta = _format_vault_result(row)

        distance = float(row["distance"])
        # pgvector cosine distance is ``1 - cosine_similarity``.
        similarity = 1.0 - distance

        # Use numeric importance_score
        imp_score = 0.5
        imp_str = meta.get("importance_score")
        if imp_str:
            try:
                imp_score = float(imp_str)
            except (ValueError, TypeError):
                pass

        # Unified scoring (Layer 4): one formula for every schema. Memories get
        # decay-adjusted importance; vault chunks use raw importance_score. No
        # recency term — updated_at is the reindex timestamp, not an authoring
        # date, so it would add a constant cross-schema offset after reindexes.
        # Note: remote retrieval_count stays frozen (mirrors are read-only) — this
        # is intentional; remote retrieval patterns should not influence local ranking.
        importance_weight = ranking_cfg.get(
            "importance_weight", DEFAULT_IMPORTANCE_WEIGHT
        )
        if use_decay and schema in core_like:
            relevance, _ = score_memory(
                similarity=similarity,
                base_importance=imp_score,
                created_at=_parse_row_datetime(row.get("created_at") or meta.get("created_at")),
                last_retrieved_at=_parse_optional_row_datetime(
                    meta.get("last_retrieved_at")
                ),
                retrieval_count=int(float(meta.get("retrieval_count", 0))),
                importance_weight=importance_weight,
                decay_config=decay_config,
            )
        else:
            relevance = compute_unified_score(
                similarity, imp_score, importance_weight=importance_weight
            )

        # Determine parent file for chunk dedup
        parent_file = meta.get("parent_file")
        if not parent_file:
            parsed = parse_id(row["id"])
            parent_file = parsed.content_id

        raw_entries.append(
            {
                "doc_id": row["id"],
                "parent_file": parent_file,
                "relevance": relevance,
                "similarity": similarity,
                "document": row["document"],
                "metadata": meta,
                "_schema": schema,
                "_channel": row.get("_channel", "semantic"),
                "_lexical_rank": row.get("_lexical_rank"),
                "_query_window": row.get("_query_window", query),
                "_query_window_index": row.get("_query_window_index", 0),
                "_pre_score": relevance,
                "_terminal_reason": None,
            }
        )

    # Staleness annotation: penalize observations whose tracked files changed
    staleness_config = get_staleness_config()
    if staleness_config.get("enabled", True):
        _annotate_staleness(raw_entries, staleness_config)

    # D9: Compound dedup key (parent_file, schema) — avoids collapsing entries
    # from different schemas that happen to share a parent_file name.
    best_per_file: dict = {}
    for entry in raw_entries:
        key = (entry["parent_file"], entry["_schema"])
        if key not in best_per_file:
            best_per_file[key] = entry
        elif entry["relevance"] > best_per_file[key]["relevance"]:
            best_per_file[key]["_terminal_reason"] = "parent_dedup"
            best_per_file[key] = entry
        else:
            entry["_terminal_reason"] = "parent_dedup"

    # D9: Cross-schema dedup — if same doc_id appears in both local and a remote
    # mirror (echo dedup miss), prefer the local copy.
    id_to_key: dict = {}
    keys_to_remove: set = set()
    for key, entry in list(best_per_file.items()):
        doc_id = entry["doc_id"]
        if doc_id in id_to_key:
            existing_key = id_to_key[doc_id]
            if entry["_schema"] == SCHEMA_LOCAL:
                keys_to_remove.add(existing_key)
                id_to_key[doc_id] = key
            else:
                keys_to_remove.add(key)
        else:
            id_to_key[doc_id] = key
    for k in keys_to_remove:
        removed = best_per_file.pop(k, None)
        if removed:
            removed["_terminal_reason"] = "parent_dedup"

    parent_dedup_count = len(best_per_file)

    # Cross-encoder reranking (applied only when enabled and >1 candidate)
    reranking_applied = False
    rerank_detail = None
    if reranking_config.get("enabled"):
        candidate_count = max(1, int(reranking_config.get("candidate_count", 20)))
        lexical_slots = max(
            0, int(get_lexical_config().get("lexical_rerank_slots", 10))
        )
        retained_keys = _select_rerank_retained(
            best_per_file, candidate_count, lexical_slots, result_limit=n_results
        )
        for key, entry in best_per_file.items():
            if key not in retained_keys:
                entry["_terminal_reason"] = "candidate_cap"
        best_per_file = {
            key: entry for key, entry in best_per_file.items() if key in retained_keys
        }
    if reranking_config.get("enabled") and len(best_per_file) > 1:
        from .reranking import (
            clear_last_rerank_result, get_last_rerank_result, rerank, rerank_multi,
        )

        deduped_list = sorted(
            best_per_file.values(), key=lambda e: e["relevance"], reverse=True
        )
        contextual_mode = get_contextual_augmentation_mode()
        batch_summaries = _rerank_summaries(deduped_list)
        docs = [
            _rerank_doc_text(e, contextual_mode, batch_summaries) for e in deduped_list
        ]
        vscores = [e["relevance"] for e in deduped_list]
        query_texts = [e.get("_query_window", query) for e in deduped_list]
        clear_last_rerank_result()
        if len(set(query_texts)) == 1:
            blended = rerank(query_texts[0], docs, vscores, reranking_config)
        else:
            blended = rerank_multi(query_texts, docs, vscores, reranking_config)
        rerank_detail = get_last_rerank_result()
        if blended is not vscores:
            for index, (entry, score) in enumerate(zip(deduped_list, blended)):
                entry["relevance"] = score
                if rerank_detail and rerank_detail.applied:
                    entry["_raw_bge_logit"] = rerank_detail.raw_logits[index]
                    entry["_bge_probability"] = rerank_detail.sigmoid_scores[index]
                entry["_blended_score"] = score
            reranking_applied = True

    # Sort by relevance descending and trim to the caller's n_results —
    # reranking rescores candidates but never expands the requested count.
    final_count = n_results
    ranked_all = sorted(
        best_per_file.values(), key=lambda e: e["relevance"], reverse=True
    )
    deduped = ranked_all[:final_count]
    for entry in ranked_all[final_count:]:
        entry["_terminal_reason"] = "result_cap"
    for entry in deduped:
        entry["_terminal_reason"] = "selected"

    results = []
    all_ids = []
    for rank, entry in enumerate(deduped, start=1):
        meta = entry["metadata"]
        doc_id = entry["doc_id"]
        entry_fmt = _detect_format_from_entry(entry)
        preview = (
            _extract_preview(entry["document"], fmt=entry_fmt)
            if entry["document"]
            else ""
        )
        title = meta.get("title", doc_id)
        doc_type = meta.get("vault_type") or meta.get("type", "unknown")
        doc_importance = meta.get("importance", "medium")
        tags = meta.get("tags", "")
        chunk_heading = meta.get("chunk_heading", "")

        schema = entry.get("_schema", "obsidian")
        if schema == "obsidian":
            source = "vault"
        else:
            source = meta.get("category", meta.get("type", "memory"))

        result_entry = {
            "rank": rank,
            "id": doc_id,
            "path": entry["parent_file"],
            "title": title,
            "type": doc_type,
            "importance": doc_importance,
            "relevance": round(entry["relevance"], 3),
            "similarity": round(entry.get("similarity", 0.0), 3),
            "preview": preview,
            "tags": tags,
            "schema": schema,
            "source": source,
        }
        # Provenance: tag results from remote mirror schemas
        if schema.startswith("remote_"):
            result_entry["source_remote"] = schema
        if chunk_heading:
            result_entry["chunk_heading"] = chunk_heading
        if entry.get("is_stale"):
            result_entry["stale"] = True
            stale_info = entry.get("staleness_info", {})
            if stale_info.get("stale_files"):
                result_entry["stale_files"] = stale_info["stale_files"]
        results.append(result_entry)
        all_ids.append(doc_id)

    # Increment retrieval counts for core results (best-effort, non-blocking)
    _increment_retrieval_counts(all_ids)

    response = {
        "success": True,
        "query": query,
        "results": results,
        "total_in_collection": total,
    }

    # Include expansion metadata when terms were added
    if expansion["terms_added"]:
        response["expansion"] = {
            "terms_added": expansion["terms_added"],
            "intent": expansion["intent"],
        }

    if expansion.get("base_window_count", 1) > 1:
        response["query_windows"] = {
            "input": expansion["base_window_count"],
            "inference": expansion["inference_window_count"],
        }

    # Include reranking metadata when applied
    if reranking_applied:
        response["reranking"] = {
            "applied": True,
            "alpha": reranking_config.get("alpha", 0.7),
            "candidates": len(best_per_file),
            "top_k": final_count,
        }

    # One aggregate event per semantic operation. This deliberately happens
    # after response construction and is fail-open inside record_event().
    try:
        from .config import get_embedding_config
        from .retrieval_telemetry import CandidateTrace, record_event

        candidate_traces = []
        final_ranks = {
            (entry.get("_schema"), entry["doc_id"]): rank
            for rank, entry in enumerate(deduped, 1)
        }
        for vector_rank, entry in enumerate(raw_entries, 1):
            identity = (entry.get("_schema"), entry["doc_id"])
            is_returned = identity in final_ranks
            candidate_traces.append(CandidateTrace(
                schema_name=entry.get("_schema", "obsidian"),
                doc_id=str(entry["doc_id"]),
                parent_id=str(entry.get("parent_file") or ""),
                parent_file=str(entry.get("parent_file") or ""),
                chunk_index=int(entry.get("metadata", {}).get("chunk_index", 0) or 0),
                query_window_index=int(entry.get("_query_window_index", 0) or 0),
                vector_rank=vector_rank,
                final_rank=final_ranks.get(identity),
                similarity=float(entry.get("similarity", 0)),
                pre_score=float(entry.get("_pre_score", entry.get("relevance", 0))),
                raw_bge_logit=entry.get("_raw_bge_logit"),
                bge_probability=entry.get("_bge_probability"),
                blended_score=entry.get("_blended_score"),
                terminal_reason=entry.get("_terminal_reason") or "result_cap",
                returned=is_returned,
                channel=entry.get("_channel", "semantic"),
            ))
        trace_id = record_event(
            purpose=purpose, query=query, candidates=candidate_traces,
            funnel={
                "query_windows": expansion.get("inference_window_count", 1),
                "ann_unique": ann_unique, "parent_dedup": parent_dedup_count,
                "lexical_candidates": lexical_stats.get("lexical_candidates", 0),
                "lexical_added": lexical_stats.get("lexical_added", 0),
                "candidate_cap": len(best_per_file),
                "live_reranker_input": len(best_per_file) if reranking_config.get("enabled") else 0,
                "live_reranker_applied": bool(rerank_detail and rerank_detail.applied),
                "live_reranker_fallback": rerank_detail.fallback_reason if rerank_detail and not rerank_detail.applied else None,
                "returned": len(results),
            },
            latency={
                "search_ms": round(search_ms, 2),
                "rerank_ms": round(rerank_detail.latency_ms, 2) if rerank_detail else 0,
                "total_ms": round((time.perf_counter() - retrieval_started) * 1000, 2),
            },
            outcome="results" if results else "empty", user_name=user,
            user_facing=telemetry_user_facing, query_ref=telemetry_query_ref,
            query_window_count=expansion.get("inference_window_count", 1),
            model_snapshot={
                "embedding_model": get_embedding_config().get("model_id"),
                "embedding_backend": get_embedding_config().get("backend"),
                "reranker_model": reranking_config.get("model"),
                "reranker_backend": reranking_config.get("backend"),
            },
            config_snapshot={
                "n_results": n_results,
                "reranking_enabled": bool(reranking_config.get("enabled")),
                "reranking_alpha": reranking_config.get("alpha"),
                # Both keys: the boolean stays for pre-3.6 readers, the mode is
                # what shadow scoring's augmentation-identity guard compares
                # (mechanical and summary are different embedding spaces).
                "contextual_embeddings": get_contextual_embeddings_enabled(),
                "contextual_augmentation": get_contextual_augmentation_mode(),
                "schemas": schemas or ["all"],
            },
            shadow_eligible=telemetry_user_facing,
        )
        if trace_id:
            response["trace_id"] = trace_id
    except Exception:
        pass

    return response


def semantic_context(
    query: str,
    threshold: float = 0.85,
    budget: int = 8000,
    skip_retrieval_increment: bool = False,
    schemas: Optional[list[str]] = None,
    max_results: int = 20,
) -> dict:
    """Search vault memories for per-prompt context injection.

    Searches both local.memories and obsidian.documents. Differs from query_vault():
    - Applies a minimum raw-cosine threshold (filters low-scoring results)
    - Caps the injected result count independently of the character budget
    - Budget-based allocation: core content shown in full, vault as references
    - Fractionally increments retrieval counts (configurable, default 0.01)
    - Filters out sensitive directories (documents/, people/)
    - Returns query_ms timing for performance monitoring
    - Silently returns empty on any error (graceful degradation)

    Budget strategy:
    - Total budget is split 50/50 between core and vault
    - Core items (observations, learnings, etc.) are shown with full content
    - Vault items are shown as references (file path + heading, ~120 chars)
    - Each half gets a guaranteed budget; unused budget overflows to the other
    - Results are processed in relevance order, highest first

    Args:
        query: User's raw prompt text
        threshold: Minimum raw cosine similarity (default 0.85)
        budget: Total character budget for injection (default 8000, split 50/50)
        max_results: Maximum matches to inject after ranking/deduplication
            (default 20, clamped to 1-100). The budget may reduce this further.
        skip_retrieval_increment: If True, skip the retrieval count write-back.
            Surfaced IDs are returned in the result dict so the caller can
            bump them externally (e.g., via HTTP POST to the MCP server).

    Returns:
        Dict with matches list and metadata. When skip_retrieval_increment=True,
        includes 'surfaced_ids' list for deferred bumping.
    """
    start = time.time()
    retrieval_started = time.perf_counter()
    try:
        max_results = int(max_results)
    except (TypeError, ValueError):
        max_results = 20
    max_results = min(max(max_results, 1), 100)

    try:
        from .embedding import get_embedding_service
        from .schema import _get_pool

        core_count = execute_query(
            "SELECT count(*) AS cnt FROM local.memories WHERE status = 'active'",
            fetch="one",
        )
        vault_count = execute_query(
            "SELECT count(*) AS cnt FROM obsidian.documents", fetch="one"
        )
        total = (core_count["cnt"] if core_count else 0) + (
            vault_count["cnt"] if vault_count else 0
        )
    except Exception as exc:
        response = {"matches": [], "query_ms": 0, "total_searched": 0}
        if is_db_unavailable_error(exc):
            response.update(degraded=True, error=safe_db_error(exc))
        return response

    # D4: Add remote schema counts (best-effort)
    from .schema_registry import get_searchable_schemas as _gss, SchemaKind as _SK, is_valid_pg_identifier as _vpi
    for _re in _gss(kind=_SK.REMOTE):
        if not _vpi(_re.name) or not _vpi(_re.table):
            continue
        try:
            total += _remote_count(_re)
        except Exception:
            pass

    if total == 0:
        response = {"matches": [], "query_ms": 0, "total_searched": 0}
        trace_id = _record_empty_trace("context_injection", query)
        if trace_id:
            response["trace_id"] = trace_id
        return response

    # Over-fetch to account for chunk dedup + threshold filtering
    fetch_count = min(100, total)

    search_started = time.perf_counter()
    try:
        service = get_embedding_service()
        rows, expansion = _search_query_windows(
            query, service, fetch_count, schemas=schemas
        )
    except Exception as exc:
        response = {"matches": [], "query_ms": 0, "total_searched": total}
        if is_db_unavailable_error(exc):
            # No error trace: writing it needs the database that just failed.
            response.update(degraded=True, error=safe_db_error(exc))
            return response
        trace_id = _record_empty_trace(
            "context_injection", query, outcome="error", error=str(exc)
        )
        if trace_id:
            response["trace_id"] = trace_id
        return response
    search_ms = (time.perf_counter() - search_started) * 1000

    # Capture the true ANN-unique count BEFORE the lexical union mutates `rows`
    # in place — otherwise funnel.ann_unique would double-count lexical rows.
    ann_unique = len(rows)

    # Union the lexical recall channel into the ANN pool. Per-prompt injection is
    # single-user (anonymous scope) with no metadata filter, so the lexical
    # channel runs unrestricted here.
    rows, lexical_stats = _augment_rows_with_lexical(
        rows, query, service, schemas=schemas
    )

    # Build raw entries with relevance scores
    decay_config = get_decay_config()
    ranking_cfg = get_ranking_config()
    use_decay = decay_config.get("enabled", True)

    # Fetch reranking + enrichment config up front: the recall-additive logit
    # gate (Part D) needs to know, inside the raw loop, whether a cosine-failing
    # candidate can still be judged (reranking on) or must be dropped now
    # (reranking off → degrade to exactly today's cosine-only behavior).
    reranking_config = get_reranking_config()
    reranking_enabled = bool(reranking_config.get("enabled"))
    enrichment_config = get_context_enrichment_config()
    bge_logit_threshold = float(enrichment_config.get("bge_logit_threshold", -4.0))

    # D12: Pre-compute core-like set for scoring gate (LOCAL + REMOTE schemas)
    from .schema_registry import _core_like_schemas as _get_core_like_sc
    core_like_sc = _get_core_like_sc()

    raw_entries = []
    trace_candidates = []
    trace_by_key = {}
    skipped_sensitive = 0

    from .retrieval_telemetry import CandidateTrace
    for vector_rank, row in enumerate(rows, start=1):
        schema = row.get("_schema", "obsidian")

        # Remote mirrors share the local.memories structure; obsidian has its own
        if schema in core_like_sc:
            meta = _format_core_result(row)
        else:
            meta = _format_vault_result(row)

        distance = float(row["distance"])
        # pgvector cosine distance is ``1 - cosine_similarity``.
        similarity = 1.0 - distance

        parent_file_for_trace = meta.get("parent_file")
        if not parent_file_for_trace:
            parent_file_for_trace = parse_id(row["id"]).content_id
        trace = CandidateTrace(
            schema_name=schema,
            doc_id=str(row["id"]),
            parent_id=str(parent_file_for_trace or ""),
            parent_file=str(parent_file_for_trace or ""),
            chunk_index=int(meta.get("chunk_index", 0) or 0),
            query_window_index=int(row.get("_query_window_index", 0) or 0),
            vector_rank=vector_rank,
            similarity=similarity,
            channel=row.get("_channel", "semantic"),
        )
        trace_candidates.append(trace)
        trace_by_key[trace.candidate_key] = trace

        # Filter sensitive directories
        directory = meta.get("directory", "")
        if directory in SENSITIVE_PATHS:
            skipped_sensitive += 1
            trace.terminal_reason = "sensitive"
            continue

        # Filter superseded core entries (enforced by active_memories view,
        # but also check metadata for safety)
        if meta.get("status") == "superseded":
            trace.terminal_reason = "superseded"
            continue

        # Use numeric importance_score
        imp_score = 0.5
        imp_str = meta.get("importance_score")
        if imp_str:
            try:
                imp_score = float(imp_str)
            except (ValueError, TypeError):
                pass

        # Recall-additive gate (Part D), Phase-1 scope: a candidate is injectable
        # iff its RAW cosine clears the threshold OR (it came from the LEXICAL
        # recall channel AND reranking applied AND its BGE logit clears
        # bge_logit_threshold). A cosine-failing row survives past this point
        # only when it is LEXICAL and reranking is enabled — it must reach the
        # reranker to earn a logit, then face the OR-gate after rescoring.
        # Cosine-failing SEMANTIC (ANN) rows are dropped here exactly as the
        # pre-hybrid path did (their logit rescue is Phase 2); this keeps the
        # BGE batch bounded and makes the lexical failer the only rescue path.
        # The gate compares RAW similarity (not the importance-boosted
        # relevance), keeping the gate and the ranking boost decoupled.
        cosine_ok = similarity >= threshold
        channel = row.get("_channel", "semantic")
        if not cosine_ok and not (reranking_enabled and channel == "lexical"):
            trace.terminal_reason = "cosine_rejected"
            continue

        # Unified scoring (Layer 4): one formula for every schema. Memories get
        # decay-adjusted importance; vault chunks use raw importance_score. No
        # recency term (updated_at is the reindex timestamp, not authoring date).
        # Remote retrieval_count stays frozen — mirrors are read-only.
        importance_weight = ranking_cfg.get(
            "importance_weight", DEFAULT_IMPORTANCE_WEIGHT
        )
        if use_decay and schema in core_like_sc:
            relevance, _ = score_memory(
                similarity=similarity,
                base_importance=imp_score,
                created_at=_parse_row_datetime(row.get("created_at") or meta.get("created_at")),
                last_retrieved_at=_parse_optional_row_datetime(
                    meta.get("last_retrieved_at")
                ),
                retrieval_count=int(float(meta.get("retrieval_count", 0))),
                importance_weight=importance_weight,
                decay_config=decay_config,
            )
        else:
            relevance = compute_unified_score(
                similarity, imp_score, importance_weight=importance_weight
            )

        # Determine parent file for chunk dedup
        parent_file = meta.get("parent_file")
        if not parent_file:
            parsed = parse_id(row["id"])
            parent_file = parsed.content_id

        raw_entries.append(
            {
                "doc_id": row["id"],
                "parent_file": parent_file,
                "relevance": relevance,
                "similarity": similarity,
                "document": row["document"],
                "metadata": meta,
                "_schema": schema,
                "_channel": row.get("_channel", "semantic"),
                "_lexical_rank": row.get("_lexical_rank"),
                "_cosine_ok": cosine_ok,
                "_query_window": row.get("_query_window", query),
                "_query_window_index": row.get("_query_window_index", 0),
                "_trace_key": trace.candidate_key,
            }
        )
        trace.pre_score = relevance

    # Staleness annotation: penalize observations whose tracked files changed
    staleness_config = get_staleness_config()
    if staleness_config.get("enabled", True):
        _annotate_staleness(raw_entries, staleness_config)

    # D9: Compound dedup key (parent_file, schema). Passer-priority: a
    # cosine-passing chunk must never lose its file's slot to a cosine-failing
    # (lexical) chunk — if the failer later gets logit-rejected the whole file
    # would vanish, removing content today's pipeline injects.
    best_per_file: dict = {}
    for entry in raw_entries:
        key = (entry["parent_file"], entry["_schema"])
        if key not in best_per_file:
            best_per_file[key] = entry
            continue
        incumbent = best_per_file[key]
        challenger_wins = (
            bool(entry.get("_cosine_ok", True)), entry["relevance"]
        ) > (
            bool(incumbent.get("_cosine_ok", True)), incumbent["relevance"]
        )
        if challenger_wins:
            trace_by_key[incumbent["_trace_key"]].terminal_reason = "parent_dedup"
            best_per_file[key] = entry
        else:
            trace_by_key[entry["_trace_key"]].terminal_reason = "parent_dedup"

    # D9: Cross-schema dedup for same doc_id. Passer-priority first (a local
    # lexical failer must not evict a remote passer), then local-wins.
    id_to_key_sc: dict = {}
    keys_to_remove_sc: set = set()
    for key, entry in list(best_per_file.items()):
        doc_id = entry["doc_id"]
        if doc_id in id_to_key_sc:
            existing_key = id_to_key_sc[doc_id]
            incumbent = best_per_file[existing_key]
            challenger_wins = (
                bool(entry.get("_cosine_ok", True)), entry["_schema"] == SCHEMA_LOCAL
            ) > (
                bool(incumbent.get("_cosine_ok", True)), incumbent["_schema"] == SCHEMA_LOCAL
            )
            if challenger_wins:
                keys_to_remove_sc.add(existing_key)
                id_to_key_sc[doc_id] = key
            else:
                keys_to_remove_sc.add(key)
        else:
            id_to_key_sc[doc_id] = key
    for k in keys_to_remove_sc:
        removed = best_per_file.pop(k, None)
        if removed:
            trace_by_key[removed["_trace_key"]].terminal_reason = "parent_dedup"

    parent_dedup_count_sc = len(best_per_file)

    # Cross-encoder reranking: rescore all candidates regardless of schema.
    # Runs after dedup (fewer candidates = faster inference). reranking_config
    # was fetched up front for the recall-additive gate.
    if reranking_config.get("enabled"):
        candidate_count = max(1, int(reranking_config.get("candidate_count", 20)))
        lexical_slots = max(
            0, int(get_lexical_config().get("lexical_rerank_slots", 10))
        )
        retained_keys = _select_rerank_retained(
            best_per_file, candidate_count, lexical_slots, result_limit=max_results
        )
        for key, entry in best_per_file.items():
            if key not in retained_keys:
                trace_by_key[entry["_trace_key"]].terminal_reason = "candidate_cap"
        best_per_file = {
            key: entry for key, entry in best_per_file.items() if key in retained_keys
        }
    rerank_detail = None
    if reranking_config.get("enabled") and len(best_per_file) > 1:
        from .reranking import (
            clear_last_rerank_result, get_last_rerank_result, rerank, rerank_multi,
        )

        candidates = sorted(
            best_per_file.values(), key=lambda e: e["relevance"], reverse=True
        )
        contextual_mode = get_contextual_augmentation_mode()
        batch_summaries = _rerank_summaries(candidates)
        docs = [
            _rerank_doc_text(e, contextual_mode, batch_summaries) for e in candidates
        ]
        vscores = [e["relevance"] for e in candidates]
        query_texts = [e.get("_query_window", query) for e in candidates]
        clear_last_rerank_result()
        if len(set(query_texts)) == 1:
            blended = rerank(query_texts[0], docs, vscores, reranking_config)
        else:
            blended = rerank_multi(query_texts, docs, vscores, reranking_config)
        rerank_detail = get_last_rerank_result()
        if blended is not vscores:
            for index, (entry, score) in enumerate(zip(candidates, blended)):
                entry["relevance"] = score
                if rerank_detail and rerank_detail.applied:
                    trace = trace_by_key[entry["_trace_key"]]
                    trace.raw_bge_logit = rerank_detail.raw_logits[index]
                    trace.bge_probability = rerank_detail.sigmoid_scores[index]
                    trace.blended_score = score
                    entry["_raw_bge_logit"] = rerank_detail.raw_logits[index]

    # Recall-additive OR-gate (Part D). Cosine-passing rows are always kept and
    # are given membership PRIORITY downstream (the max_results cut and the
    # character budget serve them before any rescued row), so enabling the
    # reranker can only ADD cosine-failing rescues, never displace a passer —
    # enabled-injected ⊇ disabled-injected.
    # A cosine-failing row (only ever LEXICAL here — semantic failers were
    # dropped at the threshold) survives only if reranking APPLIED and produced a
    # logit at or above bge_logit_threshold; otherwise it is dropped:
    #   - logit present but below threshold → terminal_reason 'logit_rejected'
    #   - no logit (reranker off / fell back / too few candidates / unscored)
    #     → dropped as 'cosine_rejected' (degrades to exactly today's behavior),
    #     counted separately in funnel.logit_unjudged_dropped.
    # logit_rescued is counted HERE at the gate (not over the final selection),
    # so rescues later suppressed by dedup / result_cap / budget stay visible in
    # the funnel; their terminal_reason carries the later drop stage.
    reranking_applied = bool(rerank_detail and rerank_detail.applied)
    logit_rescued = 0
    logit_rejected = 0
    logit_unjudged_dropped = 0
    gated_best_per_file: dict = {}
    for key, entry in best_per_file.items():
        if entry.get("_cosine_ok", True):
            gated_best_per_file[key] = entry
            continue
        logit = entry.get("_raw_bge_logit")
        if reranking_applied and logit is not None and float(logit) >= bge_logit_threshold:
            entry["_logit_rescued"] = True
            gated_best_per_file[key] = entry
            logit_rescued += 1
        elif reranking_applied and logit is not None:
            trace_by_key[entry["_trace_key"]].terminal_reason = "logit_rejected"
            logit_rejected += 1
        else:
            trace_by_key[entry["_trace_key"]].terminal_reason = "cosine_rejected"
            logit_unjudged_dropped += 1
    best_per_file = gated_best_per_file

    # Suppress semantically redundant local records before the final cap. This
    # targets auto-extraction overlap (memory + observation + worklog) without
    # collapsing distinct vault references or records from different projects.
    semantic_duplicates_suppressed = 0
    candidates = list(best_per_file.values())
    if enrichment_config.get("semantic_dedup_enabled", True):
        before_semantic_keys = {entry["_trace_key"] for entry in candidates}
        candidates, semantic_duplicates_suppressed = _semantic_deduplicate_context(
            candidates,
            service,
            enrichment_config.get("semantic_dedup_threshold", 0.86),
        )
        after_semantic_keys = {entry["_trace_key"] for entry in candidates}
        for key in before_semantic_keys - after_semantic_keys:
            trace_by_key[key].terminal_reason = "semantic_duplicate"

    # Passers-first membership priority (recall-additivity, fix for finding 1):
    # cosine-passing rows claim the max_results seats (and, below, the character
    # budget) before any logit-rescued cosine-failing row. Within each group,
    # rank by relevance. A rescued row can therefore only take a seat/budget a
    # passer left free — it can never displace one. The enforced invariant is
    # HYBRID additivity: with reranking on, enabling the lexical channel +
    # logit gate never removes an injection the non-hybrid reranked path makes.
    # (Reranking itself reorders passers by blended score — pre-existing, by
    # design — so a binding max_results can select a different passer subset
    # than the reranking-DISABLED path; that is the reranker's job, not a
    # hybrid regression.)
    passers = sorted(
        (e for e in candidates if e.get("_cosine_ok", True)),
        key=lambda e: e["relevance"], reverse=True,
    )
    rescued = sorted(
        (e for e in candidates if not e.get("_cosine_ok", True)),
        key=lambda e: e["relevance"], reverse=True,
    )
    prioritized = passers + rescued
    deduped = prioritized[:max_results]
    for entry in prioritized[max_results:]:
        trace_by_key[entry["_trace_key"]].terminal_reason = "result_cap"

    # Budget-based selection: 3-way split (local / obsidian / remote).
    # Each bucket gets a guaranteed share; unused budget overflows to others.
    VAULT_REF_COST = 120  # estimated chars for "See: path + heading"

    # Check if any remote schemas are in play
    has_remote = any(e.get("_schema", "").startswith("remote_") for e in deduped)
    if has_remote:
        third = budget // 3
        local_remaining = third
        vault_remaining = third
        remote_remaining = budget - 2 * third  # absorb rounding remainder
    else:
        half = budget // 2
        local_remaining = half
        vault_remaining = half
        remote_remaining = 0

    selected = []

    def _try_spend(cost: int, primary: str) -> bool:
        """Try to spend from primary bucket, overflow to others, or split across buckets.

        Phase 1: Try each bucket individually (prefer primary, then others).
        Phase 2: If no single bucket suffices, split the cost across all buckets
        that have remaining capacity (prevents large entries from being silently
        dropped when total remaining budget is sufficient but fragmented).
        """
        nonlocal local_remaining, vault_remaining, remote_remaining
        buckets = {"local": local_remaining, "vault": vault_remaining, "remote": remote_remaining}
        order = [primary] + [b for b in ("local", "vault", "remote") if b != primary]

        # Phase 1: single-bucket fit
        for bucket in order:
            if buckets[bucket] >= cost:
                if bucket == "local":
                    local_remaining -= cost
                elif bucket == "vault":
                    vault_remaining -= cost
                else:
                    remote_remaining -= cost
                return True

        # Phase 2: split across buckets when total remaining >= cost
        total_available = local_remaining + vault_remaining + remote_remaining
        if total_available >= cost:
            remaining_cost = cost
            for bucket in order:
                take = min(remaining_cost, buckets[bucket])
                if take > 0:
                    if bucket == "local":
                        local_remaining -= take
                    elif bucket == "vault":
                        vault_remaining -= take
                    else:
                        remote_remaining -= take
                    remaining_cost -= take
                if remaining_cost <= 0:
                    break
            return True

        return False

    for entry in deduped:
        schema = entry.get("_schema", "obsidian")
        is_vault = schema == "obsidian"
        is_remote = schema.startswith("remote_")

        if is_vault:
            cost = VAULT_REF_COST
            trace_by_key[entry["_trace_key"]].display_cost = cost
            if _try_spend(cost, "vault"):
                entry["display_mode"] = "reference"
                selected.append(entry)
            else:
                trace_by_key[entry["_trace_key"]].terminal_reason = "budget_rejected"
        else:
            content_len = len(entry["document"] or "")
            cost = max(content_len, 50)  # minimum 50 chars cost
            trace_by_key[entry["_trace_key"]].display_cost = cost
            bucket = "remote" if is_remote else "local"
            if _try_spend(cost, bucket):
                entry["display_mode"] = "full"
                selected.append(entry)
            else:
                trace_by_key[entry["_trace_key"]].terminal_reason = "budget_rejected"

    # Membership was decided passers-first (above); re-sort the survivors by
    # relevance for display so the injected order still reads best-first.
    selected.sort(key=lambda e: e["relevance"], reverse=True)

    # Fractional retrieval bump for passively surfaced results
    surfaced_ids = [entry["doc_id"] for entry in selected] if selected else []
    if selected and not skip_retrieval_increment:
        passive_increment = enrichment_config.get("passive_retrieval_increment", 0.01)
        if passive_increment > 0:
            _increment_retrieval_counts(surfaced_ids, increment=passive_increment)

    matches = []
    for final_rank, entry in enumerate(selected, start=1):
        selected_trace = trace_by_key[entry["_trace_key"]]
        selected_trace.final_rank = final_rank
        selected_trace.returned = True
        selected_trace.terminal_reason = "selected"
        meta = entry["metadata"]
        doc_type = meta.get("vault_type") or meta.get("type", "unknown")
        chunk_heading = meta.get("chunk_heading", "")

        if entry["display_mode"] == "reference":
            # Vault reference: path + heading only, no content
            ref_text = entry["parent_file"]
            if chunk_heading:
                ref_text += f" (section: {chunk_heading})"
            content = ref_text
        else:
            # Core memory: full content, no truncation
            entry_fmt = _detect_format_from_entry(entry)
            content = (
                _extract_preview(entry["document"], max_len=10000, fmt=entry_fmt)
                if entry["document"]
                else ""
            )

        match = {
            "id": entry["parent_file"],
            "relevance": round(entry["relevance"], 3),
            "similarity": round(entry.get("similarity", 0.0), 3),
            "type": doc_type,
            "content": content,
            "display_mode": entry["display_mode"],
            "schema": entry.get("_schema", "local"),
            "candidate_key": selected_trace.candidate_key,
        }
        if chunk_heading:
            match["heading"] = chunk_heading
        if entry.get("is_stale"):
            match["stale"] = True

        # Attribution for remote memories: surface origin so the LLM
        # can distinguish "my memory" from "synced from another user".
        schema = entry.get("_schema", "")
        if schema.startswith("remote_"):
            match["source_remote"] = schema
            # Try to extract author hint from project_path metadata
            project_path = meta.get("project_path", "")
            if "/Users/" in project_path:
                # /Users/<username>/... → extract username
                parts = project_path.split("/Users/", 1)
                if len(parts) > 1:
                    author = parts[1].split("/", 1)[0]
                    if author:
                        match["origin_user"] = author

        matches.append(match)

    query_ms = round((time.time() - start) * 1000, 1)

    result = {
        "matches": matches,
        "query_ms": query_ms,
        "total_searched": total,
        "skipped_sensitive": skipped_sensitive,
        "semantic_duplicates_suppressed": semantic_duplicates_suppressed,
        "budget_used": {
            "local": (third if has_remote else half) - local_remaining,
            "vault": (third if has_remote else half) - vault_remaining,
            "remote": (budget - 2 * third if has_remote else 0) - remote_remaining if has_remote else 0,
            "total": budget,
        },
    }

    try:
        from .config import get_embedding_config
        from .retrieval_telemetry import record_event

        emb_cfg = get_embedding_config()
        terminal_counts = {}
        for candidate in trace_candidates:
            reason = candidate.terminal_reason or "candidate_cap"
            candidate.terminal_reason = reason
            terminal_counts[reason] = terminal_counts.get(reason, 0) + 1
        trace_id = record_event(
            purpose="context_injection", query=query, candidates=trace_candidates,
            funnel={
                "query_windows": expansion.get("inference_window_count", 1),
                "ann_unique": ann_unique, "sensitive_rejected": skipped_sensitive,
                "cosine_passed": sum(
                    1 for c in trace_candidates
                    if c.similarity is not None and c.similarity >= threshold
                    and c.terminal_reason not in {"sensitive", "superseded"}
                ),
                "cosine_rejected": sum(
                    1 for c in trace_candidates if c.terminal_reason == "cosine_rejected"
                ),
                "lexical_candidates": lexical_stats.get("lexical_candidates", 0),
                "lexical_added": lexical_stats.get("lexical_added", 0),
                # logit_rescued counts gate-stage rescues (visible even when a
                # rescue is later dropped by dedup / result_cap / budget);
                # logit_rescued_injected counts those that reached the output.
                "logit_rescued": logit_rescued,
                "logit_rescued_injected": sum(1 for e in selected if e.get("_logit_rescued")),
                "logit_rejected": logit_rejected,
                "logit_unjudged_dropped": logit_unjudged_dropped,
                "parent_dedup": parent_dedup_count_sc,
                "candidate_cap": len(best_per_file),
                "live_reranker_input": len(best_per_file) if reranking_config.get("enabled") else 0,
                "live_reranker_applied": bool(rerank_detail and rerank_detail.applied),
                "live_reranker_fallback": rerank_detail.fallback_reason if rerank_detail and not rerank_detail.applied else None,
                "semantic_duplicates_suppressed": semantic_duplicates_suppressed,
                "result_cap": len(deduped), "budget_selected": len(selected),
                "terminal_reasons": terminal_counts,
            },
            latency={
                "search_ms": round(search_ms, 2),
                "rerank_ms": round(rerank_detail.latency_ms, 2) if rerank_detail else 0,
                "total_ms": round((time.perf_counter() - retrieval_started) * 1000, 2),
            },
            outcome="results" if matches else "empty",
            query_window_count=expansion.get("inference_window_count", 1),
            model_snapshot={
                "embedding_model": emb_cfg.get("model_id"),
                "embedding_backend": emb_cfg.get("backend"),
                "reranker_model": reranking_config.get("model"),
                "reranker_backend": reranking_config.get("backend"),
            },
            config_snapshot={
                "cosine_threshold": threshold,
                "bge_logit_threshold": bge_logit_threshold,
                "budget": budget,
                "max_results": max_results,
                "reranking_enabled": bool(reranking_config.get("enabled")),
                "reranking_alpha": reranking_config.get("alpha"),
                # Both keys: the boolean stays for pre-3.6 readers, the mode is
                # what shadow scoring's augmentation-identity guard compares
                # (mechanical and summary are different embedding spaces).
                "contextual_embeddings": get_contextual_embeddings_enabled(),
                "contextual_augmentation": get_contextual_augmentation_mode(),
                "schemas": schemas or ["all"],
            },
        )
        if trace_id:
            result["trace_id"] = trace_id
    except Exception:
        pass

    if expansion.get("base_window_count", 1) > 1:
        result["query_windows"] = {
            "input": expansion["base_window_count"],
            "inference": expansion["inference_window_count"],
        }

    # Include surfaced IDs when caller needs to bump externally
    if skip_retrieval_increment and surfaced_ids:
        result["surfaced_ids"] = surfaced_ids

    return result


def doc_read(ids: list, include_metadata: bool = True) -> dict:
    """Read specific documents from PostgreSQL by ID.

    Routes by ID prefix: vault:: → obsidian.documents, else → local.memories.

    Accepts both namespaced IDs (vault::notes/my-note.md) and bare paths
    (notes/my-note.md). Bare paths are automatically prefixed with vault::.

    Args:
        ids: Document IDs (vault-relative paths or namespaced IDs)
        include_metadata: Whether to include metadata in response

    Returns:
        Documents with optional metadata, plus not_found list
    """
    if not ids:
        return {"success": False, "error": "No IDs provided"}

    # Normalize IDs: bare paths get vault:: prefix
    vault_ids = []
    core_ids = []
    id_map = {}  # lookup_id -> original_id (for display)

    for doc_id in ids:
        if "::" not in doc_id:
            namespaced = f"vault::{doc_id}"
            id_map[namespaced] = doc_id
            vault_ids.append(namespaced)
        elif doc_id.startswith("vault::"):
            id_map[doc_id] = doc_id
            vault_ids.append(doc_id)
        else:
            id_map[doc_id] = doc_id
            core_ids.append(doc_id)

    found = []
    found_ids = set()

    try:
        # Read vault documents
        if vault_ids:
            meta_cols = ", metadata, parent_file, directory, vault_type, title, chunk_index, chunk_total, chunk_heading, importance_score" if include_metadata else ""
            sql = f"SELECT id, document{meta_cols} FROM obsidian.documents WHERE id = ANY(%s)"
            rows = execute_query(sql, (vault_ids,))
            for row in rows:
                entry = {
                    "id": row["id"],
                    "path": _display_path(row["id"]),
                    "document": row["document"],
                }
                if include_metadata:
                    entry["metadata"] = _format_vault_result(row)
                found.append(entry)
                found_ids.add(row["id"])

        # Read core memories
        if core_ids:
            meta_cols = ", metadata, category, scope, source, importance_score" if include_metadata else ""
            sql = f"SELECT id, document{meta_cols} FROM local.memories WHERE id = ANY(%s)"
            rows = execute_query(sql, (core_ids,))
            for row in rows:
                entry = {
                    "id": row["id"],
                    "path": _display_path(row["id"]),
                    "document": row["document"],
                }
                if include_metadata:
                    entry["metadata"] = _format_core_result(row)
                found.append(entry)
                found_ids.add(row["id"])

    except Exception as e:
        return {"success": False, "error": f"Read failed: {e}"}

    all_lookup_ids = vault_ids + core_ids
    not_found = [
        id_map.get(lid, lid) for lid in all_lookup_ids if lid not in found_ids
    ]

    return {
        "success": True,
        "documents": found,
        "not_found": not_found,
    }


def collection_stats(sample_size: int = 5, detailed: bool = False) -> dict:
    """Get memory system health and statistics.

    Queries both local.memories and obsidian.documents.

    Args:
        sample_size: Number of sample entries to peek
        detailed: Include per-type/namespace breakdowns and storage size

    Returns:
        Stats dict with count, samples, type distribution
    """
    try:
        core_count = execute_query(
            "SELECT count(*) AS cnt FROM local.memories WHERE status = 'active'",
            fetch="one",
        )
        vault_count = execute_query(
            "SELECT count(*) AS cnt FROM obsidian.documents", fetch="one"
        )
        core_total = core_count["cnt"] if core_count else 0
        vault_total = vault_count["cnt"] if vault_count else 0
        total = core_total + vault_total
    except Exception as e:
        return {"success": False, "error": f"Database unavailable: {e}"}

    if total == 0:
        return {
            "success": True,
            "total_documents": 0,
            "samples": [],
            "message": "No documents indexed. Ask Jarvis to 'index my vault' or use jarvis_index_vault tool.",
        }

    # Peek at samples from both schemas
    samples = []
    half_sample = max(1, sample_size // 2)

    try:
        # Core samples
        core_rows = execute_query(
            "SELECT id, metadata, category FROM local.memories WHERE status = 'active' LIMIT %s",
            (half_sample,),
        )
        for row in core_rows:
            meta = jsonb_to_metadata(row.get("metadata", {}))
            samples.append(
                {
                    "id": row["id"],
                    "path": _display_path(row["id"]),
                    "title": meta.get("title", row["id"]),
                    "type": row.get("category", meta.get("type", "unknown")),
                    "schema": "local",
                }
            )

        # Vault samples
        vault_rows = execute_query(
            "SELECT id, metadata, vault_type, title FROM obsidian.documents LIMIT %s",
            (half_sample,),
        )
        for row in vault_rows:
            samples.append(
                {
                    "id": row["id"],
                    "path": _display_path(row["id"]),
                    "title": row.get("title", row["id"]),
                    "type": row.get("vault_type", "document"),
                    "schema": "obsidian",
                }
            )
    except Exception as e:
        return {"success": False, "error": f"Stats failed: {e}"}

    result = {
        "success": True,
        "total_documents": total,
        "core_documents": core_total,
        "vault_documents": vault_total,
        "samples": samples,
    }

    # Remote schema stats (Step 8: provenance + freshness)
    from .schema_registry import get_searchable_schemas as _gss_cs, SchemaKind as _SK_cs, is_valid_pg_identifier as _vpi_cs
    from .schema import get_meta as _get_meta_cs
    remote_entries = _gss_cs(kind=_SK_cs.REMOTE)
    if remote_entries:
        remote_stats = []
        for entry in remote_entries:
            if not _vpi_cs(entry.name) or not _vpi_cs(entry.table):
                continue
            try:
                count = _remote_count(entry)
                pull_meta = _get_meta_cs(f"pull_sync_ts:{entry.remote_name}")
                last_pull = pull_meta.get("timestamp") if pull_meta else None
                remote_stats.append({
                    "schema": entry.name,
                    "remote_name": entry.remote_name,
                    "count": count,
                    "last_pull_ts": last_pull,
                })
            except Exception as e:
                remote_stats.append({
                    "schema": entry.name,
                    "remote_name": entry.remote_name,
                    "error": str(e),
                })
        result["remote_schemas"] = remote_stats

    # Detailed breakdown
    if detailed:
        try:
            # Core breakdown by category
            cat_rows = execute_query(
                """SELECT category, count(*) AS cnt
                   FROM local.memories
                   WHERE status = 'active'
                   GROUP BY category"""
            )
            category_counts = {row["category"]: row["cnt"] for row in cat_rows}

            # Vault breakdown by vault_type
            vt_rows = execute_query(
                """SELECT vault_type, count(*) AS cnt
                   FROM obsidian.documents
                   GROUP BY vault_type"""
            )
            vault_type_counts = {row["vault_type"]: row["cnt"] for row in vt_rows}

            result["category_breakdown"] = category_counts
            result["vault_type_breakdown"] = vault_type_counts

            # Storage size (both tables)
            for table, label in [
                ("local.memories", "core_storage"),
                ("obsidian.documents", "vault_storage"),
            ]:
                try:
                    size_result = execute_query(
                        f"SELECT pg_total_relation_size('{table}') AS total_bytes",
                        fetch="one",
                    )
                    if size_result:
                        result[f"{label}_bytes"] = size_result["total_bytes"]
                        result[f"{label}_mb"] = round(
                            size_result["total_bytes"] / (1024 * 1024), 2
                        )
                except Exception:
                    pass

        except Exception as e:
            result["detailed_error"] = str(e)

    return result


# Backward-compatible aliases (will be removed in future version)
memory_read = doc_read
memory_stats = collection_stats
