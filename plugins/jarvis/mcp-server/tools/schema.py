"""PostgreSQL + pgvector schema and connection management.

Provides the singleton connection pool and schema initialization for
the local.memories and obsidian.documents tables. Replaces the single-table
public.jarvis design (v2.x) with dual-schema architecture (v3.0).

Schemas:
- local: memories (observations, patterns, strategic, etc.)
- obsidian: indexed vault file chunks
"""

from __future__ import annotations

import contextlib
import contextvars
import json
import logging
import os
import re
import threading
import time
from typing import Any
from urllib.parse import unquote

import psycopg
import psycopg_pool

logger = logging.getLogger("jarvis-core")

# Singleton pool state
_pool = None
_pool_cache_key: tuple | None = None
# Hook and MCP handlers run in worker threads, so first use and a config-driven
# replacement can race; the lock keeps it to one pool.
_pool_lock = threading.Lock()

# ── Schema SQL ────────────────────────────────────────────────────────

LOCAL_SCHEMA_SQL = """\
CREATE EXTENSION IF NOT EXISTS vector;

CREATE SCHEMA IF NOT EXISTS local;

CREATE TABLE IF NOT EXISTS local.memories (
    id TEXT PRIMARY KEY,
    document TEXT NOT NULL,
    embedding halfvec({dimensions}) NOT NULL,

    -- Classification columns
    category TEXT NOT NULL DEFAULT 'observation'
        CHECK (category IN ('observation', 'pattern', 'learning', 'decision',
                            'summary', 'code', 'relationship', 'hint', 'plan',
                            'worklog', 'memory')),
    scope TEXT NOT NULL DEFAULT 'global'
        CHECK (scope IN ('global', 'project')),
    project TEXT,
    source TEXT NOT NULL DEFAULT 'auto-extract',
    importance_score FLOAT NOT NULL DEFAULT 0.5
        CHECK (importance_score >= 0.0 AND importance_score <= 1.0),
    retrieval_count FLOAT NOT NULL DEFAULT 0.0,

    -- Lifecycle
    status TEXT NOT NULL DEFAULT 'active'
        CHECK (status IN ('active', 'superseded', 'deleted')),
    superseded_by TEXT,
    deleted_at TIMESTAMPTZ,

    -- Remaining flexible metadata
    metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb,

    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Cross-field integrity
DO $$ BEGIN
    ALTER TABLE local.memories ADD CONSTRAINT chk_scope_project
        CHECK ((scope = 'project' AND project IS NOT NULL) OR (scope = 'global'));
EXCEPTION WHEN duplicate_object THEN NULL;
END $$;

DO $$ BEGIN
    ALTER TABLE local.memories ADD CONSTRAINT chk_superseded_by
        CHECK ((status = 'superseded' AND superseded_by IS NOT NULL) OR (status != 'superseded'));
EXCEPTION WHEN duplicate_object THEN NULL;
END $$;

-- Indexes
CREATE INDEX IF NOT EXISTS idx_local_embedding ON local.memories
    USING hnsw (embedding halfvec_cosine_ops) WITH (m = 16, ef_construction = 200);
CREATE INDEX IF NOT EXISTS idx_local_metadata ON local.memories USING gin (metadata jsonb_path_ops);
CREATE INDEX IF NOT EXISTS idx_local_category ON local.memories (category);
CREATE INDEX IF NOT EXISTS idx_local_active ON local.memories (status) WHERE status = 'active';
CREATE INDEX IF NOT EXISTS idx_local_importance ON local.memories (importance_score DESC);

-- Search-only windows for canonical memories that exceed an inference context.
-- The full document remains in local.memories and ID reads always return it.
CREATE TABLE IF NOT EXISTS local.memory_chunks (
    parent_id TEXT NOT NULL REFERENCES local.memories(id)
        ON DELETE CASCADE ON UPDATE CASCADE,
    chunk_index INTEGER NOT NULL,
    chunk_total INTEGER NOT NULL,
    document TEXT NOT NULL,
    embedding halfvec({dimensions}) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (parent_id, chunk_index)
);
CREATE INDEX IF NOT EXISTS idx_local_memory_chunks_embedding
    ON local.memory_chunks
    USING hnsw (embedding halfvec_cosine_ops) WITH (m = 16, ef_construction = 200);
CREATE INDEX IF NOT EXISTS idx_local_memory_chunks_parent
    ON local.memory_chunks (parent_id);

-- Active view (query default — excludes superseded + deleted)
CREATE OR REPLACE VIEW local.active_memories AS
    SELECT * FROM local.memories WHERE status = 'active';

-- updated_at trigger function (shared by both schemas)
CREATE OR REPLACE FUNCTION update_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = now();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_trigger WHERE tgname = 'trg_local_memories_updated_at'
    ) THEN
        CREATE TRIGGER trg_local_memories_updated_at
            BEFORE UPDATE ON local.memories
            FOR EACH ROW EXECUTE FUNCTION update_updated_at();
    END IF;
END;
$$;
"""

# Deprecated alias
CORE_SCHEMA_SQL = LOCAL_SCHEMA_SQL


# Retrieval observability is intentionally kept in its own tables. Candidate
# text never belongs here: locators and scores are enough to replay a trace and
# prevent the telemetry store from becoming a second copy of the vault.
RETRIEVAL_TELEMETRY_SCHEMA_SQL = """\
CREATE TABLE IF NOT EXISTS local.retrieval_events (
    id UUID PRIMARY KEY,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    expires_at TIMESTAMPTZ NOT NULL,
    user_name TEXT,
    purpose TEXT NOT NULL,
    pipeline TEXT NOT NULL DEFAULT 'semantic',
    status TEXT NOT NULL DEFAULT 'complete',
    outcome TEXT NOT NULL DEFAULT 'unknown',
    query_text TEXT,
    query_sha256 TEXT NOT NULL,
    query_ref TEXT,
    query_length INTEGER NOT NULL DEFAULT 0,
    query_window_count INTEGER NOT NULL DEFAULT 1,
    model_snapshot JSONB NOT NULL DEFAULT '{}'::jsonb,
    config_snapshot JSONB NOT NULL DEFAULT '{}'::jsonb,
    funnel JSONB NOT NULL DEFAULT '{}'::jsonb,
    latency JSONB NOT NULL DEFAULT '{}'::jsonb,
    delivery JSONB NOT NULL DEFAULT '{}'::jsonb,
    shadow_status TEXT NOT NULL DEFAULT 'disabled'
        CHECK (shadow_status IN ('disabled', 'pending', 'running', 'complete',
                                 'partial', 'failed', 'skipped')),
    shadow_attempts INTEGER NOT NULL DEFAULT 0,
    shadow_started_at TIMESTAMPTZ,
    shadow_finished_at TIMESTAMPTZ,
    shadow_error TEXT
);

CREATE TABLE IF NOT EXISTS local.retrieval_candidates (
    event_id UUID NOT NULL REFERENCES local.retrieval_events(id) ON DELETE CASCADE,
    candidate_key TEXT NOT NULL,
    schema_name TEXT NOT NULL,
    doc_id TEXT NOT NULL,
    parent_id TEXT,
    parent_file TEXT,
    chunk_index INTEGER,
    query_window_index INTEGER NOT NULL DEFAULT 0,
    vector_rank INTEGER,
    final_rank INTEGER,
    similarity DOUBLE PRECISION,
    pre_score DOUBLE PRECISION,
    raw_bge_logit DOUBLE PRECISION,
    bge_probability DOUBLE PRECISION,
    blended_score DOUBLE PRECISION,
    display_cost INTEGER,
    terminal_reason TEXT,
    returned BOOLEAN NOT NULL DEFAULT false,
    delivered BOOLEAN NOT NULL DEFAULT false,
    PRIMARY KEY (event_id, candidate_key)
);

CREATE TABLE IF NOT EXISTS local.retrieval_feedback (
    event_id UUID PRIMARY KEY REFERENCES local.retrieval_events(id) ON DELETE CASCADE,
    verdict TEXT NOT NULL CHECK (verdict IN ('useful', 'mixed', 'noisy', 'missed', 'unsure')),
    expected_missing_ids JSONB NOT NULL DEFAULT '[]'::jsonb,
    note TEXT,
    user_name TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS local.retrieval_candidate_feedback (
    event_id UUID NOT NULL,
    candidate_key TEXT NOT NULL,
    verdict TEXT NOT NULL CHECK (verdict IN ('relevant', 'irrelevant', 'unsure')),
    note TEXT,
    user_name TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (event_id, candidate_key),
    FOREIGN KEY (event_id, candidate_key)
        REFERENCES local.retrieval_candidates(event_id, candidate_key)
        ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_retrieval_events_created
    ON local.retrieval_events (created_at DESC);
CREATE INDEX IF NOT EXISTS idx_retrieval_events_expires
    ON local.retrieval_events (expires_at);
CREATE INDEX IF NOT EXISTS idx_retrieval_events_purpose
    ON local.retrieval_events (purpose, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_retrieval_events_shadow
    ON local.retrieval_events (shadow_status, created_at)
    WHERE shadow_status IN ('pending', 'running');
-- idx_retrieval_candidates_event (event_id, vector_rank) duplicated the primary
-- key's leading event_id column: per-event lookups, the FK cascade and the
-- feedback FK all use the PK, and sorting <=100 rows by vector_rank is trivial.
-- It was ~10% of the database and a third of candidate write amplification (and
-- the file that first hit ENOSPC). Idempotent drop for existing installs.
DROP INDEX IF EXISTS local.idx_retrieval_candidates_event;
CREATE INDEX IF NOT EXISTS idx_retrieval_candidate_doc
    ON local.retrieval_candidates (schema_name, doc_id);
"""


OBSIDIAN_SCHEMA_SQL = """\
CREATE SCHEMA IF NOT EXISTS obsidian;

CREATE TABLE IF NOT EXISTS obsidian.documents (
    id TEXT PRIMARY KEY,
    document TEXT NOT NULL,
    embedding halfvec({dimensions}) NOT NULL,

    -- Vault-specific columns
    parent_file TEXT NOT NULL,
    directory TEXT NOT NULL DEFAULT '',
    vault_type TEXT NOT NULL DEFAULT 'document',
    title TEXT NOT NULL DEFAULT '',
    chunk_index INTEGER NOT NULL DEFAULT 0,
    chunk_total INTEGER NOT NULL DEFAULT 1,
    chunk_heading TEXT NOT NULL DEFAULT '',
    importance_score FLOAT NOT NULL DEFAULT 0.5,

    -- Remaining flexible metadata
    metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb,

    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_obsidian_embedding ON obsidian.documents
    USING hnsw (embedding halfvec_cosine_ops) WITH (m = 16, ef_construction = 200);
CREATE INDEX IF NOT EXISTS idx_obsidian_metadata ON obsidian.documents USING gin (metadata jsonb_path_ops);
CREATE INDEX IF NOT EXISTS idx_obsidian_parent_file ON obsidian.documents (parent_file);
CREATE INDEX IF NOT EXISTS idx_obsidian_directory ON obsidian.documents (directory);
CREATE INDEX IF NOT EXISTS idx_obsidian_importance ON obsidian.documents (importance_score DESC);

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_trigger WHERE tgname = 'trg_obsidian_documents_updated_at'
    ) THEN
        CREATE TRIGGER trg_obsidian_documents_updated_at
            BEFORE UPDATE ON obsidian.documents
            FOR EACH ROW EXECUTE FUNCTION update_updated_at();
    END IF;
END;
$$;
"""

# Deprecated alias
VAULT_SCHEMA_SQL = OBSIDIAN_SCHEMA_SQL


# LLM-generated document context: ONE situating sentence per FILE, cached.
#
# A separate table (not a column on obsidian.documents) because the summary is a
# property of the FILE, not of the fragment: a 90-chunk document would otherwise
# store 90 identical copies and every reindex would have to keep them in sync.
# ``content_hash`` is the sha256 of the document text the summary was generated
# from, so an unchanged file never re-calls the LLM across reindexes; a changed
# file misses the cache and regenerates. ``model`` records which model wrote it.
#
# There is deliberately no FK to obsidian.documents: rows survive a
# force-reindex (which deletes and reinserts every chunk), which is exactly the
# cache-reuse property the table exists for. Orphans are harmless — they are
# only ever read by parent_file lookup.
DOCUMENT_CONTEXT_SCHEMA_SQL = """\
CREATE TABLE IF NOT EXISTS obsidian.document_context (
    parent_file TEXT PRIMARY KEY,
    summary TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    model TEXT NOT NULL,
    generated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
"""


# Phase 1 hybrid retrieval — statistical (lexical) recall channel.
# Generated STORED tsvector columns + GIN indexes give a full-text recall
# channel alongside the bi-encoder ANN channel. `to_tsvector('english', …)`
# (explicit regconfig) is IMMUTABLE — required for a GENERATED column; the
# one-argument form is only STABLE and would be rejected here.
#
# NOTE: to_tsvector emits a harmless NOTICE ("word is too long to be indexed")
# for any single token longer than 2047 bytes; the token is skipped, not an
# error, and the DDL still succeeds.
#
# BODY CAP: a single tsvector may hold at most 1MB of lexeme data. A stored,
# indexed memory can legitimately hold a whole pasted transcript/log (many
# unique tokens — hashes, ids, code), whose to_tsvector would exceed that limit
# and make BOTH the generated-column ALTER (rewritten for every existing row)
# and every future INSERT of such a row FAIL. The body input is therefore capped
# with ``left(document, 200000)`` (~200KB → well under 1MB even for all-unique
# tokens) inside the generated expression; title/heading are short and left
# uncapped. Lexical recall reads the head of the document, which is where the
# informative material lives.
#
# All statements are additive and idempotent (ADD COLUMN IF NOT EXISTS /
# CREATE INDEX IF NOT EXISTS), so ensure_schema can run this on every startup.
LEXICAL_SCHEMA_SQL = """\
-- obsidian.documents: title (A) + chunk_heading (B) + body (D), weighted.
ALTER TABLE obsidian.documents ADD COLUMN IF NOT EXISTS tsv tsvector
    GENERATED ALWAYS AS (
        setweight(to_tsvector('english', coalesce(title, '')), 'A') ||
        setweight(to_tsvector('english', coalesce(chunk_heading, '')), 'B') ||
        setweight(to_tsvector('english', left(document, 200000)), 'D')
    ) STORED;
CREATE INDEX IF NOT EXISTS idx_obsidian_tsv
    ON obsidian.documents USING gin (tsv);

-- local.memories: body only (D).
ALTER TABLE local.memories ADD COLUMN IF NOT EXISTS tsv tsvector
    GENERATED ALWAYS AS (
        setweight(to_tsvector('english', left(document, 200000)), 'D')
    ) STORED;
CREATE INDEX IF NOT EXISTS idx_local_tsv
    ON local.memories USING gin (tsv);

-- Retrieval channel provenance: 'semantic' | 'lexical' | 'both'.
ALTER TABLE local.retrieval_candidates ADD COLUMN IF NOT EXISTS channel TEXT;

-- Shadow retry backoff. Without a next-attempt gate a failed job is reclaimed
-- on the very next poll, so max_attempts burns in seconds and a brief model-host
-- outage permanently censors those events from the calibration corpus.
ALTER TABLE local.retrieval_events
    ADD COLUMN IF NOT EXISTS shadow_next_attempt_at TIMESTAMPTZ;
"""


LOCAL_META_SQL = """\
CREATE TABLE IF NOT EXISTS local.meta (
    key TEXT PRIMARY KEY,
    value JSONB NOT NULL DEFAULT '{}'::jsonb,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_trigger WHERE tgname = 'trg_local_meta_updated_at'
    ) THEN
        CREATE TRIGGER trg_local_meta_updated_at
            BEFORE UPDATE ON local.meta
            FOR EACH ROW EXECUTE FUNCTION update_updated_at();
    END IF;
END;
$$;
"""

# Deprecated alias
CORE_META_SQL = LOCAL_META_SQL


MIGRATION_SQL = """\
-- Step 1: Vault rows → obsidian.documents
INSERT INTO obsidian.documents (id, document, embedding, parent_file, directory,
    vault_type, title, chunk_index, chunk_total, chunk_heading,
    importance_score, metadata, created_at, updated_at)
SELECT id, document, embedding,
    COALESCE(metadata->>'parent_file', REPLACE(id, 'vault::', '')),
    COALESCE(metadata->>'directory', ''),
    COALESCE(metadata->>'vault_type', 'document'),
    COALESCE(metadata->>'title', ''),
    CASE WHEN metadata->>'chunk_index' ~ '^\\d+$'
         THEN (metadata->>'chunk_index')::int ELSE 0 END,
    CASE WHEN metadata->>'chunk_total' ~ '^\\d+$'
         THEN (metadata->>'chunk_total')::int ELSE 1 END,
    COALESCE(metadata->>'chunk_heading', ''),
    CASE WHEN metadata->>'importance_score' ~ '^\\d+\\.?\\d*$'
         THEN LEAST((metadata->>'importance_score')::float, 1.0) ELSE 0.5 END,
    metadata - 'parent_file' - 'directory' - 'vault_type' - 'title'
            - 'chunk_index' - 'chunk_total' - 'chunk_heading'
            - 'importance_score' - 'tier' - 'namespace' - 'type'
            - 'promoted' - 'source',
    created_at, updated_at
FROM jarvis WHERE id LIKE 'vault::%%'
ON CONFLICT (id) DO NOTHING;

-- Step 2: Memory/content rows → local.memories
INSERT INTO local.memories (id, document, embedding, category, scope, project,
    source, importance_score, retrieval_count, status, superseded_by,
    metadata, created_at, updated_at)
SELECT id, document, embedding,
    CASE WHEN id LIKE 'memory::%%' THEN 'memory'
         WHEN metadata->>'type' IN ('observation','pattern','learning','decision',
              'summary','code','relationship','hint','plan','worklog','memory')
         THEN metadata->>'type'
         ELSE 'observation' END,
    CASE WHEN metadata->>'scope' IN ('global','project')
         THEN metadata->>'scope' ELSE 'global' END,
    NULLIF(metadata->>'project', ''),
    COALESCE(metadata->>'source', 'auto-extract'),
    CASE WHEN metadata->>'importance_score' ~ '^\\d+\\.?\\d*$'
         THEN LEAST((metadata->>'importance_score')::float, 1.0) ELSE 0.5 END,
    CASE WHEN metadata->>'retrieval_count' ~ '^\\d+\\.?\\d*$'
         THEN (metadata->>'retrieval_count')::float ELSE 0.0 END,
    CASE WHEN metadata->>'status' IN ('active','superseded','deleted')
         THEN metadata->>'status' ELSE 'active' END,
    NULLIF(metadata->>'superseded_by', ''),
    metadata - 'type' - 'scope' - 'project' - 'source' - 'importance_score'
            - 'retrieval_count' - 'status' - 'superseded_by' - 'tier'
            - 'namespace' - 'promoted' - 'promoted_at' - 'original_tier2_id',
    created_at, updated_at
FROM jarvis WHERE id NOT LIKE 'vault::%%'
ON CONFLICT (id) DO NOTHING;

-- Step 3: Migrate meta table
INSERT INTO local.meta (key, value, updated_at)
SELECT key, value, updated_at FROM jarvis_meta
ON CONFLICT (key) DO NOTHING;

-- Step 4: Bump schema version
INSERT INTO local.meta (key, value) VALUES ('schema_version', '{{"version": 3}}')
ON CONFLICT (key) DO UPDATE SET value = '{{"version": 3}}'::jsonb;
"""


CONSOLIDATION_SCHEMA_SQL = """\
-- Phase 8: LLM-driven consolidation support
ALTER TABLE local.memories ADD COLUMN IF NOT EXISTS
    consolidation_run_id TEXT;

-- Self-supersession prevention
DO $$ BEGIN
    ALTER TABLE local.memories ADD CONSTRAINT chk_no_self_supersession
        CHECK (id != superseded_by);
EXCEPTION WHEN duplicate_object THEN NULL;
END $$;

-- Cycle prevention trigger (A→B→A and longer chains)
CREATE OR REPLACE FUNCTION local.prevent_supersession_cycle() RETURNS trigger AS $$
BEGIN
    IF NEW.superseded_by IS NULL THEN
        RETURN NEW;
    END IF;
    IF EXISTS (
        WITH RECURSIVE chain AS (
            SELECT NEW.superseded_by AS node_id
            UNION ALL
            SELECT m.superseded_by
            FROM local.memories m
            JOIN chain c ON m.id = c.node_id
            WHERE m.superseded_by IS NOT NULL
        )
        SELECT 1 FROM chain WHERE node_id = NEW.id
    ) THEN
        RAISE EXCEPTION 'Supersession cycle detected: % would create a loop', NEW.id;
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_trigger WHERE tgname = 'trg_supersession_cycle'
    ) THEN
        CREATE TRIGGER trg_supersession_cycle
            BEFORE INSERT OR UPDATE OF superseded_by ON local.memories
            FOR EACH ROW WHEN (NEW.superseded_by IS NOT NULL)
            EXECUTE FUNCTION local.prevent_supersession_cycle();
    END IF;
END;
$$;

-- Index for consolidation run queries
CREATE INDEX IF NOT EXISTS idx_local_consolidation_run
    ON local.memories (consolidation_run_id)
    WHERE consolidation_run_id IS NOT NULL;
"""


SYNC_SCHEMA_SQL = """\
-- Phase 7: Multi-remote sync columns on local.memories
ALTER TABLE local.memories ADD COLUMN IF NOT EXISTS
    synced_to TEXT[] NOT NULL DEFAULT '{}';
ALTER TABLE local.memories ADD COLUMN IF NOT EXISTS
    origin TEXT NOT NULL DEFAULT 'local';

-- Routing composite index (only local, active memories need routing)
CREATE INDEX IF NOT EXISTS idx_local_routing
    ON local.memories (category, scope, project)
    WHERE status = 'active' AND origin = 'local';

-- Sync outbox queue
CREATE TABLE IF NOT EXISTS local.sync_queue (
    id SERIAL PRIMARY KEY,
    memory_id TEXT NOT NULL REFERENCES local.memories(id) ON DELETE CASCADE,
    destination TEXT NOT NULL,
    version INT NOT NULL DEFAULT 1,
    status TEXT NOT NULL DEFAULT 'pending'
        CHECK (status IN ('pending', 'sending', 'done', 'failed', 'dlq')),
    attempts INT NOT NULL DEFAULT 0,
    max_attempts INT NOT NULL DEFAULT 5,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    last_attempt TIMESTAMPTZ,
    next_retry_at TIMESTAMPTZ DEFAULT now(),
    error TEXT,
    UNIQUE (memory_id, destination, version)
);

CREATE INDEX IF NOT EXISTS idx_sync_queue_pending
    ON local.sync_queue (next_retry_at) WHERE status = 'pending';
CREATE INDEX IF NOT EXISTS idx_sync_queue_dlq
    ON local.sync_queue (destination) WHERE status = 'dlq';
"""


REMOTE_SCHEMA_SQL = """\
CREATE EXTENSION IF NOT EXISTS vector;

CREATE SCHEMA IF NOT EXISTS {schema};

-- Drop legacy flat table (nuke existing data on first CAS deployment)
DROP TABLE IF EXISTS {schema}.memories CASCADE;

-- CAS content store (immutable, deduped by hash)
CREATE TABLE IF NOT EXISTS {schema}.content (
    hash TEXT PRIMARY KEY,
    content TEXT NOT NULL,
    embedding halfvec({dimensions}) NOT NULL,
    embedding_model TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_{schema}_content_embedding ON {schema}.content
    USING hnsw (embedding halfvec_cosine_ops) WITH (m = 16, ef_construction = 200);

-- Mutable metadata references (FK to content)
CREATE TABLE IF NOT EXISTS {schema}.memory_refs (
    id TEXT PRIMARY KEY,
    content_hash TEXT NOT NULL REFERENCES {schema}.content(hash) ON DELETE RESTRICT,
    category TEXT NOT NULL DEFAULT 'observation'
        CHECK (category IN ('observation', 'pattern', 'learning', 'decision',
                            'summary', 'code', 'relationship', 'hint', 'plan',
                            'worklog', 'memory')),
    scope TEXT NOT NULL DEFAULT 'global'
        CHECK (scope IN ('global', 'project')),
    project TEXT,
    source TEXT NOT NULL DEFAULT 'auto-extract',
    importance_score FLOAT NOT NULL DEFAULT 0.5
        CHECK (importance_score >= 0.0 AND importance_score <= 1.0),
    retrieval_count FLOAT NOT NULL DEFAULT 0.0,
    status TEXT NOT NULL DEFAULT 'active'
        CHECK (status IN ('active', 'superseded', 'deleted')),
    superseded_by TEXT,
    deleted_at TIMESTAMPTZ,
    synced_to TEXT[] NOT NULL DEFAULT '{{}}',
    origin TEXT NOT NULL DEFAULT 'local',
    metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_{schema}_refs_content_hash ON {schema}.memory_refs (content_hash);
CREATE INDEX IF NOT EXISTS idx_{schema}_refs_metadata ON {schema}.memory_refs USING gin (metadata jsonb_path_ops);
CREATE INDEX IF NOT EXISTS idx_{schema}_refs_category ON {schema}.memory_refs (category);
CREATE INDEX IF NOT EXISTS idx_{schema}_refs_active ON {schema}.memory_refs (status) WHERE status = 'active';
CREATE INDEX IF NOT EXISTS idx_{schema}_refs_importance ON {schema}.memory_refs (importance_score DESC);

-- Backward-compat view for direct remote queries
CREATE OR REPLACE VIEW {schema}.active_memories AS
    SELECT r.id, c.content AS document, c.embedding,
           r.category, r.scope, r.project, r.source,
           r.importance_score, r.retrieval_count,
           r.status, r.superseded_by, r.deleted_at,
           r.synced_to, r.origin, r.metadata,
           r.created_at, r.updated_at,
           r.content_hash, c.embedding_model
    FROM {schema}.memory_refs r
    JOIN {schema}.content c ON c.hash = r.content_hash
    WHERE r.status = 'active';

-- updated_at trigger function (idempotent — may already exist on remote)
CREATE OR REPLACE FUNCTION update_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = now();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_trigger WHERE tgname = 'trg_{schema}_refs_updated_at'
    ) THEN
        CREATE TRIGGER trg_{schema}_refs_updated_at
            BEFORE UPDATE ON {schema}.memory_refs
            FOR EACH ROW EXECUTE FUNCTION update_updated_at();
    END IF;
END;
$$;
"""


LOCAL_MIRROR_SQL = """\
CREATE SCHEMA IF NOT EXISTS {schema};

CREATE TABLE IF NOT EXISTS {schema}.memories (
    id TEXT PRIMARY KEY,
    document TEXT NOT NULL,
    embedding halfvec({dimensions}),

    -- Classification columns (same as local.memories)
    category TEXT DEFAULT 'observation',
    scope TEXT DEFAULT 'global',
    project TEXT,
    source TEXT DEFAULT 'auto-extract',
    importance_score FLOAT DEFAULT 0.5,
    retrieval_count FLOAT DEFAULT 0,

    -- Lifecycle
    status TEXT DEFAULT 'active',
    superseded_by TEXT,
    deleted_at TIMESTAMPTZ,

    -- Sync metadata
    synced_to TEXT[] DEFAULT '{{}}',
    origin TEXT DEFAULT 'local',
    consolidation_run_id TEXT,

    -- Remaining flexible metadata
    metadata JSONB DEFAULT '{{}}'::jsonb,

    created_at TIMESTAMPTZ DEFAULT now(),
    updated_at TIMESTAMPTZ DEFAULT now()
);

-- Indexes (match local.memories for consistent HNSW search performance)
CREATE INDEX IF NOT EXISTS idx_{schema}_embedding ON {schema}.memories
    USING hnsw (embedding halfvec_cosine_ops) WITH (m = 16, ef_construction = 200);
CREATE INDEX IF NOT EXISTS idx_{schema}_active ON {schema}.memories (status)
    WHERE status = 'active';

-- Active view for query convenience
CREATE OR REPLACE VIEW {schema}.active_memories AS
    SELECT * FROM {schema}.memories WHERE status = 'active';

-- Reuse the shared updated_at trigger function (created by LOCAL_SCHEMA_SQL)
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_trigger WHERE tgname = 'trg_{schema}_memories_updated_at'
    ) THEN
        CREATE TRIGGER trg_{schema}_memories_updated_at
            BEFORE UPDATE ON {schema}.memories
            FOR EACH ROW EXECUTE FUNCTION update_updated_at();
    END IF;
END;
$$;
"""


# ── Outage resilience: fail-fast pool, circuit breaker, cached probe ──
#
# 2026-09 outage: the Docker VM disk filled and the embedded PostgreSQL sat in
# a crash/recovery loop for ~15h. Every checkout waited psycopg_pool's default
# 30s, and one hook ingest chains up to 8 checkouts (240s). Three layers now
# bound that:
#   1. the pool fails fast: 10s by default (memory.pool_timeout_seconds) and
#      HOOK_CONN_TIMEOUT on hook paths (checkout_timeout());
#   2. a circuit breaker: once checkouts fail against an unreachable database,
#      callers get DatabaseUnavailable immediately instead of paying the
#      timeout again. One half-open trial every _BREAKER_OPEN_SECONDS, and any
#      successful checkout (or probe) closes it;
#   3. a cached probe (probe_db_status / get_db_status) that /health reports
#      without touching the database on the request path.

HOOK_CONN_TIMEOUT = 1.5  # seconds; below the hook clients' 2.5s deadline
_DEFAULT_POOL_TIMEOUT = 10.0
_POOL_MAX_WAITING = 32
_POOL_RECONNECT_TIMEOUT = 60.0
_POOL_CONNECT_TIMEOUT = 3
# TCP keepalives and a TCP user timeout drop a connection whose peer vanished
# (VM paused, network gone) instead of blocking on it forever. A peer that is
# alive but frozen still ACKs, so callers keep their own deadlines too.
_POOL_CONNECT_KWARGS = {
    "connect_timeout": _POOL_CONNECT_TIMEOUT,
    "keepalives": 1,
    "keepalives_idle": 30,
    "keepalives_interval": 10,
    "keepalives_count": 3,
}
if psycopg.pq.version() >= 120000:  # libpq 12 added it; ignored where the OS lacks it
    _POOL_CONNECT_KWARGS["tcp_user_timeout"] = 30_000  # ms
_BREAKER_OPEN_SECONDS = 10.0
_PROBE_CONNECT_TIMEOUT = 2
_DISK_FULL_WINDOW_SECONDS = 300.0
# A recent disk-full error stops forcing "disk_full" once this long has passed
# without another one AND the database has shown it can write again (a
# connected probe with measured free space above the minimum, or a committed
# write). The full window then only applies when neither can be observed.
_DISK_FULL_CLEAR_SECONDS = 60.0
_DISK_FULL_MIN_FREE_BYTES = 64 * 1024 * 1024
_DISK_FULL_LOG_INTERVAL_SECONDS = 60.0
_DEFAULT_PGDATA = "/var/lib/postgresql/data"

# SQLSTATE classes that mean "the server cannot serve this right now" rather
# than "this statement is wrong": connection exceptions (08), transaction
# rollback such as deadlock (40), insufficient resources incl. disk full (53),
# operator intervention incl. recovery mode (57), system/IO errors (58). A
# connection lost mid-query has no SQLSTATE at all.
_TRANSIENT_SQLSTATE_CLASSES = ("08", "40", "53", "57", "58")
_RECOVERY_MARKERS = (
    "in recovery mode",
    "the database system is starting up",
    "not yet accepting connections",
)
_DSN_CREDENTIALS_RE = re.compile(r"(://[^/\s:@]*:)\S*@")
# libpq names the server it failed to reach ('connection to server at "h"
# (ip), port N failed: FATAL:  ...'). /health is unauthenticated, so the
# reason is kept and the target dropped.
_CONN_TARGET_RE = re.compile(
    r'connection to server (?:at "[^"]*"(?: \([^)]*\))?, port \d+|on socket "[^"]*") failed:\s*'
)
_SEVERITY_PREFIX_RE = re.compile(r"^(?:FATAL|ERROR|PANIC):\s+")
_PASSWORD_KV_RE = re.compile(r"(password\s*=\s*)('[^']*'|\S+)", re.IGNORECASE)
# libpq's conninfo parser quotes the offending component verbatim; with an
# unencoded '%' or space in the password that component IS the password
# ('invalid percent-encoded token: "<password>"'). None of these messages is
# worth showing, so they collapse to one fixed text.
_CONNINFO_PARSE_ERROR_RE = re.compile(
    r"invalid percent-encoded token|forbidden value %00 in percent-encoded value"
    r'|unexpected spaces found in|missing "=" after|unterminated quoted string'
    r"|invalid connection option|invalid URI|in URI\b|URI query parameter"
    r'|invalid integer value .* for connection option|invalid \S+ value: "',
    re.IGNORECASE,
)
INVALID_CONNINFO_MESSAGE = (
    "invalid PostgreSQL connection string (check POSTGRES_URL / memory.postgres_url)"
)
_URI_USERINFO_RE = re.compile(
    r"^[a-z][a-z0-9+.-]*://([^:/@\s]*):(.+)@[^@]*$", re.IGNORECASE | re.DOTALL
)
_KV_VALUE_RE = re.compile(r"(?:^|\s)(user|password)\s*=\s*('(?:[^'\\]|\\.)*'|\S+)", re.IGNORECASE)
_MIN_REDACTED_SECRET_CHARS = 6
# Passwords of the connection strings this process connects with (see
# register_conninfo_secret); safe_db_error scrubs them from any text.
_known_secrets: set[str] = set()


def _conninfo_credentials(url: str) -> tuple[str | None, str | None]:
    """(user, password) of a URI or key/value connection string, without libpq.

    libpq cannot be asked: the strings that leak are the ones it can't parse.
    """
    text = url.strip()
    match = _URI_USERINFO_RE.match(text)
    if match:
        return match.group(1), match.group(2)
    values = {}
    for key, raw in _KV_VALUE_RE.findall(text):
        if len(raw) >= 2 and raw[0] == raw[-1] == "'":
            raw = raw[1:-1].replace("\\'", "'").replace("\\\\", "\\")
        values[key.lower()] = raw
    return values.get("user"), values.get("password")


def register_conninfo_secret(url: str | None) -> None:
    """Remember the password of a connection string this process uses.

    A last line of defense behind the pattern-based redaction: safe_db_error
    replaces it (raw and percent-decoded) in any text. Short passwords and one
    equal to the user name are skipped — scrubbing "jarvis" would mangle every
    message that names the user or the database.
    """
    if not url:
        return
    user, password = _conninfo_credentials(url)
    if not password:
        return
    for secret in {password, unquote(password)}:
        if len(secret) >= _MIN_REDACTED_SECRET_CHARS and secret != user:
            _known_secrets.add(secret)


def redact_known_secrets(text: str) -> str:
    """Replace every registered connection-string password in ``text``."""
    for secret in sorted(_known_secrets, key=len, reverse=True):
        if secret in text:
            text = text.replace(secret, "***")
    return text


def is_conninfo_parse_error(text: str) -> bool:
    """True for libpq's "can't parse this connection string" messages."""
    return bool(_CONNINFO_PARSE_ERROR_RE.search(text))


def display_conninfo(url: str | None) -> str:
    """``host:port/dbname`` of a connection string, for logs and status output.

    Never the password: a key/value DSN has no '@' to split on, and one libpq
    can't parse is not echoed at all.
    """
    try:
        from psycopg.conninfo import conninfo_to_dict

        info = conninfo_to_dict(url or "")
    except Exception:
        return "(unparseable connection string)"
    host = info.get("host") or info.get("hostaddr") or "localhost"
    return f"{host}:{info.get('port') or 5432}/{info.get('dbname') or ''}"


class DatabaseUnavailable(RuntimeError):
    """PostgreSQL cannot serve this call right now.

    Raised when the circuit breaker is open or a pool checkout fails. ``str()``
    is sanitized (no conninfo, no password), so it is safe to return to clients.
    """


_checkout_timeout_var: contextvars.ContextVar[float | None] = contextvars.ContextVar(
    "jarvis_db_checkout_timeout", default=None
)

_breaker_lock = threading.Lock()
_db_down_until: float | None = None
_db_down_reason: str | None = None
# Most recent failed connect, cleared by a successful connect or checkout.
# Distinguishes "database unreachable" from "pool busy" on a PoolTimeout.
_last_connect_error: str | None = None
_last_disk_full_at: float | None = None
_last_disk_full_log_at: float | None = None
# Monotonic time of the most recent committed execute_write/execute_batch.
_last_write_ok_at: float | None = None


def _unknown_db_status() -> dict:
    return {"status": "unknown", "error": None, "checked_at": None, "free_bytes": None}


_db_status: dict = _unknown_db_status()


@contextlib.contextmanager
def checkout_timeout(seconds: float | None = HOOK_CONN_TIMEOUT):
    """Bound every pool checkout made in this context to ``seconds``.

    Hook paths wrap their work in this so a sick database costs them
    ``HOOK_CONN_TIMEOUT`` per checkout instead of the pool default. Context
    variables follow the caller into ``contextvars.copy_context().run`` and
    ``asyncio.to_thread``; ``None`` restores the pool default.
    """
    token = _checkout_timeout_var.set(seconds)
    try:
        yield
    finally:
        _checkout_timeout_var.reset(token)


def safe_db_error(exc: BaseException | str) -> str:
    """One-line, credential-free rendering of a database error for clients/logs."""
    msg = redact_known_secrets(" ".join(str(exc).split()))
    if is_conninfo_parse_error(msg):
        return INVALID_CONNINFO_MESSAGE
    if msg.startswith("connection failed: "):
        msg = msg[len("connection failed: "):]
    msg = _CONN_TARGET_RE.sub("", msg)
    msg = _SEVERITY_PREFIX_RE.sub("", msg)
    msg = _DSN_CREDENTIALS_RE.sub(r"\1***@", msg)
    msg = _PASSWORD_KV_RE.sub(r"\1***", msg)
    return msg[:300] or type(exc).__name__


class _SanitizePoolLogFilter(logging.Filter):
    """psycopg_pool logs every failed connect verbatim ("error connecting in
    'pool-1': <libpq error>"), conninfo parse errors quoting the password
    included; its records go through safe_db_error like everything else."""

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            record.msg = safe_db_error(record.getMessage())
            record.args = None
        except Exception:
            pass
        return True


logging.getLogger("psycopg.pool").addFilter(_SanitizePoolLogFilter())


def _is_recovery_message(message: str) -> bool:
    lowered = message.lower()
    return any(marker in lowered for marker in _RECOVERY_MARKERS)


def _is_disk_full(exc: BaseException) -> bool:
    return (
        isinstance(exc, psycopg.errors.DiskFull)
        or getattr(exc, "sqlstate", None) == "53100"
        or "no space left on device" in str(exc).lower()
    )


def is_db_unavailable_error(exc: BaseException) -> bool:
    """True when ``exc`` means "retry later", not "this request is wrong".

    Callers use it to tell a retryable outage (keep the payload) from a
    permanent failure such as a constraint violation (drop it).
    """
    if isinstance(exc, DatabaseUnavailable):
        return True
    if isinstance(exc, (psycopg_pool.PoolTimeout, psycopg_pool.TooManyRequests)):
        return True
    if isinstance(exc, psycopg.OperationalError):
        sqlstate = getattr(exc, "sqlstate", None)
        return sqlstate is None or sqlstate[:2] in _TRANSIENT_SQLSTATE_CLASSES
    return False


def db_available() -> bool:
    """False while the circuit breaker is open. Never blocks, never queries."""
    with _breaker_lock:
        return _db_down_until is None or time.monotonic() >= _db_down_until


def db_unavailable_reason() -> str:
    """Sanitized reason the breaker last opened, for client-facing errors."""
    with _breaker_lock:
        return _db_down_reason or "PostgreSQL is unreachable"


def _trip_breaker(reason: str) -> None:
    global _db_down_until, _db_down_reason
    with _breaker_lock:
        was_open = _db_down_until is not None
        _db_down_until = time.monotonic() + _BREAKER_OPEN_SECONDS
        _db_down_reason = reason
    if not was_open:
        logger.error(
            "PostgreSQL unavailable; failing fast, retrying every %gs: %s",
            _BREAKER_OPEN_SECONDS, reason,
        )


def _mark_db_up() -> None:
    """A checkout or probe reached the database: close the breaker."""
    global _db_down_until, _db_down_reason, _last_connect_error
    with _breaker_lock:
        was_open = _db_down_until is not None
        _db_down_until = None
        _db_down_reason = None
        _last_connect_error = None
    if was_open:
        logger.warning("PostgreSQL reachable again; circuit breaker closed")


def _enter_breaker() -> None:
    """Raise while the breaker is open; let exactly one half-open trial through."""
    global _db_down_until
    with _breaker_lock:
        if _db_down_until is None:
            return
        now = time.monotonic()
        if now < _db_down_until:
            raise DatabaseUnavailable(
                f"{_db_down_reason or 'PostgreSQL is unreachable'} "
                f"(circuit open, next retry in {_db_down_until - now:.0f}s)"
            )
        # Half-open: this caller is the trial. Everyone else keeps failing
        # fast until it succeeds (_mark_db_up) or fails (_trip_breaker).
        _db_down_until = now + _BREAKER_OPEN_SECONDS


def _record_connect_error(exc: BaseException) -> None:
    global _last_connect_error
    note_db_error(exc)
    with _breaker_lock:
        _last_connect_error = safe_db_error(exc)


def _record_connect_ok() -> None:
    global _last_connect_error
    with _breaker_lock:
        _last_connect_error = None


def _checkout_failed(exc: psycopg.OperationalError) -> DatabaseUnavailable:
    """Classify a failed checkout, trip the breaker if the DB is the cause."""
    with _breaker_lock:
        connect_error = _last_connect_error
    # TooManyRequests (max_waiting reached) is contention by definition; a
    # PoolTimeout is too when the database still answers new connects.
    if isinstance(exc, psycopg_pool.TooManyRequests) or (
        isinstance(exc, psycopg_pool.PoolTimeout) and connect_error is None
    ):
        # Every connection is checked out and the database answers new
        # connects: contention, not an outage. Fail this call only, so a slow
        # reindex holding the pool cannot trip everyone else into fail-fast.
        return DatabaseUnavailable(f"connection pool busy ({safe_db_error(exc)})")
    reason = connect_error or safe_db_error(exc)
    _trip_breaker(reason)
    return DatabaseUnavailable(reason)


def note_db_error(exc: BaseException) -> None:
    """Report disk-full loudly and feed it into get_db_status().

    The first ENOSPC of the 2026-09 outage left a single DEBUG line. Any caller
    that swallows database errors should pass them through here first.
    """
    global _last_disk_full_at, _last_disk_full_log_at
    if not _is_disk_full(exc):
        return
    now = time.monotonic()
    with _breaker_lock:
        _last_disk_full_at = now
        should_log = (
            _last_disk_full_log_at is None
            or now - _last_disk_full_log_at >= _DISK_FULL_LOG_INTERVAL_SECONDS
        )
        if should_log:
            _last_disk_full_log_at = now
    reason = safe_db_error(exc)
    _set_db_status(status="disk_full", error=f"PostgreSQL disk full: {reason}")
    if should_log:
        logger.critical(
            "PostgreSQL DISK FULL (SQLSTATE 53100): writes are failing. Free "
            "space on the PostgreSQL volume (docker system df) and restart "
            "Jarvis. %s", reason,
        )


def _disk_full_seen_recently() -> bool:
    with _breaker_lock:
        seen = _last_disk_full_at
    return seen is not None and time.monotonic() - seen < _DISK_FULL_WINDOW_SECONDS


def note_db_write_ok() -> None:
    """A write committed: evidence (for probe_db_status) that disk-full is over.

    execute_write/execute_batch call it; so should code that commits on its
    own pool connection.
    """
    global _last_write_ok_at
    now = time.monotonic()
    with _breaker_lock:
        _last_write_ok_at = now


def _clear_disk_full_if_recovered(connected: bool, space_ok: bool) -> bool:
    """Forget a recent disk-full error once the database writes again.

    Without this /health (and so the statusline and launcher) kept saying
    "disk full" for the whole 5-minute window after space had been freed and
    writes were succeeding. Requires _DISK_FULL_CLEAR_SECONDS without a new
    disk-full error, so a volume that is still full does not flap to "ok"
    between failed writes. Returns True when it cleared the error.
    """
    global _last_disk_full_at
    if not connected:
        return False
    with _breaker_lock:
        seen = _last_disk_full_at
        if seen is None or time.monotonic() - seen < _DISK_FULL_CLEAR_SECONDS:
            return False
        wrote_since = _last_write_ok_at is not None and _last_write_ok_at > seen
        if not (space_ok or wrote_since):
            return False
        _last_disk_full_at = None
    logger.warning(
        "PostgreSQL disk-full condition cleared: no disk-full error for %gs and %s",
        _DISK_FULL_CLEAR_SECONDS,
        "a write has succeeded since" if wrote_since else "the volume has free space again",
    )
    return True


def _set_db_status(**fields: Any) -> None:
    global _db_status
    with _breaker_lock:
        status = dict(_db_status)
        status.update(fields)
        status["checked_at"] = time.time()
        _db_status = status


def get_db_status() -> dict:
    """Cached PostgreSQL status for /health. Never blocks, never queries.

    Shape: ``{"status": "ok"|"recovering"|"unreachable"|"disk_full"|"unknown",
    "error": str|None, "checked_at": float|None, "free_bytes": int|None}``.
    Refreshed by probe_db_status() (and immediately by note_db_error).
    """
    with _breaker_lock:
        return dict(_db_status)


def _is_local_db(url: str) -> bool:
    try:
        from psycopg.conninfo import conninfo_to_dict

        host = str(conninfo_to_dict(url).get("host") or "")
    except Exception:
        return False
    return all(
        not h or h.startswith("/") or h in ("localhost", "127.0.0.1", "::1")
        for h in (part.strip() for part in host.split(","))
    )


def _pgdata_free_bytes(url: str | None) -> int | None:
    """Free bytes on the embedded PGDATA volume, or None when not embedded."""
    if not url or not _is_local_db(url):
        return None
    path = os.environ.get("PGDATA") or _DEFAULT_PGDATA
    try:
        if not os.path.isdir(path):
            return None
        st = os.statvfs(path)
    except (OSError, AttributeError):
        return None
    return int(st.f_bavail * st.f_frsize)


def probe_db_status() -> dict:
    """BLOCKING probe of PostgreSQL health — run it in a worker thread.

    Opens a direct connection (never the pool, whose checkout would wait the
    pool timeout), asks ``pg_is_in_recovery()``, and reads free space on an
    embedded PGDATA volume. Updates the get_db_status() cache and the circuit
    breaker, and returns the new status. Never raises.
    """
    url = None
    connected = False
    status, error = "unreachable", None
    try:
        from .config import get_postgres_config

        url = get_postgres_config()["url"]
        register_conninfo_secret(url)
        with psycopg.connect(
            url, connect_timeout=_PROBE_CONNECT_TIMEOUT, autocommit=True
        ) as conn:
            row = conn.execute("SELECT pg_is_in_recovery()").fetchone()
        connected = True
        if row and row[0]:
            status, error = "recovering", "the database is in recovery (read-only)"
        else:
            status = "ok"
    except Exception as exc:
        note_db_error(exc)
        error = safe_db_error(exc)
        status = "recovering" if _is_recovery_message(error) else "unreachable"

    try:
        free_bytes = _pgdata_free_bytes(url)
    except Exception:
        free_bytes = None
    low_space = free_bytes is not None and free_bytes < _DISK_FULL_MIN_FREE_BYTES
    _clear_disk_full_if_recovered(
        connected and status == "ok",
        space_ok=free_bytes is not None and not low_space,
    )
    if low_space or _disk_full_seen_recently():
        detail = (
            f"PostgreSQL volume has {free_bytes // (1024 * 1024)} MiB free"
            if low_space
            else "PostgreSQL reported disk full (SQLSTATE 53100) in the last "
                 f"{_DISK_FULL_WINDOW_SECONDS / 60:.0f} min"
        )
        status, error = "disk_full", detail if error is None else f"{detail}; {error}"

    if connected:
        _mark_db_up()
    else:
        _trip_breaker(error or "PostgreSQL is unreachable")
    _set_db_status(status=status, error=error, free_bytes=free_bytes)
    return get_db_status()


def mark_db_probe_stalled(seconds: float) -> None:
    """The probe got no answer within ``seconds`` and its thread is stuck.

    connect_timeout bounds only the connect; a server that accepts and then
    never answers blocks the probe's query. The cached status must not keep
    reporting the previous verdict meanwhile, and the breaker opens: without
    it every hook would wait out its deadline on the frozen server, and a
    deadline miss with the breaker closed reads as "slow", not "down".
    """
    reason = f"PostgreSQL did not answer the status probe within {seconds:g}s"
    _trip_breaker(reason)
    _set_db_status(status="unreachable", error=reason)


def reset_breaker() -> None:
    """Forget breaker, disk-full and probe state. Used by tests and reset_pool."""
    global _db_down_until, _db_down_reason, _last_connect_error
    global _last_disk_full_at, _last_disk_full_log_at, _last_write_ok_at, _db_status
    with _breaker_lock:
        _db_down_until = None
        _db_down_reason = None
        _last_connect_error = None
        _last_disk_full_at = None
        _last_disk_full_log_at = None
        _last_write_ok_at = None
        _db_status = _unknown_db_status()


class _TrackedConnection(psycopg.Connection):
    """Pool connection class that records why connects fail.

    psycopg_pool only logs connect errors from its worker threads and reports a
    bare "couldn't get a connection" to the caller; the recorded reason (e.g.
    "the database system is in recovery mode") becomes the breaker's reason.
    """

    @classmethod
    def connect(cls, conninfo: str = "", **kwargs: Any):
        try:
            conn = super().connect(conninfo, **kwargs)
        except Exception as exc:
            _record_connect_error(exc)
            raise
        _record_connect_ok()
        return conn


class _JarvisPool(psycopg_pool.ConnectionPool):
    """ConnectionPool with context-scoped checkout timeouts and the breaker."""

    def getconn(self, timeout: float | None = None):
        if timeout is None:
            timeout = _checkout_timeout_var.get()
        _enter_breaker()
        try:
            conn = super().getconn(timeout=timeout)
        except psycopg_pool.PoolClosed:
            raise
        except psycopg.OperationalError as exc:
            # PoolTimeout and TooManyRequests are OperationalError subclasses
            # (TooManyRequests is NOT a PoolTimeout).
            raise _checkout_failed(exc) from exc
        _mark_db_up()
        return conn

    @contextlib.contextmanager
    def connection(self, timeout: float | None = None):
        conn = None
        try:
            with super().connection(timeout=timeout) as conn:
                yield conn
        except DatabaseUnavailable:
            raise
        except psycopg.Error as exc:
            note_db_error(exc)
            # A connection that died mid-statement (server crash, admin
            # shutdown) is the same outage signal as a failed checkout.
            if isinstance(exc, psycopg.OperationalError) and conn is not None and conn.broken:
                _trip_breaker(safe_db_error(exc))
            raise


def _configure_connection(conn) -> None:
    """Pool configure step (pgvector types); a failure here is a connect failure.

    _TrackedConnection records the connect as successful before this runs, so
    without recording it a missing vector extension would read as "pool busy"
    on every checkout and never trip the breaker.
    """
    from pgvector.psycopg import register_vector

    try:
        register_vector(conn)
    except Exception as exc:
        _record_connect_error(exc)
        raise


def _on_reconnect_failed(pool) -> None:
    """psycopg_pool gave up reconnecting after reconnect_timeout."""
    with _breaker_lock:
        reason = _last_connect_error
    _trip_breaker(reason or f"reconnection failed after {_POOL_RECONNECT_TIMEOUT:.0f}s")


def _pool_timeout_seconds() -> float:
    from .config import get_memory_config

    try:
        value = float(get_memory_config().get("pool_timeout_seconds", _DEFAULT_POOL_TIMEOUT))
    except (TypeError, ValueError):
        return _DEFAULT_POOL_TIMEOUT
    return value if value > 0 else _DEFAULT_POOL_TIMEOUT


def _get_pool():
    """Get or create singleton connection pool with config-based invalidation.

    Cache-key pattern ensures singleton per connection string.
    Pool is created with pgvector type registration on each connection.
    """
    global _pool, _pool_cache_key
    from .config import get_postgres_config, get_embedding_config

    cfg = get_postgres_config()
    emb = get_embedding_config()
    key = (cfg["url"], emb["dimensions"])

    pool = _pool
    if pool is not None and _pool_cache_key == key:
        return pool

    with _pool_lock:
        if _pool is not None and _pool_cache_key == key:
            return _pool

        if _pool is not None:
            try:
                _pool.close()
            except Exception:
                pass
            reset_breaker()  # breaker state described the previous database

        register_conninfo_secret(cfg["url"])
        _pool = _JarvisPool(
            conninfo=cfg["url"],
            min_size=1,
            max_size=5,
            open=True,
            configure=_configure_connection,
            connection_class=_TrackedConnection,
            kwargs=dict(_POOL_CONNECT_KWARGS),
            timeout=_pool_timeout_seconds(),
            max_waiting=_POOL_MAX_WAITING,
            reconnect_timeout=_POOL_RECONNECT_TIMEOUT,
            reconnect_failed=_on_reconnect_failed,
            check=psycopg_pool.ConnectionPool.check_connection,
        )
        _pool_cache_key = key
        logger.info("PostgreSQL connection pool created for %s", display_conninfo(cfg["url"]))
        return _pool


def reset_pool() -> None:
    """Close and reset the connection pool. Used in tests and config changes."""
    global _pool, _pool_cache_key
    with _pool_lock:
        if _pool is not None:
            try:
                _pool.close()
            except Exception:
                pass
        _pool = None
        _pool_cache_key = None
    reset_breaker()


RENAME_SCHEMA_SQL = """\
-- Rename old schema names to new ones (idempotent)
DO $$ BEGIN
    IF EXISTS (SELECT 1 FROM pg_namespace WHERE nspname = 'core')
       AND NOT EXISTS (SELECT 1 FROM pg_namespace WHERE nspname = 'local')
    THEN
        ALTER SCHEMA core RENAME TO local;
    END IF;
END $$;

DO $$ BEGIN
    IF EXISTS (SELECT 1 FROM pg_namespace WHERE nspname = 'vault')
       AND NOT EXISTS (SELECT 1 FROM pg_namespace WHERE nspname = 'obsidian')
    THEN
        ALTER SCHEMA vault RENAME TO obsidian;
    END IF;
END $$;
"""


def ensure_schema() -> None:
    """Create both schemas, tables, and indexes. Run migration if needed.

    Safe to call multiple times (all DDL uses IF NOT EXISTS guards).
    Called at server startup to handle first-run setup and migrations.
    Renames core→local and vault→obsidian on existing databases.
    """
    from .config import get_embedding_config

    emb = get_embedding_config()
    dims = emb["dimensions"]

    pool = _get_pool()
    with pool.connection() as conn:
        # Advisory lock to prevent concurrent migration
        conn.execute("SELECT pg_advisory_lock(42424242)")
        try:
            # pgvector preflight
            row = conn.execute(
                "SELECT extversion FROM pg_extension WHERE extname = 'vector'"
            ).fetchone()
            if row:
                logger.info("pgvector version: %s", row[0])

            # Rename old schemas if they exist (core→local, vault→obsidian)
            conn.execute(RENAME_SCHEMA_SQL)

            # Create both schemas and tables (idempotent)
            conn.execute(LOCAL_SCHEMA_SQL.format(dimensions=dims))
            conn.execute(OBSIDIAN_SCHEMA_SQL.format(dimensions=dims))
            # Per-file LLM summary cache (idempotent; needs the obsidian schema)
            conn.execute(DOCUMENT_CONTEXT_SCHEMA_SQL)
            conn.execute(LOCAL_META_SQL)
            conn.execute(RETRIEVAL_TELEMETRY_SCHEMA_SQL)

            # Phase 1 hybrid retrieval: lexical tsvector columns + channel
            # column (idempotent, additive). Runs after obsidian.documents,
            # local.memories, and local.retrieval_candidates exist.
            conn.execute(LEXICAL_SCHEMA_SQL)

            # Phase 7: sync columns + queue table (idempotent)
            conn.execute(SYNC_SCHEMA_SQL)

            # Phase 8: consolidation support (idempotent)
            conn.execute(CONSOLIDATION_SCHEMA_SQL)

            # Check if migration is needed (from legacy public.jarvis)
            old_table = conn.execute(
                "SELECT to_regclass('public.jarvis')"
            ).fetchone()
            has_old_table = old_table and old_table[0] is not None

            if has_old_table:
                # Check if migration already done
                schema_ver = conn.execute(
                    "SELECT value FROM local.meta WHERE key = 'schema_version'"
                ).fetchone()
                already_migrated = (
                    schema_ver
                    and isinstance(schema_ver[0], dict)
                    and schema_ver[0].get("version", 0) >= 3
                )

                if not already_migrated:
                    logger.info("Migrating data from public.jarvis to local/obsidian schemas...")
                    conn.execute(MIGRATION_SQL.format(dimensions=dims))
                    logger.info("Migration complete")

            conn.commit()
        finally:
            conn.execute("SELECT pg_advisory_unlock(42424242)")

    logger.info("Schema verified (dimensions=%d)", dims)


# ── Query helpers ─────────────────────────────────────────────────────


def _checkout(pool, conn_timeout: float | None):
    """``pool.connection()``, bounded by ``conn_timeout`` when given."""
    if conn_timeout is None:
        return pool.connection()
    return pool.connection(timeout=conn_timeout)


def execute_query(
    sql: str,
    params: tuple | dict | None = None,
    *,
    fetch: str = "all",
    conn_timeout: float | None = None,
) -> Any:
    """Execute a SQL query and return results.

    Args:
        sql: SQL query string with %s or %(name)s placeholders.
        params: Query parameters.
        fetch: "all" returns list of dicts, "one" returns single dict or None,
               "none" returns None (for INSERT/UPDATE/DELETE without RETURNING).
        conn_timeout: Max seconds to wait for a pool connection (default: the
            checkout_timeout() context, else the pool timeout).

    Returns:
        Query results as list of dicts, single dict, or None.
    """
    pool = _get_pool()
    with _checkout(pool, conn_timeout) as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            if fetch == "none":
                conn.commit()
                return None
            columns = [desc.name for desc in cur.description]
            if fetch == "one":
                row = cur.fetchone()
                return dict(zip(columns, row)) if row else None
            rows = cur.fetchall()
            return [dict(zip(columns, row)) for row in rows]


def execute_write(
    sql: str,
    params: tuple | dict | None = None,
    *,
    returning: bool = False,
    conn_timeout: float | None = None,
) -> dict | None:
    """Execute a write query (INSERT/UPDATE/DELETE).

    Args:
        sql: SQL statement.
        params: Query parameters.
        returning: If True, fetch and return the RETURNING row as dict.
        conn_timeout: Max seconds to wait for a pool connection.

    Returns:
        Dict of the RETURNING row if returning=True, else None.
    """
    pool = _get_pool()
    with _checkout(pool, conn_timeout) as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            result = None
            if returning and cur.description:
                row = cur.fetchone()
                if row:
                    columns = [desc.name for desc in cur.description]
                    result = dict(zip(columns, row))
            conn.commit()
            note_db_write_ok()
            return result


def execute_batch(
    sql: str,
    params_list: list[tuple],
    *,
    conn_timeout: float | None = None,
) -> int:
    """Execute a parameterized query for multiple parameter sets.

    Uses executemany for efficient batch operations.

    Returns:
        Number of rows affected.
    """
    if not params_list:
        return 0
    pool = _get_pool()
    with _checkout(pool, conn_timeout) as conn:
        with conn.cursor() as cur:
            cur.executemany(sql, params_list)
            count = cur.rowcount
            conn.commit()
            note_db_write_ok()
            return count


def metadata_to_jsonb(metadata: dict) -> str:
    """Serialize metadata dict to JSONB-compatible JSON string."""
    return json.dumps(metadata, default=str)


def jsonb_to_metadata(jsonb_val) -> dict:
    """Deserialize JSONB value to a Python dict.

    psycopg auto-deserializes JSONB to dict, but this handles
    edge cases (None, string).
    """
    if jsonb_val is None:
        return {}
    if isinstance(jsonb_val, str):
        return json.loads(jsonb_val)
    return dict(jsonb_val)


# ── local.meta CRUD ──────────────────────────────────────────────────


def get_meta(key: str) -> dict | None:
    """Get a value from local.meta by key.

    Returns the JSONB value as a dict, or None if the key doesn't exist.
    """
    result = execute_query(
        "SELECT value FROM local.meta WHERE key = %s",
        (key,),
        fetch="one",
    )
    if result is None:
        return None
    val = result["value"]
    if isinstance(val, str):
        return json.loads(val)
    return dict(val) if val else {}


def set_meta(key: str, value: dict) -> None:
    """Upsert a value into local.meta.

    Uses ON CONFLICT DO UPDATE for atomic upsert.
    """
    execute_write(
        """INSERT INTO local.meta (key, value)
           VALUES (%s, %s)
           ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value""",
        (key, json.dumps(value, default=str)),
    )


def get_all_meta() -> dict[str, dict]:
    """Get all rows from local.meta as a {key: value} dict."""
    rows = execute_query("SELECT key, value FROM local.meta", fetch="all")
    result = {}
    for row in rows:
        val = row["value"]
        if isinstance(val, str):
            val = json.loads(val)
        result[row["key"]] = dict(val) if val else {}
    return result


# ── Model consistency ────────────────────────────────────────────────


class ModelMismatchError(Exception):
    """Raised when the embedding config doesn't match what's stored in local.meta.

    Mixed embedding spaces produce garbage search results silently.
    This error forces the operator to either align the config or
    re-embed (bin/reindex_embeddings.py).
    """
    pass


# Identities recorded by retired deployments. Pre-3.5 Docker images embedded
# in-container (ONNX INT8 weights at a fixed path); those vectors measurably
# differ from the host llama.cpp space (host-inference/PROOF.md: 0.99 cosine
# agreement), so upgrades must re-embed rather than relabel.
_LEGACY_MODEL_IDENTITIES = {"/app/models/embedding"}

_REINDEX_REMEDY = (
    "Re-embed with bin/reindex_embeddings.py (Docker: docker exec --user postgres "
    "--env 'POSTGRES_URL=postgresql:///jarvis?host=/var/run/postgresql' "
    "-w /app/jarvis-core <container> python bin/reindex_embeddings.py). "
    "bin/init_db.py --force-model-record keeps the EXISTING vectors under the new "
    "identity without re-embedding (mixes embedding spaces; degrades ranking), and "
    "JARVIS_SKIP_MODEL_CHECK=1 only silences this check."
)


def _augmentation_remedy(target_mode: str) -> str:
    """The remediation that can actually reach ``target_mode``.

    This text used to name ``bin/reindex_embeddings.py`` for every case. That
    script reads the summary cache and NEVER generates, so for a summary-mode
    target with an empty cache it re-embeds everything mechanically and then
    relabels local.meta as 'summary' — disarming the very warning that sent the
    operator there. Reaching a summary space is inherently TWO steps: fill the
    cache out of band, then re-embed.
    """
    from .chunk_context import MODE_SUMMARY

    if target_mode == MODE_SUMMARY:
        return (
            "Reaching 'summary' mode takes TWO steps, in this order: "
            "(1) generate the per-file summaries out of band — "
            "bin/generate_summaries.py (Docker: docker exec -e ANTHROPIC_API_KEY "
            "-w /app/jarvis-core <container> python bin/generate_summaries.py); "
            "(2) re-embed with jarvis_index_vault(force=true). Step 2 alone "
            "(or bin/reindex_embeddings.py, which only READS the summary cache) "
            "cannot produce a summary-mode space. To stay mechanical instead, "
            "set memory.chunking.contextual_summaries.enabled=false and re-embed."
        )
    return (
        "Re-embed with jarvis_index_vault(force=true) or "
        "bin/reindex_embeddings.py --store all (or restore "
        "memory.chunking.contextual_embeddings / "
        "memory.chunking.contextual_summaries.enabled to match the stored space)."
    )


def check_model_consistency() -> None:
    """Verify embedding config matches what's stored in local.meta.

    First run: records the current config + schema version.
    Subsequent runs: compares model name and dimensions.
    Bypass: set JARVIS_SKIP_MODEL_CHECK=1 env var.
    """
    if os.environ.get("JARVIS_SKIP_MODEL_CHECK") == "1":
        logger.info("Model consistency check skipped (JARVIS_SKIP_MODEL_CHECK=1)")
        return

    from .chunk_context import (
        MODE_PARTIAL_SUMMARY, MODE_SUMMARY, normalize_recorded_augmentation,
    )
    from .config import get_contextual_augmentation_mode, get_embedding_config
    from .embedding import get_embedding_model_identity

    emb = get_embedding_config()
    model_identity = get_embedding_model_identity(emb)
    contextual = get_contextual_augmentation_mode()
    stored = get_meta("embedding_config")

    if stored is None:
        # First run — record current config
        set_meta("embedding_config", {
            "model": model_identity,
            "dimensions": emb["dimensions"],
            "vector_type": "halfvec",
            "contextual_chunks": contextual,
        })
        set_meta("schema_version", {"version": 6})
        logger.info("Recorded embedding config in local.meta: %s (%dd)",
                     emb["model"], emb["dimensions"])
        return

    # Compare model and dimensions
    if stored.get("model") != model_identity:
        if stored.get("model") in _LEGACY_MODEL_IDENTITIES:
            raise ModelMismatchError(
                f"Database vectors were built by the retired in-container model "
                f"'{stored.get('model')}' (pre-3.5 image); config now specifies "
                f"'{model_identity}'. The two embedding spaces are close but not "
                f"identical, so search quality silently degrades without a "
                f"re-embed. {_REINDEX_REMEDY}"
            )
        raise ModelMismatchError(
            f"Embedding model mismatch: database has '{stored.get('model')}' "
            f"but config specifies '{model_identity}'. {_REINDEX_REMEDY}"
        )

    stored_dims = int(stored.get("dimensions", 0))
    if stored_dims != emb["dimensions"]:
        raise ModelMismatchError(
            f"Embedding dimensions mismatch: database has {stored_dims} "
            f"but config specifies {emb['dimensions']}. {_REINDEX_REMEDY}"
        )

    # Chunk-context augmentation is part of the embedding-space identity for
    # vault chunks: changing the MODE (none/mechanical/summary) without
    # re-embedding leaves a mixed space. A two-value flag could not distinguish
    # mechanical from summary augmentation, so the recorded value is the mode;
    # legacy booleans are migrated on read (True → 'mechanical', False → 'none').
    # WARN rather than refuse — the degradation is gradual ranking skew, not
    # garbage, and refusing would take the embedded PostgreSQL down with the
    # server, leaving no way to run the reindex.
    stored_contextual = normalize_recorded_augmentation(
        stored.get("contextual_chunks", False)
    )
    coverage = stored.get("contextual_coverage") or {}
    if stored_contextual != contextual:
        logger.critical(
            "Chunk-context augmentation mismatch: vault vectors were indexed "
            "in '%s' mode but config now says '%s'. Vault ranking is skewed "
            "until re-embedded. %s",
            stored_contextual, contextual,
            _augmentation_remedy(contextual),
        )
    if stored_contextual == MODE_PARTIAL_SUMMARY:
        # Distinct from a mismatch, and NOT fixable by re-embedding alone: some
        # chunked files carry their LLM summary and some do not, so the vault
        # occupies two embedding spaces at once. The cache has to be filled
        # BEFORE the re-embed or the same partial state comes straight back.
        logger.critical(
            "Chunk-context augmentation is PARTIAL: only %s of %s chunked vault "
            "files were embedded with their LLM summary, so vault vectors span "
            "two embedding spaces. %s",
            coverage.get("files_with_summary", "?"),
            coverage.get("chunked_files", "?"),
            _augmentation_remedy(MODE_SUMMARY),
        )
