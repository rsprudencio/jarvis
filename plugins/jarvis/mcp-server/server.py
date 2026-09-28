#!/usr/bin/env python3
"""
Jarvis Core MCP Server

Unified content API and vault access for JARVIS protocol.

Tools - Content Lifecycle (unified API):
- jarvis_store: Write any content (vault file, memory, or content)
- jarvis_retrieve: Read/search any content (semantic, by ID, by name, list)
- jarvis_remove: Delete any content (by ID or name)

Tools - Vault Filesystem:
- jarvis_read_vault_file, jarvis_list_vault_dir, jarvis_file_exists

Tools - Memory Maintenance:
- jarvis_index_vault, jarvis_index_file, jarvis_collection_stats

Tools - Path Configuration:
- jarvis_resolve_path

Tools - Format Support:
- jarvis_get_format_reference

Note: Git operations (commit, status, push, etc.) have moved to jarvis-obsidian.
PKM-specific tools (index_vault, index_file, get_format_reference)
are conditionally visible based on jarvis-obsidian availability.
"""
import asyncio
import contextvars
import functools
import inspect
import json
import logging
import os
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

from tools.file_ops import read_vault_file, list_vault_dir, file_exists_in_vault
from tools.memory import index_vault, index_file
from tools.paths import (
    get_path,
    get_relative_path,
    PathNotConfiguredError,
)
from tools.query import collection_stats
from tools.schema import db_available, db_unavailable_reason, safe_db_error
from tools.store import store
from tools.retrieve import retrieve
from tools.remove import remove

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger("jarvis-core")

import system_prompt

server = Server("core", instructions=system_prompt.instructions)

# Tool definitions
TOOLS = [
    # Unified content API
    Tool(
        name="jarvis_store",
        description="Store content in Jarvis. Provide ONE routing param: id (update existing from retrieve), relative_path (new vault file), or type (new memory/content). Auto-indexes .md files.",
        inputSchema={
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "Content to store (required for write/append modes and type-based writes)",
                },
                "id": {
                    "type": "string",
                    "description": "Document ID from jarvis_retrieve — update existing content. Routes by prefix: vault::* -> file, memory::* -> memory, obs::/pattern::/* -> content.",
                },
                "relative_path": {
                    "type": "string",
                    "description": "Vault-relative path for NEW file writes (e.g., 'journal/2026/02/entry.md'). Use when creating content with no prior ID.",
                },
                "type": {
                    "type": "string",
                    "enum": [
                        "memory",
                        "observation",
                        "pattern",
                        "learning",
                        "decision",
                        "summary",
                        "code",
                        "relationship",
                        "hint",
                        "plan",
                        "worklog",
                    ],
                    "description": "Content type for NEW content. 'memory' = strategic (file-backed). Others = ephemeral (pgvector).",
                },
                "name": {
                    "type": "string",
                    "description": "Name/slug for addressable content. Required for: memory, pattern, plan, decision.",
                },
                "mode": {
                    "type": "string",
                    "enum": ["write", "append", "edit"],
                    "default": "write",
                    "description": "For vault file writes: 'write' (create/overwrite), 'append' (add to existing), 'edit' (find-and-replace)",
                },
                "old_string": {
                    "type": "string",
                    "description": "For edit mode: exact string to find",
                },
                "new_string": {
                    "type": "string",
                    "description": "For edit mode: replacement string",
                },
                "separator": {
                    "type": "string",
                    "default": "\n",
                    "description": "For append mode: prepended before content",
                },
                "replace_all": {
                    "type": "boolean",
                    "default": False,
                    "description": "For edit mode: replace all occurrences",
                },
                "importance": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "description": "Importance score 0.0-1.0",
                },
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Tags for categorization",
                },
                "scope": {
                    "type": "string",
                    "enum": ["global", "project"],
                    "default": "global",
                    "description": "For memory type: scope",
                },
                "project": {
                    "type": "string",
                    "description": "For project-scoped memories",
                },
                "source": {
                    "type": "string",
                    "description": "Source label (default varies by route)",
                },
                "session_id": {"type": "string", "description": "Session identifier"},
                "extra_metadata": {
                    "type": "object",
                    "description": "Additional metadata key-value pairs",
                },
                "overwrite": {
                    "type": "boolean",
                    "default": False,
                    "description": "For memory type: allow overwriting (auto-set to true for id-based updates)",
                },
                "auto_index": {
                    "type": "boolean",
                    "default": True,
                    "description": "Auto-index .md files for semantic search",
                },
                "skip_secret_scan": {
                    "type": "boolean",
                    "default": False,
                    "description": "Skip secret detection",
                },
            },
            "required": ["content"],
        },
    ),
    Tool(
        name="jarvis_retrieve",
        description="Retrieve content from Jarvis. Provide ONE of: query (semantic search), id (read by ID), name (memory by name), or list_type ('content'/'memory' to browse).",
        inputSchema={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Semantic search query (searches all indexed content)",
                },
                "id": {
                    "type": "string",
                    "description": "Document ID to read (routes automatically by ID prefix)",
                },
                "name": {
                    "type": "string",
                    "description": "Strategic memory name to read",
                },
                "list_type": {
                    "type": "string",
                    "enum": ["content", "tier2", "memory"],
                    "description": "List content: 'content' (ephemeral) or 'memory' (strategic)",
                },
                "n_results": {
                    "type": "integer",
                    "default": 5,
                    "description": "Max results for query mode",
                },
                "type_filter": {
                    "type": "string",
                    "description": "Filter by content type when listing",
                },
                "min_importance": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "description": "Min importance score for content listing",
                },
                "source": {
                    "type": "string",
                    "description": "Filter by source for content listing",
                },
                "scope": {
                    "type": "string",
                    "enum": ["global", "project", "all"],
                    "default": "global",
                    "description": "Scope for memory reads/lists",
                },
                "project": {
                    "type": "string",
                    "description": "Project name for scoped memories",
                },
                "tag": {
                    "type": "string",
                    "description": "Filter by tag for memory listing",
                },
                "importance": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "description": "Filter by importance for memory listing",
                },
                "limit": {
                    "type": "integer",
                    "default": 20,
                    "description": "Max results for list mode",
                },
                "filter": {
                    "type": "object",
                    "description": "Metadata filter for query and list modes. "
                        "Known keys: type, importance, tags, directory. "
                        "Any other key filters by metadata JSONB field "
                        "(e.g. git_branch, workstream). "
                        "Note: project and scope have dedicated columns — use "
                        "the top-level project/scope params instead of filter.",
                },
                "include_metadata": {
                    "type": "boolean",
                    "default": True,
                    "description": "Include metadata in ID-based reads",
                },
                "include_content": {
                    "type": "boolean",
                    "default": False,
                    "description": "Include document content in list results (for list_type='memory' and 'content')",
                },
                "sort_by": {
                    "type": "string",
                    "enum": [
                        "importance_desc",
                        "importance_asc",
                        "created_at_desc",
                        "created_at_asc",
                        "none",
                    ],
                    "default": "importance_desc",
                    "description": "Sort order for content list mode (default: importance_desc)",
                },
                "session_id": {
                    "type": "string",
                    "description": "Filter content results by session ID",
                },
                "user": {
                    "type": "string",
                    "description": "Filter results by user (for multi-user deployments)",
                },
                "schemas": {
                    "type": "string",
                    "description": (
                        "Which schemas to search: 'all' (default, searches every registered schema), "
                        "'local' (only local memories), 'obsidian' (only vault), "
                        "or comma-separated names (e.g. 'local,remote_personio'). "
                        "Use 'remote_<name>' to target a specific remote mirror."
                    ),
                },
            },
        },
    ),
    Tool(
        name="jarvis_remove",
        description="Delete content from Jarvis. Provide id (document ID from retrieve results) or name (strategic memory name).",
        inputSchema={
            "type": "object",
            "properties": {
                "id": {
                    "type": "string",
                    "description": "Document ID to delete (from jarvis_retrieve results). Works for vault and content.",
                },
                "name": {
                    "type": "string",
                    "description": "Strategic memory name to delete",
                },
                "scope": {
                    "type": "string",
                    "enum": ["global", "project"],
                    "default": "global",
                },
                "project": {
                    "type": "string",
                    "description": "Project name for scoped memories",
                },
                "confirm": {
                    "type": "boolean",
                    "default": False,
                    "description": "Required for global memory deletion (safety gate)",
                },
            },
        },
    ),
    # Vault file operations (read-only filesystem access)
    Tool(
        name="jarvis_read_vault_file",
        description="Read a file from within the vault directory.",
        inputSchema={
            "type": "object",
            "properties": {
                "relative_path": {
                    "type": "string",
                    "description": "Path relative to vault root",
                }
            },
            "required": ["relative_path"],
        },
    ),
    Tool(
        name="jarvis_list_vault_dir",
        description="List contents of a directory within the vault.",
        inputSchema={
            "type": "object",
            "properties": {
                "relative_path": {
                    "type": "string",
                    "description": "Path relative to vault root (default: vault root)",
                }
            },
        },
    ),
    Tool(
        name="jarvis_file_exists",
        description="Check if a file or directory exists within the vault.",
        inputSchema={
            "type": "object",
            "properties": {
                "relative_path": {
                    "type": "string",
                    "description": "Path relative to vault root",
                }
            },
            "required": ["relative_path"],
        },
    ),
    # Memory operations (pgvector semantic indexing)
    Tool(
        name="jarvis_index_vault",
        description="Bulk index all .md files in the vault into PostgreSQL for semantic search.",
        inputSchema={
            "type": "object",
            "properties": {
                "force": {
                    "type": "boolean",
                    "description": "Re-index all files, even already indexed (default: false)",
                },
                "directory": {
                    "type": "string",
                    "description": "Only index files in this subdirectory (optional)",
                },
                "include_sensitive": {
                    "type": "boolean",
                    "description": "Include documents/ and people/ directories (default: false)",
                },
            },
        },
    ),
    Tool(
        name="jarvis_index_file",
        description="Index a single vault file into PostgreSQL (for incremental indexing after journal creation).",
        inputSchema={
            "type": "object",
            "properties": {
                "relative_path": {
                    "type": "string",
                    "description": "Path relative to vault root",
                }
            },
            "required": ["relative_path"],
        },
    ),
    # Memory stats
    Tool(
        name="jarvis_collection_stats",
        description="Get memory system health: document count, sample entries, and index status.",
        inputSchema={
            "type": "object",
            "properties": {
                "sample_size": {
                    "type": "integer",
                    "description": "Number of sample entries to include (default: 5)",
                    "default": 5,
                },
                "detailed": {
                    "type": "boolean",
                    "description": "Include per-type/namespace breakdowns and storage size (default: false)",
                    "default": False,
                },
            },
        },
    ),
    # Path configuration tools
    Tool(
        name="jarvis_resolve_path",
        description="Resolve a named path to its absolute filesystem location. Use for configurable vault paths.",
        inputSchema={
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Path identifier (e.g., 'journal_jarvis', 'inbox', 'strategic')",
                },
                "substitutions": {
                    "type": "object",
                    "description": 'Template variable replacements (e.g., {"YYYY": "2026", "MM": "02"})',
                },
                "ensure_exists": {
                    "type": "boolean",
                    "description": "Create directory if it does not exist (default: false)",
                },
            },
            "required": ["name"],
        },
    ),
    Tool(
        name="jarvis_get_format_reference",
        description="Get the active file format reference (syntax guide + journal entry template). Returns the format guide content and configured extension. Call this before creating new vault files to know the correct syntax.",
        inputSchema={"type": "object", "properties": {}},
    ),
]


# Conditional PKM tool visibility — these tools are only useful when
# jarvis-obsidian provides the git audit layer for PKM workflows.
_obsidian_cache = {"available": None, "checked_at": 0.0}
_OBSIDIAN_HEALTH_TTL = 30  # seconds

_PKM_TOOLS = {
    "jarvis_index_vault",
    "jarvis_index_file",
    "jarvis_get_format_reference",
}


def _is_obsidian_available() -> bool:
    """Check if jarvis-obsidian server is reachable (with TTL cache)."""
    now = time.time()
    if now - _obsidian_cache["checked_at"] < _OBSIDIAN_HEALTH_TTL:
        if _obsidian_cache["available"] is not None:
            return _obsidian_cache["available"]
    url = os.environ.get("JARVIS_OBSIDIAN_URL", "http://localhost:8744")
    try:
        req = urllib.request.Request(f"{url}/health", method="GET")
        with urllib.request.urlopen(req, timeout=2) as resp:
            data = json.loads(resp.read())
            available = data.get("status") == "ok"
    except Exception:
        available = False
    _obsidian_cache.update(available=available, checked_at=now)
    return available


def _cached_obsidian_availability() -> bool | None:
    """Fresh cached availability, or None when a health check is due."""
    if time.time() - _obsidian_cache["checked_at"] < _OBSIDIAN_HEALTH_TTL:
        return _obsidian_cache["available"]
    return None


@server.list_tools()
async def list_tools() -> list[Tool]:
    available = _cached_obsidian_availability()
    if available is None:
        # urlopen blocks for up to its 2s timeout — never on the event loop,
        # where it would stall /health and every hook with it.
        available = await asyncio.to_thread(_is_obsidian_available)
    if available:
        return TOOLS
    return [t for t in TOOLS if t.name not in _PKM_TOOLS]


def handle_resolve_path(args: dict) -> dict:
    """Handle jarvis_resolve_path."""
    name = args.get("name", "")
    substitutions = args.get("substitutions")
    ensure_exists = args.get("ensure_exists", False)

    try:
        resolved = get_path(
            name, substitutions=substitutions, ensure_exists=ensure_exists
        )
        is_vault_relative = name not in {"project_memories_path"}
        result = {
            "success": True,
            "name": name,
            "resolved": resolved,
            "is_vault_relative": is_vault_relative,
            "exists": os.path.exists(resolved),
        }
        if is_vault_relative:
            result["relative"] = get_relative_path(name)
        return result
    except PathNotConfiguredError as e:
        return {"success": False, "error": str(e)}
    except ValueError as e:
        return {"success": False, "error": str(e)}


def handle_get_format_reference() -> dict:
    """Handle jarvis_get_format_reference.

    Reads the configured file format and returns the corresponding
    syntax reference guide with extension info.
    """
    from tools.config import get_file_format
    import os

    fmt = get_file_format()
    ext = ".org" if fmt == "org" else ".md"
    ref_filename = "org.md" if fmt == "org" else "markdown.md"

    # Look for format reference in plugin defaults
    ref_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "defaults", "formats", ref_filename
    )
    ref_path = os.path.normpath(ref_path)

    if os.path.isfile(ref_path):
        with open(ref_path, "r", encoding="utf-8") as f:
            reference_content = f.read()
    else:
        reference_content = f"Format reference file not found: {ref_path}"

    return {
        "success": True,
        "format": fmt,
        "extension": ext,
        "reference": reference_content,
    }


# Tool calls run concurrently on _tool_executor; a second full reindex racing
# the first would redo the work and interleave its chunk deletes/inserts.
_INDEX_VAULT_LOCK = threading.Lock()


def handle_index_vault(args: dict) -> dict:
    """Handle jarvis_index_vault, one reindex at a time."""
    with _INDEX_VAULT_LOCK:
        return index_vault(
            force=args.get("force", False),
            directory=args.get("directory"),
            include_sensitive=args.get("include_sensitive", False),
        )


# Tool name -> handler mapping (module-level to avoid per-call allocation)
_HANDLERS = {
    # Unified content API
    "jarvis_store": lambda args: store(**args),
    "jarvis_retrieve": lambda args: retrieve(**args),
    "jarvis_remove": lambda args: remove(**args),
    # Vault file operations (read-only)
    "jarvis_read_vault_file": lambda args: read_vault_file(
        args.get("relative_path", "")
    ),
    "jarvis_list_vault_dir": lambda args: list_vault_dir(
        args.get("relative_path", ".")
    ),
    "jarvis_file_exists": lambda args: file_exists_in_vault(
        args.get("relative_path", "")
    ),
    # Memory indexing operations
    "jarvis_index_vault": lambda args: handle_index_vault(args),
    "jarvis_index_file": lambda args: index_file(args.get("relative_path", "")),
    "jarvis_collection_stats": lambda args: collection_stats(
        sample_size=args.get("sample_size", 5),
        detailed=args.get("detailed", False),
    ),
    # Path configuration
    "jarvis_resolve_path": lambda args: handle_resolve_path(args),
    "jarvis_get_format_reference": lambda args: handle_get_format_reference(),
}


# Sync tool handlers do blocking DB / embedding / filesystem work. They run on
# this bounded executor: never on the event loop (a 30s pool wait there
# freezes /health and every hook), and apart from http_app's hook executor so
# a long reindex can't starve hooks or vice versa.
_TOOL_WORKERS = 4
_tool_executor: ThreadPoolExecutor | None = None

# A PostgreSQL that accepts connections but never answers (frozen, not
# crash-looping) blocks a tool's thread with no timeout of its own. The caller
# is answered at the deadline (the thread runs on); None = no deadline, for a
# full reindex that legitimately runs for minutes, one at a time.
TOOL_DEADLINE_SECONDS = 60.0
_TOOL_DEADLINES = {"jarvis_index_vault": None}
# Tools that never touch PostgreSQL: never refused for a DB outage.
_NON_DB_TOOLS = frozenset({
    "jarvis_read_vault_file",
    "jarvis_list_vault_dir",
    "jarvis_file_exists",
    "jarvis_resolve_path",
    "jarvis_get_format_reference",
})
# Start time (monotonic) of each running tool call, by worker thread. With
# every worker busy for this long while the breaker is open, the workers are
# stuck on the database rather than failing fast against it.
_WEDGED_AFTER_SECONDS = 2.0
_tool_calls_lock = threading.Lock()
_tool_calls_running: dict[int, float] = {}


def _get_tool_executor() -> ThreadPoolExecutor:
    """Lazily create the tool executor (again after a shutdown)."""
    global _tool_executor
    if _tool_executor is None:
        _tool_executor = ThreadPoolExecutor(
            max_workers=_TOOL_WORKERS, thread_name_prefix="mcp-tool"
        )
    return _tool_executor


def shutdown_tool_executor() -> None:
    """Drop queued tool calls without waiting on running ones."""
    global _tool_executor
    executor, _tool_executor = _tool_executor, None
    if executor is not None:
        executor.shutdown(wait=False, cancel_futures=True)


def _tracked(call):
    """Wrap a tool call so its worker thread is listed while it runs."""
    def run():
        ident = threading.get_ident()
        with _tool_calls_lock:
            _tool_calls_running[ident] = time.monotonic()
        try:
            return call()
        finally:
            with _tool_calls_lock:
                _tool_calls_running.pop(ident, None)

    return run


def _tool_workers_wedged() -> bool:
    """Breaker open and every worker stuck for _WEDGED_AFTER_SECONDS: a new
    call would only queue behind them. Not refused otherwise — with a free
    worker the tool's own checkouts fail fast, and some tools (memory reads)
    fall back to files while the database is down."""
    with _tool_calls_lock:
        starts = list(_tool_calls_running.values())
    if len(starts) < _TOOL_WORKERS or db_available():
        return False
    return time.monotonic() - max(starts) >= _WEDGED_AFTER_SECONDS


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    # Keys only: values are memory contents and may carry secrets (the
    # secret scan runs later, inside content_write, and never sees this log).
    logger.info("Tool: %s, arg_keys: %s", name, sorted((arguments or {}).keys()))

    try:
        handler = _HANDLERS.get(name)
        if handler and name not in _NON_DB_TOOLS and _tool_workers_wedged():
            result = {
                "success": False,
                "retryable": True,
                "error_kind": "db_unavailable",
                "error": f"database unavailable: {safe_db_error(db_unavailable_reason())}"
                         f" ({_TOOL_WORKERS} tool calls still waiting on it)",
            }
        elif handler:
            call = functools.partial(
                contextvars.copy_context().run, handler, arguments or {}
            )
            deadline = _TOOL_DEADLINES.get(name, TOOL_DEADLINE_SECONDS)
            loop = asyncio.get_running_loop()
            future = loop.run_in_executor(_get_tool_executor(), _tracked(call))
            try:
                result = await asyncio.wait_for(future, deadline)
            except TimeoutError:
                if not future.cancelled():
                    raise  # the tool's own TimeoutError
                logger.warning("Tool %s exceeded its %gs deadline", name, deadline)
                result = {
                    "success": False,
                    "retryable": True,
                    "error_kind": "deadline",
                    "error": f"{name} did not finish within {deadline:g}s and may "
                             "still complete; check before retrying",
                }
            if inspect.isawaitable(result):
                result = await result
        else:
            result = {"success": False, "error": f"Unknown tool: {name}"}

        return [TextContent(type="text", text=json.dumps(result, indent=2))]

    except Exception as e:
        logger.error("Error: %s", safe_db_error(e), exc_info=True)
        return [
            TextContent(
                type="text",
                text=json.dumps({"success": False, "error": safe_db_error(e)}),
            )
        ]


def get_background_tasks():
    """Registry of background async tasks to run alongside the MCP server.

    Both stdio (main) and HTTP (http_app lifespan) transports consume this,
    ensuring no drift between transport modes.
    """
    from tools.patterns import pattern_detection_loop
    from tools.todoist_sync import todoist_sync_loop
    from tools.sync_worker import sync_worker_loop
    from tools.sync_pull import get_pull_sync_tasks
    from tools.retrieval_telemetry import retrieval_telemetry_loop

    return [
        pattern_detection_loop(),
        todoist_sync_loop(),
        sync_worker_loop(),
        retrieval_telemetry_loop(),
        *get_pull_sync_tasks(),
        db_status_probe_loop(),
    ]


DB_STATUS_PROBE_INTERVAL_SECONDS = 10
DB_STATUS_PROBE_TIMEOUT_SECONDS = 5


async def db_status_probe_loop():
    """Keep the cached DB status (/health "postgres") and the breaker fresh.

    probe_db_status() connects directly with a short connect timeout, so it
    stays bounded even when the pool is wedged; it still blocks, so it runs
    in a worker thread and the loop itself only sleeps. connect_timeout does
    not bound the query after the connect, so the wait is capped too, and a
    stuck probe is not started again until it returns (no thread pile-up).
    """
    from tools.schema import mark_db_probe_stalled, probe_db_status

    probe = None
    while True:
        if probe is None or probe.done():
            probe = asyncio.ensure_future(asyncio.to_thread(probe_db_status))
        try:
            await asyncio.wait_for(asyncio.shield(probe), DB_STATUS_PROBE_TIMEOUT_SECONDS)
        except TimeoutError:
            logger.warning(
                "DB status probe still blocked after %ss", DB_STATUS_PROBE_TIMEOUT_SECONDS
            )
            mark_db_probe_stalled(DB_STATUS_PROBE_TIMEOUT_SECONDS)
        except Exception as e:  # never raises by contract; keep the loop alive regardless
            logger.warning("DB status probe failed: %s", e)
        await asyncio.sleep(DB_STATUS_PROBE_INTERVAL_SECONDS)


async def main():
    logger.info("Starting Jarvis Core MCP Server")

    # Initialize pgvector schema (idempotent — safe to call every startup)
    from tools.schema import ensure_schema, check_model_consistency, ModelMismatchError
    try:
        ensure_schema()
        check_model_consistency()
    except ModelMismatchError as mme:
        # Mixed embedding spaces silently corrupt every search — refuse to
        # serve, same as the HTTP transport.
        logger.critical("FATAL: %s", mme)
        raise SystemExit(1) from mme
    except Exception as e:
        logger.warning("Schema initialization deferred (database may not be ready): %s", e)

    # D6: Rebuild schema registry, auto-discovering existing remote_* schemas
    try:
        from tools.schema_registry import rebuild_registry
        rebuild_registry()
    except Exception as e:
        logger.warning("Schema registry rebuild deferred: %s", e)

    # Stdio clients have the same first-query latency constraint as HTTP hooks.
    # A down model host must not prevent serving: retrieval fails open at
    # runtime, so log loudly and continue degraded.
    from tools.embedding import warm_embedding_service
    try:
        warm_embedding_service()
    except Exception as e:
        logger.critical(
            "Embedding warmup failed: %s — serving in DEGRADED mode; "
            "embedding-dependent operations fail open until the model host "
            "is reachable again", e,
        )

    async with stdio_server() as (read_stream, write_stream):
        server_task = server.run(
            read_stream, write_stream, server.create_initialization_options()
        )
        await asyncio.gather(
            server_task, *get_background_tasks(), return_exceptions=True
        )


def main_sync():
    """Synchronous entry point for uvx/pip scripts."""
    asyncio.run(main())


if __name__ == "__main__":
    main_sync()
