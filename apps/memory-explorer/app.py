"""Jarvis Admin — web UI for Jarvis memory stores and sync status.

Standalone FastAPI app. Run with:
    cd apps/memory-explorer
    uv run uvicorn app:app --reload --host 127.0.0.1 --port 8750

Safety:
    - Localhost-only bind (enforced by uvicorn --host 127.0.0.1)
    - Read-only sessions (SET TRANSACTION READ ONLY per query)
    - Write path: DELETE /api/memories/{id} performs soft-delete on local.memories
      (gated by jarvis_common.auth, same as admin CRUD endpoints)
    - sql.Identifier() for all dynamic schema/table names
    - Source whitelist (only sources from config are accepted)
    - DSN redaction in logs via redact_dsn()
    - XSS prevention: all user content via textContent in SPA
    - CSP header on SPA route
    - statement_timeout = 10s to prevent runaway queries
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import re
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import psycopg
import psycopg_pool
from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, JSONResponse
from pgvector.psycopg import register_vector
from psycopg import sql
from pydantic import BaseModel

# ── Import tools from MCP server via sys.path ──────────────────────────────
# In Docker: tools live at /app/jarvis-core. In local dev: relative path.
_CONTAINER_PATH = Path("/app/jarvis-core")
_DEV_PATH = Path(__file__).resolve().parent.parent.parent / "plugins" / "jarvis" / "mcp-server"
_MCP = _CONTAINER_PATH if _CONTAINER_PATH.exists() else _DEV_PATH
if str(_MCP) not in sys.path:
    sys.path.insert(0, str(_MCP))

from tools.config import get_embedding_config, get_postgres_config, get_sync_config  # noqa: E402
from tools.embedding import get_embedding_service  # noqa: E402
from tools.remote_connection import get_remote_pool  # noqa: E402
from tools.schema import (  # noqa: E402
    INVALID_CONNINFO_MESSAGE,
    is_conninfo_parse_error,
    redact_known_secrets,
    register_conninfo_secret,
)
from jarvis_common.sync_validation import redact_dsn  # noqa: E402
from jarvis_common.routing import evaluate_routing, parse_routing_rule  # noqa: E402
from app_admin import admin_router, require_auth  # noqa: E402
from tools.sync_queue import enqueue_sync  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("memory-explorer")

_SNIPPET_LEN = 300
_MAX_PAGE_SIZE = 100

# ── Module state ────────────────────────────────────────────────────────────
_local_pool: Optional[psycopg_pool.ConnectionPool] = None
_sources: dict[str, dict] = {}
# Monotonic time of the last discovery attempt. None until lifespan's first
# discovery — lazy re-discovery only maintains a cache lifespan populated.
_sources_at: Optional[float] = None
_sources_lock = asyncio.Lock()
_discovery_runs = 0
_refresh_task: Optional[asyncio.Task] = None
_db_probe_task: Optional[asyncio.Task] = None
_last_disk_full_at: Optional[float] = None
# Cached result of _probe_db_status(), served by /health without blocking.
_db_status: dict = {"status": "unknown", "error": None, "checked_at": None, "free_bytes": None}
# Monotonic time the status probe last failed to reach the local database
# (connect refused/rejected, or no answer); None once a probe gets through.
_db_unreachable_at: Optional[float] = None

# An interactive UI must fail in seconds: psycopg_pool's default 30s wait
# turned a Postgres outage into a 30s spinner plus a raw PoolTimeout.
_POOL_TIMEOUT = 5.0
_PROBE_TIMEOUT = 3.0          # per source-discovery probe checkout
_SOURCES_TTL = 60.0           # re-discover sources lazily after this
_DB_PROBE_INTERVAL = 10.0
_DB_PROBE_WAIT = 5.0          # cap on one background probe (connect + query)
_DB_CONNECT_TIMEOUT = 2       # direct connect used to learn the real cause
_LOW_DISK_BYTES = 64 * 1024 * 1024
_DISK_FULL_WINDOW = 300.0
# A recent disk-full error stops forcing "disk_full" after this long without
# another one, once a probe connects and sees enough free space (core: same).
_DISK_FULL_CLEAR = 60.0
# While the probe's "unreachable" verdict is younger than this (one probe
# cycle), local-database requests get their 503 at once instead of waiting
# out _POOL_TIMEOUT each.
_FAST_FAIL_WINDOW = _DB_PROBE_INTERVAL + _DB_PROBE_WAIT
_RETRY_AFTER = {"Retry-After": "10"}
# A Postgres that accepts connections but never answers blocks a request's
# thread with no timeout of its own (statement_timeout is enforced by the very
# server that froze). The request is answered 504 at this deadline.
_DB_CALL_DEADLINE = 15.0
_DB_ANALYSIS_DEADLINE = 60.0  # simulate / export scan all of telemetry
# Forced re-discovery (unknown source) is bounded: it probes every source and
# every configured remote, and any GET — even a cross-site one — can ask.
_FORCED_DISCOVERY_INTERVAL = 10.0
_forced_discovery_at: Optional[float] = None


# ── Pool helpers ─────────────────────────────────────────────────────────────

def _make_local_pool() -> psycopg_pool.ConnectionPool:
    url = get_postgres_config()["url"]
    register_conninfo_secret(url)
    logger.info("Connecting to local PG: %s", redact_dsn(url))
    return psycopg_pool.ConnectionPool(
        conninfo=url,
        min_size=1,
        max_size=5,
        open=True,
        timeout=_POOL_TIMEOUT,
        # Replace connections a PG crash/restart left broken instead of
        # handing them to a request ("server closed the connection").
        check=psycopg_pool.ConnectionPool.check_connection,
        kwargs={"connect_timeout": 5},
        configure=lambda conn: register_vector(conn),
    )


# ── Database status ─────────────────────────────────────────────────────────

# 57P03 (cannot_connect_now) texts; libpq reports no SQLSTATE on a failed
# connect, so the server's message is the only signal.
_RECOVERING_RE = re.compile(
    r"the database system is (in recovery mode|starting up|shutting down"
    r"|not yet accepting connections|not accepting connections)"
)
_CONN_TARGET_RE = re.compile(
    r'connection to server (?:at "[^"]*"(?: \([^)]*\))?, port \d+|on socket "[^"]*") failed:\s*'
)


def _db_err_reason(e: Exception) -> str:
    """The server's reason for a DB error, without host/port/socket or DSN."""
    msg = str(e).strip().split("\n", 1)[0]
    msg = _CONN_TARGET_RE.sub("", msg)
    msg = re.sub(r"^(?:connection failed|connection is bad):\s*", "", msg)
    msg = re.sub(r"^(?:FATAL|ERROR|PANIC):\s+", "", msg)
    return _safe_err(Exception(" ".join(msg.split()) or type(e).__name__))


def _note_db_error(e: Exception) -> None:
    """Remember a disk-full error so /health can report disk_full for a while."""
    global _last_disk_full_at
    if (
        isinstance(e, psycopg.errors.DiskFull)
        or getattr(e, "sqlstate", None) == "53100"
        or "No space left on device" in str(e)
    ):
        _last_disk_full_at = time.monotonic()


def _pgdata_free_bytes() -> Optional[int]:
    """Free bytes on an embedded Postgres data dir, or None if not local."""
    try:
        st = os.statvfs(os.environ.get("PGDATA") or "/var/lib/postgresql/data")
    except OSError:
        return None
    return st.f_bavail * st.f_frsize


def _probe_db_status() -> dict:
    """BLOCKING (run in a thread): classify Postgres health and cache it.

    Uses a direct connection, not the pool: the pool only ever reports that
    it waited, while the server's FATAL ('the database system is in recovery
    mode') is what explains the outage. Never raises.
    """
    global _db_status, _db_unreachable_at, _last_disk_full_at
    status, error = "ok", None
    try:
        url = get_postgres_config()["url"]
        register_conninfo_secret(url)
        with psycopg.connect(url, connect_timeout=_DB_CONNECT_TIMEOUT) as conn:
            row = conn.execute("SELECT pg_is_in_recovery()").fetchone()
            if row and row[0]:
                status, error = "recovering", "the database system is in recovery"
        _db_unreachable_at = None
    except Exception as e:
        _db_unreachable_at = time.monotonic()
        _note_db_error(e)
        error = _db_err_reason(e)
        recovering = getattr(e, "sqlstate", None) == "57P03" or _RECOVERING_RE.search(str(e))
        status = "recovering" if recovering else "unreachable"
    free = _pgdata_free_bytes()
    low_space = free is not None and free < _LOW_DISK_BYTES
    now = time.monotonic()
    if (
        _last_disk_full_at is not None
        and status == "ok"
        and free is not None
        and not low_space
        and now - _last_disk_full_at >= _DISK_FULL_CLEAR
    ):
        _last_disk_full_at = None  # writing again: space is back, errors stopped
    recent_disk_full = (
        _last_disk_full_at is not None and now - _last_disk_full_at < _DISK_FULL_WINDOW
    )
    if recent_disk_full or low_space:
        status = "disk_full"
        detail = (
            f"PGDATA free space low ({free // (1024 * 1024)} MiB)" if low_space
            else "disk full (No space left on device)"
        )
        # Keep the connect error too: "disk full; Connection refused" says
        # both what happened and why the server is down.
        error = detail if error is None else f"{detail}; {error}"
    _db_status = {"status": status, "error": error, "checked_at": time.time(), "free_bytes": free}
    return _db_status


class _LocalDbDown(psycopg.OperationalError):
    """The status probe could not reach the local database moments ago."""


def _check_local_db() -> None:
    """Fail at once while the local database is known to be unreachable.

    Every request used to wait out _POOL_TIMEOUT (5s) before its 503 during
    an outage. Only a verdict younger than _FAST_FAIL_WINDOW counts, so a
    recovered database is used again within one probe cycle.
    """
    down_at = _db_unreachable_at
    if down_at is not None and time.monotonic() - down_at < _FAST_FAIL_WINDOW:
        raise _LocalDbDown(_db_status.get("error") or "PostgreSQL is unreachable")


def _local_connection():
    """A local-pool checkout, or an immediate _LocalDbDown during an outage."""
    _check_local_db()
    return _local_pool.connection()


def _db_unavailable_detail(src: Optional[dict] = None) -> str:
    """BLOCKING: explain a pool timeout with the server's actual reason."""
    if src is not None and src.get("type") != "local":
        return "Database unavailable: remote connection timed out"
    st = _probe_db_status()
    if st["status"] == "ok":
        return "Database unavailable: connection pool exhausted"
    return "Database unavailable: " + (st["error"] or st["status"])


def _core_db_unavailable(e: Exception) -> bool:
    """Core's circuit breaker error (semantic mode runs on core's pool).

    Resolved at call time from the loaded tools.schema module, so a reloaded
    module's class still matches.
    """
    schema = sys.modules.get("tools.schema")
    cls = getattr(schema, "DatabaseUnavailable", None)
    return isinstance(cls, type) and isinstance(e, cls)


async def _db_unavailable_error(e: Exception, src: Optional[dict] = None) -> HTTPException:
    """Map DB unavailability to a fast 503 that carries the real cause."""
    if isinstance(e, _LocalDbDown):
        # The probe's verdict (already sanitized); no second probe per request.
        detail = "Database unavailable: " + str(e)
    elif isinstance(e, (psycopg_pool.PoolTimeout, psycopg_pool.TooManyRequests)) or _core_db_unavailable(e):
        detail = await asyncio.to_thread(_db_unavailable_detail, src)
    else:
        _note_db_error(e)
        detail = "Database unavailable: " + _db_err_reason(e)
    logger.warning("%s", detail)
    return HTTPException(503, detail, headers=_RETRY_AFTER)


class _DbCallTimedOut(psycopg.errors.QueryCanceled):
    """A DB call outlived its deadline; handled (504) like statement_timeout."""


async def _run_db(fn, *args, deadline: Optional[float] = None, local_db: bool = False):
    """Run a blocking DB call in a thread, bounded by ``deadline``
    (default _DB_CALL_DEADLINE).

    Raises _DbCallTimedOut at the deadline. The thread is abandoned, not
    killed: it ends when the database answers or the connection breaks.
    ``local_db`` marks a call on the local database made through core's pool
    (the Retrieval tab): it fails fast like _local_connection() does.
    """
    if local_db:
        _check_local_db()
    if deadline is None:
        deadline = _DB_CALL_DEADLINE
    loop = asyncio.get_running_loop()
    try:
        return await asyncio.wait_for(loop.run_in_executor(None, fn, *args), deadline)
    except TimeoutError:
        raise _DbCallTimedOut(f"database did not answer within {deadline:g}s") from None


async def _db_status_loop() -> None:
    """Refresh the cached DB status for /health off the event loop.

    connect_timeout does not bound the query after the connect, so the wait
    is capped and a stuck probe is not started again until it returns.
    """
    global _db_status, _db_unreachable_at
    probe: Optional[asyncio.Future] = None
    while True:
        if probe is None or probe.done():
            probe = asyncio.ensure_future(asyncio.to_thread(_probe_db_status))
        try:
            await asyncio.wait_for(asyncio.shield(probe), _DB_PROBE_WAIT)
        except TimeoutError:
            _db_unreachable_at = time.monotonic()
            _db_status = {
                **_db_status,
                "status": "unreachable",
                "error": f"no answer to the status probe within {_DB_PROBE_WAIT:g}s",
                "checked_at": time.time(),
            }
        except Exception as e:  # executor gone during shutdown
            logger.debug("DB status probe skipped: %s", e)
        await asyncio.sleep(_DB_PROBE_INTERVAL)


def _probe_source(pool_fn, schema: str, table: str) -> bool:
    """Return True if the table exists and is queryable."""
    try:
        with pool_fn() as conn:
            conn.execute("SET TRANSACTION READ ONLY")
            conn.execute(
                sql.SQL("SELECT 1 FROM {}.{} LIMIT 1").format(
                    sql.Identifier(schema), sql.Identifier(table)
                )
            )
        return True
    except Exception as e:
        logger.warning("Probe failed %s.%s: %s", schema, table, e)
        return False


def _probe_semantic(pool_fn, schema: str, table: str, dims: int) -> bool:
    """Return True if a pgvector cosine query succeeds with current dimensions."""
    dummy = "[" + ",".join(["0.0"] * dims) + "]"
    try:
        with pool_fn() as conn:
            conn.execute("SET TRANSACTION READ ONLY")
            conn.execute(
                sql.SQL(
                    "SELECT 1 FROM {}.{} ORDER BY embedding <=> %s::vector LIMIT 1"
                ).format(sql.Identifier(schema), sql.Identifier(table)),
                [dummy],
            )
        return True
    except Exception as e:
        logger.warning("Semantic probe failed %s.%s: %s", schema, table, e)
        return False


def _discover_sources(local_pool: psycopg_pool.ConnectionPool) -> dict[str, dict]:
    sources: dict[str, dict] = {}
    dims = get_embedding_config()["dimensions"]

    def lconn():
        # Short wait: with PG down every failed probe waits this long, at
        # startup (before uvicorn binds) and on each re-discovery.
        return local_pool.connection(timeout=_PROBE_TIMEOUT)

    # local.memories
    if _probe_source(lconn, "local", "memories"):
        sem = _probe_semantic(lconn, "local", "memories", dims)
        sources["local"] = {
            "id": "local",
            "label": "Local Memories",
            "type": "local",
            "schema": "local",
            "table": "memories",
            "has_retrieval_count": True,
            "has_status": True,
            "capabilities": ["text", "metadata"] + (["semantic"] if sem else []),
            "metadata_filters": ["category", "scope", "project", "source", "status"],
        }

    # obsidian.documents
    if _probe_source(lconn, "obsidian", "documents"):
        sem = _probe_semantic(lconn, "obsidian", "documents", dims)
        sources["obsidian"] = {
            "id": "obsidian",
            "label": "Obsidian Vault",
            "type": "local",
            "schema": "obsidian",
            "table": "documents",
            "has_retrieval_count": False,
            "has_status": False,
            "capabilities": ["text", "metadata"] + (["semantic"] if sem else []),
            "metadata_filters": ["vault_type", "directory"],
        }

    # Remotes from sync config (skip disabled and template entries)
    sync_cfg = get_sync_config()
    for rname, rcfg in sync_cfg.get("remotes", {}).items():
        if not rcfg.get("enabled", True):
            continue
        if rname.startswith("_"):
            continue
        src_id = f"remote:{rname}"
        schema = rcfg.get("schema", rname)
        try:
            pool = get_remote_pool(rname)

            def rconn(p=pool):
                return p.connection()

            ok = _probe_source(rconn, schema, "memory_refs")
            sem = ok and _probe_semantic(rconn, schema, "content", dims)
            caps = (["text", "metadata"] + (["semantic"] if sem else [])) if ok else []
        except Exception as e:
            logger.warning("Remote %r unavailable: %s", rname, e)
            ok, caps = False, []

        display_name = rcfg.get("schema", rname)
        sources[src_id] = {
            "id": src_id,
            "label": f"Remote: {display_name}",
            "type": "remote",
            "remote_name": rname,
            "schema": schema,
            "available": ok,
            "capabilities": caps,
            "metadata_filters": ["category", "scope", "project", "source", "status"],
        }

    return sources


async def _rediscover_sources() -> None:
    """Re-run source discovery in a thread, one run at a time.

    An explorer started while Postgres was down cached an empty source list
    forever (400 "Unknown source: 'local'" until restart).
    """
    global _sources, _sources_at, _discovery_runs
    runs = _discovery_runs
    async with _sources_lock:
        if _discovery_runs != runs:
            return  # a concurrent request just re-discovered; reuse its result
        try:
            found = await asyncio.to_thread(_discover_sources, _local_pool)
        except Exception as e:
            logger.warning("Source re-discovery failed: %s", _db_err_reason(e))
            return
        finally:
            _discovery_runs += 1
            _sources_at = time.monotonic()
        # During an outage every local probe fails: keep the last good list,
        # whose handlers then answer 503 with the cause.
        if "local" in found or "local" not in _sources:
            if found.keys() != _sources.keys():
                logger.info("Sources re-discovered: %s", list(found.keys()))
            _sources = found


def _discoverable(source: str) -> bool:
    """Whether a re-discovery could produce ``source`` at all."""
    if source in ("local", "obsidian"):
        return True
    if not source.startswith("remote:"):
        return False
    name = source[len("remote:"):]
    rcfg = get_sync_config().get("remotes", {}).get(name)
    return isinstance(rcfg, dict) and bool(rcfg.get("enabled", True)) and not name.startswith("_")


async def _ensure_sources(source: Optional[str] = None) -> None:
    """Lazily refresh the source cache before it is used.

    'local' missing, or a requested source that discovery could produce but
    hasn't, waits for a re-discovery — at most one forced run per
    _FORCED_DISCOVERY_INTERVAL (requests arriving while one runs share it).
    Any other unknown id is answered straight away. A merely stale cache
    refreshes in the background so the healthy path never waits on it.
    """
    global _refresh_task, _forced_discovery_at
    if _sources_at is None or _local_pool is None:
        return
    wanted = source is not None and source not in _sources and _discoverable(source)
    if "local" not in _sources or wanted:
        now = time.monotonic()
        if _sources_lock.locked():
            await _rediscover_sources()  # join the run in flight
        elif (
            _forced_discovery_at is None
            or now - _forced_discovery_at >= _FORCED_DISCOVERY_INTERVAL
        ):
            _forced_discovery_at = now
            await _rediscover_sources()
    elif time.monotonic() - _sources_at > _SOURCES_TTL and not _sources_lock.locked():
        if _refresh_task is None or _refresh_task.done():
            _refresh_task = asyncio.create_task(_rediscover_sources())


def _fresh_db_status() -> Optional[dict]:
    """The background probe's verdict while it is still current, else None."""
    checked_at = _db_status.get("checked_at")
    if checked_at is not None and time.time() - checked_at < _DB_PROBE_INTERVAL + _DB_PROBE_WAIT:
        return _db_status
    return None


async def _unknown_source_error(status: int, detail: str) -> HTTPException:
    """Unknown source after re-discovery: 503 if the local DB is why."""
    if _sources_at is not None and "local" not in _sources:
        # The cached verdict when current: no direct connect per request.
        st = _fresh_db_status() or await asyncio.to_thread(_probe_db_status)
        if st["status"] != "ok":
            reason = "Database unavailable: " + (st["error"] or st["status"])
            return HTTPException(503, reason, headers=_RETRY_AFTER)
    return HTTPException(status, detail)


# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _local_pool, _sources, _sources_at, _sources_lock, _db_probe_task
    _local_pool = _make_local_pool()
    _sources = _discover_sources(_local_pool)
    _sources_at = time.monotonic()
    _sources_lock = asyncio.Lock()
    _db_probe_task = asyncio.create_task(_db_status_loop())
    logger.info("Ready. Sources: %s", list(_sources.keys()))
    yield
    for task in (_db_probe_task, _refresh_task):
        if task is not None and not task.done():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
    _local_pool.close()


app = FastAPI(title="Jarvis Admin", lifespan=lifespan)
app.include_router(admin_router)


# Routes without their own mapping (the Retrieval tab runs on core's pool)
# would otherwise turn a DB outage into a bare "Internal Server Error".
@app.exception_handler(psycopg.OperationalError)
async def _operational_error_handler(request, exc: psycopg.OperationalError):
    if isinstance(exc, psycopg.errors.QueryCanceled):
        return JSONResponse({"detail": "query timed out"}, status_code=504)
    err = await _db_unavailable_error(exc)
    return JSONResponse({"detail": err.detail}, status_code=503, headers=err.headers)


@app.exception_handler(RuntimeError)
async def _core_unavailable_handler(request, exc: RuntimeError):
    """Core's DatabaseUnavailable (a RuntimeError) → 503; anything else as before."""
    if not _core_db_unavailable(exc):
        raise exc
    err = await _db_unavailable_error(exc)
    return JSONResponse({"detail": err.detail}, status_code=503, headers=err.headers)

_CSP = (
    "default-src 'self'; "
    "script-src 'self' 'unsafe-inline'; "
    "style-src 'self' 'unsafe-inline'; "
    "connect-src 'self';"
)


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "server": "memory-explorer",
        "sources": sorted(_sources),
        # Cached by _db_status_loop — /health never touches the DB itself.
        "postgres": dict(_db_status),
    }


@app.get("/", response_class=HTMLResponse)
async def spa():
    return HTMLResponse(content=_HTML, headers={"Content-Security-Policy": _CSP})


@app.get("/api/sources")
async def list_sources():
    await _ensure_sources()
    out = []
    for s in _sources.values():
        item = {k: v for k, v in s.items() if k != "remote_name"}
        item["sort_options"] = _sort_options_for(s)
        item["deletable"] = (s["type"] == "local" and s["schema"] == "local")
        out.append(item)
    return out


@app.get("/api/stats")
async def get_stats(source: Optional[str] = Query(default=None)):
    targets = (
        [_sources[source]] if (source and source in _sources)
        else list(_sources.values())
    )
    # Concurrently: during an outage each source waits out the pool timeout.
    counts = await asyncio.gather(
        *(_run_db(_count_sync, src) for src in targets),
        return_exceptions=True,
    )
    out = {}
    for src, n in zip(targets, counts):
        if isinstance(n, Exception):
            out[src["id"]] = {"count": None, "error": _safe_err(n)}
        else:
            out[src["id"]] = {"count": n}
    return out


def _count_sync(src: dict) -> int:
    if src["type"] == "local":
        with _local_connection() as conn:
            conn.execute("SET TRANSACTION READ ONLY")
            conn.execute("SET statement_timeout = '10000'")
            row = conn.execute(
                sql.SQL("SELECT COUNT(*) FROM {}.{}").format(
                    sql.Identifier(src["schema"]), sql.Identifier(src["table"])
                )
            ).fetchone()
            return row[0] if row else 0
    else:
        pool = get_remote_pool(src["remote_name"])
        with pool.connection() as conn:
            conn.execute("SET TRANSACTION READ ONLY")
            conn.execute("SET statement_timeout = '10000'")
            row = conn.execute(
                sql.SQL("SELECT COUNT(*) FROM {}.memory_refs").format(
                    sql.Identifier(src["schema"])
                )
            ).fetchone()
            return row[0] if row else 0


class SearchRequest(BaseModel):
    source: str
    mode: str
    query: str = ""
    filters: dict = {}
    page: int = 0
    page_size: int = 20
    sort_by: str = "date_desc"


class RetrievalFeedbackRequest(BaseModel):
    verdict: str
    expected_missing_ids: list[str] = []
    note: str = ""


class CandidateFeedbackRequest(BaseModel):
    verdict: str
    note: str = ""


class SimulationRequest(BaseModel):
    policy: str = "cosine-only"
    cosine_threshold: float = 0.85
    bge_logit_threshold: float = -2.5
    # Augmentation era filter ('' = all eras pooled). Mechanical-era and
    # summary-era BGE logits come from different rerank input spaces, so a
    # threshold swept across both is correct for neither.
    contextual_augmentation: str = ""


# Sort definitions per context: local (single table) vs remote (r/c JOIN aliases)
_SORT_OPTIONS = {
    "date_desc":       {"local": "created_at DESC",                       "remote": "r.created_at DESC",                       "label": "Newest first"},
    "date_asc":        {"local": "created_at ASC",                        "remote": "r.created_at ASC",                        "label": "Oldest first"},
    "updated_desc":    {"local": "updated_at DESC NULLS LAST",            "remote": "r.updated_at DESC NULLS LAST",            "label": "Recently updated"},
    "importance_desc": {"local": "importance_score DESC NULLS LAST",      "remote": "r.importance_score DESC NULLS LAST",      "label": "Most important"},
    "importance_asc":  {"local": "importance_score ASC NULLS LAST",       "remote": "r.importance_score ASC NULLS LAST",       "label": "Least important"},
    "size_desc":       {"local": "LENGTH(document) DESC",                 "remote": "LENGTH(c.content) DESC",                  "label": "Largest first"},
    "size_asc":        {"local": "LENGTH(document) ASC",                  "remote": "LENGTH(c.content) ASC",                   "label": "Smallest first"},
    "retrieval_desc":  {"local": "retrieval_count DESC NULLS LAST",       "remote": "r.retrieval_count DESC NULLS LAST",       "label": "Most retrieved",
                        "sources": {"local", "remote"}},
}


def _sort_options_for(src: dict) -> list[dict]:
    """Return sort options available for a given source."""
    src_type = "obsidian" if src.get("schema") == "obsidian" else src["type"]
    result = []
    for key, opt in _SORT_OPTIONS.items():
        allowed = opt.get("sources")
        if allowed and src_type not in allowed:
            continue
        result.append({"value": key, "label": opt["label"]})
    return result


def _order_sql(sort_by: str, remote: bool = False, has_rc: bool = True) -> sql.SQL:
    """Build ORDER BY SQL fragment. Falls back to date_desc."""
    opt = _SORT_OPTIONS.get(sort_by, _SORT_OPTIONS["date_desc"])
    # Fall back if sort references retrieval_count on a table that lacks it
    if not has_rc and "retrieval" in sort_by:
        opt = _SORT_OPTIONS["date_desc"]
    return sql.SQL(opt["remote"] if remote else opt["local"])


def _local_cols(has_rc: bool) -> str:
    """Build SELECT column list for local tables."""
    base = "id, document, created_at, updated_at, importance_score"
    if has_rc:
        base += ", retrieval_count"
    base += ", LENGTH(document) AS doc_size"
    return base


def _local_sem_cols(has_rc: bool) -> str:
    """Build SELECT column list for local semantic queries (with score)."""
    base = "id, document, created_at, 1 - (embedding <=> %s::vector) AS score, updated_at, importance_score"
    if has_rc:
        base += ", retrieval_count"
    base += ", LENGTH(document) AS doc_size"
    return base


@app.post("/api/search")
async def search(req: SearchRequest):
    await _ensure_sources(req.source)
    if req.source not in _sources:
        raise await _unknown_source_error(400, f"Unknown source: {req.source!r}")
    src = _sources[req.source]
    if req.mode not in ("text", "semantic", "metadata"):
        raise HTTPException(400, f"Invalid mode: {req.mode!r}")
    if req.mode not in src.get("capabilities", []):
        raise HTTPException(400, f"Source {req.source!r} does not support {req.mode!r}")
    if not (0 < req.page_size <= _MAX_PAGE_SIZE):
        raise HTTPException(400, "page_size must be between 1 and 100")

    try:
        # Production host inference uses the shared core primitive. Local
        # development with the legacy in-process backend keeps the direct SQL
        # path, avoiding a second implicit model lifecycle in this UI process.
        shared_semantic = (
            req.mode == "semantic"
            and src["type"] == "local"
            and get_embedding_config().get("backend") == "host"
        )
        if shared_semantic:
            result = await _run_db(_semantic_search_sync, src, req)
        else:
            result = await _run_db(_search_sync, src, req)
            if req.mode == "semantic":
                result["trace_id"] = await _run_db(
                    _trace_remote_explorer_search, src, req, result
                )
    # QueryCanceled (statement_timeout) is an OperationalError subclass:
    # it must be matched first or it would read as "Database unavailable".
    except psycopg.errors.QueryCanceled:
        raise HTTPException(504, "query timed out")
    except psycopg.OperationalError as e:  # incl. PoolTimeout
        raise await _db_unavailable_error(e, src)
    except Exception as e:
        if _core_db_unavailable(e):
            raise await _db_unavailable_error(e, src)
        logger.exception("Search error")
        raise HTTPException(500, _safe_err(e))
    return {
        "results": result["rows"],
        "total": result["total"],
        "page": req.page,
        "source": req.source,
        "allowed_filters": src.get("metadata_filters", []),
        "sort_options": _sort_options_for(src),
        "trace_id": result.get("trace_id"),
    }


def _semantic_search_sync(src: dict, req: SearchRequest) -> dict:
    """Route Explorer semantic mode through Jarvis core's shared recall."""
    from tools.query import semantic_candidate_search

    _check_local_db()  # core's pool, same local database

    schema_name = src["schema"]
    result = semantic_candidate_search(
        req.query,
        limit=req.page_size,
        offset=req.page * req.page_size,
        schemas=[schema_name],
        purpose="memory_explorer",
    )
    rows = []
    for item in result["results"]:
        document = item.pop("document", "")
        rows.append({
            "id": item["id"],
            "snippet": document[:_SNIPPET_LEN] + ("…" if len(document) > _SNIPPET_LEN else ""),
            "created_at": item["created_at"].isoformat() if hasattr(item["created_at"], "isoformat") else item["created_at"],
            "updated_at": item["updated_at"].isoformat() if hasattr(item["updated_at"], "isoformat") else item["updated_at"],
            "importance_score": item["importance_score"],
            "retrieval_count": item["retrieval_count"],
            "doc_size": len(document),
            "score": round(float(item["score"]), 4),
        })
    return {"rows": rows, "total": _count_sync(src), "trace_id": result.get("trace_id")}


def _trace_remote_explorer_search(src: dict, req: SearchRequest, result: dict) -> Optional[str]:
    """Trace remote Explorer search without persisting returned content."""
    try:
        # Direct Explorer searches use their own pool; establish the shared
        # repository pool only when the real app pool is healthy (test mocks
        # must never trigger an external connection).
        if isinstance(_local_pool, psycopg_pool.ConnectionPool):
            from tools.schema import _get_pool

            _get_pool()
        from tools.retrieval_telemetry import CandidateTrace, record_event

        candidates = [
            CandidateTrace(
                schema_name=str(src.get("schema", "remote")),
                doc_id=str(row.get("id", "")), vector_rank=index,
                final_rank=index, similarity=row.get("score"),
                pre_score=row.get("score"), terminal_reason="selected", returned=True,
            )
            for index, row in enumerate(result.get("rows", []), 1)
        ]
        return record_event(
            purpose="memory_explorer", query=req.query, candidates=candidates,
            funnel={"ann_unique": len(candidates), "returned": len(candidates)},
            latency={}, outcome="results" if candidates else "empty",
            pipeline="explorer-remote-semantic",
            config_snapshot={"source": src.get("id"), "page": req.page, "page_size": req.page_size},
        )
    except Exception:
        return None


@app.get("/api/retrieval/summary")
async def retrieval_summary(days: int = Query(default=7, ge=1, le=90)):
    from tools.retrieval_telemetry import get_summary

    return await _run_db(lambda: get_summary(days), local_db=True)


@app.get("/api/retrieval/events")
async def retrieval_events(
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    purpose: str = Query(default=""),
    outcome: str = Query(default=""),
):
    from tools.retrieval_telemetry import list_events

    return await _run_db(
        lambda: list_events(limit=limit, offset=offset, purpose=purpose, outcome=outcome),
        local_db=True,
    )


@app.get("/api/retrieval/events/{event_id}")
async def retrieval_event(event_id: str):
    from tools.retrieval_telemetry import get_event

    event = await _run_db(lambda: get_event(event_id), local_db=True)
    if not event:
        raise HTTPException(404, "Retrieval event not found")
    return event


@app.get("/api/retrieval/events/{event_id}/documents")
async def retrieval_event_documents(
    event_id: str,
    preview_chars: int = Query(default=240, ge=0, le=4000),
    candidate_key: str = Query(default=""),
):
    """Resolve candidate bodies on demand — telemetry stores only locators."""
    from tools.retrieval_telemetry import get_event_documents

    return await _run_db(
        lambda: get_event_documents(
            event_id, preview_chars=preview_chars, candidate_key=candidate_key or None
        ),
        local_db=True,
    )


@app.put("/api/retrieval/events/{event_id}/feedback")
async def retrieval_event_feedback(
    event_id: str,
    req: RetrievalFeedbackRequest,
    user: str = Depends(require_auth),
):
    from tools.retrieval_telemetry import put_event_feedback

    try:
        await _run_db(
            lambda: put_event_feedback(event_id, req.model_dump(), user), local_db=True
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    return {"ok": True}


@app.put("/api/retrieval/events/{event_id}/candidates/{candidate_key}/feedback")
async def retrieval_candidate_feedback(
    event_id: str,
    candidate_key: str,
    req: CandidateFeedbackRequest,
    user: str = Depends(require_auth),
):
    from tools.retrieval_telemetry import put_candidate_feedback

    try:
        await _run_db(
            lambda: put_candidate_feedback(event_id, candidate_key, req.model_dump(), user),
            local_db=True,
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    return {"ok": True}


@app.post("/api/retrieval/simulate")
async def retrieval_simulate(req: SimulationRequest):
    from tools.retrieval_telemetry import simulate_policy

    try:
        return await _run_db(
            lambda: simulate_policy(req.model_dump()),
            deadline=_DB_ANALYSIS_DEADLINE, local_db=True,
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc))


@app.get("/api/retrieval/export")
async def retrieval_export():
    from tools.retrieval_telemetry import export_labeled_events

    rows = await _run_db(
        export_labeled_events, deadline=_DB_ANALYSIS_DEADLINE, local_db=True
    )
    return {"schema_version": 1, "exported_at": __import__("datetime").datetime.now(__import__("datetime").timezone.utc), "events": rows}


def _count_where(conn, schema: str, table: str, where_sql, params: list) -> int:
    """Count rows matching a WHERE clause."""
    row = conn.execute(
        sql.SQL("SELECT COUNT(*) FROM {}.{} WHERE {}").format(
            sql.Identifier(schema), sql.Identifier(table), where_sql
        ),
        params,
    ).fetchone()
    return row[0] if row else 0


def _search_sync(src: dict, req: SearchRequest) -> dict:
    offset = req.page * req.page_size
    allowed_filters = set(src.get("metadata_filters", []))
    total = 0

    if src["type"] == "local":
        schema, table = src["schema"], src["table"]
        sch = sql.Identifier(schema)
        tbl = sql.Identifier(table)
        has_rc = src.get("has_retrieval_count", False)
        has_status = src.get("has_status", False)

        # Exclude soft-deleted items unless user explicitly filters by status
        # (only applies to sources that actually have a status column)
        _has_status_filter = "status" in req.filters
        _base_where = (
            sql.SQL("TRUE")
            if (_has_status_filter or not has_status)
            else sql.SQL("status != 'deleted'")
        )

        with _local_connection() as conn:
            conn.execute("SET TRANSACTION READ ONLY")
            conn.execute("SET statement_timeout = '10000'")

            order = _order_sql(req.sort_by, has_rc=has_rc)

            if req.mode == "text":
                q = req.query.strip()
                if not q:
                    where = _base_where
                    wparams: list = []
                else:
                    where = sql.SQL("{} AND document ILIKE %s").format(_base_where)
                    wparams = [f"%{q}%"]
                total = _count_where(conn, schema, table, where, wparams)
                rows = conn.execute(
                    sql.SQL(
                        "SELECT " + _local_cols(has_rc) + " FROM {}.{}"
                        " WHERE {} ORDER BY {} LIMIT %s OFFSET %s"
                    ).format(sch, tbl, where, order),
                    wparams + [req.page_size, offset],
                ).fetchall()

            elif req.mode == "semantic":
                vec = _vec_str(req.query)
                total = _count_where(conn, schema, table, _base_where, [])
                use_sim = req.sort_by in ("similarity", "date_desc")
                sem_order = sql.SQL("embedding <=> %s::vector") if use_sim else order
                sem_params = [vec] if use_sim else []
                rows = conn.execute(
                    sql.SQL(
                        "SELECT " + _local_sem_cols(has_rc)
                        + " FROM {}.{}"
                        " WHERE {} ORDER BY {} LIMIT %s OFFSET %s"
                    ).format(sch, tbl, _base_where, sem_order),
                    [vec] + sem_params + [req.page_size, offset],
                ).fetchall()

            else:  # metadata
                conds, params = _build_conds(req.filters, allowed_filters)
                if has_status and not _has_status_filter:
                    conds.append(sql.SQL("status != 'deleted'"))
                where_sql = sql.SQL(" AND ").join(conds)
                total = _count_where(conn, schema, table, where_sql, list(params))
                rows = conn.execute(
                    sql.SQL(
                        "SELECT " + _local_cols(has_rc) + " FROM {}.{}"
                        " WHERE {} ORDER BY {} LIMIT %s OFFSET %s"
                    ).format(sch, tbl, where_sql, order),
                    params + [req.page_size, offset],
                ).fetchall()

    else:  # remote — memory_refs JOIN content
        pool = get_remote_pool(src["remote_name"])
        s = sql.Identifier(src["schema"])
        with pool.connection() as conn:
            conn.execute("SET TRANSACTION READ ONLY")
            conn.execute("SET statement_timeout = '10000'")

            order = _order_sql(req.sort_by, remote=True)

            if req.mode == "text":
                q = req.query.strip()
                if not q:
                    where = sql.SQL("TRUE")
                    wparams = []
                else:
                    where = sql.SQL("c.content ILIKE %s")
                    wparams = [f"%{q}%"]
                row = conn.execute(
                    sql.SQL(
                        "SELECT COUNT(*) FROM {0}.memory_refs r"
                        " JOIN {0}.content c ON r.content_hash = c.hash"
                        " WHERE {1}"
                    ).format(s, where),
                    wparams,
                ).fetchone()
                total = row[0] if row else 0
                rows = conn.execute(
                    sql.SQL(
                        "SELECT r.id, c.content, r.created_at,"
                        " r.updated_at, r.importance_score, r.retrieval_count, LENGTH(c.content) AS doc_size,"
                        " r.metadata->>'project_path' AS project_path"
                        " FROM {0}.memory_refs r JOIN {0}.content c ON r.content_hash = c.hash"
                        " WHERE {1} ORDER BY {2} LIMIT %s OFFSET %s"
                    ).format(s, where, order),
                    wparams + [req.page_size, offset],
                ).fetchall()

            elif req.mode == "semantic":
                row = conn.execute(
                    sql.SQL("SELECT COUNT(*) FROM {0}.memory_refs").format(s)
                ).fetchone()
                total = row[0] if row else 0
                vec = _vec_str(req.query)
                use_sim = req.sort_by in ("similarity", "date_desc")
                sem_order = sql.SQL("c.embedding <=> %s::vector") if use_sim else order
                sem_params = [vec] if use_sim else []
                rows = conn.execute(
                    sql.SQL(
                        "SELECT r.id, c.content, r.created_at,"
                        " 1 - (c.embedding <=> %s::vector) AS score,"
                        " r.updated_at, r.importance_score, r.retrieval_count, LENGTH(c.content) AS doc_size,"
                        " r.metadata->>'project_path' AS project_path"
                        " FROM {0}.memory_refs r JOIN {0}.content c ON r.content_hash = c.hash"
                        " ORDER BY {1} LIMIT %s OFFSET %s"
                    ).format(s, sem_order),
                    [vec] + sem_params + [req.page_size, offset],
                ).fetchall()

            else:  # metadata
                conds, params = _build_remote_conds(req.filters, allowed_filters)
                where_sql = sql.SQL(" AND ").join(conds)
                row = conn.execute(
                    sql.SQL(
                        "SELECT COUNT(*) FROM {0}.memory_refs r"
                        " JOIN {0}.content c ON r.content_hash = c.hash"
                        " WHERE {1}"
                    ).format(s, where_sql),
                    list(params),
                ).fetchone()
                total = row[0] if row else 0
                rows = conn.execute(
                    sql.SQL(
                        "SELECT r.id, c.content, r.created_at,"
                        " r.updated_at, r.importance_score, r.retrieval_count, LENGTH(c.content) AS doc_size,"
                        " r.metadata->>'project_path' AS project_path"
                        " FROM {0}.memory_refs r JOIN {0}.content c ON r.content_hash = c.hash"
                        " WHERE {1} ORDER BY {2} LIMIT %s OFFSET %s"
                    ).format(s, where_sql, order),
                    params + [req.page_size, offset],
                ).fetchall()

    has_rc = src.get("has_retrieval_count", src["type"] == "remote")
    is_remote = src["type"] == "remote"
    return {"rows": _rows_to_dicts(rows, req.mode, has_rc=has_rc, extract_user=is_remote), "total": total}


@app.get("/api/content")
async def get_content(source: str = Query(...), id: str = Query(...)):
    await _ensure_sources(source)
    if source not in _sources:
        raise await _unknown_source_error(404, "Source not found")
    src = _sources[source]
    try:
        item = await _run_db(_fetch_content_sync, src, id)
    except psycopg.errors.QueryCanceled:
        raise HTTPException(504, "query timed out")
    except psycopg.OperationalError as e:
        raise await _db_unavailable_error(e, src)
    except Exception as e:
        logger.exception("Content fetch error")
        raise HTTPException(500, _safe_err(e))
    if item is None:
        raise HTTPException(404, "Item not found")
    return item


def _fetch_content_sync(src: dict, item_id: str) -> Optional[dict]:
    if src["type"] == "local":
        schema, table = src["schema"], src["table"]
        with _local_connection() as conn:
            conn.execute("SET TRANSACTION READ ONLY")
            conn.execute("SET statement_timeout = '10000'")
            cur = conn.execute(
                sql.SQL(
                    "SELECT * FROM {}.{} WHERE id = %s"
                ).format(sql.Identifier(schema), sql.Identifier(table)),
                [item_id],
            )
            row = cur.fetchone()
            if not row:
                return None
            cols = [desc.name for desc in cur.description]
            return _row_to_detail(cols, row)
    else:
        pool = get_remote_pool(src["remote_name"])
        s = sql.Identifier(src["schema"])
        with pool.connection() as conn:
            conn.execute("SET TRANSACTION READ ONLY")
            conn.execute("SET statement_timeout = '10000'")
            cur = conn.execute(
                sql.SQL(
                    "SELECT r.*, c.content"
                    " FROM {0}.memory_refs r JOIN {0}.content c ON r.content_hash = c.hash"
                    " WHERE r.id = %s"
                ).format(s),
                [item_id],
            )
            row = cur.fetchone()
            if not row:
                return None
            cols = [desc.name for desc in cur.description]
            return _row_to_detail(cols, row)


def _delete_sync(item_id: str) -> dict:
    """Soft-delete a memory from local.memories and enqueue remote sync."""
    with _local_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """UPDATE local.memories
                   SET status = 'deleted', deleted_at = now(), updated_at = now()
                   WHERE id = %s AND status != 'deleted'
                   RETURNING id, synced_to""",
                [item_id],
            )
            row = cur.fetchone()
            if row is None:
                conn.commit()
                return {"deleted": False, "id": item_id}

            # Propagate deletion to synced remotes
            synced_remotes = row[1]
            if synced_remotes:
                try:
                    enqueue_sync(cur, item_id, synced_remotes)
                except Exception as e:
                    logger.debug("Delete sync propagation skipped: %s", e)

            conn.commit()
            return {"deleted": True, "id": item_id}


@app.delete("/api/memories/{item_id}")
async def delete_memory(
    item_id: str,
    source: str = Query(...),
    _user: str = Depends(require_auth),
):
    """Soft-delete a memory from local.memories."""
    await _ensure_sources(source)
    if source not in _sources:
        raise await _unknown_source_error(404, "Source not found")
    src = _sources[source]
    if not (src["type"] == "local" and src["schema"] == "local"):
        raise HTTPException(403, "Deletion only supported for local memories")

    try:
        result = await _run_db(_delete_sync, item_id)
    except psycopg.errors.QueryCanceled:
        raise HTTPException(504, "query timed out")
    except psycopg.OperationalError as e:
        raise await _db_unavailable_error(e, src)
    except Exception as e:
        logger.exception("Delete error")
        raise HTTPException(500, _safe_err(e))

    if not result["deleted"]:
        raise HTTPException(404, "Memory not found or already deleted")
    return {"ok": True, "id": item_id}


def _row_to_detail(cols: list[str], row) -> dict:
    """Convert a full row into a detail response with content + all metadata."""
    data = dict(zip(cols, row))
    # Extract primary fields
    content = data.pop("document", None) or data.pop("content", None) or ""
    item_id = data.pop("id", "")
    # Remove binary/large fields not useful in UI
    data.pop("embedding", None)
    # Format timestamps
    created = data.pop("created_at", None)
    updated = data.pop("updated_at", None)
    # Merge JSONB metadata into top-level for flat display
    jsonb_meta = data.pop("metadata", None) or {}
    if isinstance(jsonb_meta, dict):
        for k, v in jsonb_meta.items():
            if k not in data:
                data[k] = v
    # Build clean metadata dict (skip None values)
    metadata = {}
    for k, v in sorted(data.items()):
        if v is not None and v != "" and v != []:
            if hasattr(v, "isoformat"):
                metadata[k] = v.isoformat()
            elif isinstance(v, (list, dict)):
                metadata[k] = v
            else:
                metadata[k] = v
    return {
        "id": item_id,
        "content": content,
        "metadata": metadata,
        "created_at": created.isoformat() if hasattr(created, "isoformat") else created,
        "updated_at": updated.isoformat() if hasattr(updated, "isoformat") else updated,
    }


# ── Helpers ───────────────────────────────────────────────────────────────────

def _vec_str(query: str) -> str:
    """Encode a query string to a pgvector-compatible string literal."""
    vec = get_embedding_service().encode(query)
    return "[" + ",".join(str(x) for x in vec) + "]"


def _build_conds(filters: dict, allowed: set) -> tuple[list, list]:
    """Build WHERE conditions for local tables (bare column names)."""
    conds = [sql.SQL("TRUE")]
    params: list = []
    for col, val in filters.items():
        if col in allowed and isinstance(val, str):
            conds.append(sql.SQL("{} = %s").format(sql.Identifier(col)))
            params.append(val)
    return conds, params


def _build_remote_conds(filters: dict, allowed: set) -> tuple[list, list]:
    """Build WHERE conditions for remote tables (prefixed with r.)."""
    conds = [sql.SQL("TRUE")]
    params: list = []
    for col, val in filters.items():
        if col in allowed and isinstance(val, str):
            conds.append(sql.SQL("r.{} = %s").format(sql.Identifier(col)))
            params.append(val)
    return conds, params


def _extract_user_from_path(project_path: str) -> Optional[str]:
    """Extract OS username from a project path like /Users/alice/dev/foo."""
    if not project_path:
        return None
    parts = project_path.split("/")
    for prefix in ("Users", "home"):
        if prefix in parts:
            idx = parts.index(prefix)
            if idx + 1 < len(parts):
                return parts[idx + 1]
    return None


def _rows_to_dicts(rows: list, mode: str, has_rc: bool = True, extract_user: bool = False) -> list[dict]:
    out = []
    for row in rows:
        doc = row[1] or ""
        entry: dict = {
            "id": row[0],
            "snippet": doc[:_SNIPPET_LEN] + ("…" if len(doc) > _SNIPPET_LEN else ""),
            "created_at": row[2].isoformat() if row[2] else None,
        }
        if mode == "semantic":
            if len(row) > 3:
                entry["score"] = round(float(row[3]), 4)
            idx = 4
        else:
            idx = 3
        # Common columns: updated_at, importance_score
        if len(row) > idx:
            entry["updated_at"] = row[idx].isoformat() if row[idx] else None
        if len(row) > idx + 1:
            entry["importance_score"] = float(row[idx + 1]) if row[idx + 1] is not None else None
        # retrieval_count only present on some tables
        if has_rc:
            if len(row) > idx + 2:
                entry["retrieval_count"] = float(row[idx + 2]) if row[idx + 2] is not None else None
            size_idx = idx + 3
        else:
            size_idx = idx + 2
        if len(row) > size_idx:
            entry["doc_size"] = int(row[size_idx]) if row[size_idx] is not None else 0
        # Remote: extract user from project_path (column after doc_size)
        if extract_user and len(row) > size_idx + 1:
            user = _extract_user_from_path(row[size_idx + 1])
            if user:
                entry["user"] = user
        out.append(entry)
    return out


def _safe_err(e: Exception) -> str:
    """Return a sanitized error message safe for API responses."""
    msg = redact_known_secrets(str(e))
    # libpq quotes an unparseable conninfo component verbatim: the password.
    if is_conninfo_parse_error(msg):
        return INVALID_CONNINFO_MESSAGE
    # Strip any potential DSN/credential leakage
    if "@" in msg and "//" in msg:
        return "Database error (credentials redacted)"
    return msg[:200]


# ── Admin API ────────────────────────────────────────────────────────────────

@app.get("/api/admin")
async def admin_data():
    """Return sync system status for the admin dashboard."""
    try:
        data = await _run_db(_admin_sync)
    except psycopg.errors.QueryCanceled:
        raise HTTPException(504, "query timed out")
    except psycopg.OperationalError as e:
        raise await _db_unavailable_error(e)
    except Exception as e:
        logger.exception("Admin data error")
        raise HTTPException(500, _safe_err(e))
    return data


def _admin_sync() -> dict:
    sync_cfg = get_sync_config()
    enabled = sync_cfg.get("enabled", False)

    # Remotes (no URLs or credentials; skip templates)
    remotes = []
    for rname, rcfg in sync_cfg.get("remotes", {}).items():
        if rname.startswith("_"):
            continue
        remote_ok = False
        is_enabled = rcfg.get("enabled", True)
        if is_enabled:
            try:
                pool = get_remote_pool(rname)
                with pool.connection() as conn:
                    conn.execute("SELECT 1")
                remote_ok = True
            except Exception:
                pass
        remotes.append({
            "name": rname,
            "schema": rcfg.get("schema", rname),
            "auth_method": rcfg.get("auth_method", "password"),
            "enabled": is_enabled,
            "connected": remote_ok,
        })

    # Queue stats + DLQ + recent activity from sync_queue
    queue_stats = {}
    dlq_entries = []
    recent_syncs = []
    last_push = None
    last_fail = None

    try:
        with _local_connection() as conn:
            conn.execute("SET TRANSACTION READ ONLY")
            conn.execute("SET statement_timeout = '10000'")

            # Per-destination status counts
            rows = conn.execute(
                "SELECT destination, status, count(*) FROM local.sync_queue "
                "GROUP BY destination, status ORDER BY destination, status"
            ).fetchall()
            for dest, status, cnt in rows:
                if dest not in queue_stats:
                    queue_stats[dest] = {}
                queue_stats[dest][status] = cnt

            # DLQ entries (limit 50)
            dlq_rows = conn.execute(
                "SELECT id, memory_id, destination, attempts, error, "
                "created_at, last_attempt "
                "FROM local.sync_queue WHERE status = 'dlq' "
                "ORDER BY last_attempt DESC NULLS LAST LIMIT 50"
            ).fetchall()
            for row in dlq_rows:
                dlq_entries.append({
                    "id": row[0],
                    "memory_id": row[1],
                    "destination": row[2],
                    "attempts": row[3],
                    "error": str(row[4])[:200] if row[4] else None,
                    "created_at": row[5].isoformat() if row[5] else None,
                    "last_attempt": row[6].isoformat() if row[6] else None,
                })

            # Last successful sync (most recent 'done' entry)
            done_row = conn.execute(
                "SELECT last_attempt, destination FROM local.sync_queue "
                "WHERE status = 'done' ORDER BY last_attempt DESC NULLS LAST LIMIT 1"
            ).fetchone()
            if done_row and done_row[0]:
                last_push = {
                    "at": done_row[0].isoformat(),
                    "destination": done_row[1],
                }

            # Last failed sync
            fail_row = conn.execute(
                "SELECT last_attempt, destination, error FROM local.sync_queue "
                "WHERE status IN ('dlq', 'pending') AND error IS NOT NULL "
                "ORDER BY last_attempt DESC NULLS LAST LIMIT 1"
            ).fetchone()
            if fail_row and fail_row[0]:
                last_fail = {
                    "at": fail_row[0].isoformat(),
                    "destination": fail_row[1],
                    "error": str(fail_row[2])[:200] if fail_row[2] else None,
                }

            # Recent sync activity (last 20) — enriched with memory preview
            recent_rows = conn.execute(
                "SELECT q.id, q.memory_id, q.destination, q.status, q.last_attempt, "
                "  m.category, m.scope, m.project, m.importance_score, "
                "  LEFT(m.document, 120) AS preview "
                "FROM local.sync_queue q "
                "LEFT JOIN local.memories m ON m.id = q.memory_id "
                "ORDER BY q.last_attempt DESC NULLS LAST LIMIT 20"
            ).fetchall()
            for row in recent_rows:
                recent_syncs.append({
                    "id": row[0],
                    "memory_id": row[1],
                    "destination": row[2],
                    "status": row[3],
                    "at": row[4].isoformat() if row[4] else None,
                    "category": row[5],
                    "scope": row[6],
                    "project": row[7],
                    "importance": row[8],
                    "preview": row[9],
                })
    except Exception as e:
        logger.warning("Failed to read sync_queue: %s", e)

    # Routing rules
    rules = []
    for raw in sync_cfg.get("rules", []):
        rules.append({
            "name": raw.get("name", "unnamed"),
            "action": raw.get("action", "route-to"),
            "destinations": raw.get("destinations", []),
            "match": raw.get("match", {}),
        })

    # Enrich recent activity with matched routing rules
    parsed_rules = []
    for raw_rule in sync_cfg.get("rules", []):
        try:
            parsed_rules.append(parse_routing_rule(raw_rule))
        except Exception:
            pass
    if recent_syncs and parsed_rules:
        strategy = sync_cfg.get("strategy", "first-match")
        project_groups = sync_cfg.get("project_groups", {})
        for entry in recent_syncs:
            if entry.get("category"):
                memory_meta = {
                    "category": entry["category"],
                    "scope": entry.get("scope", "global"),
                    "project": entry.get("project"),
                    "importance_score": entry.get("importance", 0.5),
                    "tags": "",
                }
                try:
                    result = evaluate_routing(
                        memory_meta, parsed_rules, strategy, project_groups
                    )
                    entry["matched_rules"] = result.matched_rules
                except Exception:
                    entry["matched_rules"] = []

    # Memory stats
    mem_stats = {}
    try:
        with _local_connection() as conn:
            conn.execute("SET TRANSACTION READ ONLY")
            conn.execute("SET statement_timeout = '10000'")
            # Total across all statuses (matches sidebar count)
            total_row = conn.execute(
                "SELECT count(*) FROM local.memories"
            ).fetchone()
            mem_stats["total"] = total_row[0] if total_row else 0
            # By status
            status_rows = conn.execute(
                "SELECT status, count(*) FROM local.memories "
                "GROUP BY status ORDER BY count(*) DESC"
            ).fetchall()
            mem_stats["by_status"] = {r[0]: r[1] for r in status_rows}
            # Active breakdown by category
            rows = conn.execute(
                "SELECT category, count(*) FROM local.memories "
                "WHERE status = 'active' GROUP BY category ORDER BY count(*) DESC"
            ).fetchall()
            mem_stats["by_category"] = {r[0]: r[1] for r in rows}
            mem_stats["total_active"] = sum(r[1] for r in rows)
    except Exception as e:
        logger.warning("Failed to read memory stats: %s", e)

    return {
        "sync": {
            "enabled": enabled,
            "strategy": sync_cfg.get("strategy", "first-match"),
            "worker_interval": sync_cfg.get("worker_interval_seconds", 30),
            "pull_interval": sync_cfg.get("pull_interval_seconds", 300),
        },
        "remotes": remotes,
        "queue": queue_stats,
        "dlq": dlq_entries,
        "last_push": last_push,
        "last_fail": last_fail,
        "recent_activity": recent_syncs,
        "rules": rules,
        "memory_stats": mem_stats,
    }


# ── Inline SPA ────────────────────────────────────────────────────────────────

_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Jarvis Admin</title>
<style>
:root {
  --bg: #0d1117; --surface: #161b22; --surface2: #21262d;
  --border: #30363d; --text: #e6edf3; --muted: #8b949e;
  --accent: #58a6ff; --badge: #1f6feb; --green: #238636; --orange: #f0883e;
  --red: #da3633;
}
* { box-sizing: border-box; margin: 0; padding: 0; }
body { background: var(--bg); color: var(--text); font: 13px/1.5 ui-monospace, "SF Mono", Consolas, monospace; display: flex; flex-direction: column; height: 100vh; overflow: hidden; }

/* Top tab bar */
#tabbar { display: flex; background: var(--surface); border-bottom: 1px solid var(--border); padding: 0 14px; gap: 0; flex-shrink: 0; align-items: stretch; }
#tabbar .tab-brand { font-size: 12px; font-weight: 700; color: var(--accent); padding: 9px 14px 9px 4px; letter-spacing: 0.5px; display: flex; align-items: center; }
.tab { padding: 9px 16px; font-size: 12px; color: var(--muted); cursor: pointer; border-bottom: 2px solid transparent; transition: all .15s; }
.tab:hover { color: var(--text); }
.tab.active { color: var(--text); border-bottom-color: var(--accent); font-weight: 600; }

/* Main content area below tabs */
#main { flex: 1; display: flex; overflow: hidden; }
#tab-memories { display: flex; flex: 1; overflow: hidden; }
#tab-admin { display: none; flex: 1; overflow-y: auto; padding: 20px 24px; }
#tab-admin.active { display: block; }
#tab-retrieval { display: none; flex: 1; overflow-y: auto; padding: 20px 24px; }
#tab-retrieval.active { display: block; }
.rt-grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:10px; margin-bottom:16px; }
.rt-card,.rt-panel { background:var(--surface); border:1px solid var(--border); border-radius:8px; padding:12px; }
.rt-value { font-size:22px; color:var(--accent); font-weight:700; }
.rt-label { color:var(--muted); font-size:10px; text-transform:uppercase; }
.rt-controls { display:flex; gap:8px; flex-wrap:wrap; margin-bottom:10px; align-items:center; }
.rt-controls input,.rt-controls select,.rt-controls textarea { background:var(--bg); color:var(--text); border:1px solid var(--border); border-radius:5px; padding:6px; font:inherit; }
.rt-table { width:100%; border-collapse:collapse; font-size:11px; }
.rt-table th,.rt-table td { padding:6px; border-bottom:1px solid var(--border); text-align:left; vertical-align:top; }
.rt-table tr[data-id] { cursor:pointer; }
.rt-table tr[data-id]:hover { background:var(--surface2); }
.rt-split { display:grid; grid-template-columns:minmax(420px,1fr) minmax(420px,1fr); gap:12px; }
.rt-panel { overflow-x:auto; }
.rt-overlay-backdrop { position:fixed; inset:0; background:rgba(0,0,0,.55); z-index:49; }
.rt-overlay { position:fixed; inset:2.5vh 2.5vw; z-index:50; background:var(--surface); border:1px solid var(--border); border-radius:10px; padding:16px 20px; overflow:auto; }
.rt-close { float:right; }
.rt-doc { white-space:pre-wrap; background:var(--bg); border:1px solid var(--border); border-radius:5px; padding:8px; font-size:11px; max-height:340px; overflow:auto; }
.rt-preview { color:var(--muted); font-size:10.5px; max-width:460px; }
.rt-row-main { cursor:pointer; }
.rt-row-main:hover { background:var(--surface2); }
.rt-bar { height:7px; background:var(--badge); border-radius:4px; min-width:2px; }
@media(max-width:1000px){.rt-split{grid-template-columns:1fr}}

/* Left sidebar */
#sidebar { width: 210px; min-width: 210px; background: var(--surface); border-right: 1px solid var(--border); display: flex; flex-direction: column; overflow: hidden; }
#sidebar-header { padding: 14px 12px 8px; font-size: 11px; text-transform: uppercase; letter-spacing: 1px; color: var(--muted); }
#source-list { flex: 1; overflow-y: auto; padding: 0 6px 8px; }
.src { padding: 7px 8px; border-radius: 6px; cursor: pointer; display: flex; align-items: center; gap: 8px; color: var(--muted); font-size: 12px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.src:hover { background: var(--surface2); color: var(--text); }
.src.active { background: var(--surface2); color: var(--text); font-weight: 600; }
.dot { width: 7px; height: 7px; min-width: 7px; border-radius: 50%; background: var(--green); }
.dot.remote { background: var(--orange); }
.dot.down { background: var(--muted); }
#stats-section { border-top: 1px solid var(--border); padding: 10px 12px; font-size: 11px; color: var(--muted); }
#stats-section strong { color: var(--text); }

/* Center: search + results */
#center { flex: 1; display: flex; flex-direction: column; min-width: 0; overflow: hidden; }
#topbar { padding: 10px 14px; background: var(--surface); border-bottom: 1px solid var(--border); display: flex; gap: 6px; align-items: center; }
#q { flex: 1; background: var(--bg); border: 1px solid var(--border); border-radius: 6px; padding: 7px 11px; color: var(--text); font: inherit; outline: none; min-width: 0; }
#q:focus { border-color: var(--accent); }
.mbtn { padding: 6px 10px; border-radius: 6px; border: 1px solid var(--border); background: var(--bg); color: var(--muted); cursor: pointer; font: 11px inherit; }
.mbtn:hover { color: var(--text); }
.mbtn.on { background: var(--badge); border-color: var(--badge); color: #fff; }
.mbtn:disabled { opacity: 0.3; cursor: default; }
#go { padding: 7px 14px; background: var(--green); border: none; border-radius: 6px; color: #fff; cursor: pointer; font: 600 12px inherit; }
#go:hover { background: #2ea043; }
#go:disabled { opacity: 0.3; cursor: default; background: var(--muted); }
#sortbar { padding: 6px 14px; background: var(--surface); border-bottom: 1px solid var(--border); display: none; gap: 10px; align-items: center; }
#sortbar.show { display: flex; }
#sortbar label { font-size: 11px; color: var(--muted); }
#sortbar select { background: var(--bg); border: 1px solid var(--border); border-radius: 4px; color: var(--text); padding: 3px 7px; font: 11px inherit; }
#sort-dir { padding: 3px 10px; border-radius: 4px; border: 1px solid var(--border); background: var(--bg); color: var(--accent); cursor: pointer; font: 600 11px inherit; min-width: 50px; }
#sort-dir:hover { border-color: var(--accent); }
#filterbar { padding: 6px 14px; background: var(--surface); border-bottom: 1px solid var(--border); display: none; gap: 10px; align-items: center; flex-wrap: wrap; }
#filterbar.show { display: flex; }
#filterbar label { font-size: 11px; color: var(--muted); }
#filterbar select { background: var(--bg); border: 1px solid var(--border); border-radius: 4px; color: var(--text); padding: 3px 7px; font: 11px inherit; }
#results { flex: 1; overflow-y: auto; padding: 12px 14px; display: flex; flex-direction: column; gap: 7px; }
.card { background: var(--surface); border: 1px solid var(--border); border-radius: 8px; padding: 11px 14px; cursor: pointer; transition: border-color .15s; }
.card:hover { border-color: #58a6ff44; }
.card.selected { border-color: var(--accent); background: var(--surface2); }
.cid { font-size: 10px; color: var(--muted); margin-bottom: 5px; font-family: inherit; word-break: break-all; }
.csnip { color: var(--text); font-size: 12px; line-height: 1.6; white-space: pre-wrap; word-break: break-word; }
.cmeta { display: flex; gap: 6px; margin-top: 7px; flex-wrap: wrap; }
.badge { font-size: 10px; padding: 2px 6px; border-radius: 4px; background: var(--surface2); color: var(--muted); }
.sbadge { background: #1c2d3f; color: var(--accent); }
.ubadge { background: #2d1c3f; color: #c49bff; }
.empty { color: var(--muted); text-align: center; padding: 60px 20px; font-size: 13px; }
#statusbar { padding: 5px 14px; border-top: 1px solid var(--border); font-size: 11px; color: var(--muted); min-height: 24px; }
#more-btn { display: block; margin: 4px auto 8px; padding: 7px 20px; background: var(--surface); border: 1px solid var(--border); border-radius: 6px; color: var(--text); cursor: pointer; font: 12px inherit; }
#more-btn:hover { border-color: var(--accent); }

/* Right detail panel */
#detail { width: 380px; min-width: 380px; background: var(--surface); border-left: 1px solid var(--border); display: flex; flex-direction: column; overflow: hidden; }
#detail-header { padding: 12px 14px 8px; font-size: 11px; text-transform: uppercase; letter-spacing: 1px; color: var(--muted); display: flex; justify-content: space-between; align-items: center; }
#detail-close { background: none; border: none; color: var(--muted); cursor: pointer; font-size: 16px; padding: 0 4px; }
#detail-close:hover { color: var(--text); }
#detail-body { flex: 1; overflow-y: auto; padding: 0 14px 14px; }
#detail.empty-state #detail-body { display: flex; align-items: center; justify-content: center; }
.detail-section { margin-bottom: 14px; }
.detail-section-title { font-size: 10px; text-transform: uppercase; letter-spacing: 0.5px; color: var(--muted); margin-bottom: 6px; padding-bottom: 4px; border-bottom: 1px solid var(--border); }
.detail-content { background: var(--bg); border: 1px solid var(--border); border-radius: 6px; padding: 12px; font-size: 12px; line-height: 1.7; white-space: pre-wrap; word-break: break-word; max-height: 50vh; overflow-y: auto; }
.meta-table { width: 100%; border-collapse: collapse; }
.meta-table tr { border-bottom: 1px solid var(--border); }
.meta-table tr:last-child { border-bottom: none; }
.meta-table td { padding: 5px 0; font-size: 12px; vertical-align: top; }
.meta-key { color: var(--accent); width: 120px; font-weight: 600; padding-right: 10px; }
.meta-val { color: var(--text); word-break: break-all; }
.meta-val.nested { font-size: 11px; color: var(--muted); white-space: pre-wrap; }

/* Admin panel */
.admin-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(340px, 1fr)); gap: 16px; margin-bottom: 20px; }
.admin-card { background: var(--surface); border: 1px solid var(--border); border-radius: 8px; padding: 16px; }
.admin-card h3 { font-size: 12px; text-transform: uppercase; letter-spacing: 0.5px; color: var(--muted); margin-bottom: 12px; padding-bottom: 6px; border-bottom: 1px solid var(--border); }
.admin-card .val { font-size: 22px; font-weight: 700; color: var(--text); }
.admin-card .lbl { font-size: 11px; color: var(--muted); margin-top: 2px; }
.kv { display: flex; justify-content: space-between; align-items: center; padding: 5px 0; border-bottom: 1px solid var(--border); font-size: 12px; }
.kv:last-child { border-bottom: none; }
.kv .k { color: var(--muted); }
.kv .v { color: var(--text); font-weight: 600; }
.pill { display: inline-block; padding: 2px 8px; border-radius: 10px; font-size: 10px; font-weight: 600; }
.pill-green { background: #23863622; color: var(--green); }
.pill-red { background: #da363322; color: var(--red); }
.pill-orange { background: #f0883e22; color: var(--orange); }
.pill-blue { background: #1f6feb22; color: var(--accent); }
.pill-muted { background: var(--surface2); color: var(--muted); }
.admin-table { width: 100%; border-collapse: collapse; font-size: 12px; }
.admin-table th { text-align: left; color: var(--muted); font-size: 10px; text-transform: uppercase; letter-spacing: 0.5px; padding: 6px 8px; border-bottom: 1px solid var(--border); }
.admin-table td { padding: 6px 8px; border-bottom: 1px solid var(--border); color: var(--text); vertical-align: top; }
.admin-table tr:last-child td { border-bottom: none; }
.admin-table .err { font-size: 11px; color: var(--red); max-width: 300px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.admin-section { margin-bottom: 20px; }
.admin-section h2 { font-size: 13px; color: var(--accent); margin-bottom: 10px; padding-bottom: 6px; border-bottom: 1px solid var(--border); }
.stat-row { display: flex; gap: 16px; flex-wrap: wrap; margin-bottom: 12px; }
.stat-box { background: var(--bg); border: 1px solid var(--border); border-radius: 6px; padding: 10px 14px; min-width: 100px; text-align: center; }
.stat-box .num { font-size: 20px; font-weight: 700; }
.stat-box .slbl { font-size: 10px; color: var(--muted); text-transform: uppercase; letter-spacing: 0.5px; }
.match-chip { display: inline-block; background: var(--surface2); border-radius: 4px; padding: 1px 6px; margin: 1px 2px; font-size: 11px; color: var(--muted); }
.ts { font-size: 11px; color: var(--muted); }

/* Admin CRUD forms */
.admin-form { background: var(--bg); border: 1px solid var(--accent); border-radius: 8px; padding: 16px; margin: 10px 0; }
.admin-form h4 { font-size: 12px; color: var(--accent); margin-bottom: 12px; }
.form-row { display: flex; gap: 10px; align-items: center; margin-bottom: 8px; flex-wrap: wrap; }
.form-row label { font-size: 11px; color: var(--muted); min-width: 90px; cursor: help; }
.form-row input, .form-row select { background: var(--surface); border: 1px solid var(--border); border-radius: 4px; color: var(--text); padding: 5px 8px; font: 12px inherit; min-width: 150px; }
.form-row input:focus, .form-row select:focus { border-color: var(--accent); outline: none; }
.form-row input[type="checkbox"] { min-width: auto; width: 16px; height: 16px; }
.form-actions { display: flex; gap: 8px; margin-top: 12px; }
.btn-save { padding: 6px 16px; background: var(--green); border: none; border-radius: 4px; color: #fff; cursor: pointer; font: 600 12px inherit; }
.btn-save:hover { opacity: 0.9; }
.btn-save:disabled { opacity: 0.4; cursor: default; }
.btn-cancel { padding: 6px 16px; background: var(--surface2); border: 1px solid var(--border); border-radius: 4px; color: var(--text); cursor: pointer; font: 12px inherit; }
.btn-cancel:hover { border-color: var(--muted); }
.btn-sm { padding: 3px 8px; font-size: 11px; border-radius: 4px; border: 1px solid var(--border); background: var(--bg); color: var(--muted); cursor: pointer; }
.btn-sm:hover { color: var(--text); border-color: var(--muted); }
.btn-sm.btn-danger { color: var(--red); }
.btn-sm.btn-danger:hover { background: #da363322; }
.btn-sm:disabled { opacity: 0.3; cursor: default; }
.btn-delete { padding: 6px 16px; background: transparent; border: 1px solid var(--red); border-radius: 4px; color: var(--red); cursor: pointer; font: 600 12px inherit; transition: all .15s; }
.btn-delete:hover { background: var(--red); color: #fff; }
.btn-delete:disabled { opacity: 0.3; cursor: default; }
.admin-error { background: #da363322; border: 1px solid var(--red); border-radius: 4px; padding: 8px 12px; margin: 8px 0; color: var(--red); font-size: 12px; }
.admin-success { background: #23863622; border: 1px solid var(--green); border-radius: 4px; padding: 8px 12px; margin: 8px 0; color: var(--green); font-size: 12px; }
.test-result { display: inline-block; padding: 3px 10px; border-radius: 10px; font-size: 11px; font-weight: 600; margin-left: 8px; }
.test-result.ok { background: #23863622; color: var(--green); }
.test-result.fail { background: #da363322; color: var(--red); }
.test-result.loading { background: var(--surface2); color: var(--muted); }
.rule-tester { background: var(--surface); border: 1px solid var(--border); border-radius: 8px; padding: 16px; margin-top: 12px; }
.rule-tester h4 { font-size: 12px; color: var(--accent); margin-bottom: 10px; cursor: pointer; }
.rule-tester-body { display: none; }
.rule-tester-body.show { display: block; }
.test-results { margin-top: 10px; padding: 10px; background: var(--bg); border: 1px solid var(--border); border-radius: 4px; font-size: 12px; }
</style>
</head>
<body>
<div id="tabbar">
  <div class="tab-brand">JARVIS</div>
  <div class="tab active" data-tab="memories" onclick="switchTab('memories')">Memories</div>
  <div class="tab" data-tab="retrieval" onclick="switchTab('retrieval')">Retrieval</div>
  <div class="tab" data-tab="admin" onclick="switchTab('admin')">Admin</div>
</div>
<div id="main">
<div id="tab-memories">
<div id="sidebar">
  <div id="sidebar-header">Sources</div>
  <div id="source-list"><div class="empty" style="padding:20px 8px">Loading...</div></div>
  <div id="stats-section"></div>
</div>
<div id="center">
  <div id="topbar">
    <input id="q" type="text" placeholder="Search memories..." autocomplete="off">
    <button class="mbtn on" data-mode="text" onclick="setMode('text')">Text</button>
    <button class="mbtn" data-mode="semantic" onclick="setMode('semantic')">Semantic</button>
    <button class="mbtn" data-mode="metadata" onclick="setMode('metadata')">Filter</button>
    <button id="go" onclick="run()">Search</button>
    <button id="mem-auto-refresh" onclick="toggleMemAutoRefresh()" style="padding:4px 10px;background:var(--bg2);border:1px solid var(--border);border-radius:4px;color:var(--fg);cursor:pointer;font-size:12px" title="Auto-refresh every 60s when browser is focused">Auto 60s</button>
  </div>
  <div id="filterbar">
    <label>Category</label>
    <select id="f-cat">
      <option value="">Any</option>
      <option>observation</option><option>pattern</option><option>learning</option>
      <option>decision</option><option>summary</option><option>code</option>
      <option>relationship</option><option>hint</option><option>plan</option>
      <option>worklog</option><option>memory</option>
    </select>
    <label>Scope</label>
    <select id="f-scope"><option value="">Any</option><option>global</option><option>project</option></select>
    <label>Status</label>
    <select id="f-status"><option value="">Any</option><option>active</option><option>superseded</option><option>deleted</option></select>
  </div>
  <div id="sortbar">
    <label>Sort by</label>
    <select id="sort-field"></select>
    <button id="sort-dir" onclick="toggleSortDir()">DESC</button>
  </div>
  <div id="results"><div class="empty">Select a source and search.</div></div>
  <div id="statusbar"></div>
</div>
<div id="detail" class="empty-state">
  <div id="detail-header">
    <span>Detail</span>
    <button id="detail-close" onclick="closeDetail()">&times;</button>
  </div>
  <div id="detail-body"><span class="empty" style="padding:20px">Click a result to view details</span></div>
</div>
</div><!-- /tab-memories -->
<div id="tab-admin"></div>
<div id="tab-retrieval"></div>
</div><!-- /main -->

<script>
let cur = null, mode = 'text', page = 0, allCaps = [], selectedCard = null, curFilters = [];
let cachedResults = [], cachedTotal = 0, sortDir = 'desc', curDeletable = false;

fetch('/api/sources').then(r => r.json()).then(srcs => {
  const el = document.getElementById('source-list');
  el.innerHTML = '';
  srcs.forEach(s => {
    const d = document.createElement('div');
    d.className = 'src';
    d.dataset.id = s.id;
    const dot = document.createElement('span');
    const caps = s.capabilities || [];
    dot.className = 'dot' + (caps.length === 0 ? ' down' : '');
    const lbl = document.createElement('span');
    lbl.textContent = s.label;
    d.appendChild(dot); d.appendChild(lbl);
    d.onclick = () => pick(s.id, caps, s.metadata_filters || [], s.sort_options || [], s.deletable);
    el.appendChild(d);
  });
  if (srcs.length) pick(srcs[0].id, srcs[0].capabilities || [], srcs[0].metadata_filters || [], srcs[0].sort_options || [], srcs[0].deletable);
}).catch(err => {
  document.getElementById('source-list').innerHTML = '<div class="empty" style="color:#ef4444">Failed: ' + esc(String(err)) + '</div>';
});

var sourceLabels = {};
fetch('/api/sources').then(r => r.json()).then(srcs2 => {
  srcs2.forEach(function(s) { sourceLabels[s.id] = s.label; });
  return fetch('/api/stats');
}).then(r => r.json()).then(stats => {
  const el = document.getElementById('stats-section');
  let html = '';
  Object.entries(stats).forEach(([id, s]) => {
    const label = sourceLabels[id] || id.replace('remote:', '');
    const n = s.count !== null ? s.count.toLocaleString() : '?';
    html += '<div style="display:flex;justify-content:space-between;margin-bottom:3px"><span>' + esc(label) + '</span><strong>' + n + '</strong></div>';
  });
  el.innerHTML = html || '<span>No data</span>';
}).catch(() => {});

function pick(id, caps, mfilt, sorts, deletable) {
  cur = id;
  allCaps = caps;
  curFilters = mfilt || [];
  curDeletable = !!deletable;
  document.querySelectorAll('.src').forEach(el => el.classList.toggle('active', el.dataset.id === id));
  ['text','semantic','metadata'].forEach(m => {
    document.querySelector('.mbtn[data-mode="' + m + '"]').disabled = !caps.includes(m);
  });
  // Show/hide filter dropdowns based on source capabilities
  var filterMap = {'f-cat':'category', 'f-scope':'scope', 'f-status':'status'};
  Object.entries(filterMap).forEach(function(e) {
    var sel = document.getElementById(e[0]);
    var show = curFilters.includes(e[1]);
    if (sel) {
      sel.style.display = show ? '' : 'none';
      // Also hide the preceding label sibling
      var lbl = sel.previousElementSibling;
      if (lbl && lbl.tagName === 'LABEL') lbl.style.display = show ? '' : 'none';
    }
  });
  // Populate sort field dropdown
  var sf = document.getElementById('sort-field');
  sf.innerHTML = '';
  (sorts || []).forEach(function(s) {
    var opt = document.createElement('option');
    // Strip direction from value (date_desc -> date) — we control dir separately
    opt.value = s.value; opt.textContent = s.label;
    sf.appendChild(opt);
  });
  sf.onchange = function() { clientSort(); };
  cachedResults = []; cachedTotal = 0;
  document.getElementById('sortbar').classList.remove('show');
  document.getElementById('go').disabled = !caps.length;
  if (!caps.includes(mode) && caps.length) setMode(caps[0]);
  if (!caps.length) {
    document.getElementById('results').innerHTML = '<div class="empty">Source unavailable — cannot connect.</div>';
  } else {
    document.getElementById('results').innerHTML = '<div class="empty">Press Search to see results.</div>';
  }
  document.getElementById('statusbar').textContent = '';
  closeDetail();
}

function setMode(m) {
  mode = m;
  document.querySelectorAll('.mbtn').forEach(b => b.classList.toggle('on', b.dataset.mode === m));
  document.getElementById('filterbar').classList.toggle('show', m === 'metadata');
  // Add/remove similarity option for semantic mode
  var sf = document.getElementById('sort-field');
  var hasSim = sf.querySelector('option[value="similarity"]');
  if (m === 'semantic' && !hasSim) {
    var opt = document.createElement('option');
    opt.value = 'similarity'; opt.textContent = 'Similarity';
    sf.insertBefore(opt, sf.firstChild);
    sf.value = 'similarity';
  } else if (m !== 'semantic' && hasSim) {
    hasSim.remove();
    if (sf.value === 'similarity') sf.value = 'date_desc';
  }
  document.getElementById('q').placeholder =
    m === 'semantic' ? 'Describe what you are looking for...' :
    m === 'metadata' ? 'Optional text filter...' : 'Search memories...';
}

function filters() {
  const f = {};
  const cat = document.getElementById('f-cat').value;
  const scope = document.getElementById('f-scope').value;
  const status = document.getElementById('f-status').value;
  if (cat) f.category = cat;
  if (scope) f.scope = scope;
  if (status) f.status = status;
  return f;
}

function closeDetail() {
  const panel = document.getElementById('detail');
  panel.classList.add('empty-state');
  document.getElementById('detail-body').innerHTML = '<span class="empty" style="padding:20px">Click a result to view details</span>';
  if (selectedCard) { selectedCard.classList.remove('selected'); selectedCard = null; }
}

function showDetail(id) {
  const panel = document.getElementById('detail');
  const body = document.getElementById('detail-body');
  panel.classList.remove('empty-state');
  body.innerHTML = '<div class="empty" style="padding:20px">Loading...</div>';

  fetch('/api/content?source=' + encodeURIComponent(cur) + '&id=' + encodeURIComponent(id))
    .then(r => { if (!r.ok) return apiError(r); return r.json(); })
    .then(d => {
      let html = '';

      // ID
      html += '<div class="detail-section"><div class="detail-section-title">ID</div>';
      html += '<div style="font-size:11px;color:var(--muted);word-break:break-all">' + esc(d.id || id) + '</div></div>';

      // Content
      html += '<div class="detail-section"><div class="detail-section-title">Content</div>';
      html += '<div class="detail-content">' + esc(d.content || '(empty)') + '</div></div>';

      // Metadata
      if (d.metadata && typeof d.metadata === 'object') {
        html += '<div class="detail-section"><div class="detail-section-title">Metadata</div>';
        html += '<table class="meta-table">';
        var keys = Object.keys(d.metadata).sort();
        keys.forEach(k => {
          var v = d.metadata[k];
          var isObj = typeof v === 'object' && v !== null;
          html += '<tr><td class="meta-key">' + esc(k) + '</td>';
          html += '<td class="meta-val' + (isObj ? ' nested' : '') + '">' + esc(isObj ? JSON.stringify(v, null, 2) : String(v)) + '</td></tr>';
        });
        html += '</table></div>';
      }

      // Timestamps
      html += '<div class="detail-section"><div class="detail-section-title">Timestamps</div>';
      html += '<table class="meta-table">';
      html += '<tr><td class="meta-key">created_at</td><td class="meta-val">' + esc(d.created_at || 'N/A') + '</td></tr>';
      if (d.updated_at) html += '<tr><td class="meta-key">updated_at</td><td class="meta-val">' + esc(d.updated_at) + '</td></tr>';
      html += '</table></div>';

      // Delete button — only for deletable sources, non-deleted items
      var itemStatus = (d.metadata && d.metadata.status) || 'active';
      if (curDeletable && itemStatus !== 'deleted') {
        html += '<div class="detail-section" style="margin-top:16px;padding-top:12px;border-top:1px solid var(--border)">';
        html += '<button class="btn-delete" data-delete-id="">Delete Memory</button>';
        html += '</div>';
      }

      body.innerHTML = html;

      // Wire delete button via DOM (no inline onclick — XSS safe)
      var delBtn = body.querySelector('.btn-delete');
      if (delBtn) delBtn.dataset.deleteId = d.id || id;
    })
    .catch(err => {
      body.innerHTML = '<div class="empty" style="color:#ef4444">Failed to load: ' + esc(String(err)) + '</div>';
    });
}

function makeCard(r) {
  var card = document.createElement('div');
  card.className = 'card';
  card.dataset.rid = r.id;
  var cid = document.createElement('div'); cid.className = 'cid'; cid.textContent = r.id;
  var csnip = document.createElement('div'); csnip.className = 'csnip'; csnip.textContent = r.snippet;
  var cmeta = document.createElement('div'); cmeta.className = 'cmeta';
  if (r.created_at) { var b = document.createElement('span'); b.className = 'badge'; b.textContent = r.created_at.slice(0,10); cmeta.appendChild(b); }
  if (r.score !== undefined) { var b2 = document.createElement('span'); b2.className = 'badge sbadge'; b2.textContent = 'sim ' + r.score; cmeta.appendChild(b2); }
  if (r.user) { var bu = document.createElement('span'); bu.className = 'badge ubadge'; bu.textContent = r.user; cmeta.appendChild(bu); }
  card.onclick = function() {
    if (selectedCard) selectedCard.classList.remove('selected');
    card.classList.add('selected');
    selectedCard = card;
    showDetail(r.id);
  };
  card.append(cid, csnip, cmeta);
  return card;
}

function renderResults() {
  var el = document.getElementById('results');
  el.innerHTML = '';
  selectedCard = null;
  if (!cachedResults.length) {
    el.innerHTML = '<div class="empty">No results found.</div>';
    document.getElementById('statusbar').textContent = '';
    return;
  }
  cachedResults.forEach(function(r) { el.appendChild(makeCard(r)); });
  document.getElementById('statusbar').textContent = 'Showing ' + cachedResults.length + ' of ' + cachedTotal + ' results';
  if (cachedResults.length >= (page + 1) * 100) {
    var btn = document.createElement('button');
    btn.id = 'more-btn'; btn.textContent = 'Load more';
    btn.onclick = function() { page++; run(true); };
    el.appendChild(btn);
  }
}

function getSortKey(r) {
  var field = document.getElementById('sort-field').value;
  if (field === 'similarity') return r.score != null ? r.score : -1;
  if (field === 'date_desc' || field === 'date_asc') return r.created_at || '';
  if (field === 'updated_desc') return r.updated_at || r.created_at || '';
  if (field === 'importance_desc' || field === 'importance_asc') return r.importance_score != null ? r.importance_score : -1;
  if (field === 'size_desc' || field === 'size_asc') return r.doc_size != null ? r.doc_size : 0;
  if (field === 'retrieval_desc') return r.retrieval_count != null ? r.retrieval_count : -1;
  return r.created_at || '';
}

function clientSort() {
  if (!cachedResults.length) return;
  var dir = sortDir === 'asc' ? 1 : -1;
  cachedResults.sort(function(a, b) {
    var ka = getSortKey(a), kb = getSortKey(b);
    if (ka < kb) return -1 * dir;
    if (ka > kb) return 1 * dir;
    return 0;
  });
  renderResults();
}

function toggleSortDir() {
  sortDir = sortDir === 'desc' ? 'asc' : 'desc';
  document.getElementById('sort-dir').textContent = sortDir.toUpperCase();
  clientSort();
}

function run(append) {
  if (!cur) return;
  if (!append) page = 0;
  var q = document.getElementById('q').value;
  var sortVal = document.getElementById('sort-field').value || 'date_desc';
  var serverSort = sortVal;
  // Map field + direction to server sort key
  if (sortVal === 'similarity') serverSort = 'similarity';
  else if (sortVal.indexOf('_') === -1) serverSort = sortVal + '_' + sortDir;
  var body = { source: cur, mode: mode, query: q, filters: filters(), page: page, page_size: 100, sort_by: serverSort };
  document.getElementById('statusbar').textContent = 'Searching...';
  fetch('/api/search', { method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify(body) })
    .then(r => { if (!r.ok) return apiError(r); return r.json(); })
    .then(data => {
      var res = data.results || [];
      if (append) {
        cachedResults = cachedResults.concat(res);
      } else {
        cachedResults = res;
      }
      cachedTotal = data.total || 0;
      document.getElementById('sortbar').classList.add('show');
      renderResults();
    })
    .catch(err => {
      document.getElementById('statusbar').textContent = 'Error: ' + err;
    });
}

function esc(s) { return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;'); }

// Reject with a readable message for a non-2xx response. The body is read
// exactly once (as text), then FastAPI's {"detail": ...} is used if present.
// Callers must render the message via textContent or esc().
function apiError(r) {
  return r.text().then(function(t) {
    var m = t || ('HTTP ' + r.status);
    try {
      var d = JSON.parse(t);
      if (d && d.detail) m = typeof d.detail === 'string' ? d.detail : JSON.stringify(d.detail);
    } catch (_) {}
    if (r.status === 503 && m.indexOf('Database unavailable') !== 0) m = 'Database unavailable: ' + m;
    return Promise.reject(m);
  });
}

document.getElementById('q').addEventListener('keydown', function(e) { if (e.key === 'Enter') run(); });

/* ── Memory auto-refresh ─────────────────────────────────────── */
var memAutoRefresh = null;
function toggleMemAutoRefresh() {
  if (memAutoRefresh) {
    clearInterval(memAutoRefresh);
    memAutoRefresh = null;
  } else {
    memAutoRefresh = setInterval(function() {
      if (document.hasFocus() && cur && cachedResults.length > 0) run();
    }, 60000);
    // Also run immediately on enable
    if (cur && cachedResults.length > 0) run();
  }
  var btn = document.getElementById('mem-auto-refresh');
  if (btn) {
    btn.style.background = memAutoRefresh ? 'var(--green)' : 'var(--bg2)';
    btn.style.color = memAutoRefresh ? '#000' : 'var(--fg)';
  }
}

/* ── Delete memory ────────────────────────────────────────────── */

document.addEventListener('click', function(e) {
  var btn = e.target.closest('.btn-delete');
  if (btn && btn.dataset.deleteId) deleteMemory(btn.dataset.deleteId);
});

function deleteMemory(id) {
  if (!confirm('Delete this memory? (soft-delete — sets status to deleted)')) return;
  fetch('/api/memories/' + encodeURIComponent(id) + '?source=' + encodeURIComponent(cur), { method: 'DELETE' })
    .then(function(r) {
      if (!r.ok) return apiError(r);
      return r.json();
    })
    .then(function() {
      cachedResults = cachedResults.filter(function(r) { return r.id !== id; });
      cachedTotal = Math.max(0, cachedTotal - 1);
      renderResults();
      closeDetail();
      refreshStats();
    })
    .catch(function(err) { alert('Delete failed: ' + err); });
}

function refreshStats() {
  fetch('/api/sources').then(function(r) { return r.json(); }).then(function(srcs) {
    srcs.forEach(function(s) { sourceLabels[s.id] = s.label; });
    return fetch('/api/stats');
  }).then(function(r) { return r.json(); }).then(function(stats) {
    var el = document.getElementById('stats-section');
    var html = '';
    Object.entries(stats).forEach(function(e) {
      var label = sourceLabels[e[0]] || e[0].replace('remote:', '');
      var n = e[1].count !== null ? e[1].count.toLocaleString() : '?';
      html += '<div style="display:flex;justify-content:space-between;margin-bottom:3px"><span>' + esc(label) + '</span><strong>' + n + '</strong></div>';
    });
    el.innerHTML = html || '<span>No data</span>';
  }).catch(function() {});
}

/* ── Tab switching + Admin panel ──────────────────────────────── */

function switchTab(tab) {
  document.querySelectorAll('.tab').forEach(t => t.classList.toggle('active', t.dataset.tab === tab));
  var mem = document.getElementById('tab-memories');
  var adm = document.getElementById('tab-admin');
  var ret = document.getElementById('tab-retrieval');
  mem.style.display = 'none'; adm.style.display = 'none'; ret.style.display = 'none';
  adm.classList.remove('active'); ret.classList.remove('active');
  if (tab === 'admin') {
    adm.style.display = 'block';
    adm.classList.add('active');
    loadAdmin();
    if (memAutoRefresh) { clearInterval(memAutoRefresh); memAutoRefresh = null; var mb = document.getElementById('mem-auto-refresh'); if (mb) { mb.style.background = 'var(--bg2)'; mb.style.color = 'var(--fg)'; } }
  } else if (tab === 'retrieval') {
    ret.style.display = 'block'; ret.classList.add('active'); loadRetrieval();
    if (adminAutoRefresh) { clearInterval(adminAutoRefresh); adminAutoRefresh = null; }
  } else {
    mem.style.display = 'flex';
    if (adminAutoRefresh) { clearInterval(adminAutoRefresh); adminAutoRefresh = null; }
  }
  // Persist tab in URL hash so page refresh stays on the same tab
  history.replaceState(null, '', '#' + tab);
}

// On page load, restore tab from hash
(function() {
  var hash = location.hash.replace('#', '');
  if (hash === 'admin' || hash === 'retrieval') switchTab(hash);
})();

var retrievalLoaded = false;
function loadRetrieval() {
  var el = document.getElementById('tab-retrieval');
  if (!retrievalLoaded) el.innerHTML = '<div class="empty">Loading retrieval telemetry...</div>';
  var purpose = document.getElementById('rt-purpose');
  var outcome = document.getElementById('rt-outcome');
  var filters = { purpose: purpose ? purpose.value : '', outcome: outcome ? outcome.value : '' };
  var query = '?limit=100' + (filters.purpose ? '&purpose='+encodeURIComponent(filters.purpose) : '') + (filters.outcome ? '&outcome='+encodeURIComponent(filters.outcome) : '');
  Promise.all([fetch('/api/retrieval/summary').then(r=>r.ok?r.json():apiError(r)), fetch('/api/retrieval/events'+query).then(r=>r.ok?r.json():apiError(r))])
    .then(parts => { retrievalLoaded=true; renderRetrieval(parts[0], parts[1], filters); })
    .catch(err => { el.innerHTML='<div class="empty" style="color:var(--red)">'+esc(String(err))+'</div>'; });
}

function escA(s) { return esc(String(s)).replace(/"/g,'&quot;'); }
function rtOpt(value, label, current) { return '<option value="'+escA(value)+'"'+(current===value?' selected':'')+'>'+esc(label)+'</option>'; }

function renderRetrieval(s, events, filters) {
  filters = filters || {purpose:'', outcome:''};
  var el=document.getElementById('tab-retrieval');
  var zero=s.requests ? Math.round(100*s.zero_results/s.requests) : 0;
  var h='<div class="rt-grid">'+rtCard('Requests (7d)',s.requests||0)+rtCard('Zero result',zero+'%')+rtCard('p50 latency',(s.p50_ms==null?'—':Math.round(s.p50_ms)+' ms'))+rtCard('p95 latency',(s.p95_ms==null?'—':Math.round(s.p95_ms)+' ms'))+rtCard('Returned',(s.funnel||{}).returned||0)+rtCard('Delivered',(s.delivery||{}).delivered||0)+rtCard('Shadow pending',s.shadow_pending||0)+rtCard('Shadow failed',s.shadow_failed||0)+'</div>';
  h+='<div class="rt-split" style="margin-bottom:12px"><div class="rt-panel"><div class="detail-section-title">Aggregate funnel</div>'+rtFunnel(s.funnel||{})+'</div><div class="rt-panel"><div class="detail-section-title">Score distributions</div><div class="rt-split">'+rtHistogram('cosine',(s.histograms||{}).cosine||[])+rtHistogram('raw BGE logit',(s.histograms||{}).raw_bge_logit||[])+'</div></div></div>';
  h+='<div class="rt-controls"><select id="rt-purpose" onchange="loadRetrieval()">'+rtOpt('','All purposes',filters.purpose);
  (s.purposes||[]).forEach(p=>{h+=rtOpt(p.purpose,p.purpose+' ('+(p.requests||0)+')',filters.purpose);});
  h+='</select><select id="rt-outcome" onchange="loadRetrieval()">'+rtOpt('','All outcomes',filters.outcome)+rtOpt('results','results',filters.outcome)+rtOpt('empty','empty',filters.outcome)+rtOpt('error','error',filters.outcome)+'</select><button class="mbtn" onclick="loadRetrieval()">Refresh</button><a class="mbtn" href="/api/retrieval/export" target="_blank">Export labels</a></div>';
  h+='<div class="rt-split"><div class="rt-panel"><div class="detail-section-title">Events</div><table class="rt-table"><tr><th>Time</th><th>Purpose</th><th>Outcome</th><th>ANN → returned</th><th>Shadow</th></tr>';
  (events||[]).forEach(e=>{var f=e.funnel||{}; h+='<tr data-id="'+escA(e.id)+'" onclick="loadRetrievalEvent(this.dataset.id)"><td>'+esc(fmtTime(e.created_at))+'</td><td>'+esc(e.purpose)+'</td><td>'+esc(e.outcome)+'</td><td>'+esc(String(f.ann_unique||0))+' → '+esc(String(f.budget_selected!=null?f.budget_selected:(f.returned||0)))+'</td><td>'+esc(e.shadow_status||'')+'</td></tr>';});
  h+='</table></div><div id="rt-detail" class="rt-panel"><div class="empty">Select an event to inspect its funnel and candidates.</div></div></div>';
  h+='<div class="rt-panel" style="margin-top:12px"><div class="detail-section-title">Policy simulator (read-only)</div><div class="rt-controls"><select id="sim-policy"><option>cosine-only</option><option>bge-only</option><option>coarse+bge</option><option>cosine-or-bge</option></select><label>cosine <input id="sim-cos" type="number" step="0.01" value="0.85" style="width:75px"></label><label>BGE logit <input id="sim-bge" type="number" step="0.1" value="-2.5" style="width:75px"></label><label>era <select id="sim-era"><option value="">all (mixed)</option><option value="summary">summary</option><option value="mechanical">mechanical</option><option value="none">none</option><option value="unstamped">unstamped</option></select></label><button class="mbtn" onclick="runSimulation()">Simulate</button></div><pre id="sim-result" class="detail-content">No live configuration is changed. Pick an augmentation era before calibrating a threshold — mechanical-era and summary-era logits are not comparable.</pre></div>';
  el.innerHTML=h;
  if (rtEvent) renderRetrievalEvent();
}
function rtCard(label,value){return '<div class="rt-card"><div class="rt-value">'+esc(String(value))+'</div><div class="rt-label">'+esc(label)+'</div></div>';}
function rtFunnel(f){var keys=['candidates','cosine_rejected','logit_rejected','sensitive_rejected','parent_dedup','semantic_duplicates','candidate_cap','result_cap','budget_rejected','returned','delivered'];var max=Math.max(1,...keys.map(k=>Number(f[k]||0)));return keys.map(k=>'<div>'+esc(k)+' '+Number(f[k]||0)+'<div class="rt-bar" style="width:'+Math.round(100*Number(f[k]||0)/max)+'%"></div></div>').join('');}
function rtHistogram(label,rows){var max=Math.max(1,...rows.map(r=>Number(r.count||0)));return '<div><div class="rt-label">'+esc(label)+'</div>'+rows.map(r=>'<div>'+esc(String(r.bucket))+' <span class="rt-bar" style="display:inline-block;width:'+Math.round(100*Number(r.count||0)/max)+'px"></span> '+r.count+'</div>').join('')+'</div>';}
function fmtTime(v){try{return new Date(v).toLocaleString();}catch(e){return String(v||'');}}

var rtEvent = null;
var rtCandFilter = '';
var rtDocs = null;
var rtOverlayOpen = false;
var rtSortKey = '__ranked__';
var rtSortDir = -1;
function loadRetrievalEvent(id){
 fetch('/api/retrieval/events/'+encodeURIComponent(id)).then(r=>r.json()).then(e=>{
  if (!rtEvent || rtEvent.id !== e.id) { rtCandFilter = ''; rtDocs = null; rtSortKey = '__ranked__'; rtSortDir = -1; }
  rtEvent = e;
  renderRetrievalEvent();
  if (rtOverlayOpen) openRtOverlay();
 });
}
function rtDescNullsLast(a, b){
  if (a == null && b == null) return 0;
  if (a == null) return 1;
  if (b == null) return -1;
  return b - a;
}
function rtRankedCmp(a, b){
  return rtDescNullsLast(a.blended_score, b.blended_score)
      || rtDescNullsLast(a.raw_bge_logit, b.raw_bge_logit)
      || rtDescNullsLast(a.similarity, b.similarity);
}
function rtSortRows(rows){
  var sorted = rows.slice();
  if (rtSortKey === '__ranked__') { sorted.sort(rtRankedCmp); return sorted; }
  var dir = rtSortDir;
  function val(c){
    if (rtSortKey === 'candidate') return c.schema_name + ':' + c.doc_id;
    if (rtSortKey === 'returned' || rtSortKey === 'delivered') return c[rtSortKey] ? 1 : 0;
    if (rtSortKey === 'terminal_reason') return c.terminal_reason || '';
    return c[rtSortKey];
  }
  sorted.sort(function(a, b){
    var x = val(a), y = val(b);
    if (x == null && y == null) return 0;
    if (x == null) return 1;
    if (y == null) return -1;
    if (x < y) return -1 * dir;
    if (x > y) return 1 * dir;
    return 0;
  });
  return sorted;
}
function rtSetSort(key){
  if (rtSortKey === key) { rtSortDir = -rtSortDir; }
  else { rtSortKey = key; rtSortDir = (key === 'vector_rank' || key === 'candidate' || key === 'terminal_reason') ? 1 : -1; }
  renderRtOverlay();
}
function rtResetSort(){
  rtSortKey = '__ranked__';
  rtSortDir = -1;
  renderRtOverlay();
}
function rtTh(key, label){
  var arrow = rtSortKey === key ? (rtSortDir === 1 ? ' ▲' : ' ▼') : '';
  return '<th data-key="'+escA(key)+'" onclick="rtSetSort(this.dataset.key)" style="cursor:pointer" title="Sort by '+escA(label)+'">'+esc(label)+arrow+'</th>';
}
function rtCandVisible(c){
  if (rtCandFilter === '') return true;
  if (rtCandFilter === '__accepted__') return !!c.returned;
  if (rtCandFilter === '__labeled__') return !!(c.feedback && c.feedback.verdict);
  return (c.terminal_reason || 'unknown') === rtCandFilter;
}
function renderRetrievalEvent(){
  var e = rtEvent, target = document.getElementById('rt-detail');
  if (!e || !target) return;
  var f=e.funnel||{}, reasons=f.terminal_reasons||{}, max=Math.max(1,...Object.values(reasons),1);
  var h='<div class="detail-section-title">'+esc(e.purpose)+' · '+esc(e.outcome)+'</div><div class="cid">'+esc(e.id)+'</div><div class="detail-content">'+esc(e.query_text||('[private query '+e.query_sha256+']'))+'</div><div style="margin:10px 0">';
  Object.keys(reasons).forEach(k=>{h+='<div>'+esc(k)+' '+reasons[k]+'<div class="rt-bar" style="width:'+Math.round(100*reasons[k]/max)+'%"></div></div>';}); h+='</div>';
  var fb=e.feedback||{};
  h+='<div class="rt-controls"><select id="rt-verdict">'+['useful','mixed','noisy','missed','unsure'].map(v=>rtOpt(v,v,fb.verdict||'useful')).join('')+'</select>';
  h+='<input id="rt-missing" placeholder="expected missing IDs (comma-separated)" style="flex:1" value="'+escA((fb.expected_missing_ids||[]).join(', '))+'">';
  h+='<input id="rt-note" placeholder="note" value="'+escA(fb.note||'')+'">';
  h+='<button class="mbtn" onclick="saveRetrievalFeedback()">Save label</button>';
  if (fb.updated_at) h+='<span class="rt-label">labeled '+esc(fmtTime(fb.updated_at))+'</span>';
  h+='</div>';
  var cands=e.candidates||[];
  var accepted=cands.filter(c=>c.returned);
  var labeled=cands.filter(c=>c.feedback&&c.feedback.verdict).length;
  var scored=cands.filter(c=>c.raw_bge_logit!=null).length;
  h+='<div class="rt-controls"><button class="mbtn" onclick="openRtOverlay()">⤢ Full candidate view</button><span class="rt-label">'+cands.length+' stored · '+accepted.length+' returned · '+scored+' BGE-scored · '+labeled+' labeled</span></div>';
  h+='<div class="detail-section-title">Returned candidates (final rank order)</div>';
  h+='<table class="rt-table"><tr><th>Rank</th><th>Candidate</th><th>cosine</th><th>raw logit</th><th>blend</th></tr>';
  accepted.slice().sort(rtRankedCmp).forEach(c=>{h+='<tr><td>'+esc(String(c.vector_rank||''))+'</td><td>'+esc(c.schema_name+':'+c.doc_id)+'</td><td>'+score(c.similarity)+'</td><td>'+score(c.raw_bge_logit)+'</td><td>'+score(c.blended_score)+'</td></tr>';});
  h+='</table>';
  target.innerHTML=h;
}

function openRtOverlay(){
  if(!rtEvent) return;
  rtOverlayOpen = true;
  if(!document.getElementById('rt-overlay')){
    var bd=document.createElement('div'); bd.id='rt-overlay-backdrop'; bd.className='rt-overlay-backdrop'; bd.onclick=closeRtOverlay; document.body.appendChild(bd);
    var ov=document.createElement('div'); ov.id='rt-overlay'; ov.className='rt-overlay'; document.body.appendChild(ov);
  }
  if(rtDocs===null){
    document.getElementById('rt-overlay').innerHTML='<div class="empty">Loading candidate contents...</div>';
    fetch('/api/retrieval/events/'+encodeURIComponent(rtEvent.id)+'/documents?preview_chars=240')
      .then(r=>r.json()).then(d=>{rtDocs=d;renderRtOverlay();})
      .catch(()=>{rtDocs={};renderRtOverlay();});
  } else renderRtOverlay();
}
function closeRtOverlay(){
  rtOverlayOpen=false;
  var ov=document.getElementById('rt-overlay'); if(ov)ov.remove();
  var bd=document.getElementById('rt-overlay-backdrop'); if(bd)bd.remove();
}
document.addEventListener('keydown', function(ev){ if(ev.key==='Escape') closeRtOverlay(); });
function renderRtOverlay(){
  var ov=document.getElementById('rt-overlay'); var e=rtEvent;
  if(!ov||!e) return;
  var cands=e.candidates||[];
  var byReason={}; cands.forEach(c=>{var k=c.terminal_reason||'unknown';byReason[k]=(byReason[k]||0)+1;});
  var accepted=cands.filter(c=>c.returned).length;
  var labeled=cands.filter(c=>c.feedback&&c.feedback.verdict).length;
  var rows=cands.filter(rtCandVisible);
  var h='<button class="mbtn rt-close" onclick="closeRtOverlay()">✕ Close (Esc)</button>';
  h+='<div class="detail-section-title">'+esc(e.purpose)+' · '+esc(e.outcome)+' · '+esc(fmtTime(e.created_at))+'</div>';
  h+='<div class="detail-content" style="margin-bottom:10px">'+esc(e.query_text||('[private query '+e.query_sha256+']'))+'</div>';
  h+='<div class="rt-controls"><select onchange="rtCandFilter=this.value;renderRtOverlay()">';
  h+=rtOpt('','All candidates ('+cands.length+')',rtCandFilter);
  h+=rtOpt('__accepted__','Accepted / returned ('+accepted+')',rtCandFilter);
  h+=rtOpt('__labeled__','Labeled ('+labeled+')',rtCandFilter);
  Object.keys(byReason).sort().forEach(k=>{h+=rtOpt(k,k+' ('+byReason[k]+')',rtCandFilter);});
  h+='</select><button class="mbtn" onclick="rtResetSort()"'+(rtSortKey==='__ranked__'?' style="background:var(--badge)"':'')+'>Ranked order</button><span class="rt-label">showing '+rows.length+' of '+cands.length+' · click a row for full content · click headers to sort</span></div>';
  h+='<table class="rt-table"><tr>'+rtTh('vector_rank','Rank')+rtTh('candidate','Candidate')+'<th>Content</th>'+rtTh('similarity','cosine')+rtTh('raw_bge_logit','raw logit')+rtTh('blended_score','blend')+rtTh('terminal_reason','Reason')+rtTh('returned','Ret')+rtTh('delivered','Del')+'<th>Label</th></tr>';
  rtSortRows(rows).forEach(c=>{
    var lab=(c.feedback&&c.feedback.verdict)||'';
    var d=(rtDocs||{})[c.candidate_key];
    var prev = !d ? '…' : (d.found ? esc(d.text||'')+(d.truncated?'…':'') : '<i>body not mirrored locally</i>');
    h+='<tr class="rt-row-main" data-key="'+escA(c.candidate_key)+'" onclick="rtToggleDoc(event,this)"'+(c.returned?' style="background:var(--surface2)"':'')+'>';
    h+='<td>'+esc(String(c.vector_rank||''))+'</td><td>'+esc(c.schema_name+':'+c.doc_id)+'</td>';
    h+='<td class="rt-preview">'+prev+'</td>';
    h+='<td>'+score(c.similarity)+'</td><td>'+score(c.raw_bge_logit)+'</td><td>'+score(c.blended_score)+'</td><td>'+esc(c.terminal_reason||'')+'</td><td>'+(c.returned?'✓':'·')+'</td><td>'+(c.delivered?'✓':'·')+'</td>';
    h+='<td><select data-key="'+escA(c.candidate_key)+'" onchange="saveCandidateFeedback(this)"><option value="">—</option>'+['relevant','irrelevant','unsure'].map(v=>rtOpt(v,v,lab)).join('')+'</select></td></tr>';
    h+='<tr id="rt-doc-'+escA(c.candidate_key)+'" style="display:none"><td colspan="10"><div class="rt-doc"></div></td></tr>';
  });
  h+='</table>';
  ov.innerHTML=h;
}
function rtToggleDoc(ev, tr){
  if (ev.target.closest('select,option,button,input,a')) return;
  var key=tr.dataset.key;
  var docRow=document.getElementById('rt-doc-'+key);
  if(!docRow) return;
  if(docRow.style.display!=='none'){ docRow.style.display='none'; return; }
  docRow.style.display='';
  var box=docRow.querySelector('.rt-doc');
  var cached=(rtDocs||{})[key];
  if(cached && cached.full!=null){ box.textContent=cached.full; return; }
  if(cached && cached.found===false){ box.textContent='Body not mirrored locally (remote schema).'; return; }
  box.textContent='Loading full content...';
  fetch('/api/retrieval/events/'+encodeURIComponent(rtEvent.id)+'/documents?candidate_key='+encodeURIComponent(key))
    .then(r=>r.json())
    .then(d=>{ var item=d[key]||{}; if(rtDocs&&rtDocs[key])rtDocs[key].full=item.text; box.textContent=item.found?(item.text||''):'Body not mirrored locally (remote schema).'; })
    .catch(err=>{ box.textContent='Failed to load: '+err; });
}
function score(v){return v==null?'—':Number(v).toFixed(3);}
function saveRetrievalFeedback(){
  if(!rtEvent)return;
  var payload={verdict:document.getElementById('rt-verdict').value,expected_missing_ids:document.getElementById('rt-missing').value.split(',').map(x=>x.trim()).filter(Boolean),note:document.getElementById('rt-note').value};
  fetch('/api/retrieval/events/'+encodeURIComponent(rtEvent.id)+'/feedback',{method:'PUT',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)})
    .then(r=>{if(!r.ok)throw Error('HTTP '+r.status);loadRetrievalEvent(rtEvent.id);})
    .catch(err=>alert('Label save failed: '+err));
}
function saveCandidateFeedback(sel){
  if(!sel.value||!rtEvent)return;
  var key=sel.dataset.key;
  fetch('/api/retrieval/events/'+encodeURIComponent(rtEvent.id)+'/candidates/'+encodeURIComponent(key)+'/feedback',{method:'PUT',headers:{'Content-Type':'application/json'},body:JSON.stringify({verdict:sel.value})})
    .then(r=>{
      if(!r.ok)throw Error('HTTP '+r.status);
      sel.style.borderColor='var(--green)';
      (rtEvent.candidates||[]).forEach(c=>{if(c.candidate_key===key)c.feedback={verdict:sel.value};});
    })
    .catch(err=>{sel.style.borderColor='var(--red)';alert('Label save failed: '+err);});
}
function runSimulation(){var payload={policy:document.getElementById('sim-policy').value,cosine_threshold:Number(document.getElementById('sim-cos').value),bge_logit_threshold:Number(document.getElementById('sim-bge').value),contextual_augmentation:document.getElementById('sim-era').value};fetch('/api/retrieval/simulate',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)}).then(r=>r.json()).then(x=>{var warn=x.augmentation_eras_mixed?'\\n\\nWARNING: this pool mixes augmentation eras ('+JSON.stringify(x.augmentation_eras)+'). Pick one era before trusting a threshold.':'';document.getElementById('sim-result').textContent=JSON.stringify(x,null,2)+warn+'\\n\\nCopy config snippet: '+JSON.stringify(x.config_snippet);});}

var adminLoaded = false;
var adminAutoRefresh = null;

function loadAdmin() {
  var el = document.getElementById('tab-admin');
  // Only show full loading on first load; subsequent refreshes keep content visible
  if (!adminLoaded) {
    el.innerHTML = '<div class="empty" style="padding:40px">Loading admin data...</div>';
  }
  fetch('/api/admin').then(r => r.json()).then(renderAdmin).catch(err => {
    el.innerHTML = '<div class="empty" style="color:var(--red)">Failed: ' + esc(String(err)) + '</div>';
  });
}

function toggleAutoRefresh() {
  if (adminAutoRefresh) {
    clearInterval(adminAutoRefresh);
    adminAutoRefresh = null;
  } else {
    adminAutoRefresh = setInterval(loadAdmin, 10000);
  }
  // Update button state
  var btn = document.getElementById('admin-auto-refresh');
  if (btn) btn.classList.toggle('active', !!adminAutoRefresh);
}

function renderAdmin(d) {
  adminLoaded = true;
  var el = document.getElementById('tab-admin');
  var h = '';

  /* ── Auth bar (hidden — auth handled server-side) ── */

  /* ── Toolbar ──────────────────────────────────────── */
  var autoClass = adminAutoRefresh ? 'active' : '';
  h += '<div style="display:flex;gap:8px;align-items:center;margin-bottom:16px;justify-content:flex-end">';
  h += '<button onclick="loadAdmin()" style="padding:4px 12px;background:var(--bg2);border:1px solid var(--border);border-radius:4px;color:var(--fg);cursor:pointer;font-size:13px">Refresh</button>';
  h += '<button id="admin-auto-refresh" onclick="toggleAutoRefresh()" class="' + autoClass + '" style="padding:4px 12px;background:' + (adminAutoRefresh ? 'var(--green)' : 'var(--bg2)') + ';border:1px solid var(--border);border-radius:4px;color:' + (adminAutoRefresh ? '#000' : 'var(--fg)') + ';cursor:pointer;font-size:13px">Auto 10s</button>';
  h += '</div>';

  /* ── Overview cards ──────────────────────────────── */
  h += '<div class="admin-grid">';

  // Sync status card
  var syncPill = d.sync.enabled
    ? '<span class="pill pill-green">ENABLED</span>'
    : '<span class="pill pill-red">DISABLED</span>';
  h += '<div class="admin-card"><h3>Sync Engine</h3>';
  h += '<div style="margin-bottom:10px">' + syncPill + '</div>';
  h += '<div class="kv"><span class="k">Strategy</span><span class="v">' + esc(d.sync.strategy) + '</span></div>';
  h += '<div class="kv"><span class="k">Push interval</span><span class="v">' + d.sync.worker_interval + 's</span></div>';
  h += '<div class="kv"><span class="k">Pull interval</span><span class="v">' + d.sync.pull_interval + 's</span></div>';
  h += '</div>';

  // Last activity card
  h += '<div class="admin-card"><h3>Last Activity</h3>';
  if (d.last_push) {
    h += '<div class="kv"><span class="k">Last push</span><span class="v"><span class="pill pill-green">OK</span> ' + fmtTime(d.last_push.at) + '</span></div>';
    h += '<div class="kv"><span class="k">Destination</span><span class="v">' + esc(d.last_push.destination) + '</span></div>';
  } else {
    h += '<div class="kv"><span class="k">Last push</span><span class="v ts">No syncs yet</span></div>';
  }
  if (d.last_fail) {
    h += '<div class="kv"><span class="k">Last failure</span><span class="v"><span class="pill pill-red">FAIL</span> ' + fmtTime(d.last_fail.at) + '</span></div>';
    h += '<div class="kv"><span class="k">Error</span><span class="v" style="font-size:11px;color:var(--red);max-width:200px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">' + esc(d.last_fail.error || '') + '</span></div>';
  }
  h += '</div>';

  // Memory stats card
  h += '<div class="admin-card"><h3>Memory Stats</h3>';
  if (d.memory_stats && d.memory_stats.total != null) {
    h += '<div style="margin-bottom:10px"><span class="val">' + d.memory_stats.total.toLocaleString() + '</span> <span class="lbl">total memories</span></div>';
    if (d.memory_stats.by_status) {
      Object.entries(d.memory_stats.by_status).forEach(function(e) {
        var pill = e[0] === 'active' ? 'pill-green' : e[0] === 'deleted' ? 'pill-red' : 'pill-orange';
        h += '<div class="kv"><span class="k"><span class="pill ' + pill + '">' + esc(e[0]) + '</span></span><span class="v">' + e[1] + '</span></div>';
      });
    }
  }
  h += '</div>';

  // Category breakdown card
  h += '<div class="admin-card"><h3>Active by Category</h3>';
  if (d.memory_stats && d.memory_stats.by_category) {
    h += '<div style="margin-bottom:8px"><span class="val">' + (d.memory_stats.total_active || 0).toLocaleString() + '</span> <span class="lbl">active</span></div>';
    Object.entries(d.memory_stats.by_category).forEach(function(e) {
      h += '<div class="kv"><span class="k">' + esc(e[0]) + '</span><span class="v">' + e[1] + '</span></div>';
    });
  }
  h += '</div>';

  h += '</div>'; // admin-grid

  /* ── Remotes ──────────────────────────────── */
  h += '<div class="admin-section"><h2>Remotes <button class="btn-sm" onclick="showRemoteForm(null)">+ Add</button></h2>';
  h += '<div id="remote-form-area"></div>';
  if (d.remotes.length === 0) {
    h += '<div class="empty" style="padding:10px">No remotes configured</div>';
  } else {
    h += '<table class="admin-table"><tr><th>Name</th><th>Schema</th><th>Auth</th><th>Status</th><th>Actions</th></tr>';
    d.remotes.forEach(function(r) {
      var st = r.connected
        ? '<span class="pill pill-green">Connected</span>'
        : '<span class="pill pill-red">Disconnected</span>';
      if (!r.enabled) st = '<span class="pill pill-muted">Disabled</span>';
      h += '<tr><td><strong>' + esc(r.name) + '</strong>';
      if (r.description) h += '<br><span class="ts">' + esc(r.description) + '</span>';
      h += '</td><td>' + esc(r.schema_name || r.schema || r.name) + '</td>';
      h += '<td><span class="pill pill-blue">' + esc(r.auth_method) + '</span></td><td>' + st + '</td>';
      h += '<td style="white-space:nowrap">';
      h += '<button class="btn-sm" onclick="showRemoteForm(\\'' + esc(r.name) + '\\')">Edit</button> ';
      h += '<button class="btn-sm" onclick="testRemote(\\'' + esc(r.name) + '\\', this)">Test</button> ';
      h += '<button class="btn-sm btn-danger" onclick="deleteRemote(\\'' + esc(r.name) + '\\')">Delete</button>';
      h += '</td></tr>';
    });
    h += '</table>';
  }
  h += '</div>';

  /* ── Queue stats ──────────────────────────────── */
  h += '<div class="admin-section"><h2>Sync Queue</h2>';
  var hasQueue = Object.keys(d.queue).length > 0;
  if (!hasQueue) {
    h += '<div class="stat-row">';
    h += '<div class="stat-box"><div class="num" style="color:var(--green)">0</div><div class="slbl">Queue Empty</div></div>';
    h += '</div>';
  } else {
    Object.entries(d.queue).forEach(function(e) {
      var dest = e[0], stats = e[1];
      h += '<div style="margin-bottom:8px"><strong>' + esc(dest) + '</strong></div>';
      h += '<div class="stat-row">';
      ['pending','sending','done','dlq'].forEach(function(s) {
        var n = stats[s] || 0;
        var color = s === 'dlq' && n > 0 ? 'var(--red)' : s === 'done' ? 'var(--green)' : s === 'pending' ? 'var(--orange)' : 'var(--text)';
        h += '<div class="stat-box"><div class="num" style="color:' + color + '">' + n + '</div><div class="slbl">' + s + '</div></div>';
      });
      h += '</div>';
    });
  }
  h += '</div>';

  /* ── DLQ entries ──────────────────────────────── */
  if (d.dlq.length > 0) {
    h += '<div class="admin-section"><h2>Dead Letter Queue (' + d.dlq.length + ')</h2>';
    h += '<table class="admin-table"><tr><th>ID</th><th>Memory</th><th>Dest</th><th>Attempts</th><th>Error</th><th>Last Attempt</th></tr>';
    d.dlq.forEach(function(e) {
      h += '<tr><td>' + e.id + '</td>';
      h += '<td style="font-size:11px;max-width:150px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">' + esc(e.memory_id || '') + '</td>';
      h += '<td>' + esc(e.destination) + '</td>';
      h += '<td>' + e.attempts + '</td>';
      h += '<td class="err" title="' + esc(e.error || '') + '">' + esc(e.error || '') + '</td>';
      h += '<td class="ts">' + fmtTime(e.last_attempt) + '</td></tr>';
    });
    h += '</table></div>';
  }

  /* ── Routing rules ──────────────────────────────── */
  h += '<div class="admin-section"><h2>Routing Rules <button class="btn-sm" onclick="showRuleForm(null)">+ Add</button></h2>';
  h += '<div id="rule-form-area"></div>';
  if (d.rules.length === 0) {
    h += '<div class="empty" style="padding:10px">No routing rules configured</div>';
  } else {
    h += '<table class="admin-table"><tr><th>#</th><th>Name</th><th>Action</th><th>Match</th><th>Destinations</th><th>Actions</th></tr>';
    d.rules.forEach(function(r, i) {
      var actionPill = r.action === 'deny'
        ? '<span class="pill pill-red">' + esc(r.action) + '</span>'
        : '<span class="pill pill-green">' + esc(r.action) + '</span>';
      var matchHtml = '';
      Object.entries(r.match).forEach(function(m) {
        var vals = Array.isArray(m[1]) ? m[1] : [m[1]];
        matchHtml += '<span class="match-chip">' + esc(m[0]) + ': ' + vals.map(esc).join(', ') + '</span> ';
      });
      h += '<tr><td>' + (i + 1) + '</td><td><strong>' + esc(r.name) + '</strong></td>';
      h += '<td>' + actionPill + '</td>';
      h += '<td>' + (matchHtml || '<span class="ts">any</span>') + '</td>';
      h += '<td>' + r.destinations.map(function(dd) { return '<span class="match-chip">' + esc(dd) + '</span>'; }).join(' ') + '</td>';
      h += '<td style="white-space:nowrap">';
      if (i > 0) h += '<button class="btn-sm" onclick="moveRule(\\'' + esc(r.name) + '\\',-1)">&#9650;</button> ';
      if (i < d.rules.length - 1) h += '<button class="btn-sm" onclick="moveRule(\\'' + esc(r.name) + '\\',1)">&#9660;</button> ';
      h += '<button class="btn-sm" onclick="showRuleForm(\\'' + esc(r.name) + '\\')">Edit</button> ';
      h += '<button class="btn-sm btn-danger" onclick="deleteRule(\\'' + esc(r.name) + '\\')">Delete</button>';
      h += '</td></tr>';
    });
    h += '</table>';
  }

  /* Rule tester */
  h += '<div class="rule-tester">';
  h += '<h4 onclick="document.getElementById(\\'rule-tester-body\\').classList.toggle(\\'show\\')">Rule Tester (click to expand)</h4>';
  h += '<div id="rule-tester-body" class="rule-tester-body">';
  h += '<div class="form-row"><label>Category</label><select id="rt-cat"><option value="observation">observation</option><option value="pattern">pattern</option><option value="learning">learning</option><option value="decision">decision</option><option value="summary">summary</option><option value="code">code</option><option value="relationship">relationship</option><option value="hint">hint</option><option value="plan">plan</option><option value="worklog">worklog</option><option value="memory">memory</option></select></div>';
  h += '<div class="form-row"><label>Project</label><input id="rt-project" placeholder="e.g. jarvis-plugin"></div>';
  h += '<div class="form-row"><label>Importance</label><input id="rt-importance" type="number" step="0.1" min="0" max="1" value="0.5"></div>';
  h += '<div class="form-row"><label>Tags</label><input id="rt-tags" placeholder="comma-separated"></div>';
  h += '<div class="form-actions"><button class="btn-save" onclick="testRouting()">Test Rules</button></div>';
  h += '<div id="rt-result"></div>';
  h += '</div></div>';

  h += '</div>';

  /* ── Recent activity ──────────────────────────────── */
  if (d.recent_activity.length > 0) {
    h += '<div class="admin-section"><h2>Recent Sync Activity</h2>';
    h += '<table class="admin-table"><tr><th>Time</th><th>Memory</th><th>Dest</th><th>Rule</th><th>Status</th></tr>';
    d.recent_activity.forEach(function(a) {
      var stPill = a.status === 'done' ? '<span class="pill pill-green">done</span>'
        : a.status === 'dlq' ? '<span class="pill pill-red">dlq</span>'
        : a.status === 'pending' ? '<span class="pill pill-orange">pending</span>'
        : a.status === 'sending' ? '<span class="pill pill-blue">sending</span>'
        : '<span class="pill pill-blue">' + esc(a.status) + '</span>';
      var catPill = a.category ? '<span class="pill pill-blue" style="font-size:10px">' + esc(a.category) + '</span> ' : '';
      var preview = a.preview ? esc(a.preview) : '<span class="ts">' + esc(a.memory_id || '?') + '</span>';
      var projTag = a.project ? ' <span class="ts">(' + esc(a.project) + ')</span>' : '';
      var rules = (a.matched_rules || []).map(function(r) { return '<span class="match-chip">' + esc(r) + '</span>'; }).join(' ');
      if (!rules) rules = '<span class="ts">-</span>';
      h += '<tr><td class="ts" style="white-space:nowrap">' + fmtTime(a.at) + '</td>';
      h += '<td style="max-width:350px">' + catPill + preview + projTag + '</td>';
      h += '<td>' + esc(a.destination) + '</td>';
      h += '<td>' + rules + '</td>';
      h += '<td>' + stPill + '</td></tr>';
    });
    h += '</table></div>';
  }

  el.innerHTML = h;

  // Cache admin data for form population
  window._adminData = d;
}

/* ── Admin fetch helper ───────────────────────────── */
function adminFetch(url, opts) {
  opts = opts || {};
  opts.headers = opts.headers || {};
  if (opts.body && typeof opts.body === 'object') {
    opts.headers['Content-Type'] = 'application/json';
    opts.body = JSON.stringify(opts.body);
  }
  return fetch(url, opts).then(function(r) {
    if (!r.ok) return r.json().catch(function() { return r.text(); }).then(function(d) {
      var msg = typeof d === 'object' ? (d.detail ? (typeof d.detail === 'object' ? JSON.stringify(d.detail) : d.detail) : JSON.stringify(d)) : d;
      return Promise.reject(msg);
    });
    return r.json();
  });
}

/* ── Remote CRUD ──────────────────────────────── */
function showRemoteForm(name) {
  var area = document.getElementById('remote-form-area');
  var d = window._adminData;
  var remote = null;
  if (name && d) {
    d.remotes.forEach(function(r) { if (r.name === name) remote = r; });
  }
  var h = '<div class="admin-form"><h4>' + (name ? 'Edit Remote: ' + esc(name) : 'Add Remote') + '</h4>';
  h += '<div id="remote-form-error"></div>';
  if (!name) h += '<div class="form-row"><label title="Unique lowercase identifier (a-z, 0-9, underscores)">Name</label><input id="rf-name" placeholder="my_remote" value=""></div>';
  h += '<div class="form-row"><label title="PostgreSQL server hostname or IP address">Host</label><input id="rf-host" value="' + esc(remote ? remote.host || 'localhost' : 'localhost') + '"></div>';
  h += '<div class="form-row"><label title="PostgreSQL server port">Port</label><input id="rf-port" type="number" value="' + (remote ? remote.port || 5432 : 5432) + '"></div>';
  h += '<div class="form-row"><label title="PostgreSQL database name">Database</label><input id="rf-db" value="' + esc(remote ? remote.database || 'jarvis' : 'jarvis') + '"></div>';
  h += '<div class="form-row"><label title="PostgreSQL username for authentication">User</label><input id="rf-user" value="' + esc(remote ? remote.user || 'jarvis' : 'jarvis') + '"></div>';
  h += '<div class="form-row"><label title="Literal password or $ENV_VAR reference (e.g. $AURORA_PASSWORD). Leave empty on edit to keep existing.">Password</label><input id="rf-pw" type="password" placeholder="' + (name ? '*** (unchanged)' : 'password or $ENV_VAR') + '" value=""></div>';
  h += '<div class="form-row"><label title="password = standard PostgreSQL auth. iam = AWS IAM database authentication.">Auth Method</label><select id="rf-auth"><option' + (remote && remote.auth_method === 'password' ? ' selected' : '') + '>password</option><option' + (remote && remote.auth_method === 'iam' ? ' selected' : '') + '>iam</option></select></div>';
  h += '<div class="form-row"><label title="PostgreSQL schema for this remote (isolates memories per remote). Defaults to the remote name.">Schema</label><input id="rf-schema" placeholder="defaults to remote name" value="' + esc(remote ? remote.schema_name || '' : '') + '"></div>';
  h += '<div class="form-row"><label title="SSL connection mode: require (encrypted), verify-full (encrypted + cert verification), disable (plaintext)">SSL Mode</label><select id="rf-ssl"><option' + (remote && remote.sslmode === 'require' ? ' selected' : '') + '>require</option><option' + (remote && remote.sslmode === 'verify-full' ? ' selected' : '') + '>verify-full</option><option' + (remote && remote.sslmode === 'disable' ? ' selected' : '') + '>disable</option></select></div>';
  h += '<div class="form-row"><label title="Disabled remotes are skipped during sync">Enabled</label><input id="rf-enabled" type="checkbox"' + (remote ? (remote.enabled !== false ? ' checked' : '') : ' checked') + '></div>';
  h += '<div class="form-row"><label title="Human-readable description for this remote">Description</label><input id="rf-desc" value="' + esc(remote ? remote.description || '' : '') + '"></div>';
  h += '<div class="form-actions"><button class="btn-save" onclick="saveRemote(' + (name ? "'" + esc(name) + "'" : 'null') + ')">Save</button>';
  h += '<button class="btn-cancel" onclick="document.getElementById(\\'remote-form-area\\').innerHTML=\\'\\'">Cancel</button></div>';
  h += '</div>';
  area.innerHTML = h;
}

function saveRemote(existingName) {
  var name = existingName || (document.getElementById('rf-name') ? document.getElementById('rf-name').value : '');
  var pw = document.getElementById('rf-pw').value;
  var body = {
    name: existingName ? undefined : name,
    host: document.getElementById('rf-host').value,
    port: parseInt(document.getElementById('rf-port').value) || 5432,
    database: document.getElementById('rf-db').value,
    user: document.getElementById('rf-user').value,
    password: pw || (existingName ? '***' : null),
    auth_method: document.getElementById('rf-auth').value,
    schema_name: document.getElementById('rf-schema').value || null,
    sslmode: document.getElementById('rf-ssl').value,
    enabled: document.getElementById('rf-enabled').checked,
    description: document.getElementById('rf-desc').value,
  };
  var url = existingName ? '/api/admin/remotes/' + encodeURIComponent(existingName) : '/api/admin/remotes';
  var method = existingName ? 'PUT' : 'POST';
  adminFetch(url, { method: method, body: body }).then(function() {
    document.getElementById('remote-form-area').innerHTML = '<div class="admin-success">Remote saved.</div>';
    setTimeout(loadAdmin, 500);
  }).catch(function(err) {
    var el = document.getElementById('remote-form-error');
    if (el) el.innerHTML = '<div class="admin-error">' + esc(String(err)) + '</div>';
  });
}

function deleteRemote(name) {
  if (!confirm('Delete remote "' + name + '"?')) return;
  adminFetch('/api/admin/remotes/' + encodeURIComponent(name), { method: 'DELETE' }).then(function() {
    loadAdmin();
  }).catch(function(err) {
    alert('Cannot delete: ' + err);
  });
}

function testRemote(name, btn) {
  var orig = btn.textContent;
  btn.textContent = '...';
  btn.disabled = true;
  // Remove any existing result pill next to button
  var existing = btn.parentNode.querySelector('.test-result');
  if (existing) existing.remove();
  adminFetch('/api/admin/remotes/' + encodeURIComponent(name) + '/test', { method: 'POST' }).then(function(d) {
    var pill = document.createElement('span');
    pill.className = 'test-result ' + (d.connected ? 'ok' : 'fail');
    pill.textContent = d.connected ? 'OK' : (d.error || 'Failed');
    btn.parentNode.appendChild(pill);
  }).catch(function(err) {
    var pill = document.createElement('span');
    pill.className = 'test-result fail';
    pill.textContent = String(err).substring(0, 40);
    btn.parentNode.appendChild(pill);
  }).finally(function() {
    btn.textContent = orig;
    btn.disabled = false;
  });
}

/* ── Rule CRUD ──────────────────────────────── */
function showRuleForm(name) {
  var area = document.getElementById('rule-form-area');
  var d = window._adminData;
  var rule = null;
  if (name && d) {
    d.rules.forEach(function(r) { if (r.name === name) rule = r; });
  }
  var match = rule ? rule.match || {} : {};
  var cats = ['observation','pattern','learning','decision','summary','code','relationship','hint','plan','worklog','memory'];
  var remoteNames = d && d.remotes ? d.remotes.map(function(r) { return r.name; }) : [];
  var selDests = rule ? (rule.destinations || []) : [];
  var h = '<div class="admin-form"><h4>' + (name ? 'Edit Rule: ' + esc(name) : 'Add Rule') + '</h4>';
  h += '<div id="rule-form-error"></div>';
  h += '<div class="form-row"><label title="Unique identifier for this rule (used in URLs and logs)">Name</label><input id="rlf-name" value="' + esc(rule ? rule.name : '') + '"' + (name ? ' readonly style="opacity:0.6"' : '') + ' placeholder="e.g. sync-observations"></div>';
  h += '<div class="form-row"><label title="route-to: send matching memories to destinations. deny: block matching memories from being synced.">Action</label><select id="rlf-action"><option value="route-to"' + (rule && rule.action === 'route-to' ? ' selected' : '') + '>route-to</option><option value="deny"' + (rule && rule.action === 'deny' ? ' selected' : '') + '>deny</option></select></div>';
  h += '<div class="form-row"><label title="Which remote databases matching memories will be synced to (required for route-to rules)">Destinations</label>';
  if (remoteNames.length) {
    h += '<div id="rlf-dest-list" style="display:flex;flex-wrap:wrap;gap:6px">';
    remoteNames.forEach(function(rn) {
      var checked = selDests.indexOf(rn) >= 0 ? ' checked' : '';
      h += '<label style="display:flex;align-items:center;gap:4px;font-size:12px;cursor:pointer"><input type="checkbox" class="rlf-dest-cb" value="' + esc(rn) + '"' + checked + '>' + esc(rn) + '</label>';
    });
    h += '</div>';
  } else {
    h += '<input id="rlf-dest" placeholder="no remotes configured" disabled>';
  }
  h += '</div>';
  h += '<div class="form-row"><label title="Only sync memories of these content types (empty = match any category)">Categories</label><input id="rlf-cat" placeholder="empty = any" value="' + esc((match.category || []).join(', ')) + '"></div>';
  h += '<div class="form-row"><label title="Glob patterns (e.g. work-*) or @group references (e.g. @work) to match memory project field">Projects</label><input id="rlf-proj" placeholder="e.g. work-*, @work" value="' + esc((match.project || []).join(', ')) + '"></div>';
  h += '<div class="form-row"><label title="Only match memories that have ALL of these tags">Tags</label><input id="rlf-tags" placeholder="comma-separated" value="' + esc((match.tags || []).join(', ')) + '"></div>';
  h += '<div class="form-row"><label title="Only sync memories with importance score >= this value (0.0 to 1.0)">Min Importance</label><input id="rlf-imp" type="number" step="0.1" min="0" max="1" value="' + (match.importance_min != null ? match.importance_min : '') + '" placeholder="0.0-1.0"></div>';
  h += '<div class="form-row"><label title="Only match memories whose vault path starts with one of these prefixes">Path Prefixes</label><input id="rlf-path" placeholder="e.g. work/, notes/" value="' + esc((match.path_prefix || []).join(', ')) + '"></div>';
  h += '<div class="form-actions"><button class="btn-save" onclick="saveRule(' + (name ? "'" + esc(name) + "'" : 'null') + ')">Save</button>';
  h += '<button class="btn-cancel" onclick="document.getElementById(\\'rule-form-area\\').innerHTML=\\'\\'">Cancel</button></div>';
  h += '</div>';
  area.innerHTML = h;
}

function _csvToList(s) { return s ? s.split(',').map(function(x) { return x.trim(); }).filter(Boolean) : []; }

function saveRule(existingName) {
  var rname = document.getElementById('rlf-name').value.trim();
  if (!rname) { document.getElementById('rule-form-error').innerHTML = '<div class="admin-error">Name is required</div>'; return; }
  var match = {};
  var cats = _csvToList(document.getElementById('rlf-cat').value);
  var projs = _csvToList(document.getElementById('rlf-proj').value);
  var tags = _csvToList(document.getElementById('rlf-tags').value);
  var imp = document.getElementById('rlf-imp').value;
  var paths = _csvToList(document.getElementById('rlf-path').value);
  if (cats.length) match.category = cats;
  if (projs.length) match.project = projs;
  if (tags.length) match.tags = tags;
  if (imp !== '') match.importance_min = parseFloat(imp);
  if (paths.length) match.path_prefix = paths;
  var body = {
    name: rname,
    action: document.getElementById('rlf-action').value,
    destinations: document.getElementById('rlf-dest-list')
      ? Array.from(document.querySelectorAll('.rlf-dest-cb:checked')).map(function(cb) { return cb.value; })
      : _csvToList((document.getElementById('rlf-dest') || {value:''}).value),
    match: match,
  };
  var url = existingName ? '/api/admin/rules/' + encodeURIComponent(existingName) : '/api/admin/rules';
  var method = existingName ? 'PUT' : 'POST';
  adminFetch(url, { method: method, body: body }).then(function() {
    document.getElementById('rule-form-area').innerHTML = '<div class="admin-success">Rule saved.</div>';
    setTimeout(loadAdmin, 500);
  }).catch(function(err) {
    var el = document.getElementById('rule-form-error');
    if (el) el.innerHTML = '<div class="admin-error">' + esc(String(err)) + '</div>';
  });
}

function deleteRule(name) {
  if (!confirm('Delete rule "' + name + '"?')) return;
  adminFetch('/api/admin/rules/' + encodeURIComponent(name), { method: 'DELETE' }).then(function() {
    loadAdmin();
  }).catch(function(err) {
    alert('Cannot delete: ' + err);
  });
}

function moveRule(name, dir) {
  var d = window._adminData;
  if (!d || !d.rules) return;
  var names = d.rules.map(function(r) { return r.name; });
  var idx = names.indexOf(name);
  if (idx < 0) return;
  var newIdx = idx + dir;
  if (newIdx < 0 || newIdx >= names.length) return;
  // Swap
  var tmp = names[idx];
  names[idx] = names[newIdx];
  names[newIdx] = tmp;
  adminFetch('/api/admin/rules/reorder', { method: 'POST', body: { order: names } }).then(function() {
    loadAdmin();
  }).catch(function(err) {
    alert('Reorder failed: ' + err);
  });
}

/* ── Rule tester ──────────────────────────────── */
function testRouting() {
  var proj = document.getElementById('rt-project').value;
  var body = {
    category: document.getElementById('rt-cat').value,
    project: proj,
    scope: proj ? 'project' : 'global',
    importance_score: parseFloat(document.getElementById('rt-importance').value) || 0.5,
    tags: document.getElementById('rt-tags').value,
  };
  var el = document.getElementById('rt-result');
  el.innerHTML = '<div class="test-results" style="color:var(--muted)">Testing...</div>';
  fetch('/api/admin/rules/test', { method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify(body) })
    .then(function(r) { return r.json(); })
    .then(function(d) {
      var h = '<div class="test-results">';
      if (d.destinations.length > 0) {
        h += '<div style="margin-bottom:6px"><strong>Destinations:</strong> ' + d.destinations.map(function(x) { return '<span class="pill pill-green">' + esc(x) + '</span>'; }).join(' ') + '</div>';
      } else {
        h += '<div style="margin-bottom:6px"><strong>Destinations:</strong> <span class="pill pill-muted">none (local only)</span></div>';
      }
      if (d.matched_rules.length > 0) {
        h += '<div style="margin-bottom:6px"><strong>Matched rules:</strong> ' + d.matched_rules.map(function(x) { return '<span class="match-chip">' + esc(x) + '</span>'; }).join(' ') + '</div>';
      }
      if (d.denied.length > 0) {
        h += '<div><strong>Denied:</strong> ' + d.denied.map(function(x) { return '<span class="pill pill-red">' + esc(x) + '</span>'; }).join(' ') + '</div>';
      }
      h += '</div>';
      el.innerHTML = h;
    })
    .catch(function(err) {
      el.innerHTML = '<div class="admin-error">' + esc(String(err)) + '</div>';
    });
}

function fmtTime(iso) {
  if (!iso) return '<span class="ts">N/A</span>';
  try {
    var d = new Date(iso);
    var now = new Date();
    var diff = (now - d) / 1000;
    if (diff < 60) return Math.floor(diff) + 's ago';
    if (diff < 3600) return Math.floor(diff / 60) + 'm ago';
    if (diff < 86400) return Math.floor(diff / 3600) + 'h ago';
    return d.toLocaleDateString() + ' ' + d.toLocaleTimeString([], {hour:'2-digit',minute:'2-digit'});
  } catch(e) { return esc(iso); }
}
</script>
</body>
</html>"""
