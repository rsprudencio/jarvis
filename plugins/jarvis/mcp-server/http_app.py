"""
Streamable HTTP transport for Jarvis Core MCP Server.

Thin ASGI wrapper around the existing stdio-based server.py,
enabling Docker deployment via uvicorn.

Usage:
    uvicorn http_app:app --host 0.0.0.0 --port 8741
"""

import asyncio
import contextvars
import functools
import json
import logging
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from typing import Any

# Mirror the sys.path setup from server.py so all tool imports resolve
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from jarvis_common.auth import authenticate, current_user
from jarvis_common.mtls import patch_uvicorn_transport
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
from server import server
from system_prompt import _version as _VERSION
from tools.schema import (
    db_available as _db_available,
    display_conninfo,
    is_db_unavailable_error,
    safe_db_error as _safe_err,
)

logger = logging.getLogger("jarvis-core")

# Patch uvicorn to expose transport in ASGI scope (required for mTLS CN extraction)
_mtls_patch_ok = patch_uvicorn_transport()
if os.environ.get("JARVIS_TLS_CA") and not _mtls_patch_ok:
    logger.error("JARVIS_TLS_CA is set but uvicorn transport patch failed — cannot verify client certs")
    sys.exit(1)

session_manager = StreamableHTTPSessionManager(
    app=server,
    stateless=True,
    json_response=True,
)

# Hook clients give up at 2.5s (hook_http_client.DEFAULT_TIMEOUT_SECONDS).
# Answer before they do, so a stalled DB yields a 503 the client can queue
# instead of a timeout it can't tell apart from a lost write.
HOOK_DEADLINE_SECONDS = 2.0
_TELEMETRY_DEADLINE_SECONDS = 10.0
_RETRY_AFTER_SECONDS = 10
MAX_REQUEST_BODY_BYTES = 1024 * 1024
# /mcp carries whole documents (jarvis_store content), so its cap is larger.
MAX_MCP_BODY_BYTES = 16 * 1024 * 1024
# Past a cap the rest of the body is still read (and dropped), up to this
# much more, so the client gets its 413: answering with the body unread makes
# the server close with data pending, which the client sees as a reset or a
# broken pipe — indistinguishable from core being down.
_MAX_DRAIN_BYTES = 16 * 1024 * 1024

# Blocking hook/telemetry work (DB, model host) runs here: never on the event
# loop — one 30s pool wait there froze /health and every hook for the whole
# 2026-09-24 outage — and never on the default executor the background loops
# share. wait_for abandons a stuck call at the deadline, but its thread keeps
# its slot until the pool's own checkout timeout fires; the bound keeps a DB
# outage from spawning threads without limit.
_HOOK_WORKERS = 4
_hook_executor: ThreadPoolExecutor | None = None


def _get_hook_executor() -> ThreadPoolExecutor:
    """Lazily create the hook executor (again after a lifespan shutdown)."""
    global _hook_executor
    if _hook_executor is None:
        _hook_executor = ThreadPoolExecutor(
            max_workers=_HOOK_WORKERS, thread_name_prefix="hook-db"
        )
    return _hook_executor


def _shutdown_hook_executor() -> None:
    """Drop queued hook work; don't wait on threads stuck behind a dead DB."""
    global _hook_executor
    executor, _hook_executor = _hook_executor, None
    if executor is not None:
        executor.shutdown(wait=False, cancel_futures=True)


class _DeadlineExceeded(Exception):
    """A blocking call outlived its deadline (its thread may still be running)."""


class _ClientDisconnected(Exception):
    """The client hung up before sending the whole request body."""


class _BodyTooLarge(Exception):
    """The request body exceeds its cap (``limit``, in bytes)."""

    def __init__(self, limit: int = MAX_REQUEST_BODY_BYTES):
        super().__init__(limit)
        self.limit = limit


async def _run_blocking(fn, *args, deadline: float = HOOK_DEADLINE_SECONDS, **kwargs):
    """Run a sync function on the hook executor, bounded by ``deadline``.

    The caller's contextvars (current_user) are copied into the thread.
    Raises _DeadlineExceeded on timeout; the function's own exceptions
    (including its own TimeoutError) propagate unchanged.
    """
    loop = asyncio.get_running_loop()
    call = functools.partial(contextvars.copy_context().run, fn, *args, **kwargs)
    future = loop.run_in_executor(_get_hook_executor(), call)
    try:
        return await asyncio.wait_for(future, deadline)
    except TimeoutError:
        if future.cancelled():
            raise _DeadlineExceeded() from None
        raise


# Headers worth an access-log line. Everything else — authorization, cookies,
# x-jarvis-internal-token, session ids — never reaches the container log. The
# deny pattern guards against a sensitive name being added to the allowlist.
_ACCESS_LOG_HEADERS = frozenset({
    "user-agent",
    "content-type",
    "content-length",
    "accept",
    "mcp-method",
    "mcp-protocol-version",
})
_SENSITIVE_HEADER_RE = re.compile(r"auth|token|secret|key|cookie|passw|session", re.IGNORECASE)


def _loggable_headers(scope) -> dict[str, str]:
    """Allowlisted request headers for the access log."""
    safe = {}
    for raw_name, raw_value in scope.get("headers", []):
        name = raw_name.decode("latin-1").lower()
        if name in _ACCESS_LOG_HEADERS and not _SENSITIVE_HEADER_RE.search(name):
            safe[name] = raw_value.decode("latin-1")
    return safe


# --- ASGI helpers ---


async def _json_response(send, data: dict, status: int = 200, headers: list | None = None):
    """Send a JSON response."""
    body = json.dumps(data).encode()
    await send(
        {
            "type": "http.response.start",
            "status": status,
            "headers": [[b"content-type", b"application/json"], *(headers or [])],
        }
    )
    await send({"type": "http.response.body", "body": body})


async def _send_unavailable(send, data: dict):
    """503 + Retry-After: nothing was processed, the caller should retry later."""
    await _json_response(
        send,
        data,
        status=503,
        headers=[[b"retry-after", str(_RETRY_AFTER_SECONDS).encode()]],
    )


async def _send_hook_result(send, response):
    """Send a hook result; a retryable one means "not done" and maps to 503."""
    if isinstance(response, dict) and response.get("retryable") is True:
        await _send_unavailable(send, response)
        return
    await _json_response(send, response)


async def _send_hook_error(
    send, exc: Exception, status: int = 500, timeout_response: dict | None = None
):
    """Map a hook failure to 503 (retryable) or ``status`` with sanitized text.

    ``error_kind`` tells the hook client whether the database is the cause
    ("db_unavailable": back off everywhere) or this one request was merely
    slow ("deadline": retry this payload, keep serving the next prompt). A
    deadline miss only counts as a DB outage while the breaker is open; with
    a healthy database it is a slow model host or a busy executor, and the
    endpoint's ``timeout_response`` (if any) is answered instead.
    """
    if isinstance(exc, _DeadlineExceeded):
        if not _db_available():
            await _send_unavailable(send, {
                "success": False, "retryable": True, "error_kind": "db_unavailable",
                "error": "database unavailable (timeout)",
            })
        elif timeout_response is not None:
            await _json_response(send, timeout_response)
        else:
            await _send_unavailable(send, {
                "success": False, "retryable": True, "error_kind": "deadline",
                "error": f"request timed out after {HOOK_DEADLINE_SECONDS:g}s",
            })
    elif is_db_unavailable_error(exc):
        reason = _safe_err(exc)
        if reason.lower().startswith("database unavailable:"):
            reason = reason[len("database unavailable:"):].strip()
        await _send_unavailable(send, {
            "success": False, "retryable": True, "error_kind": "db_unavailable",
            "error": f"database unavailable: {reason}",
        })
    else:
        await _json_response(send, {"success": False, "error": _safe_err(exc)}, status=status)


async def _read_request_body(receive, limit: int = MAX_REQUEST_BODY_BYTES) -> bytes:
    """Read full HTTP request body from ASGI receive channel.

    Raises _ClientDisconnected on http.disconnect — uvicorn returns that
    message without suspending once the client is gone, so looping on it
    pins the event loop at 100% CPU — and _BodyTooLarge past ``limit``, once
    the rest of the body (up to _MAX_DRAIN_BYTES more) has been drained.
    """
    chunks = []
    size = 0
    while True:
        message = await receive()
        if message["type"] == "http.disconnect":
            raise _ClientDisconnected()
        if message["type"] != "http.request":
            continue
        body = message.get("body", b"")
        if body:
            size += len(body)
            if size > limit:
                if size > limit + _MAX_DRAIN_BYTES:
                    raise _BodyTooLarge(limit)
                chunks = None  # draining: keep reading, keep nothing
            else:
                chunks.append(body)
        if not message.get("more_body", False):
            break
    if chunks is None:
        raise _BodyTooLarge(limit)
    return b"".join(chunks)


def _replay_body(body: bytes, receive):
    """An ASGI receive() that yields ``body`` once, then defers to ``receive``."""
    pending = [{"type": "http.request", "body": body, "more_body": False}]

    async def replay():
        if pending:
            return pending.pop()
        return await receive()

    return replay


async def _read_json_body(receive) -> tuple[dict[str, Any] | None, str]:
    """Read and decode JSON body from request.

    Returns:
        (data, error_message). Exactly one is non-empty.
    """
    raw = await _read_request_body(receive)
    if not raw:
        return None, "Request body is required"
    try:
        data = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None, "Malformed JSON body"
    if not isinstance(data, dict):
        return None, "JSON body must be an object"
    return data, ""


async def _send_401(send, message: str):
    """Send a 401 Unauthorized response with WWW-Authenticate header (RFC 7235)."""
    body = json.dumps({"error": message}).encode()
    await send(
        {
            "type": "http.response.start",
            "status": 401,
            "headers": [
                [b"content-type", b"application/json"],
                [b"www-authenticate", b'Bearer realm="jarvis"'],
            ],
        }
    )
    await send({"type": "http.response.body", "body": body})


# --- Endpoint handlers ---


_UNKNOWN_DB_STATUS = {"status": "unknown", "error": None, "checked_at": None, "free_bytes": None}


def _cached_db_status() -> dict:
    """The background probe's last verdict — a dict read, never a DB round-trip."""
    try:
        from tools.schema import get_db_status

        status = dict(get_db_status())
    except Exception:
        return dict(_UNKNOWN_DB_STATUS)
    if status.get("error"):
        status["error"] = _safe_err(status["error"])
    return status


async def health_response(scope, receive, send):
    """Liveness check — no DB queries, no secrets, no auth required.

    Top-level status stays "ok" whenever the process answers (entrypoint,
    compose healthcheck and statusline gate on it); DB state rides in
    ``postgres``, cached by the background probe (server.db_status_probe_loop).
    """
    await _json_response(send, {
        "status": "ok",
        "server": "jarvis-core",
        "version": _VERSION,
        "postgres": _cached_db_status(),
    })


async def not_found(scope, receive, send):
    await _json_response(send, {"error": "Not found"}, status=404)


def _collect_telemetry() -> dict:
    """Build the /telemetry payload (blocking: queries PostgreSQL)."""
    from jarvis_common.auth import get_auth_config
    from tools.config import get_postgres_config, get_sync_config

    # --- PostgreSQL status ---
    cfg = get_postgres_config()

    pg_status = "ok"
    pg_info = {"host": display_conninfo(cfg["url"])}
    try:
        from tools.schema import execute_query
        count_result = execute_query(
            "SELECT count(*) AS cnt FROM local.memories WHERE status = 'active'",
            fetch="one",
        )
        pg_info["doc_count"] = count_result["cnt"] if count_result else 0
    except Exception as e:
        pg_status = "disconnected"
        pg_info["error"] = _safe_err(e)

    data = {
        "status": "ok" if pg_status == "ok" else "degraded",
        "server": "jarvis-core",
        "version": _VERSION,
        "postgres": {**pg_info, "status": pg_status},
    }

    # --- Sync status ---
    sync_cfg = get_sync_config()
    if sync_cfg.get("enabled"):
        from tools.sync_queue import get_queue_stats
        from tools.schema import _get_pool

        remotes = sync_cfg.get("remotes", {})
        try:
            pool = _get_pool()
            queue_stats = get_queue_stats(pool)
        except Exception:
            queue_stats = {"error": "unavailable"}

        data["sync"] = {
            "enabled": True,
            "strategy": sync_cfg.get("strategy", "first-match"),
            "worker_interval_seconds": sync_cfg.get("worker_interval_seconds", 30),
            "remotes": {name: {"configured": True} for name in remotes},
            "queue": queue_stats,
        }
    else:
        data["sync"] = {"enabled": False}

    # --- Retrieval telemetry (best-effort; never degrades core health) ---
    try:
        from tools.retrieval_telemetry import get_summary

        data["retrieval"] = get_summary(days=7)
        data["retrieval"]["status"] = "ok"
    except Exception as exc:
        data["retrieval"] = {"status": "unavailable", "error": _safe_err(exc)}

    # --- Auth status ---
    auth_cfg = get_auth_config()
    if auth_cfg is not None:
        tokens = auth_cfg.get("tokens", {})
        mtls_configured = bool(os.environ.get("JARVIS_TLS_CA"))
        data["auth"] = {
            "enabled": True,
            "users": len(tokens) if isinstance(tokens, dict) else 0,
            "mtls": mtls_configured and _mtls_patch_ok,
        }
    else:
        data["auth"] = {"enabled": False}

    return data


async def telemetry_response(scope, receive, send):
    """GET /telemetry — full operational status (authenticated)."""
    try:
        data = await _run_blocking(_collect_telemetry, deadline=_TELEMETRY_DEADLINE_SECONDS)
    except _DeadlineExceeded:
        await _send_unavailable(send, {"error": "telemetry unavailable (timeout)"})
        return
    except Exception as e:
        await _json_response(send, {"error": _safe_err(e)}, status=500)
        return
    await _json_response(send, data)


async def hook_prompt_context_response(scope, receive, send):
    """POST /hook/prompt-context."""
    body, err = await _read_json_body(receive)
    if err:
        await _json_response(send, {"success": False, "error": err}, status=400)
        return

    prompt = body.get("prompt", "")
    if not isinstance(prompt, str):
        await _json_response(
            send,
            {"success": False, "error": "'prompt' must be a string"},
            status=400,
        )
        return

    try:
        from tools.hook_endpoints import get_prompt_context

        response = await _run_blocking(get_prompt_context, prompt)
    except Exception as e:
        # Slow but healthy (model host, busy workers): inject nothing this
        # once rather than make the client treat core as down.
        await _send_hook_error(send, e, timeout_response={
            "success": True, "matches": [], "timed_out": True,
            "query_ms": int(HOOK_DEADLINE_SECONDS * 1000),
        })
        return

    await _send_hook_result(send, response)


async def retrieval_delivery_response(scope, receive, send, trace_id: str):
    """PUT /telemetry/retrieval/{trace_id}/delivery."""
    body, err = await _read_json_body(receive)
    if err:
        await _json_response(send, {"success": False, "error": err}, status=400)
        return
    try:
        from tools.retrieval_telemetry import acknowledge_delivery

        updated = await _run_blocking(acknowledge_delivery, trace_id, body)
    except Exception as exc:
        await _send_hook_error(send, exc, status=400)
        return
    await _json_response(send, {"success": bool(updated), "trace_id": trace_id})


async def hook_auto_extract_context_response(scope, receive, send):
    """POST /hook/auto-extract/context."""
    body, err = await _read_json_body(receive)
    if err:
        await _json_response(send, {"success": False, "error": err}, status=400)
        return

    workstream_limit = body.get("workstream_limit", 30)
    try:
        workstream_limit = int(workstream_limit)
    except (TypeError, ValueError):
        await _json_response(
            send,
            {"success": False, "error": "'workstream_limit' must be an integer"},
            status=400,
        )
        return

    try:
        from tools.hook_endpoints import get_auto_extract_context

        response = await _run_blocking(
            get_auto_extract_context, workstream_limit=workstream_limit
        )
    except Exception as e:
        await _send_hook_error(send, e)
        return

    await _send_hook_result(send, response)


async def hook_auto_extract_ingest_response(scope, receive, send):
    """POST /hook/auto-extract/ingest."""
    body, err = await _read_json_body(receive)
    if err:
        await _json_response(send, {"success": False, "error": err}, status=400)
        return

    # Keep required shape explicit for easier client-side debugging.
    if "observations" in body and not isinstance(body.get("observations"), list):
        await _json_response(
            send,
            {"success": False, "error": "'observations' must be a list"},
            status=400,
        )
        return
    if "worklog" in body and body.get("worklog") is not None and not isinstance(
        body.get("worklog"), dict
    ):
        await _json_response(
            send,
            {"success": False, "error": "'worklog' must be an object or null"},
            status=400,
        )
        return
    if "context" in body and not isinstance(body.get("context"), dict):
        await _json_response(
            send,
            {"success": False, "error": "'context' must be an object"},
            status=400,
        )
        return
    if "dedup" in body and not isinstance(body.get("dedup"), dict):
        await _json_response(
            send,
            {"success": False, "error": "'dedup' must be an object"},
            status=400,
        )
        return

    try:
        from tools.hook_endpoints import ingest_auto_extract

        response = await _run_blocking(ingest_auto_extract, body)
    except Exception as e:
        await _send_hook_error(send, e)
        return

    await _send_hook_result(send, response)


# --- ASGI app ---


async def app(scope, receive, send):
    """ASGI application with path-based routing and opt-in auth.

    Routes:
        GET  /health  -> health check (always open — Docker healthcheck needs it)
        *    /mcp     -> MCP Streamable HTTP
        POST /hook/*  -> hook endpoints
    """
    if scope["type"] == "lifespan":
        await _handle_lifespan(scope, receive, send)
        return

    path = scope.get("path", "")
    method = scope.get("method", "")

    # Access log — allowlisted headers only. /health is polled every few
    # seconds (Docker healthcheck, statusline), so it only logs at DEBUG.
    level = logging.DEBUG if path == "/health" else logging.INFO
    if logger.isEnabledFor(level):
        logger.log(level, "[ACCESS] %s %s headers=%s", method, path, _loggable_headers(scope))

    # Health check always open (Docker healthcheck, monitoring)
    if path == "/health" and method == "GET":
        await health_response(scope, receive, send)
        return

    # Auth check for all other endpoints
    username, err = authenticate(scope)
    if err:
        await _send_401(send, err)
        return

    # Set contextvar for downstream use, reset on completion
    token = current_user.set(username)
    try:
        if path == "/telemetry" and method == "GET":
            await telemetry_response(scope, receive, send)
        elif path.startswith("/telemetry/retrieval/") and path.endswith("/delivery") and method == "PUT":
            trace_id = path[len("/telemetry/retrieval/"):-len("/delivery")].strip("/")
            await retrieval_delivery_response(scope, receive, send, trace_id)
        elif path == "/hook/prompt-context" and method == "POST":
            await hook_prompt_context_response(scope, receive, send)
        elif path == "/hook/auto-extract/context" and method == "POST":
            await hook_auto_extract_context_response(scope, receive, send)
        elif path == "/hook/auto-extract/ingest" and method == "POST":
            await hook_auto_extract_ingest_response(scope, receive, send)
        elif path == "/mcp" or path.startswith("/mcp/"):
            if method == "POST":
                # The SDK reads the whole body with no limit: read it here,
                # capped, and hand the SDK a replay of it.
                body = await _read_request_body(receive, limit=MAX_MCP_BODY_BYTES)
                receive = _replay_body(body, receive)
            await session_manager.handle_request(scope, receive, send)
        else:
            await not_found(scope, receive, send)
    except _ClientDisconnected:
        # Nobody left to answer, and nothing was processed.
        logger.debug("[ACCESS] %s %s client disconnected mid-body", method, path)
    except _BodyTooLarge as exc:
        await _json_response(
            send,
            {"success": False, "error": f"Request body exceeds {exc.limit} bytes"},
            status=413,
        )
    finally:
        current_user.reset(token)


async def _handle_lifespan(scope, receive, send):
    """Handle ASGI lifespan events (startup/shutdown) with graceful drain."""
    from server import get_background_tasks

    _run_ctx = None
    _bg_tasks = []
    while True:
        message = await receive()
        if message["type"] == "lifespan.startup":
            # Initialize pgvector schema (idempotent)
            from tools.schema import ensure_schema, check_model_consistency, ModelMismatchError
            try:
                ensure_schema()
            except Exception as e:
                logger.warning("Schema initialization deferred: %s", _safe_err(e))
            else:
                try:
                    check_model_consistency()
                except ModelMismatchError as mme:
                    # Raising alone (even SystemExit) from a lifespan handler
                    # is swallowed by uvicorn's lifespan="auto" and leaves a
                    # zombie server that answers /health but serves no MCP
                    # traffic. Send startup.failed first — the only signal
                    # that makes uvicorn abort and exit — then raise so test
                    # harnesses see the original error.
                    logger.critical("FATAL: %s", mme)
                    await send({"type": "lifespan.startup.failed", "message": str(mme)})
                    raise
                except Exception as e:
                    logger.warning("Model consistency check deferred: %s", _safe_err(e))

            # D6: Rebuild schema registry, auto-discovering existing remote_* schemas
            try:
                from tools.schema_registry import rebuild_registry
                rebuild_registry()
            except Exception as e:
                logger.warning("Schema registry rebuild deferred: %s", _safe_err(e))

            # Complete local model initialization before Uvicorn marks startup
            # complete. This keeps the first UserPromptSubmit request inside its
            # 2.5s deadline instead of paying the ONNX cold-start cost.
            # A failed warmup (e.g. host inference service not up yet after a
            # reboot) must not take the whole server down: retrieval already
            # fails open at runtime, so serve degraded and let embedding
            # recover when the host service returns.
            from tools.embedding import warm_embedding_service
            try:
                warm_embedding_service()
            except Exception as e:
                logger.critical(
                    "Embedding warmup failed: %s — serving in DEGRADED mode; "
                    "embedding-dependent operations fail open until the model "
                    "host is reachable again", e,
                )

            _run_ctx = session_manager.run()
            await _run_ctx.__aenter__()
            _bg_tasks = [asyncio.create_task(t) for t in get_background_tasks()]
            await send({"type": "lifespan.startup.complete"})
        elif message["type"] == "lifespan.shutdown":
            logger.info("[jarvis] Shutting down — cancelling background tasks...")

            # Cancel background tasks (pattern detection, DB status probe, etc.)
            for task in _bg_tasks:
                if not task.done():
                    task.cancel()

            if _run_ctx:
                await _run_ctx.__aexit__(None, None, None)

            # Queued blocking work is dropped; threads stuck behind a dead DB
            # are not waited on (they end at the pool's checkout timeout).
            from server import shutdown_tool_executor
            _shutdown_hook_executor()
            shutdown_tool_executor()
            await send({"type": "lifespan.shutdown.complete"})
            return
