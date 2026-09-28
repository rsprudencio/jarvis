"""Stdlib HTTP client for local Jarvis hook endpoints.

This module intentionally avoids any `tools.*` imports so hook handlers can run
without path-coupling to the MCP server package.

It also carries a client-side circuit breaker shared by every hook in every
session: a short-lived `core_degraded` marker in the Jarvis state directory.
While it is fresh, requests fail immediately without touching the network.
During the 2026-09-24 outage each abandoned 2.5 s hook request still cost the
server its full DB wait, so retrying on every prompt and Stop fed the freeze.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path
from urllib import error, parse, request

DEFAULT_BASE_URL = "http://localhost:8741"
DEFAULT_TIMEOUT_SECONDS = 2.5

# Written on a timeout, refused/reset connection, HTTP 502/503/504, or a
# retryable/degraded body; hooks skip the network while it is younger than TTL.
# Not for error_kind "deadline": core answered that one request slowly while
# its database was fine, which is no reason to stop serving the next prompt.
CORE_DEGRADED_MARKER = "core_degraded"
CORE_DEGRADED_TTL_SECONDS = 60.0
# A marker dated further in the future than this is treated as stale, so a
# clock jump cannot silence the hooks for longer than the TTL.
_MARKER_CLOCK_SKEW_SECONDS = 5.0
_MARKER_REASON_MAX_CHARS = 200

# The server rejected the payload itself: resending it can never succeed.
_PERMANENT_HTTP_STATUSES = frozenset({400, 413, 422})
# Core (or its database) is unavailable right now: retry later.
_UNAVAILABLE_HTTP_STATUSES = frozenset({502, 503, 504})
_ERROR_KIND_DEADLINE = "deadline"


def _strip_mcp_suffix(url: str) -> str:
    """Normalize configured MCP URL to the server base URL.

    Example:
      http://localhost:8741/mcp -> http://localhost:8741
    """
    parsed = parse.urlsplit(url.strip())
    if not parsed.scheme or not parsed.netloc:
        return DEFAULT_BASE_URL

    path = parsed.path or ""
    if path.endswith("/mcp"):
        path = path[: -len("/mcp")]
    elif path.endswith("/mcp/"):
        path = path[: -len("/mcp/")]
    normalized = parse.urlunsplit(
        (parsed.scheme, parsed.netloc, path.rstrip("/"), "", "")
    )
    return normalized or DEFAULT_BASE_URL


def resolve_base_url(mcp_json_path: str | Path | None = None) -> str:
    """Resolve local Jarvis core base URL from `.mcp.json`.

    Falls back to `http://localhost:8741` on any error.
    """
    if mcp_json_path is None:
        mcp_json_path = Path(__file__).resolve().parent.parent / ".mcp.json"

    path = Path(mcp_json_path)
    try:
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
        core = data.get("core", {})
        url = core.get("url")
        if isinstance(url, str) and url.strip():
            return _strip_mcp_suffix(url)
    except Exception:
        pass
    return DEFAULT_BASE_URL


def _join_url(base_url: str, endpoint_path: str) -> str:
    return f"{base_url.rstrip('/')}/{endpoint_path.lstrip('/')}"


# --- Core-degraded marker (client-side circuit breaker) ---


def _state_dir() -> Path:
    """Resolve Jarvis state directory with JARVIS_HOME override."""
    jarvis_home = os.environ.get("JARVIS_HOME")
    if jarvis_home:
        return Path(jarvis_home) / "state"
    return Path.home() / ".jarvis" / "state"


def core_degraded_marker_path() -> Path:
    return _state_dir() / CORE_DEGRADED_MARKER


def core_degraded_reason(ttl_seconds: float = CORE_DEGRADED_TTL_SECONDS) -> str:
    """Return why core was marked degraded, or "" when the marker is absent/stale."""
    path = core_degraded_marker_path()
    try:
        age = time.time() - path.stat().st_mtime
    except OSError:
        return ""
    if not -_MARKER_CLOCK_SKEW_SECONDS <= age < ttl_seconds:
        return ""
    reason = ""
    try:
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
        if isinstance(data, dict):
            reason = str(data.get("reason") or "")
    except (OSError, ValueError):
        pass
    return reason or "core unavailable"


def is_core_degraded(ttl_seconds: float = CORE_DEGRADED_TTL_SECONDS) -> bool:
    return bool(core_degraded_reason(ttl_seconds))


def mark_core_degraded(reason: str) -> None:
    """Atomically (re)write the marker. Never raises."""
    one_line = " ".join(str(reason or "").split())[:_MARKER_REASON_MAX_CHARS]
    try:
        directory = _state_dir()
        directory.mkdir(parents=True, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(
            dir=directory, prefix=f".{CORE_DEGRADED_MARKER}.", suffix=".tmp"
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "marked_at": time.strftime(
                            "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
                        ),
                        "reason": one_line or "core unavailable",
                    },
                    handle,
                )
            os.replace(tmp_path, core_degraded_marker_path())
        except Exception:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise
    except Exception:
        pass


def _note_core_outcome(result: dict) -> None:
    """Trip the breaker when core is unavailable or reports a degraded DB."""
    data = result.get("data")
    if result.get("retryable"):
        if result.get("error_kind") != _ERROR_KIND_DEADLINE:
            mark_core_degraded(result.get("error", ""))
    elif (
        result.get("success")
        and isinstance(data, dict)
        and data.get("degraded") is True
    ):
        mark_core_degraded(str(data.get("error") or "database unavailable"))


# --- Requests ---


def _result(
    success: bool,
    data: dict | None = None,
    error_text: str = "",
    http_status: int | None = None,
    retryable: bool = False,
    permanent: bool = False,
    skipped: bool = False,
    error_kind: str = "",
) -> dict:
    return {
        "success": success,
        "data": data,
        "error": error_text,
        "http_status": http_status,
        "retryable": retryable,
        "permanent": permanent,
        "skipped": skipped,
        "error_kind": error_kind,
    }


def _error_kind(body) -> str:
    kind = body.get("error_kind") if isinstance(body, dict) else None
    return kind if isinstance(kind, str) else ""


def _send(method: str, url: str, payload: dict, timeout_seconds: float) -> dict:
    body = json.dumps(payload).encode("utf-8")
    req = request.Request(
        url=url,
        data=body,
        method=method,
        headers={"Content-Type": "application/json"},
    )

    try:
        with request.urlopen(req, timeout=timeout_seconds) as resp:
            http_status = getattr(resp, "status", None)
            raw = resp.read()
    except error.HTTPError as exc:
        raw = b""
        try:
            raw = exc.read() or b""
        except Exception:
            pass
        detail = ""
        body_retryable = False
        error_kind = ""
        if raw:
            try:
                parsed_body = json.loads(raw.decode("utf-8"))
                detail = parsed_body.get("error") or parsed_body.get("message") or ""
                body_retryable = parsed_body.get("retryable") is True
                error_kind = _error_kind(parsed_body)
            except Exception:
                detail = raw.decode("utf-8", errors="replace").strip()
        msg = f"http_error:{exc.code}"
        if detail:
            msg = f"{msg}:{detail}"
        retryable = body_retryable or exc.code in _UNAVAILABLE_HTTP_STATUSES
        return _result(
            False,
            error_text=msg,
            http_status=exc.code,
            retryable=retryable,
            permanent=not retryable and exc.code in _PERMANENT_HTTP_STATUSES,
            error_kind=error_kind,
        )
    except Exception as exc:
        # Timeout, refused/reset connection, empty reply: nothing was delivered.
        return _result(False, error_text=str(exc), retryable=True)

    try:
        decoded = json.loads(raw.decode("utf-8"))
    except Exception:
        return _result(False, error_text="invalid_json_response", http_status=http_status)

    if not isinstance(decoded, dict):
        return _result(False, error_text="invalid_response_shape", http_status=http_status)
    # A retryable body is a delivery failure even without success:false.
    retryable = decoded.get("retryable") is True
    if decoded.get("success") is False or retryable:
        detail = decoded.get("error") or "request_failed"
        return _result(
            False,
            data=decoded,
            error_text=str(detail),
            http_status=http_status,
            retryable=retryable,
            error_kind=_error_kind(decoded),
        )
    return _result(True, data=decoded, http_status=http_status)


def _request_json(
    method: str,
    endpoint_path: str,
    payload: dict,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    mcp_json_path: str | Path | None = None,
    trip_breaker: bool = True,
) -> dict:
    """Send JSON to a local hook endpoint.

    Returns a normalized contract:
      {"success": bool, "data": dict|None, "error": str,
       "http_status": int|None, "retryable": bool, "permanent": bool,
       "skipped": bool, "error_kind": str}

    retryable: core or its database is unavailable (timeout, connection error,
      HTTP 502/503/504, or a body with retryable=true); nothing was delivered.
    permanent: the server rejected the payload (HTTP 400/413/422); resending
      it can never succeed.
    skipped: the core-degraded marker was fresh, so no request was sent.
    error_kind: core's classification of a failure ("db_unavailable",
      "deadline", ...) or "" when it gave none.

    trip_breaker=False keeps a failure from writing the marker (for
    best-effort calls with very short timeouts).
    """
    reason = core_degraded_reason()
    if reason:
        return _result(
            False,
            error_text=f"core degraded: {reason}",
            retryable=True,
            skipped=True,
        )

    url = _join_url(resolve_base_url(mcp_json_path), endpoint_path)
    result = _send(method, url, payload, timeout_seconds)
    if trip_breaker:
        _note_core_outcome(result)
    return result


def post_json(
    endpoint_path: str,
    payload: dict,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    mcp_json_path: str | Path | None = None,
    trip_breaker: bool = True,
) -> dict:
    return _request_json(
        "POST", endpoint_path, payload, timeout_seconds, mcp_json_path, trip_breaker
    )


def put_json(
    endpoint_path: str,
    payload: dict,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    mcp_json_path: str | Path | None = None,
    trip_breaker: bool = True,
) -> dict:
    return _request_json(
        "PUT", endpoint_path, payload, timeout_seconds, mcp_json_path, trip_breaker
    )
