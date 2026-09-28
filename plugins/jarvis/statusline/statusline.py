#!/usr/bin/env python3
"""Jarvis statusline for Claude Code (personal profile).

Reads JSON from stdin (piped by Claude Code), enriches with Jarvis health,
account info, and git status. Outputs ANSI-colored statusline.

Stdlib-only. No external dependencies.
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------

CACHE_DIR = Path.home() / ".jarvis" / "cache" / "statusline"
JARVIS_CACHE_TTL = 30  # seconds — short enough to catch Docker start/stop

# ---------------------------------------------------------------------------
# ANSI
# ---------------------------------------------------------------------------

RESET = "\033[0m"
BOLD = "\033[1m"
PURPLE = "\033[35m"
BLUE = "\033[34m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
RED = "\033[31m"
CYAN = "\033[36m"
GRAY = "\033[90m"

# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------


def _ensure_cache_dir():
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
    except OSError:
        pass


def _read_cache(name: str, ttl: int):
    try:
        p = CACHE_DIR / name
        if not p.exists():
            return None
        age = time.time() - p.stat().st_mtime
        if age > ttl:
            p.unlink(missing_ok=True)
            return None
        return json.loads(p.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def _write_cache(name: str, data):
    try:
        (CACHE_DIR / name).write_text(json.dumps(data))
    except OSError:
        pass


# ---------------------------------------------------------------------------
# Data collectors
# ---------------------------------------------------------------------------


def _account_name() -> str:
    """Read account name from __CLAUDE_ACCOUNT__ env var."""
    return os.environ.get("__CLAUDE_ACCOUNT__", "")


def _git_info() -> dict:
    """Git info — no cache (per-session, different repos, ~5ms)."""
    env = {**os.environ, "GIT_OPTIONAL_LOCKS": "0"}
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True, text=True, env=env, timeout=5,
        )
        if result.returncode != 0:
            return {"branch": "", "dirty": False}
        branch = result.stdout.strip()
        dirty = bool(subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True, text=True, env=env, timeout=5,
        ).stdout.strip())
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return {"branch": "", "dirty": False}
    return {"branch": branch, "dirty": dirty}


def _jarvis_health() -> dict:
    """Quick Jarvis MCP server health check (cached).

    /health stays HTTP 200 with top-level status "ok" while the server is up;
    database state rides along as the server's cached probe:
    postgres = {status: ok|recovering|unreachable|disk_full|unknown,
    error, checked_at, free_bytes}.

    Returns dict with keys: ok, pg_status, pg_free_bytes, repl_mode.
    """
    cached = _read_cache("jarvis.json", JARVIS_CACHE_TTL)
    if cached:
        return cached

    info = {"ok": False, "pg_status": "", "pg_free_bytes": None, "repl_mode": ""}
    try:
        result = subprocess.run(
            ["curl", "-sf", "--max-time", "2", "http://localhost:8741/health"],
            capture_output=True, text=True, timeout=3,
        )
        if result.returncode == 0:
            data = json.loads(result.stdout)
            if not isinstance(data, dict):
                data = {}
            info["ok"] = data.get("status") == "ok"
            pg = data.get("postgres")
            if isinstance(pg, dict):
                status = pg.get("status")
                info["pg_status"] = status if isinstance(status, str) else ""
                free = pg.get("free_bytes")
                if isinstance(free, int) and not isinstance(free, bool):
                    info["pg_free_bytes"] = free
            repl = data.get("replication")
            if isinstance(repl, dict) and isinstance(repl.get("mode"), str):
                info["repl_mode"] = repl["mode"]
    except (subprocess.TimeoutExpired, FileNotFoundError, json.JSONDecodeError):
        pass

    _write_cache("jarvis.json", info)
    return info


# ---------------------------------------------------------------------------
# Formatters
# ---------------------------------------------------------------------------


def _fmt_model(model) -> str:
    if not model:
        return f"{GRAY}unknown{RESET}"
    if isinstance(model, str):
        name = model
    else:
        name = model.get("display_name") or model.get("id") or str(model)
    return name


def _fmt_cost(cost) -> str:
    if not isinstance(cost, (int, float)) or cost == 0:
        return "$0.00"
    return f"${cost:.4f}"


def _fmt_context(data: dict) -> str:
    pct = 0.0
    ctx = data.get("context_window")
    if isinstance(ctx, dict):
        pct = ctx.get("used_percentage") or 0.0
    if pct == 0:
        return f"{GRAY}0%{RESET}"
    s = f"{pct:.0f}%"
    if pct > 65:
        return f"{RED}{s}{RESET}"
    if pct > 40:
        return f"{YELLOW}{s}{RESET}"
    return f"{GREEN}{s}{RESET}"


def _fmt_bytes(n) -> str:
    if not isinstance(n, int) or isinstance(n, bool) or n < 0:
        return ""
    units = ["B", "K", "M", "G", "T"]
    value = float(n)
    unit = 0
    while value >= 1024 and unit < len(units) - 1:
        value /= 1024
        unit += 1
    if value >= 10 or unit == 0:
        return f"{value:.0f}{units[unit]}"
    return f"{value:.1f}{units[unit]}"


# Known /health postgres.status values -> (label, color). "ok" renders nothing.
_PG_STATUS_LABELS = {
    "recovering": ("recovering", RED),
    "unreachable": ("unreachable", RED),
    "disk_full": ("disk full", RED),
    "unknown": ("unknown", YELLOW),
}


def _fmt_pg_status(status: str, free_bytes=None) -> str:
    """Red "DB: <state>" segment when the database is not ok, else ""."""
    if not status or status == "ok":
        return ""
    label, color = _PG_STATUS_LABELS.get(status, ("", RED))
    if not label:
        # Unrecognized value from the server: never echo raw text (escape
        # sequences) into the terminal.
        label = "".join(
            ch for ch in status.lower() if ch.isascii() and (ch.isalnum() or ch in "_-")
        )[:24] or "error"
    if status == "disk_full":
        free = _fmt_bytes(free_bytes)
        if free:
            label += f" ({free} free)"
    return f"{color}DB: {label}{RESET}"



# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def generate(data: dict) -> str:
    _ensure_cache_dir()

    SEP = f" {GRAY}\u2502{RESET} "

    git = _git_info()
    jarvis = _jarvis_health()
    account = _account_name()

    model = data.get("model")
    cost_obj = data.get("cost", {})
    cost_usd = cost_obj.get("total_cost_usd", 0) if isinstance(cost_obj, dict) else 0
    cwd = data.get("cwd", "")
    dirname = os.path.basename(cwd) if cwd else "?"

    # Build segments
    parts = []

    # Account name
    if account:
        parts.append(f"{GREEN}{BOLD}{account}{RESET}")

    # Jarvis branding (only when the server answers /health) plus a DB
    # segment whenever its cached Postgres probe is not ok
    if jarvis.get("ok"):
        jarvis_label = f"{BOLD}{YELLOW}\u26a1{RESET} {YELLOW}JARVIS{RESET}"
        pg_segment = _fmt_pg_status(
            jarvis.get("pg_status", ""), jarvis.get("pg_free_bytes")
        )
        if pg_segment:
            jarvis_label += f" {pg_segment}"
        # Replication indicator
        repl_mode = jarvis.get("repl_mode", "")
        if repl_mode and repl_mode != "disabled":
            jarvis_label += f" {CYAN}repl:{repl_mode}{RESET}"
        parts.append(jarvis_label)

    # Session name (from /rename) or truncated session ID
    session_label = data.get("session_name") or (data.get("session_id", "")[:8])
    if session_label:
        parts.append(session_label)

    # Model
    parts.append(_fmt_model(model))

    # Directory with folder emoji
    parts.append(f"\U0001f4c1 {dirname}")

    # Git — only if we're in a repo (branch is non-empty)
    branch = git.get("branch", "")
    if branch:
        dirty_mark = "*" if git.get("dirty") else ""
        parts.append(f"{branch}{dirty_mark}")

    # Cost
    if cost_usd:
        parts.append(_fmt_cost(cost_usd))

    # Context — always show
    parts.append(f"ctx {_fmt_context(data)}")

    return SEP.join(parts)


def main():
    try:
        # Windows: force UTF-8 for emoji/ANSI output
        if sys.platform == "win32":
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
            sys.stdin.reconfigure(encoding="utf-8", errors="replace")

        # Demo mode: --demo flag or interactive TTY with no stdin
        if "--demo" in sys.argv or sys.stdin.isatty():
            demo = {
                "session_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
                "session_name": "demo-session",
                "model": {"id": "claude-opus-4-6", "display_name": "Opus 4.6"},
                "cwd": os.getcwd(),
                "cost": {
                    "total_cost_usd": 1.23,
                    "total_duration_ms": 95000,
                },
                "context_window": {"used_percentage": 34.5},
            }
            print(generate(demo))
            return

        raw = sys.stdin.read()
        data = json.loads(raw) if raw.strip() else {}
        print(generate(data))
    except Exception:
        # Never crash — output something safe
        print(f"{GRAY}jarvis statusline error{RESET}")


if __name__ == "__main__":
    main()
