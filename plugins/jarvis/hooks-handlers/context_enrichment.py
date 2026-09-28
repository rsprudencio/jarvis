#!/usr/bin/env python3
"""Context enrichment: per-prompt semantic memory injection.

Usage:
  Hook mode:  echo '{"prompt":"..."}' | python3 context_enrichment.py --hook
  Direct:     python3 context_enrichment.py "query text here"

Called by the UserPromptSubmit hook. Outputs XML-formatted vault memories as
plain context for Claude Code or in Codex's structured hook response. Silent
on errors (exit 0, no output) to avoid disrupting the user's conversation.
"""
import json
import os
import sys
import time
import xml.sax.saxutils as saxutils
from pathlib import Path
from typing import Tuple

from hook_http_client import post_json, put_json
from harness import format_user_prompt_submit_output
from precompact_dedup import compute_content_hash, filter_already_injected, write_injection_state

# Core embeds and lexically matches only the head of a prompt (4000 chars for
# the lexical channel, the model's token window for embeddings). A huge paste
# sent whole blew past core's 1 MiB request cap, and the reset connection read
# as "core unavailable" for every hook for a minute. Even at 12 bytes per
# JSON-escaped character this stays far below the cap.
MAX_PROMPT_CHARS = 32_000


# --- Debug Logging ---

DEBUG_LOG_FILE = Path.home() / ".jarvis" / "debug.per-prompt-search.log"


def _debug_log(action: str, detail: str, prompt: str = "", injected: str = ""):
    """Append a structured debug block to the per-prompt search log.

    Uses shared ANSI colors and section dividers for visual consistency
    with the auto-extract debug log when tailing.

    Args:
        action: SKIP, ERROR, EMPTY, or FOUND
        detail: Summary line (e.g., "460ms | 3/5 | sources...")
        prompt: The user's prompt text
        injected: The full XML output injected into Claude's context (FOUND only)
    """
    try:
        from _colors import C_GREEN, C_YELLOW, C_RESET, divider_thick, divider_section

        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        status_color = C_GREEN if action == "FOUND" else C_YELLOW

        lines = []
        lines.append(divider_thick())
        lines.append(f"{ts} | {status_color}{action:5s}{C_RESET} | {detail}")

        if prompt:
            lines.append(divider_section("PROMPT"))
            lines.append(prompt)

        if injected:
            lines.append(divider_section("INJECTED"))
            lines.append(injected)

        lines.append("")  # Blank line separator

        with open(DEBUG_LOG_FILE, "a") as f:
            f.write("\n".join(lines) + "\n")
    except Exception:
        pass  # Never fail on debug logging


# --- Telemetry ---

TELEMETRY_FILE = Path.home() / ".jarvis" / "telemetry" / "prompt_search.jsonl"


def _write_telemetry(prompt: str, query_ms: int, matches: list, result: dict):
    """Append a structured JSONL line for threshold/budget analysis."""
    try:
        TELEMETRY_FILE.parent.mkdir(parents=True, exist_ok=True)
        scores = [m["relevance"] for m in matches]
        # The threshold gates on raw similarity, not relevance — log both so
        # recalibration analyzes the quantity the gate actually compares.
        similarities = [m.get("similarity", 0.0) for m in matches]
        n_vault = sum(1 for m in matches if m.get("display_mode") == "reference")
        budget = result.get("budget_used", {})
        entry = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "prompt_len": len(prompt),
            "query_ms": query_ms,
            "n_results": len(matches),
            "n_core": len(matches) - n_vault,
            "n_vault": n_vault,
            "scores": [round(s, 3) for s in scores],
            "similarities": [round(s, 3) for s in similarities],
            "budget_local_used": budget.get("local", 0),
            "budget_vault_used": budget.get("vault", 0),
            "budget_remote_used": budget.get("remote", 0),
        }
        with open(TELEMETRY_FILE, "a") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception:
        pass  # Never fail on telemetry


# --- Prompt Filtering ---

_SKIP_PATTERNS = {
    "yes",
    "no",
    "ok",
    "sure",
    "thanks",
    "thank you",
    "go ahead",
    "done",
    "next",
    "continue",
    "correct",
    "right",
    "got it",
    "sounds good",
    "perfect",
    "great",
    "fine",
    "agreed",
    "yep",
    "nope",
    "nah",
    "yeah",
    "yup",
    "okay",
}


def _should_skip_prompt(query: str) -> Tuple[bool, str]:
    """Determine if prompt is too trivial for semantic search.

    Returns:
        Tuple of (should_skip, reason). Reason is empty string if not skipped.
    """
    stripped = query.strip()

    # Too short
    if len(stripped) < 10:
        return True, "short"

    # Slash commands (have their own handlers)
    if stripped.startswith("/"):
        return True, "slash_cmd"

    # Known confirmation patterns (case-insensitive, strip trailing punctuation)
    normalized = stripped.lower().rstrip(".!?")
    if normalized in _SKIP_PATTERNS:
        return True, "confirmation"

    # Pure code blocks
    if stripped.startswith("```"):
        return True, "code_block"

    # Auto-extract Haiku prompt (fired via `claude -p` subprocess)
    if (
        "You are analyzing a conversation turn between a user and an AI assistant"
        in stripped[:100]
    ):
        return True, "auto_extract_prompt"

    return False, ""


# --- Prompt Extraction from Hook JSON ---


def _extract_prompt(hook_json: str) -> str:
    """Extract prompt text from UserPromptSubmit hook input JSON."""
    try:
        data = json.loads(hook_json)
        # Try known key names for the prompt text
        prompt = (
            data.get("prompt") or data.get("user_prompt") or data.get("message") or ""
        )
        if isinstance(prompt, dict):
            prompt = prompt.get("text", prompt.get("content", ""))
        return str(prompt)
    except (json.JSONDecodeError, TypeError, AttributeError):
        return ""


# --- Output Formatting ---


def _format_memories(matches: list, query_ms: float) -> str:
    """Format search results as XML for injection into Claude's context.

    Vault items (display_mode="reference") are shown as compact file pointers.
    Core memory items (display_mode="full") are shown with full content.
    """
    if not matches:
        return ""

    lines = [f'<relevant-vault-memories count="{len(matches)}" query_ms="{query_ms}">']

    for match in matches:
        display_mode = match.get("display_mode", "full")
        # Backward compat: accept both "id" (new) and "source" (legacy)
        mem_id = match.get("id") or match.get("source", "")
        attrs = [
            f'id="{saxutils.escape(mem_id)}"',
            f'relevance="{match["relevance"]}"',
            f'type="{saxutils.escape(match.get("type", "unknown"))}"',
        ]
        if match.get("heading"):
            attrs.append(f'heading="{saxutils.escape(match["heading"])}"')
        if match.get("schema"):
            attrs.append(f'schema="{saxutils.escape(match["schema"])}"')
        if match.get("source_remote"):
            attrs.append(f'source_remote="{saxutils.escape(match["source_remote"])}"')
        if match.get("origin_user"):
            attrs.append(f'origin_user="{saxutils.escape(match["origin_user"])}"')
        if display_mode == "reference":
            attrs.append('ref="vault"')
        if match.get("stale"):
            attrs.append('stale="true"')

        content = saxutils.escape(match.get("content", ""))
        lines.append(f'<memory {" ".join(attrs)}>')
        lines.append(content)
        lines.append("</memory>")

    lines.append("</relevant-vault-memories>")
    return "\n".join(lines)


# --- Main Entry Point ---


def _format_memory_unavailable_warning(error_text: str = "") -> str:
    """Return warning block for MCP outage / endpoint failures."""
    detail = "Jarvis memory context is temporarily unavailable. Continuing without memory injection."
    if error_text:
        safe_error = saxutils.escape(error_text[:180])
        detail += f" ({safe_error})"
    return (
        '<jarvis-warning type="memory-unavailable">'
        f"{detail}"
        "</jarvis-warning>"
    )


def main():
    """Run per-prompt semantic search and output results to stdout."""
    # Determine prompt text source
    session_id = ""
    hook_mode = len(sys.argv) >= 2 and sys.argv[1] == "--hook"
    if hook_mode:
        # Hook mode: read JSON from stdin
        try:
            hook_input = sys.stdin.read()
        except Exception:
            sys.exit(0)
        prompt_text = _extract_prompt(hook_input)
        # Extract session_id for dedup tracking
        try:
            hook_data = json.loads(hook_input)
            session_id = hook_data.get("session_id", "")
        except Exception:
            pass
    elif len(sys.argv) >= 2:
        # Direct mode: prompt text as argument
        prompt_text = sys.argv[1]
    else:
        sys.exit(0)

    if not prompt_text:
        sys.exit(0)

    # Skip trivial prompts quickly
    skip, _ = _should_skip_prompt(prompt_text)

    if skip:
        sys.exit(0)

    # Fetch semantic context + per-prompt flags from local hook endpoint.
    # This keeps retrieval bump and config resolution server-side. While the
    # core-degraded marker is fresh the client fails fast without a request,
    # which lands in the memory-unavailable warning below.
    ctx_resp = post_json(
        "/hook/prompt-context",
        {"prompt": prompt_text[:MAX_PROMPT_CHARS]},
        timeout_seconds=2.5,
    )

    output_parts = []
    config = {
        "enabled": True,
        "debug": False,
        "todoist_prompt_alerts": {"enabled": False, "max_per_category": 3},
    }
    result = {"matches": [], "query_ms": 0, "budget_used": {"core": 0, "vault": 0}}
    query_ms = 0

    if ctx_resp.get("success"):
        data = ctx_resp.get("data") or {}
        config["enabled"] = bool(data.get("enabled", True))
        config["debug"] = bool(data.get("debug", False))
        config["todoist_prompt_alerts"] = data.get(
            "todoist_prompt_alerts", config["todoist_prompt_alerts"]
        )
        result = data
        query_ms = result.get("query_ms", 0)
    else:
        output_parts.append(_format_memory_unavailable_warning(ctx_resp.get("error", "")))

    debug = bool(config.get("debug", False))

    if not config.get("enabled", True):
        if debug:
            _debug_log("SKIP", "disabled")
        sys.exit(0)

    # Core answered, but its database is down: the empty result is not a miss.
    if ctx_resp.get("success") and result.get("degraded") is True:
        output_parts.append(
            _format_memory_unavailable_warning(
                str(result.get("error") or "database unavailable")
            )
        )

    returned_matches = result.get("matches", [])
    matches = returned_matches

    # Filter out memories already injected in this compaction window
    matches = filter_already_injected(matches, session_id)

    # The server trace captures budget-selected candidates; this acknowledgement
    # closes the loop after session-level dedup so the UI shows what the model
    # actually received. It is deliberately best-effort and short-timeout.
    trace_id = result.get("trace_id")
    if trace_id:
        delivered_keys = [
            str(match.get("candidate_key")) for match in matches
            if match.get("candidate_key")
        ]
        try:
            put_json(
                f"/telemetry/retrieval/{trace_id}/delivery",
                {
                    "status": "complete",
                    "returned_count": len(returned_matches),
                    "delivered_count": len(matches),
                    "suppressed_count": len(returned_matches) - len(matches),
                    "output_chars": sum(len(str(m.get("content", ""))) for m in matches),
                    "delivered_candidate_keys": delivered_keys,
                },
                timeout_seconds=0.5,
                # A slow ack under load is not an outage; don't trip the breaker.
                trip_breaker=False,
            )
        except Exception:
            pass

    if matches:
        # Format vault/core memories
        output_parts.append(_format_memories(matches, result.get("query_ms", 0)))

        # Record injected content for dedup on subsequent prompts
        content_hashes = [compute_content_hash(m.get("content", "")) for m in matches]
        sources = [m.get("id") or m.get("source", "") for m in matches]
        write_injection_state(session_id, content_hashes, sources)

        # JSONL telemetry (always on, lightweight)
        if not trace_id:
            _write_telemetry(prompt_text, query_ms, matches, result)

        if debug:
            n_vault = sum(1 for m in matches if m.get("display_mode") == "reference")
            n_core = len(matches) - n_vault
            budget = result.get("budget_used", {})
            sources = " ".join(f'{m.get("id") or m.get("source", "")}({m["relevance"]})' for m in matches)
            _debug_log(
                "FOUND",
                f"{query_ms}ms | {len(matches)} ({n_core}c+{n_vault}v) | "
                f"budget c:{budget.get('core', 0)}/v:{budget.get('vault', 0)} | {sources}",
                prompt_text,
                injected=output_parts[0],
            )
    else:
        if debug:
            _debug_log("EMPTY", f"{query_ms}ms | 0 results", prompt_text)

    # Todoist alerts (independent of vault matches — always check)
    try:
        from todoist_check import get_todoist_alerts as _get_todoist_alerts

        todoist_cfg = config.get("todoist_prompt_alerts", {})

        if todoist_cfg.get("enabled", False):
            cache_path = str(Path.home() / ".jarvis" / "state" / "todoist_cache.json")
            # Respect JARVIS_HOME env var for Docker
            jarvis_home = os.environ.get("JARVIS_HOME")
            if jarvis_home:
                cache_path = str(Path(jarvis_home) / "state" / "todoist_cache.json")
            alerts_xml = _get_todoist_alerts(
                cache_path,
                max_per_category=todoist_cfg.get("max_per_category", 3),
            )
            if alerts_xml:
                output_parts.append(alerts_xml)
    except Exception:
        pass  # Never fail the hook

    output = "\n".join(output_parts)
    if output:
        # Direct mode remains human-readable. Hook mode selects the stdout
        # protocol using the harness environment without touching retrieval,
        # telemetry, or session-dedup behavior.
        print(format_user_prompt_submit_output(output) if hook_mode else output)

    sys.exit(0)


if __name__ == "__main__":
    main()
