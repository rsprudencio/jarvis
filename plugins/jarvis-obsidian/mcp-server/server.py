#!/usr/bin/env python3
"""
Jarvis Obsidian MCP Server

Git audit trail and PKM vault operations for JARVIS protocol.
Provides 9 git tools for vault commit management.

Tools:
- obsidian_commit, obsidian_status, obsidian_parse_last_commit, obsidian_push
- obsidian_move_files, obsidian_query_history, obsidian_rollback
- obsidian_file_history, obsidian_rewrite_commit_messages
"""
import asyncio
import inspect
import json
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

from protocol import (
    ProtocolTag,
    ProtocolValidator,
    format_commit_message,
)
from tools.commit import (
    stage_files,
    execute_commit,
    get_commit_stats,
    commit_user_prologue,
)
from tools.git_ops import (
    parse_last_commit,
    get_status,
    push_to_remote,
    move_files,
    query_history,
    rollback_commit,
    file_history,
    rewrite_commit_messages,
)

import system_prompt

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger("jarvis-obsidian")

server = Server("vault", instructions=system_prompt.instructions)

# Tool definitions
TOOLS = [
    Tool(
        name="obsidian_commit",
        description="Create a JARVIS protocol git commit with validation and formatting.",
        inputSchema={
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": ["create", "edit", "delete", "move", "user"],
                    "description": "Operation type",
                },
                "description": {"type": "string", "description": "Commit message"},
                "entry_id": {
                    "type": "string",
                    "description": "14-digit timestamp (optional)",
                },
                "trigger_mode": {
                    "type": "string",
                    "enum": ["conversational", "agent"],
                    "default": "conversational",
                },
                "files": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Files to stage (optional)",
                },
            },
            "required": ["operation", "description"],
        },
    ),
    Tool(
        name="obsidian_status",
        description="Get current git status (staged, unstaged, untracked files).",
        inputSchema={"type": "object", "properties": {}},
    ),
    Tool(
        name="obsidian_parse_last_commit",
        description="Parse info about the most recent commit.",
        inputSchema={"type": "object", "properties": {}},
    ),
    Tool(
        name="obsidian_push",
        description="Push commits to remote repository.",
        inputSchema={
            "type": "object",
            "properties": {
                "branch": {"type": "string", "description": "Branch to push (optional)"}
            },
        },
    ),
    Tool(
        name="obsidian_move_files",
        description="Move/rename files using git mv (preserves history).",
        inputSchema={
            "type": "object",
            "properties": {
                "moves": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "source": {"type": "string"},
                            "destination": {"type": "string"},
                        },
                        "required": ["source", "destination"],
                    },
                }
            },
            "required": ["moves"],
        },
    ),
    Tool(
        name="obsidian_query_history",
        description="Query Jarvis operations from git history.",
        inputSchema={
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": ["create", "edit", "delete", "move", "user", "all"],
                    "description": "Filter by operation type (default: all)",
                },
                "since": {
                    "type": "string",
                    "description": "Time filter (e.g., 'today', '1 week ago')",
                },
                "limit": {
                    "type": "integer",
                    "description": "Max results (default: 10)",
                },
                "file": {
                    "type": "string",
                    "description": "Filter by file path (optional)",
                },
            },
        },
    ),
    Tool(
        name="obsidian_rollback",
        description="Rollback a specific Jarvis commit using git revert.",
        inputSchema={
            "type": "object",
            "properties": {
                "commit_hash": {
                    "type": "string",
                    "description": "Commit hash to revert",
                }
            },
            "required": ["commit_hash"],
        },
    ),
    Tool(
        name="obsidian_file_history",
        description="Get Jarvis operation history for a specific file.",
        inputSchema={
            "type": "object",
            "properties": {
                "file_path": {"type": "string", "description": "Path to the file"},
                "limit": {
                    "type": "integer",
                    "description": "Max results (default: 10)",
                },
            },
            "required": ["file_path"],
        },
    ),
    Tool(
        name="obsidian_rewrite_commit_messages",
        description="Rewrite recent commit messages to remove unwanted text (e.g., Co-Authored-By lines). WARNING: Rewrites history - only use on unpushed commits.",
        inputSchema={
            "type": "object",
            "properties": {
                "count": {
                    "type": "integer",
                    "description": "Number of recent commits to process (default: 1)",
                },
                "patterns": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Sed regex patterns to remove. Default: ['Co-Authored-By:.*']",
                },
            },
        },
    ),
]


async def handle_commit(args: dict) -> dict:
    """Handle obsidian_commit."""
    operation = args.get("operation")
    description = args.get("description")
    entry_id = args.get("entry_id")
    trigger_mode = args.get("trigger_mode", "conversational")
    files = args.get("files")

    # Validate
    errors = ProtocolValidator.validate_all(
        operation=operation,
        description=description,
        entry_id=entry_id,
        trigger_mode=trigger_mode,
    )
    if errors:
        return {"success": False, "validation_errors": errors}

    # Auto user prologue: when explicit files are provided and this isn't a
    # user operation, automatically commit any other dirty vault files as
    # [JARVIS:U] first.
    prologue_result = None
    if operation != "user" and files:
        prologue_result = commit_user_prologue(set(files))
        if prologue_result and not prologue_result.get("success", True):
            return prologue_result

    # Stage
    stage_result = stage_files(files)
    if not stage_result["success"]:
        return stage_result

    # Build protocol tag
    tag = ProtocolTag(operation=operation, trigger_mode=trigger_mode, entry_id=entry_id)
    tag_string = tag.to_string()

    # Commit
    commit_msg = format_commit_message(operation, description, tag_string)
    commit_result = execute_commit(commit_msg)
    if not commit_result["success"]:
        return commit_result

    stats = get_commit_stats()

    response = {
        "success": True,
        "commit_hash": commit_result["commit_hash"],
        "protocol_tag": tag_string,
        "files_changed": stats["files_changed"],
        "insertions": stats["insertions"],
        "deletions": stats["deletions"],
    }
    if prologue_result and prologue_result.get("commit_hash"):
        response["user_prologue"] = prologue_result
    return response


# Tool name -> handler mapping
_HANDLERS = {
    "obsidian_commit": handle_commit,
    "obsidian_status": lambda args: get_status(),
    "obsidian_parse_last_commit": lambda args: parse_last_commit(),
    "obsidian_push": lambda args: push_to_remote(args.get("branch")),
    "obsidian_move_files": lambda args: move_files(args.get("moves", [])),
    "obsidian_query_history": lambda args: query_history(
        operation=args.get("operation", "all"),
        since=args.get("since"),
        limit=args.get("limit", 10),
        file_path=args.get("file"),
    ),
    "obsidian_rollback": lambda args: rollback_commit(args.get("commit_hash")),
    "obsidian_file_history": lambda args: file_history(
        args.get("file_path"), args.get("limit", 10)
    ),
    "obsidian_rewrite_commit_messages": lambda args: rewrite_commit_messages(
        count=args.get("count", 1), patterns=args.get("patterns")
    ),
}


@server.list_tools()
async def list_tools() -> list[Tool]:
    return TOOLS


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    # Keys only: values carry commit messages, file paths and note text.
    logger.info("Tool: %s, arg_keys: %s", name, sorted((arguments or {}).keys()))

    try:
        handler = _HANDLERS.get(name)
        if handler:
            result = handler(arguments or {})
            if inspect.isawaitable(result):
                result = await result
        else:
            result = {"success": False, "error": f"Unknown tool: {name}"}

        return [TextContent(type="text", text=json.dumps(result, indent=2))]

    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        return [
            TextContent(
                type="text", text=json.dumps({"success": False, "error": str(e)})
            )
        ]


async def main():
    logger.info("Starting Jarvis Obsidian MCP Server")
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream, write_stream, server.create_initialization_options()
        )


def main_sync():
    """Synchronous entry point for uvx/pip scripts."""
    asyncio.run(main())


if __name__ == "__main__":
    main_sync()
