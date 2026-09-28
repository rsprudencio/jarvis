"""call_tool logs which arguments a tool got, never their values."""

import asyncio
import logging

import server


def test_call_tool_logs_argument_keys_not_values(caplog):
    secret = "ghp_do-not-log-this-token"
    with caplog.at_level(logging.INFO, logger="jarvis-obsidian"):
        asyncio.run(server.call_tool("no_such_tool", {"message": secret, "files": ["a.md"]}))

    logged = "\n".join(r.getMessage() for r in caplog.records)
    assert "Tool: no_such_tool, arg_keys: ['files', 'message']" in logged
    assert secret not in logged
    assert "a.md" not in logged
