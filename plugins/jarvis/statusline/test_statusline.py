"""Tests for Jarvis statusline."""
import json
import os
import subprocess
import sys
from pathlib import Path
from unittest import mock

import pytest

# Add statusline dir to path
sys.path.insert(0, str(Path(__file__).parent))
import statusline as sl


# ---------------------------------------------------------------------------
# Formatter tests
# ---------------------------------------------------------------------------

class TestFmtModel:
    def test_display_name(self):
        result = sl._fmt_model({"display_name": "Opus 4.6"})
        assert result == "Opus 4.6"

    def test_id_fallback(self):
        result = sl._fmt_model({"id": "claude-opus-4-6"})
        assert result == "claude-opus-4-6"

    def test_none(self):
        result = sl._fmt_model(None)
        assert "unknown" in result
        assert sl.GRAY in result

    def test_string_input(self):
        result = sl._fmt_model("raw-model-string")
        assert result == "raw-model-string"

    def test_empty_dict(self):
        result = sl._fmt_model({})
        # str({}) fallback
        assert result


class TestFmtCost:
    def test_zero(self):
        assert sl._fmt_cost(0) == "$0.00"

    def test_normal(self):
        assert sl._fmt_cost(1.23) == "$1.2300"

    def test_small(self):
        assert sl._fmt_cost(0.0042) == "$0.0042"

    def test_none(self):
        assert sl._fmt_cost(None) == "$0.00"

    def test_string_input(self):
        assert sl._fmt_cost("not a number") == "$0.00"


class TestFmtContext:
    def test_zero(self):
        result = sl._fmt_context({})
        assert "0%" in result
        assert sl.GRAY in result

    def test_low(self):
        result = sl._fmt_context({"context_window": {"used_percentage": 25}})
        assert "25%" in result
        assert sl.GREEN in result

    def test_medium(self):
        result = sl._fmt_context({"context_window": {"used_percentage": 50}})
        assert "50%" in result
        assert sl.YELLOW in result

    def test_high(self):
        result = sl._fmt_context({"context_window": {"used_percentage": 80}})
        assert "80%" in result
        assert sl.RED in result

    def test_boundary_40(self):
        # At exactly 40, should be green (not yellow)
        result = sl._fmt_context({"context_window": {"used_percentage": 40}})
        assert sl.GREEN in result

    def test_boundary_41(self):
        result = sl._fmt_context({"context_window": {"used_percentage": 41}})
        assert sl.YELLOW in result

    def test_boundary_65(self):
        # At exactly 65, should be yellow (not red)
        result = sl._fmt_context({"context_window": {"used_percentage": 65}})
        assert sl.YELLOW in result

    def test_boundary_66(self):
        result = sl._fmt_context({"context_window": {"used_percentage": 66}})
        assert sl.RED in result

    def test_missing_context_window(self):
        result = sl._fmt_context({"model": "test"})
        assert "0%" in result

    def test_null_percentage(self):
        result = sl._fmt_context({"context_window": {"used_percentage": None}})
        assert "0%" in result


# ---------------------------------------------------------------------------
# Cache tests
# ---------------------------------------------------------------------------

class TestCache:
    def test_write_read_cycle(self, tmp_path):
        with mock.patch.object(sl, "CACHE_DIR", tmp_path):
            sl._write_cache("test.json", {"key": "value"})
            result = sl._read_cache("test.json", 60)
            assert result == {"key": "value"}

    def test_expired_cache(self, tmp_path):
        with mock.patch.object(sl, "CACHE_DIR", tmp_path):
            sl._write_cache("old.json", {"stale": True})
            cache_file = tmp_path / "old.json"
            old_time = os.path.getmtime(cache_file) - 120
            os.utime(cache_file, (old_time, old_time))
            result = sl._read_cache("old.json", 60)
            assert result is None

    def test_missing_cache(self, tmp_path):
        with mock.patch.object(sl, "CACHE_DIR", tmp_path):
            result = sl._read_cache("missing.json", 60)
            assert result is None

    def test_corrupt_cache(self, tmp_path):
        with mock.patch.object(sl, "CACHE_DIR", tmp_path):
            (tmp_path / "bad.json").write_text("not json{{{")
            result = sl._read_cache("bad.json", 60)
            assert result is None


# ---------------------------------------------------------------------------
# Account name tests
# ---------------------------------------------------------------------------

class TestAccountName:
    def test_env_set(self):
        with mock.patch.dict(os.environ, {"__CLAUDE_ACCOUNT__": "personal-account"}):
            assert sl._account_name() == "personal-account"

    def test_env_missing(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            assert sl._account_name() == ""


# ---------------------------------------------------------------------------
# Git info tests (mocked subprocess)
# ---------------------------------------------------------------------------

class TestGitInfo:
    @mock.patch("statusline.subprocess.run")
    def test_normal_repo(self, mock_run):
        mock_run.side_effect = [
            mock.Mock(returncode=0, stdout="main\n"),
            mock.Mock(stdout="M file.py\n"),
        ]
        result = sl._git_info()
        assert result["branch"] == "main"
        assert result["dirty"] is True

    @mock.patch("statusline.subprocess.run")
    def test_clean_repo(self, mock_run):
        mock_run.side_effect = [
            mock.Mock(returncode=0, stdout="develop\n"),
            mock.Mock(stdout=""),
        ]
        result = sl._git_info()
        assert result["branch"] == "develop"
        assert result["dirty"] is False

    @mock.patch("statusline.subprocess.run", side_effect=FileNotFoundError)
    def test_no_git(self, mock_run):
        result = sl._git_info()
        assert result["branch"] == ""
        assert result["dirty"] is False

    @mock.patch("statusline.subprocess.run")
    def test_not_a_repo(self, mock_run):
        mock_run.return_value = mock.Mock(returncode=128, stdout="", stderr="fatal")
        result = sl._git_info()
        assert result["branch"] == ""
        assert result["dirty"] is False

    @mock.patch("statusline.subprocess.run",
                side_effect=subprocess.TimeoutExpired("git", 5))
    def test_timeout(self, mock_run):
        result = sl._git_info()
        assert result["branch"] == ""


# ---------------------------------------------------------------------------
# Jarvis health tests (mocked subprocess)
# ---------------------------------------------------------------------------

class TestJarvisHealth:
    @mock.patch("statusline.subprocess.run")
    def test_healthy(self, mock_run, tmp_path):
        with mock.patch.object(sl, "CACHE_DIR", tmp_path):
            mock_run.return_value = mock.Mock(
                returncode=0,
                stdout='{"status":"ok","server":"jarvis-core","version":"3.6.0","postgres":{"status":"ok","error":null,"checked_at":1790000000.0,"free_bytes":107374182400}}',
            )
            result = sl._jarvis_health()
            assert result["ok"] is True
            assert result["pg_status"] == "ok"
            assert result["pg_free_bytes"] == 107374182400

    @mock.patch("statusline.subprocess.run")
    def test_healthy_with_replication(self, mock_run, tmp_path):
        with mock.patch.object(sl, "CACHE_DIR", tmp_path):
            mock_run.return_value = mock.Mock(
                returncode=0,
                stdout='{"status":"ok","postgres":{"status":"ok","error":null,"checked_at":1.0,"free_bytes":null},"replication":{"mode":"local"}}',
            )
            result = sl._jarvis_health()
            assert result["ok"] is True
            assert result["repl_mode"] == "local"

    @mock.patch("statusline.subprocess.run")
    def test_down(self, mock_run, tmp_path):
        with mock.patch.object(sl, "CACHE_DIR", tmp_path):
            mock_run.return_value = mock.Mock(returncode=1, stdout="")
            result = sl._jarvis_health()
            assert result["ok"] is False
            assert result["pg_status"] == ""
            assert result["pg_free_bytes"] is None

    @mock.patch("statusline.subprocess.run", side_effect=FileNotFoundError)
    def test_no_curl(self, mock_run, tmp_path):
        with mock.patch.object(sl, "CACHE_DIR", tmp_path):
            result = sl._jarvis_health()
            assert result["ok"] is False

    def test_uses_cache(self, tmp_path):
        with mock.patch.object(sl, "CACHE_DIR", tmp_path):
            sl._write_cache("jarvis.json", {"ok": True, "pg_status": "recovering", "pg_free_bytes": 42, "repl_mode": ""})
            result = sl._jarvis_health()
            assert result["ok"] is True
            assert result["pg_status"] == "recovering"
            assert result["pg_free_bytes"] == 42

    @mock.patch("statusline.subprocess.run")
    def test_caches_result(self, mock_run, tmp_path):
        with mock.patch.object(sl, "CACHE_DIR", tmp_path):
            mock_run.return_value = mock.Mock(
                returncode=0,
                stdout='{"status":"ok","postgres":{"status":"disk_full","error":"No space left on device","checked_at":1.0,"free_bytes":10}}',
            )
            sl._jarvis_health()
            cached = sl._read_cache("jarvis.json", 60)
            assert cached["ok"] is True
            assert cached["pg_status"] == "disk_full"
            assert cached["pg_free_bytes"] == 10

    @mock.patch("statusline.subprocess.run")
    def test_degraded_pg(self, mock_run, tmp_path):
        with mock.patch.object(sl, "CACHE_DIR", tmp_path):
            mock_run.return_value = mock.Mock(
                returncode=0,
                stdout='{"status":"degraded","postgres":{"status":"disconnected","error":"connection refused"}}',
            )
            result = sl._jarvis_health()
            assert result["ok"] is False
            assert result["pg_status"] == "disconnected"


# ---------------------------------------------------------------------------
# Generate tests
# ---------------------------------------------------------------------------

class TestGenerate:
    @mock.patch.object(sl, "_jarvis_health", return_value={"ok": True, "pg_status": "ok", "pg_free_bytes": None, "repl_mode": ""})
    @mock.patch.object(sl, "_git_info", return_value={"branch": "main", "dirty": False})
    @mock.patch.object(sl, "_account_name", return_value="personal-account")
    def test_full_output(self, mock_acct, mock_git, mock_jarvis, tmp_path):
        with mock.patch.object(sl, "CACHE_DIR", tmp_path):
            data = {
                "session_name": "my-session",
                "model": {"display_name": "Opus 4.6"},
                "cwd": "/Users/test/project",
                "cost": {"total_cost_usd": 0.50},
                "context_window": {"used_percentage": 45},
            }
            output = sl.generate(data)
            assert "\n" not in output
            assert "personal-account" in output
            assert "JARVIS" in output
            # A healthy database adds no segment
            assert "DB:" not in output
            assert "my-session" in output
            assert "Opus 4.6" in output
            assert "project" in output
            assert "main" in output
            assert "$0.5000" in output
            assert "45%" in output

    @mock.patch.object(sl, "_jarvis_health", return_value={"ok": True, "pg_status": "ok", "pg_free_bytes": None, "repl_mode": "local"})
    @mock.patch.object(sl, "_git_info", return_value={"branch": "main", "dirty": False})
    @mock.patch.object(sl, "_account_name", return_value="")
    def test_replication_indicator(self, mock_acct, mock_git, mock_jarvis, tmp_path):
        with mock.patch.object(sl, "CACHE_DIR", tmp_path):
            data = {"model": "test", "cwd": "/tmp", "context_window": {}}
            output = sl.generate(data)
            assert "repl:local" in output
            assert "DB:" not in output

    @mock.patch.object(sl, "_jarvis_health", return_value={"ok": True, "pg_status": "disconnected", "pg_free_bytes": None, "repl_mode": ""})
    @mock.patch.object(sl, "_git_info", return_value={"branch": "", "dirty": False})
    @mock.patch.object(sl, "_account_name", return_value="")
    def test_pg_disconnected_display(self, mock_acct, mock_git, mock_jarvis, tmp_path):
        with mock.patch.object(sl, "CACHE_DIR", tmp_path):
            data = {"model": "test", "cwd": "/tmp", "context_window": {}}
            output = sl.generate(data)
            assert "DB: disconnected" in output
            assert sl.RED in output

    @mock.patch.object(sl, "_jarvis_health", return_value={"ok": False, "pg_status": "", "pg_free_bytes": None, "repl_mode": ""})
    @mock.patch.object(sl, "_git_info", return_value={"branch": "", "dirty": False})
    @mock.patch.object(sl, "_account_name", return_value="")
    def test_minimal_data(self, mock_acct, mock_git, mock_jarvis, tmp_path):
        with mock.patch.object(sl, "CACHE_DIR", tmp_path):
            output = sl.generate({})
            assert "JARVIS" not in output
            assert "personal-account" not in output
            assert "\n" not in output

    @mock.patch.object(sl, "_jarvis_health", return_value={"ok": True})
    @mock.patch.object(sl, "_git_info", return_value={"branch": "main", "dirty": False})
    @mock.patch.object(sl, "_account_name", return_value="")
    def test_zero_cost_hidden(self, mock_acct, mock_git, mock_jarvis, tmp_path):
        with mock.patch.object(sl, "CACHE_DIR", tmp_path):
            output = sl.generate({"model": "test", "cwd": "/tmp/x"})
            assert "$" not in output

    @mock.patch.object(sl, "_jarvis_health", return_value={"ok": True})
    @mock.patch.object(sl, "_git_info", return_value={"branch": "main", "dirty": False})
    @mock.patch.object(sl, "_account_name", return_value="")
    def test_context_always_shown(self, mock_acct, mock_git, mock_jarvis, tmp_path):
        with mock.patch.object(sl, "CACHE_DIR", tmp_path):
            output = sl.generate({"model": "test", "cwd": "/tmp/x"})
            assert "ctx" in output
            assert "0%" in output

    @mock.patch.object(sl, "_jarvis_health", return_value={"ok": True})
    @mock.patch.object(sl, "_git_info", return_value={"branch": "feature/x", "dirty": True})
    @mock.patch.object(sl, "_account_name", return_value="")
    def test_dirty_marker(self, mock_acct, mock_git, mock_jarvis, tmp_path):
        with mock.patch.object(sl, "CACHE_DIR", tmp_path):
            output = sl.generate({"model": "test", "cwd": "/tmp/test"})
            assert "feature/x*" in output

    @mock.patch.object(sl, "_jarvis_health", return_value={"ok": True})
    @mock.patch.object(sl, "_git_info", return_value={"branch": "", "dirty": False})
    @mock.patch.object(sl, "_account_name", return_value="")
    def test_no_git_omitted(self, mock_acct, mock_git, mock_jarvis, tmp_path):
        with mock.patch.object(sl, "CACHE_DIR", tmp_path):
            output = sl.generate({"model": "test", "cwd": "/tmp/x"})
            # No branch info when not in a git repo
            assert "no-git" not in output
            # Should still have other segments
            assert "test" in output

    @mock.patch.object(sl, "_jarvis_health", return_value={"ok": False})
    @mock.patch.object(sl, "_git_info", return_value={"branch": "main", "dirty": False})
    @mock.patch.object(sl, "_account_name", return_value="")
    def test_jarvis_down_no_branding(self, mock_acct, mock_git, mock_jarvis, tmp_path):
        with mock.patch.object(sl, "CACHE_DIR", tmp_path):
            output = sl.generate({"model": "test", "cwd": "/tmp/x"})
            assert "JARVIS" not in output
            assert "\u26a1" not in output


class TestSessionLabel:
    @mock.patch.object(sl, "_jarvis_health", return_value={"ok": True})
    @mock.patch.object(sl, "_git_info", return_value={"branch": "main", "dirty": False})
    @mock.patch.object(sl, "_account_name", return_value="")
    def test_session_name_shown(self, mock_acct, mock_git, mock_jarvis, tmp_path):
        with mock.patch.object(sl, "CACHE_DIR", tmp_path):
            data = {"session_name": "ayo-silver", "session_id": "abc12345",
                    "model": "test", "cwd": "/tmp"}
            output = sl.generate(data)
            assert "ayo-silver" in output

    @mock.patch.object(sl, "_jarvis_health", return_value={"ok": True})
    @mock.patch.object(sl, "_git_info", return_value={"branch": "main", "dirty": False})
    @mock.patch.object(sl, "_account_name", return_value="")
    def test_session_id_fallback(self, mock_acct, mock_git, mock_jarvis, tmp_path):
        with mock.patch.object(sl, "CACHE_DIR", tmp_path):
            data = {"session_id": "92751608-e0ca-45af-b719-bc157175b6ac",
                    "model": "test", "cwd": "/tmp"}
            output = sl.generate(data)
            assert "92751608" in output
            # Full UUID should NOT be shown
            assert "92751608-e0ca" not in output

    @mock.patch.object(sl, "_jarvis_health", return_value={"ok": True})
    @mock.patch.object(sl, "_git_info", return_value={"branch": "main", "dirty": False})
    @mock.patch.object(sl, "_account_name", return_value="")
    def test_session_name_preferred_over_id(self, mock_acct, mock_git, mock_jarvis, tmp_path):
        with mock.patch.object(sl, "CACHE_DIR", tmp_path):
            data = {"session_name": "my-session", "session_id": "abc12345",
                    "model": "test", "cwd": "/tmp"}
            output = sl.generate(data)
            assert "my-session" in output
            assert "abc12345" not in output

    @mock.patch.object(sl, "_jarvis_health", return_value={"ok": True})
    @mock.patch.object(sl, "_git_info", return_value={"branch": "main", "dirty": False})
    @mock.patch.object(sl, "_account_name", return_value="")
    def test_no_session_info(self, mock_acct, mock_git, mock_jarvis, tmp_path):
        with mock.patch.object(sl, "CACHE_DIR", tmp_path):
            data = {"model": "test", "cwd": "/tmp"}
            output = sl.generate(data)
            # Should still render without session segment
            assert "test" in output


class TestFolderEmoji:
    @mock.patch.object(sl, "_jarvis_health", return_value={"ok": True})
    @mock.patch.object(sl, "_git_info", return_value={"branch": "main", "dirty": False})
    @mock.patch.object(sl, "_account_name", return_value="")
    def test_folder_emoji_present(self, mock_acct, mock_git, mock_jarvis, tmp_path):
        with mock.patch.object(sl, "CACHE_DIR", tmp_path):
            output = sl.generate({"model": "test", "cwd": "/Users/test/my-project"})
            assert "\U0001f4c1" in output
            assert "my-project" in output


# ---------------------------------------------------------------------------
# /health postgres object rendering (cached server-side DB probe)
# ---------------------------------------------------------------------------

def _health_json(pg_status, free_bytes=None, top_status="ok"):
    return json.dumps({
        "status": top_status,
        "server": "jarvis-core",
        "version": "3.6.0",
        "postgres": {
            "status": pg_status,
            "error": None if pg_status == "ok" else "the database system is in recovery mode",
            "checked_at": 1790000000.0,
            "free_bytes": free_bytes,
        },
    })


def _render_with_health(tmp_path, stdout, returncode=0):
    """Run generate() with curl mocked to return `stdout` for /health."""
    with mock.patch.object(sl, "CACHE_DIR", tmp_path), \
         mock.patch.object(sl, "_git_info", return_value={"branch": "", "dirty": False}), \
         mock.patch.object(sl, "_account_name", return_value=""), \
         mock.patch("statusline.subprocess.run",
                    return_value=mock.Mock(returncode=returncode, stdout=stdout)):
        return sl.generate({"model": "test", "cwd": "/tmp/x"})


class TestPostgresStatus:
    @pytest.mark.parametrize("pg_status,label,color", [
        ("recovering", "DB: recovering", sl.RED),
        ("unreachable", "DB: unreachable", sl.RED),
        ("disk_full", "DB: disk full", sl.RED),
        ("unknown", "DB: unknown", sl.YELLOW),
    ])
    def test_not_ok_status_renders_segment(self, pg_status, label, color, tmp_path):
        output = _render_with_health(tmp_path, _health_json(pg_status))
        assert "JARVIS" in output
        assert f"{color}{label}" in output

    def test_ok_status_renders_no_segment(self, tmp_path):
        output = _render_with_health(tmp_path, _health_json("ok", 10 ** 11))
        assert "JARVIS" in output
        assert "DB:" not in output

    def test_disk_full_shows_free_space(self, tmp_path):
        output = _render_with_health(tmp_path, _health_json("disk_full", 12 * 1024 * 1024))
        assert "DB: disk full (12M free)" in output

    def test_free_space_only_shown_for_disk_full(self, tmp_path):
        output = _render_with_health(tmp_path, _health_json("recovering", 12 * 1024 * 1024))
        assert "DB: recovering" in output
        assert "free" not in output

    def test_top_level_ok_gate_still_applies(self, tmp_path):
        """A non-ok top-level status hides branding and the DB segment."""
        output = _render_with_health(tmp_path, _health_json("recovering", top_status="starting"))
        assert "JARVIS" not in output
        assert "DB:" not in output

    def test_health_without_postgres_key(self, tmp_path):
        """Older servers without the postgres object still render JARVIS."""
        output = _render_with_health(tmp_path, '{"status":"ok","server":"jarvis-core"}')
        assert "JARVIS" in output
        assert "DB:" not in output

    @pytest.mark.parametrize("pg_value", ["null", '"recovering"', "[1, 2]", "42"])
    def test_malformed_postgres_value_ignored(self, pg_value, tmp_path):
        output = _render_with_health(tmp_path, '{"status":"ok","postgres":%s}' % pg_value)
        assert "JARVIS" in output
        assert "DB:" not in output
        assert "statusline error" not in output

    def test_non_object_health_body(self, tmp_path):
        with mock.patch.object(sl, "CACHE_DIR", tmp_path), \
             mock.patch("statusline.subprocess.run",
                        return_value=mock.Mock(returncode=0, stdout="[1, 2, 3]")):
            result = sl._jarvis_health()
        assert result["ok"] is False

    def test_unrecognized_status_is_sanitized(self, tmp_path):
        """Server text never reaches the terminal raw (no escape injection)."""
        hostile = "\x1b[2J\x1b]0;owned\x07Bad Status!"
        output = _render_with_health(tmp_path, _health_json(hostile))
        assert "\x1b[2J" not in output
        assert "\x07" not in output
        assert "]0;owned" not in output
        assert f"{sl.RED}DB: 2j0ownedbadstatus" in output

    def test_error_text_not_rendered(self, tmp_path):
        output = _render_with_health(tmp_path, _health_json("recovering"))
        assert "recovery mode" not in output

    def test_curl_timeout_drops_branding(self, tmp_path):
        with mock.patch.object(sl, "CACHE_DIR", tmp_path), \
             mock.patch("statusline.subprocess.run",
                        side_effect=subprocess.TimeoutExpired("curl", 3)):
            result = sl._jarvis_health()
        assert result["ok"] is False
        assert result["pg_status"] == ""

    def test_probe_is_time_bounded(self, tmp_path):
        with mock.patch.object(sl, "CACHE_DIR", tmp_path), \
             mock.patch("statusline.subprocess.run",
                        return_value=mock.Mock(returncode=0, stdout=_health_json("ok"))) as run:
            sl._jarvis_health()
        args, kwargs = run.call_args
        assert "--max-time" in args[0]
        assert kwargs["timeout"] <= 3


class TestFmtBytes:
    @pytest.mark.parametrize("value,expected", [
        (0, "0B"),
        (512, "512B"),
        (1536, "1.5K"),
        (12 * 1024 * 1024, "12M"),
        (int(2.5 * 1024 ** 3), "2.5G"),
        (5 * 1024 ** 5, "5120T"),
    ])
    def test_units(self, value, expected):
        assert sl._fmt_bytes(value) == expected

    @pytest.mark.parametrize("value", [None, -1, "12", 1.5, True])
    def test_invalid(self, value):
        assert sl._fmt_bytes(value) == ""


# ---------------------------------------------------------------------------
# Main tests
# ---------------------------------------------------------------------------

class TestMain:
    def test_demo_mode(self, capsys):
        with mock.patch("sys.argv", ["statusline.py", "--demo"]):
            with mock.patch.object(sl, "_jarvis_health", return_value={"ok": True}):
                with mock.patch.object(sl, "_git_info", return_value={"branch": "main", "dirty": False}):
                    sl.main()
        output = capsys.readouterr().out
        assert "Opus 4.6" in output
        assert "demo-session" in output

    def test_stdin_mode(self, capsys):
        data = json.dumps({
            "model": {"display_name": "Haiku 4.5"},
            "cwd": "/tmp/test",
            "context_window": {"used_percentage": 30},
        })
        with mock.patch("sys.stdin", mock.Mock(isatty=lambda: False, read=lambda: data)):
            with mock.patch("sys.argv", ["statusline.py"]):
                with mock.patch.object(sl, "_jarvis_health", return_value={"ok": False}):
                    with mock.patch.object(sl, "_git_info", return_value={"branch": "", "dirty": False}):
                        sl.main()
        output = capsys.readouterr().out
        assert "Haiku 4.5" in output

    def test_error_handling(self, capsys):
        with mock.patch("sys.stdin", mock.Mock(isatty=lambda: False, read=mock.Mock(side_effect=Exception("boom")))):
            with mock.patch("sys.argv", ["statusline.py"]):
                sl.main()
        output = capsys.readouterr().out
        assert "jarvis statusline error" in output
