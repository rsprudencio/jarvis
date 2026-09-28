"""Outage behavior of the `jarvis` launcher (plugins/jarvis/shell/jarvis.sh).

Runs the real script with PATH shims for claude/docker and small servers on
127.0.0.1 ephemeral ports. Every probe must be bounded, and a wedged core, a
crash-looping container, or Docker being absent/hung must never keep the
launcher from reaching `exec claude`.
"""

from __future__ import annotations

import http.server
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[4]
LAUNCHER = REPO_ROOT / "plugins/jarvis/shell/jarvis.sh"
CURL = shutil.which("curl")
SLEEP = shutil.which("sleep")

pytestmark = pytest.mark.skipif(
    sys.platform == "win32" or not Path("/bin/bash").exists() or not CURL or not SLEEP,
    reason="launcher tests need /bin/bash, curl and sleep",
)

JARVIS_PLUGINS = json.dumps([{"id": "jarvis@jarvis-plugins", "version": "3.6.0"}])

CLAUDE_SHIM = r"""#!/bin/bash
if [ "${1:-} ${2:-}" = "plugin list" ]; then
    case "${PLUGIN_LIST:-ok}" in
        hang) printf '%s\n' "$$" > "$SHIM_DIR/plugin_list.pid"; exec sleep 30 ;;
        fail) echo "plugin list failed" >&2; exit 1 ;;
        garbage) echo "not json" ;;
        *) printf '%s\n' "$PLUGIN_LIST_JSON" ;;
    esac
    exit 0
fi
if [ "$#" -gt 0 ]; then printf '%s\0' "$@"; fi > "$SHIM_DIR/claude.argv"
exit 0
"""

# `hang` mimics a wedged `docker` whose `docker-compose` plugin child keeps
# the pipe open: only a process-group kill stops both. `hang_graceful` is what
# the real `docker compose` does on SIGTERM: cancel and exit 130, not die of it.
DOCKER_SHIM = r"""#!/bin/bash
printf '%s\n' "$*" >> "$SHIM_DIR/docker.log"
hang() {
    printf '%s\n' "$$" > "$SHIM_DIR/docker_hang.pid"
    sleep 30 &
    printf '%s\n' "$!" > "$SHIM_DIR/docker_hang_child.pid"
    wait
}
hang_graceful() {
    trap 'exit 130' TERM
    hang
}
case " $* " in
    *" ps "*)
        ps_mode="${DOCKER_PS:-exited}"
        if [ -e "$SHIM_DIR/compose_up.done" ] && [ -n "${DOCKER_PS_AFTER_UP:-}" ]; then
            ps_mode="$DOCKER_PS_AFTER_UP"
        fi
        case "$ps_mode" in
            hang) hang ;;
            hang_graceful) hang_graceful ;;
            fail) echo "Cannot connect to the Docker daemon. Is the docker daemon running?" >&2; exit 1 ;;
            none) ;;
            *) printf '%s\n' "$ps_mode" ;;
        esac
        ;;
    *" up "*)
        case "${DOCKER_UP:-ok}" in
            hang) hang ;;
            hang_graceful) hang_graceful ;;
            fail) echo "no space left on device" >&2; exit 1 ;;
            serve)  # a real core coming up: serve $SHIM_DIR/www/health on the core port
                python3 -m http.server --bind 127.0.0.1 --directory "$SHIM_DIR/www" "${JARVIS_CORE_URL##*:}" \
                    </dev/null >/dev/null 2>&1 &
                printf '%s\n' "$!" > "$SHIM_DIR/core_server.pid"
                ;;
            *) : > "$SHIM_DIR/compose_up.done"; echo " Container jarvis-jarvis-1  Started" >&2 ;;
        esac
        ;;
esac
exit 0
"""


def _executable(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")
    path.chmod(0o755)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _gone(pid: int, within: float = 3.0) -> bool:
    deadline = time.monotonic() + within
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return True
        except PermissionError:
            return False
        time.sleep(0.05)
    return False


class _HealthHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):  # noqa: N802 - http.server API
        status, body = self.server.reply  # type: ignore[attr-defined]
        payload = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *args):
        pass


def _write_bin(bin_dir: Path) -> Path:
    """The launcher's whole PATH: shims plus only the real tools it needs, so
    no real docker/claude can leak in."""
    bin_dir.mkdir()
    _executable(bin_dir / "claude", CLAUDE_SHIM)
    _executable(bin_dir / "docker", DOCKER_SHIM)
    _executable(bin_dir / "curl", f'#!/bin/bash\nexec "{CURL}" "$@"\n')
    _executable(bin_dir / "sleep", f'#!/bin/bash\nexec "{SLEEP}" "$@"\n')
    _executable(bin_dir / "python3", f'#!/bin/bash\nexec "{sys.executable}" "$@"\n')
    # Pay macOS's first-exec cost for new executables here, not inside a
    # timed launcher run.
    scratch = bin_dir.parent / "warmup"
    scratch.mkdir(exist_ok=True)
    env = {"PATH": str(bin_dir), "SHIM_DIR": str(scratch), "PLUGIN_LIST_JSON": "[]"}
    for cmd in (["claude", "plugin", "list"], ["docker", "version"], ["curl", "--version"],
                ["sleep", "0"], ["python3", "-c", "pass"]):
        subprocess.run([str(bin_dir / cmd[0]), *cmd[1:]], env=env, capture_output=True, timeout=30, check=False)
    return bin_dir


class Harness:
    """Shims, fake servers and a runner for one launcher invocation. Per-run
    state lives in shim_dir; the shim scripts themselves are shared (a fresh
    executable costs ~0.5s on its first exec on macOS)."""

    def __init__(self, tmp_path: Path, shared_bin: Path):
        self.tmp = tmp_path
        self.shim_dir = tmp_path / "shim"
        self.shim_dir.mkdir()
        self.bin = shared_bin
        self.jarvis_home = tmp_path / "jarvis-home"
        self.jarvis_home.mkdir()
        (self.jarvis_home / "docker-compose.yml").write_text("services: {}\n", encoding="utf-8")
        self.core_url = f"http://127.0.0.1:{_free_port()}"  # refused until a server binds it
        self._cleanups: list = []

    # --- fake core servers -------------------------------------------------

    def serve_health(self, body: dict | None = None, status: int = 200, port: int | None = None) -> None:
        server = http.server.ThreadingHTTPServer(("127.0.0.1", port or 0), _HealthHandler)
        server.daemon_threads = True
        server.reply = (status, body if body is not None else {"status": "ok", "server": "jarvis-core"})
        threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True).start()
        self._cleanups.append(server.server_close)
        self._cleanups.append(server.shutdown)
        self.core_url = f"http://127.0.0.1:{server.server_address[1]}"

    def serve_wedged(self) -> None:
        """Completes the TCP handshake (listen backlog) but never answers,
        like a core whose event loop is frozen by blocking DB calls."""
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind(("127.0.0.1", 0))
        sock.listen(16)
        self._cleanups.append(sock.close)
        self.core_url = f"http://127.0.0.1:{sock.getsockname()[1]}"

    def serve_health_after(self, marker: str | None = None, delay: float = 0.0) -> None:
        """Bind the (currently refused) core port once `marker` appears in the
        shim dir, or after `delay` seconds: a container that finishes booting."""
        port = int(self.core_url.rsplit(":", 1)[1])
        stop = threading.Event()

        def _later():
            deadline = time.monotonic() + 15
            started = time.monotonic()
            while not stop.is_set() and time.monotonic() < deadline:
                ready = (self.shim_dir / marker).exists() if marker else time.monotonic() - started >= delay
                if ready:
                    self.serve_health(port=port)
                    return
                time.sleep(0.05)

        threading.Thread(target=_later, daemon=True).start()
        self._cleanups.append(stop.set)

    def own_bin(self) -> Path:
        """A private copy of the shims, for tests that remove or replace one."""
        if self.bin.parent != self.tmp:
            self.bin = _write_bin(self.tmp / "bin")
        return self.bin

    def close(self) -> None:
        for fn in reversed(self._cleanups):
            try:
                fn()
            except Exception:
                pass
        # Shim processes a failed test may have left behind (hung docker, core server).
        for pid_file in self.shim_dir.glob("*.pid"):
            try:
                os.kill(int(pid_file.read_text()), signal.SIGKILL)
            except (ValueError, ProcessLookupError, PermissionError):
                pass

    # --- running the launcher ----------------------------------------------

    def env(self, **overrides: str) -> dict:
        env = {
            "PATH": str(self.bin),
            "HOME": str(self.tmp / "home"),
            "JARVIS_HOME": str(self.jarvis_home),
            "JARVIS_CORE_URL": self.core_url,
            "SHIM_DIR": str(self.shim_dir),
            "PLUGIN_LIST_JSON": JARVIS_PLUGINS,
        }
        if "TMPDIR" in os.environ:
            env["TMPDIR"] = os.environ["TMPDIR"]
        env.update(overrides)
        return env

    def run(self, *args: str, timeout: float = 20, **env: str):
        started = time.monotonic()
        result = subprocess.run(
            ["/bin/bash", str(LAUNCHER), *args],
            env=self.env(**env),
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=timeout,
            start_new_session=True,
            check=False,
        )
        elapsed = time.monotonic() - started
        # Diagnostics belong on stderr; stdout stays clean for claude.
        assert result.stdout == "", result.stdout
        return result, elapsed

    def claude_argv(self) -> list[str] | None:
        path = self.shim_dir / "claude.argv"
        if not path.exists():
            return None
        raw = path.read_bytes()
        return raw.decode().split("\0")[:-1] if raw else []

    def docker_calls(self) -> list[str]:
        path = self.shim_dir / "docker.log"
        return path.read_text(encoding="utf-8").splitlines() if path.exists() else []

    def hang_pids(self, *names: str) -> list[int]:
        return [int((self.shim_dir / n).read_text()) for n in names]


@pytest.fixture(scope="module")
def shared_bin(tmp_path_factory):
    return _write_bin(tmp_path_factory.mktemp("launcher") / "bin")


@pytest.fixture
def harness(tmp_path, shared_bin):
    h = Harness(tmp_path, shared_bin)
    yield h
    h.close()


# --- static checks ------------------------------------------------------------


def test_launcher_passes_bash_syntax_check():
    result = subprocess.run(["/bin/bash", "-n", str(LAUNCHER)], capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr


@pytest.mark.skipif(shutil.which("shellcheck") is None, reason="shellcheck not installed")
def test_launcher_passes_shellcheck():
    result = subprocess.run(["shellcheck", str(LAUNCHER)], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr


def test_launcher_no_longer_uses_set_e():
    lines = [ln.strip() for ln in LAUNCHER.read_text(encoding="utf-8").splitlines()]
    assert "set -u" in lines
    assert "set -e" not in lines


# --- healthy / degraded core ----------------------------------------------------


def test_healthy_core_execs_claude_fast_with_argv_passed_through(harness):
    harness.serve_health()
    args = ["--model", "opus", "two words", "--", "-x", "", "*"]
    result, elapsed = harness.run(*args)
    assert result.returncode == 0, result.stderr
    assert harness.claude_argv() == args
    assert harness.docker_calls() == []
    assert result.stderr == ""
    assert elapsed < 1.5


def test_no_arguments_under_set_u(harness):
    harness.serve_health()
    result, _ = harness.run()
    assert result.returncode == 0, result.stderr
    assert harness.claude_argv() == []


def test_wedged_core_warns_and_execs_without_touching_docker(harness):
    harness.serve_wedged()
    result, elapsed = harness.run("hi")
    assert result.returncode == 0, result.stderr
    assert harness.claude_argv() == ["hi"]
    assert "Jarvis core is not responding (event loop busy / DB outage?)" in result.stderr
    assert harness.docker_calls() == []  # no compose, no polling
    assert elapsed < 4


def test_non_200_health_warns_and_execs(harness):
    harness.serve_health({"error": "boom"}, status=500)
    result, _ = harness.run()
    assert result.returncode == 0
    assert harness.claude_argv() == []
    assert "HTTP 500" in result.stderr
    assert harness.docker_calls() == []


@pytest.mark.parametrize("status", ["recovering", "unreachable"])
def test_degraded_postgres_prints_one_line_hint(harness, status):
    harness.serve_health({"status": "ok", "postgres": {"status": status, "error": "secret-ish detail"}})
    result, _ = harness.run()
    assert result.returncode == 0
    assert harness.claude_argv() == []
    assert f"postgres={status}" in result.stderr
    assert "secret-ish detail" not in result.stderr
    assert len(result.stderr.strip().splitlines()) == 1


def test_disk_full_postgres_points_at_docker_disk_usage(harness):
    harness.serve_health({"status": "ok", "postgres": {"status": "disk_full", "free_bytes": 0}})
    result, _ = harness.run()
    assert "postgres=disk_full" in result.stderr
    assert "docker system df" in result.stderr


@pytest.mark.parametrize(
    "body",
    [
        {"status": "ok"},
        {"status": "ok", "postgres": {"status": "ok"}},
        {"status": "ok", "postgres": {"status": "unknown", "checked_at": None}},
        {"status": "ok", "postgres": None},
        {"status": "ok", "postgres": {"status": "\u001b[31mred"}},
    ],
    ids=["absent", "ok", "unknown", "null", "escape-sequence"],
)
def test_postgres_hint_silent_when_ok_absent_or_unsafe(harness, body):
    harness.serve_health(body)
    result, _ = harness.run()
    assert result.returncode == 0
    assert harness.claude_argv() == []
    assert result.stderr == ""


# --- core down: container state ---------------------------------------------------


def test_crash_looping_container_hints_and_execs_without_compose_up(harness):
    result, elapsed = harness.run("x", DOCKER_PS="restarting")
    assert result.returncode == 0
    assert harness.claude_argv() == ["x"]
    assert "crash-looping" in result.stderr
    assert "logs --tail 30" in result.stderr
    assert "docker system df" in result.stderr
    assert not any("up -d" in c for c in harness.docker_calls())
    assert elapsed < 3


def test_stopped_container_is_started_then_waited_for(harness):
    harness.serve_health_after(marker="compose_up.done")
    result, _ = harness.run("go", DOCKER_PS="exited")
    assert result.returncode == 0, result.stderr
    assert harness.claude_argv() == ["go"]
    assert "Starting Jarvis container..." in result.stderr
    assert "ready." in result.stderr
    assert any("up -d" in c for c in harness.docker_calls())


def test_booting_container_is_waited_for_without_compose_up(harness):
    harness.serve_health_after(delay=1.2)
    result, _ = harness.run(DOCKER_PS="running")
    assert result.returncode == 0, result.stderr
    assert "ready." in result.stderr
    assert not any("up -d" in c for c in harness.docker_calls())


def test_wait_stops_early_when_container_starts_crash_looping(harness):
    result, elapsed = harness.run(
        DOCKER_PS="exited", DOCKER_PS_AFTER_UP="restarting", JARVIS_START_TIMEOUT="15"
    )
    assert result.returncode == 0
    assert harness.claude_argv() == []
    assert "crash-looping" in result.stderr
    assert elapsed < 5


def test_container_never_ready_gives_up_at_start_timeout(harness):
    result, elapsed = harness.run(DOCKER_PS="exited", JARVIS_START_TIMEOUT="2")
    assert result.returncode == 0
    assert harness.claude_argv() == []
    assert "not ready after 2s" in result.stderr
    assert "logs --tail 200" in result.stderr
    assert elapsed < 6


def test_compose_up_failure_execs_without_waiting(harness):
    result, _ = harness.run(DOCKER_PS="exited", DOCKER_UP="fail")
    assert result.returncode == 0
    assert harness.claude_argv() == []
    assert "no space left on device" in result.stderr  # compose's own error is shown
    assert "docker compose up failed (exit 1)" in result.stderr
    assert "Waiting for Jarvis core" not in result.stderr


@pytest.mark.parametrize("mode", ["hang", "hang_graceful"])
def test_hung_compose_up_is_killed_at_compose_timeout(harness, mode):
    result, elapsed = harness.run(DOCKER_PS="exited", DOCKER_UP=mode, JARVIS_COMPOSE_TIMEOUT="1")
    assert result.returncode == 0
    assert harness.claude_argv() == []
    assert "did not finish in 1s" in result.stderr
    assert elapsed < 5
    for pid in harness.hang_pids("docker_hang.pid", "docker_hang_child.pid"):
        assert _gone(pid), f"pid {pid} survived the timeout"


# --- docker absent / down / hung --------------------------------------------------


def test_docker_missing_still_execs(harness):
    (harness.own_bin() / "docker").unlink()
    result, elapsed = harness.run("a")
    assert result.returncode == 0
    assert harness.claude_argv() == ["a"]
    assert "docker not found" in result.stderr
    assert elapsed < 3


def test_docker_daemon_down_still_execs(harness):
    result, _ = harness.run(DOCKER_PS="fail")
    assert result.returncode == 0
    assert harness.claude_argv() == []
    assert "Docker is not reachable" in result.stderr
    assert not any("up -d" in c for c in harness.docker_calls())


@pytest.mark.parametrize("mode", ["hang", "hang_graceful"])
def test_hung_docker_is_bounded_and_its_process_group_killed(harness, mode):
    result, elapsed = harness.run("b", DOCKER_PS=mode, JARVIS_DOCKER_TIMEOUT="2")
    assert result.returncode == 0
    assert harness.claude_argv() == ["b"]
    assert "Docker is not responding (no answer in 2s)" in result.stderr
    assert 1.5 <= elapsed < 5
    for pid in harness.hang_pids("docker_hang.pid", "docker_hang_child.pid"):
        assert _gone(pid), f"pid {pid} survived the timeout"


def test_no_compose_file_skips_probe_and_docker(harness):
    (harness.jarvis_home / "docker-compose.yml").unlink()
    harness.serve_wedged()  # would cost 2s if it were probed
    result, elapsed = harness.run("c")
    assert result.returncode == 0
    assert harness.claude_argv() == ["c"]
    assert harness.docker_calls() == []
    assert elapsed < 1.5


# --- plugin check -----------------------------------------------------------------


def test_plugin_list_timeout_warns_and_continues(harness):
    harness.serve_health()
    result, elapsed = harness.run("d", PLUGIN_LIST="hang", JARVIS_PLUGIN_CHECK_TIMEOUT="1")
    assert result.returncode == 0
    assert harness.claude_argv() == ["d"]
    assert "Could not verify the Jarvis plugin install (timed out)" in result.stderr
    assert "not installed" not in result.stderr
    assert elapsed < 4
    assert _gone(harness.hang_pids("plugin_list.pid")[0])


@pytest.mark.parametrize("mode", ["fail", "garbage"])
def test_plugin_list_error_warns_and_continues(harness, mode):
    harness.serve_health()
    result, _ = harness.run(PLUGIN_LIST=mode)
    assert result.returncode == 0
    assert harness.claude_argv() == []
    assert "Could not verify the Jarvis plugin install" in result.stderr
    assert "not installed" not in result.stderr


def test_plugin_genuinely_not_installed_exits_1(harness):
    harness.serve_health()
    result, _ = harness.run(PLUGIN_LIST_JSON=json.dumps([{"id": "other@elsewhere"}]))
    assert result.returncode == 1
    assert harness.claude_argv() is None
    assert "Error: Jarvis core plugin not installed." in result.stderr
    assert "claude plugin install jarvis@jarvis-plugins" in result.stderr


def test_skip_plugin_check(harness):
    harness.serve_health()
    result, _ = harness.run(JARVIS_SKIP_PLUGIN_CHECK="1", PLUGIN_LIST_JSON="[]")
    assert result.returncode == 0
    assert harness.claude_argv() == []


def test_missing_claude_is_a_clear_error(harness):
    (harness.own_bin() / "claude").unlink()
    result, _ = harness.run()
    assert result.returncode == 127
    assert "claude CLI not found" in result.stderr


def test_non_numeric_timeouts_fall_back_to_defaults(harness):
    harness.serve_health()
    result, _ = harness.run(JARVIS_START_TIMEOUT="soon", JARVIS_DOCKER_TIMEOUT="-1")
    assert result.returncode == 0, result.stderr
    assert harness.claude_argv() == []


# --- terminal / job control ---------------------------------------------------------


# Runs in a fresh single-threaded interpreter (forking the multi-threaded
# pytest process for a pty is unsafe): starts the launcher as session leader on
# a new pty, types a line once claude is up, relays the output, and exits with
# the launcher's status.
PTY_DRIVER = r"""
import os, pty, select, sys, time
launcher, shim_dir = sys.argv[1], sys.argv[2]
pid, fd = pty.fork()
if pid == 0:
    os.execv("/bin/bash", ["/bin/bash", launcher])
out, status, fed, deadline = b"", None, False, time.monotonic() + 15
while status is None and time.monotonic() < deadline:
    if not fed and os.path.exists(os.path.join(shim_dir, "claude.fg")):
        os.write(fd, b"typed\n")
        fed = True
    if select.select([fd], [], [], 0.05)[0]:
        try:
            out += os.read(fd, 4096)
        except OSError:
            pass
    done, st = os.waitpid(pid, os.WNOHANG)
    if done:
        status = os.waitstatus_to_exitcode(st)
if status is None:
    os.kill(pid, 9)
sys.stdout.write(out.decode(errors="replace"))
sys.exit(99 if status is None else status)
"""


@pytest.mark.skipif(shutil.which("ps") is None, reason="needs ps")
def test_claude_owns_the_terminal_after_bounded_docker_calls(harness):
    """with_timeout toggles `set -m`; under a real tty, claude must still end up
    in the terminal's foreground process group and be able to read from it."""
    ps = shutil.which("ps")
    (harness.shim_dir / "www").mkdir()
    (harness.shim_dir / "www/health").write_text('{"status": "ok"}', encoding="utf-8")
    _executable(
        harness.own_bin() / "claude",
        f"""#!/bin/bash
if [ "${{1:-}} ${{2:-}}" = "plugin list" ]; then printf '%s\\n' "$PLUGIN_LIST_JSON"; exit 0; fi
"{ps}" -o pgid= -o tpgid= -p $$ > "$SHIM_DIR/claude.fg"
read -r -t 5 line
printf '%s\\n' "$line" > "$SHIM_DIR/claude.stdin"
exit 0
""",
    )
    result = subprocess.run(
        [sys.executable, "-c", PTY_DRIVER, str(LAUNCHER), str(harness.shim_dir)],
        env=harness.env(DOCKER_PS="exited", DOCKER_UP="serve"),
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=30,
        start_new_session=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "ready." in result.stdout
    pgid, tpgid = (harness.shim_dir / "claude.fg").read_text().split()
    assert pgid == tpgid, "claude is not in the terminal's foreground process group"
    assert (harness.shim_dir / "claude.stdin").read_text().strip() == "typed"


# --- Ctrl-C -----------------------------------------------------------------------


def test_ctrl_c_during_hung_docker_kills_its_group_and_exits_130(harness):
    proc = subprocess.Popen(
        ["/bin/bash", str(LAUNCHER)],
        env=harness.env(DOCKER_PS="hang", JARVIS_DOCKER_TIMEOUT="20"),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    try:
        child_pid_file = harness.shim_dir / "docker_hang_child.pid"
        deadline = time.monotonic() + 10
        while not child_pid_file.exists() and time.monotonic() < deadline:
            time.sleep(0.05)
        assert child_pid_file.exists(), "docker shim never started"
        time.sleep(0.1)
        started = time.monotonic()
        os.killpg(proc.pid, signal.SIGINT)  # what the terminal does on Ctrl-C
        proc.wait(timeout=5)
        elapsed = time.monotonic() - started
    finally:
        if proc.poll() is None:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
        proc.stdout.close()
        proc.stderr.close()
    assert proc.returncode == 130
    assert elapsed < 3
    assert harness.claude_argv() is None
    for pid in harness.hang_pids("docker_hang.pid", "docker_hang_child.pid"):
        assert _gone(pid), f"pid {pid} was orphaned by Ctrl-C"
