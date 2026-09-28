"""Deploy scripts: entrypoint supervision, compose exposure/logging, installer
compose generation + launcher refresh, and the `jarvis-transport.sh backup`
command.

Regression guards for the 2026-09-24/25 outage: a full Docker VM disk put the
embedded Postgres into a 15h crash/recovery loop that nothing supervised, and
every unbounded health probe in the shell tooling hung behind the frozen core.

Everything here is hermetic: bash functions are sourced from the real scripts
(via their *_LIB_ONLY guards) with fake docker/su/pg_ctl/pg_isready/df/curl
binaries on PATH, and the only network use is localhost servers started on
ephemeral ports. No Docker required.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import socket
import stat
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[4]
ENTRYPOINT = REPO_ROOT / "docker" / "entrypoint.sh"
REPO_COMPOSE = REPO_ROOT / "docker" / "docker-compose.yml"
INSTALLER = REPO_ROOT / "install.sh"
TRANSPORT = REPO_ROOT / "plugins" / "jarvis" / "scripts" / "jarvis-transport.sh"
LAUNCHER_SRC = REPO_ROOT / "plugins" / "jarvis" / "shell" / "jarvis.sh"

EDITED_SCRIPTS = [ENTRYPOINT, INSTALLER, TRANSPORT]

# Env that would change script behavior (or touch the real machine) if it
# leaked in from the developer's shell.
_SCRUB_ENV = (
    "CLAUDE_CONFIG_DIR", "JARVIS_HOME", "JARVIS_TLS_CERT", "JARVIS_TLS_KEY",
    "JARVIS_TLS_CA", "POSTGRES_URL", "PGDATA", "JARVIS_BIND_HOST",
    "JARVIS_BACKUP_KEEP", "JARVIS_BACKUP_TIMEOUT", "TODOIST_API_TOKEN",
)


def _executable(path: Path, content: str) -> Path:
    path.write_text(content, encoding="utf-8")
    path.chmod(0o755)
    return path


def _base_env(tmp_path: Path, fake_bin: Path | None = None) -> dict[str, str]:
    env = os.environ.copy()
    for key in _SCRUB_ENV:
        env.pop(key, None)
    home = tmp_path / "home"
    home.mkdir(exist_ok=True)
    env["HOME"] = str(home)
    env["JARVIS_HOME"] = str(home / ".jarvis")
    path = "/usr/bin:/bin:/usr/sbin:/sbin"
    env["PATH"] = f"{fake_bin}:{path}" if fake_bin else path
    return env


def _bash(script: str, env: dict[str, str], timeout: float = 10) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["/bin/bash", "-c", script],
        env=env,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def _mode(path: Path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


# ─── Syntax ─────────────────────────────────────────────────────────


@pytest.mark.parametrize("script", EDITED_SCRIPTS, ids=lambda p: p.name)
def test_script_parses_with_bash_n(script: Path) -> None:
    result = subprocess.run(
        ["/bin/bash", "-n", str(script)], capture_output=True, text=True, timeout=10
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("script", EDITED_SCRIPTS, ids=lambda p: p.name)
def test_script_has_no_shellcheck_errors(script: Path) -> None:
    if shutil.which("shellcheck") is None:
        pytest.skip("shellcheck not installed")
    result = subprocess.run(
        ["shellcheck", "-S", "error", "-s", "bash", str(script)],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("script", EDITED_SCRIPTS, ids=lambda p: p.name)
def test_every_curl_invocation_is_time_bounded(script: Path) -> None:
    """A bare `curl -sf` against a wedged server blocks for as long as the
    server stays wedged (0-480s in the outage). Every invocation must carry
    --max-time. The compose healthcheck array is exempt: Docker bounds it
    with the healthcheck `timeout`."""
    offenders = []
    for lineno, line in enumerate(script.read_text().splitlines(), 1):
        stripped = line.strip()
        if stripped.startswith("#") or "curl" not in stripped:
            continue
        if '"CMD", "curl"' in stripped:
            continue
        if re.search(r"(^|[\s|&;($])curl\s", stripped) and "--max-time" not in stripped:
            offenders.append(f"{script.name}:{lineno}: {stripped}")
    assert offenders == []


# ─── Repo docker-compose.yml ────────────────────────────────────────

_VAR = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?:(:?[-?])([^}]*))?\}")


def _expand(value: str, env: dict[str, str]) -> str:
    """Minimal compose-style ${VAR:-default} interpolation."""
    def sub(m: re.Match) -> str:
        name, op, arg = m.group(1), m.group(2), m.group(3) or ""
        if env.get(name):
            return env[name]
        return arg if op and op.endswith("-") else f"<{name}>"
    return _VAR.sub(sub, value)


def _repo_compose_service() -> dict:
    yaml = pytest.importorskip("yaml")
    compose = yaml.safe_load(REPO_COMPOSE.read_text())
    return compose, compose["services"]["jarvis"]


def test_repo_compose_publishes_every_port_on_loopback_by_default() -> None:
    _, svc = _repo_compose_service()
    ports = [_expand(p, {}) for p in svc["ports"]]
    assert ports == [
        "127.0.0.1:8741:8741",
        "127.0.0.1:8742:8742",
        "127.0.0.1:8744:8744",
        "127.0.0.1:8750:8750",
    ]


def test_repo_compose_bind_host_override_only_moves_mcp_ports() -> None:
    _, svc = _repo_compose_service()
    ports = [_expand(p, {"JARVIS_BIND_HOST": "0.0.0.0"}) for p in svc["ports"]]
    assert ports[:3] == ["0.0.0.0:8741:8741", "0.0.0.0:8742:8742", "0.0.0.0:8744:8744"]
    # The explorer has no auth at all: never exposed.
    assert ports[3] == "127.0.0.1:8750:8750"


def test_repo_compose_bounds_container_logs() -> None:
    _, svc = _repo_compose_service()
    assert svc["logging"] == {
        "driver": "local",
        "options": {"max-size": "10m", "max-file": "5"},
    }


def test_repo_compose_keeps_pgdata_in_a_named_volume() -> None:
    compose, svc = _repo_compose_service()
    assert "pgdata:/var/lib/postgresql/data" in svc["volumes"]
    assert "pgdata" in compose["volumes"]


# ─── install.sh helpers ─────────────────────────────────────────────


def _installer_env(tmp_path: Path, fake_bin: Path | None = None) -> dict[str, str]:
    env = _base_env(tmp_path, fake_bin)
    env["JARVIS_INSTALL_LIB_ONLY"] = "1"
    Path(env["JARVIS_HOME"]).mkdir(parents=True, exist_ok=True)
    return env


def _run_installer_lib(tmp_path: Path, body: str, fake_bin: Path | None = None):
    env = _installer_env(tmp_path, fake_bin)
    return _bash(f'source "{INSTALLER}"\n{body}', env), env


def _generated_compose(tmp_path: Path, config: dict | str | None) -> tuple[dict, str]:
    yaml = pytest.importorskip("yaml")
    env = _installer_env(tmp_path)
    config_path = Path(env["JARVIS_HOME"]) / "config.json"
    if config is not None:
        config_path.write_text(config if isinstance(config, str) else json.dumps(config))
    result = _bash(
        f'source "{INSTALLER}"\n'
        f'auth=$(read_auth_enabled "{config_path}")\n'
        f'generate_compose_file ghcr.io/example/jarvis:test "{tmp_path}/vault" '
        f'"{env["JARVIS_HOME"]}" "$auth"',
        env,
    )
    assert result.returncode == 0, result.stderr
    return yaml.safe_load(result.stdout), result.stdout


def test_generated_compose_is_loopback_only_when_auth_disabled(tmp_path: Path) -> None:
    compose, raw = _generated_compose(
        tmp_path, {"server": {"auth": {"enabled": False, "tokens": {}}}}
    )
    svc = compose["services"]["jarvis"]
    assert svc["image"] == "ghcr.io/example/jarvis:test"
    assert svc["ports"] == [
        "127.0.0.1:8741:8741",
        "127.0.0.1:8742:8742",
        "127.0.0.1:8744:8744",
        "127.0.0.1:${JARVIS_EXPLORER_PORT:-8750}:8750",
    ]
    assert svc["logging"] == {
        "driver": "local",
        "options": {"max-size": "10m", "max-file": "5"},
    }
    assert "pgdata:/var/lib/postgresql/data" in svc["volumes"]
    assert "pgdata" in compose["volumes"]
    # Runtime-interpolated vars must reach the file unexpanded.
    assert "ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY:-}" in svc["environment"]
    assert "TODOIST_API_TOKEN=${TODOIST_API_TOKEN:-}" in svc["environment"]
    assert svc["restart"] == "unless-stopped"


def test_generated_compose_exposes_mcp_ports_only_with_auth_enabled(tmp_path: Path) -> None:
    compose, _ = _generated_compose(
        tmp_path, {"server": {"auth": {"enabled": True, "tokens": {"a": "b"}}}}
    )
    ports = compose["services"]["jarvis"]["ports"]
    assert ports[:3] == ["8741:8741", "8742:8742", "8744:8744"]
    assert ports[3] == "127.0.0.1:${JARVIS_EXPLORER_PORT:-8750}:8750"


@pytest.mark.parametrize(
    "config",
    [None, "{not json", {"server": "nope"}, {"server": {"auth": "x"}}, {}],
    ids=["missing", "malformed", "server-not-dict", "auth-not-dict", "no-server"],
)
def test_unreadable_auth_config_fails_closed_to_loopback(tmp_path: Path, config) -> None:
    compose, _ = _generated_compose(tmp_path, config)
    ports = compose["services"]["jarvis"]["ports"]
    assert all(p.startswith("127.0.0.1:") for p in ports), ports


def test_install_managed_file_backs_up_and_replaces_a_changed_copy(tmp_path: Path) -> None:
    src = tmp_path / "new.sh"
    src.write_text("#!/bin/bash\necho new\n")
    dst_dir = tmp_path / "bin"
    dst_dir.mkdir()
    dst = dst_dir / "jarvis"
    dst.write_text("#!/bin/bash\necho old\n")
    dst.chmod(0o755)

    result, env = _run_installer_lib(
        tmp_path, f'install_managed_file "{src}" "{dst}" launcher'
    )
    assert result.returncode == 0, result.stderr

    assert dst.read_text() == src.read_text()
    assert _mode(dst) == 0o755
    backup_dir = Path(env["JARVIS_HOME"]) / "backups" / "installed"
    backups = list(backup_dir.glob("jarvis.*.bak"))
    assert len(backups) == 1
    assert backups[0].read_text() == "#!/bin/bash\necho old\n"
    assert _mode(backups[0]) == 0o600
    assert _mode(backup_dir) == 0o700
    assert _mode(backup_dir.parent) == 0o700
    assert not list(dst_dir.glob("jarvis.tmp.*"))


def test_install_managed_file_is_a_noop_when_identical(tmp_path: Path) -> None:
    src = tmp_path / "same.sh"
    src.write_text("#!/bin/bash\necho same\n")
    dst = tmp_path / "jarvis"
    dst.write_text(src.read_text())

    result, env = _run_installer_lib(
        tmp_path, f'install_managed_file "{src}" "{dst}" launcher'
    )
    assert result.returncode == 0, result.stderr
    assert "up to date" in result.stdout
    assert not (Path(env["JARVIS_HOME"]) / "backups").exists()


def test_install_managed_file_installs_fresh_without_backup(tmp_path: Path) -> None:
    src = tmp_path / "src.sh"
    src.write_text("#!/bin/bash\necho hi\n")
    dst = tmp_path / "jarvis"

    result, env = _run_installer_lib(
        tmp_path, f'install_managed_file "{src}" "{dst}" launcher'
    )
    assert result.returncode == 0, result.stderr
    assert dst.read_text() == src.read_text()
    assert _mode(dst) == 0o755
    assert not (Path(env["JARVIS_HOME"]) / "backups").exists()


def test_install_managed_file_leaves_symlinks_alone(tmp_path: Path) -> None:
    src = tmp_path / "src.sh"
    src.write_text("#!/bin/bash\necho new\n")
    target = tmp_path / "checkout.sh"
    target.write_text("#!/bin/bash\necho dev\n")
    dst = tmp_path / "jarvis"
    dst.symlink_to(target)

    result, _ = _run_installer_lib(
        tmp_path, f'install_managed_file "{src}" "{dst}" launcher'
    )
    assert result.returncode == 0, result.stderr
    assert dst.is_symlink()
    assert target.read_text() == "#!/bin/bash\necho dev\n"


def test_find_installed_launcher_requires_the_launcher_marker(tmp_path: Path) -> None:
    env = _installer_env(tmp_path)
    local_bin = Path(env["HOME"]) / ".local" / "bin"
    local_bin.mkdir(parents=True)
    candidate = local_bin / "jarvis"

    candidate.write_text("#!/bin/bash\necho some other jarvis\n")
    result = _bash(f'source "{INSTALLER}"\nfind_installed_launcher || echo NONE', env)
    assert result.stdout.strip() == "NONE"

    candidate.write_text("#!/bin/bash\n# Jarvis AI Assistant launcher\nexec claude\n")
    result = _bash(f'source "{INSTALLER}"\nfind_installed_launcher || echo NONE', env)
    assert result.stdout.strip() == str(candidate)


def test_shipped_launcher_keeps_the_marker_the_installer_detects() -> None:
    assert "Jarvis AI Assistant launcher" in LAUNCHER_SRC.read_text()


def _fake_installer_runtime(tmp_path: Path) -> dict[str, str]:
    """Just enough of claude/docker/curl for install.sh to run end to end."""
    fake_bin = tmp_path / "fakebin"
    fake_bin.mkdir()
    plugin_root = REPO_ROOT / "plugins" / "jarvis"
    _executable(
        fake_bin / "claude",
        """#!/bin/bash
if [ "$1 $2 $3" = "plugin marketplace list" ]; then
    printf '%s\\n' 'jarvis-plugins'
elif [ "$1 $2 $3" = "plugin list --json" ]; then
    printf '%s\\n' "$PLUGIN_LIST_JSON"
fi
exit 0
""",
    )
    _executable(fake_bin / "docker", "#!/bin/bash\nexit 0\n")
    _executable(
        fake_bin / "curl",
        """#!/bin/bash
printf '%s\\n' '{"status":"ok","postgres":{"status":"ok"}}'
exit 0
""",
    )
    env = _base_env(tmp_path, fake_bin)
    env.update(
        {
            "JARVIS_HARNESS": "claude",
            "CLAUDE_CONFIG_DIR": str(tmp_path / "claude-config"),
            "PLUGIN_LIST_JSON": json.dumps(
                [{"id": "jarvis@jarvis-plugins", "installPath": str(plugin_root)}]
            ),
        }
    )
    env.pop("ANTHROPIC_API_KEY", None)
    return env


def test_reinstall_refreshes_a_stale_launcher_with_a_backup(tmp_path: Path) -> None:
    """The launcher fix only helps if it reaches ~/.local/bin/jarvis: a re-run
    of the installer must replace an outdated copy (keeping a backup) without
    re-asking, and leave loopback-bound compose behind."""
    env = _fake_installer_runtime(tmp_path)
    local_bin = Path(env["HOME"]) / ".local" / "bin"
    local_bin.mkdir(parents=True)
    stale = "#!/bin/bash\n# Jarvis AI Assistant launcher\n# old unbounded probes\nexec claude \"$@\"\n"
    (local_bin / "jarvis").write_text(stale)
    (local_bin / "jarvis").chmod(0o755)

    result = subprocess.run(
        ["/bin/bash", str(INSTALLER)],
        env=env,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr

    installed = local_bin / "jarvis"
    assert installed.read_text() == LAUNCHER_SRC.read_text()
    backups = list((Path(env["JARVIS_HOME"]) / "backups" / "installed").glob("jarvis.*.bak"))
    assert [b.read_text() for b in backups] == [stale]
    # Refreshed without re-asking (the prompt itself only renders on a TTY).
    assert "'jarvis' command already installed" in result.stdout

    compose = (Path(env["JARVIS_HOME"]) / "docker-compose.yml").read_text()
    assert '"127.0.0.1:8741:8741"' in compose
    assert "driver: local" in compose
    # The statusline landed in the sandboxed config dir, not the real one.
    settings = json.loads((tmp_path / "claude-config" / "settings.json").read_text())
    assert settings["statusLine"]["command"].startswith(env["JARVIS_HOME"])


# ─── jarvis-transport.sh ────────────────────────────────────────────

_FAKE_DUMP = "PGDMP\x01fake-custom-archive"

_FAKE_DOCKER = """#!/bin/bash
printf '%s\\n' "$*" >> "$FAKE_DOCKER_LOG"
case "$*" in
    *"pg_dump -h 127.0.0.1 -Fc jarvis"*)
        [ -n "$FAKE_DUMP_HANG" ] && exec sleep 30
        [ -n "$FAKE_DUMP_FAIL" ] && { echo "container not running" >&2; exit 1; }
        printf 'PGDMP\\001fake-custom-archive'
        ;;
    *"pg_restore --list"*)
        [ -n "$FAKE_RESTORE_FAIL" ] && { echo "pg_restore: error: corrupt" >&2; exit 1; }
        [ "$(head -c 5)" = "PGDMP" ] || exit 1
        printf ';\\n; Archive created at 2026-09-25\\n;\\n'
        printf '215; 1259 16386 TABLE local memories postgres\\n'
        printf '216; 1259 16390 TABLE obsidian documents postgres\\n'
        ;;
    *"ps --quiet"*)
        echo 0123abcd
        ;;
esac
exit 0
"""


@pytest.fixture
def transport(tmp_path: Path):
    fake_bin = tmp_path / "fakebin"
    fake_bin.mkdir()
    _executable(fake_bin / "docker", _FAKE_DOCKER)
    env = _base_env(tmp_path, fake_bin)
    jarvis_home = Path(env["JARVIS_HOME"])
    jarvis_home.mkdir(parents=True)
    (jarvis_home / "docker-compose.yml").write_text("services: {}\n")
    env["FAKE_DOCKER_LOG"] = str(tmp_path / "docker.log")

    def run(*args: str, extra_env: dict[str, str] | None = None, timeout: float = 15):
        run_env = dict(env, **(extra_env or {}))
        started = time.monotonic()
        result = subprocess.run(
            ["/bin/bash", str(TRANSPORT), *args],
            env=run_env,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        return result, time.monotonic() - started

    run.jarvis_home = jarvis_home
    run.backup_dir = jarvis_home / "backups"
    run.docker_log = tmp_path / "docker.log"
    run.fake_bin = fake_bin
    return run


def _dumps(backup_dir: Path) -> list[str]:
    return sorted(p.name for p in backup_dir.glob("jarvis-*.dump"))


def test_backup_writes_a_verified_private_dump(transport) -> None:
    result, _ = transport("backup")
    assert result.returncode == 0, result.stdout + result.stderr

    dumps = _dumps(transport.backup_dir)
    assert len(dumps) == 1
    assert re.fullmatch(r"jarvis-\d{8}-\d{6}\.dump", dumps[0])
    dump = transport.backup_dir / dumps[0]
    assert dump.read_bytes() == _FAKE_DUMP.encode()
    assert _mode(dump) == 0o600
    assert _mode(transport.backup_dir) == 0o700
    assert not list(transport.backup_dir.glob("*.partial"))
    assert "2 archive entries" in result.stdout

    calls = transport.docker_log.read_text().splitlines()
    assert any(
        "exec -T jarvis su postgres -c pg_dump -h 127.0.0.1 -Fc jarvis" in c for c in calls
    )
    assert any("exec -T jarvis pg_restore --list" in c for c in calls)


def test_backup_tightens_a_preexisting_backup_dir(transport) -> None:
    transport.backup_dir.mkdir(mode=0o755)
    transport.backup_dir.chmod(0o755)
    result, _ = transport("backup")
    assert result.returncode == 0, result.stdout + result.stderr
    assert _mode(transport.backup_dir) == 0o700


def test_backup_rotation_keeps_only_the_newest_n(transport) -> None:
    transport.backup_dir.mkdir(mode=0o700)
    old = [f"jarvis-2020010{d}-000000.dump" for d in range(1, 6)]
    for name in old:
        (transport.backup_dir / name).write_bytes(b"PGDMPold")
    unrelated = ["notes.txt", "jarvis-20200101-000000.dump.partial"]
    for name in unrelated:
        (transport.backup_dir / name).write_text("keep me")

    result, _ = transport("backup", "3")
    assert result.returncode == 0, result.stdout + result.stderr

    remaining = _dumps(transport.backup_dir)
    assert len(remaining) == 3
    assert remaining[:2] == old[-2:]
    assert remaining[2] > old[-1]  # the new dump
    for name in unrelated:
        assert (transport.backup_dir / name).exists()
    assert "removed 3 old backup(s)" in result.stdout


def test_backup_keep_defaults_to_seven(transport) -> None:
    transport.backup_dir.mkdir(mode=0o700)
    for d in range(1, 10):
        (transport.backup_dir / f"jarvis-2020010{d}-000000.dump").write_bytes(b"x")
    result, _ = transport("backup")
    assert result.returncode == 0, result.stdout + result.stderr
    assert len(_dumps(transport.backup_dir)) == 7


def test_backup_rotation_never_deletes_hand_made_dumps(transport) -> None:
    """jarvis-2026-09-25.dump (a manual pg_dump taken during the outage) also
    matched jarvis-*.dump and, sorting before every timestamped name, was the
    first file rotation deleted."""
    transport.backup_dir.mkdir(mode=0o700)
    manual = ["jarvis-2026-09-25.dump", "jarvis-before-upgrade.dump", "jarvis-1.dump"]
    for name in manual:
        (transport.backup_dir / name).write_bytes(b"PGDMPmanual")
    old = [f"jarvis-2020010{d}-000000.dump" for d in range(1, 4)]
    for name in old:
        (transport.backup_dir / name).write_bytes(b"PGDMPold")

    result, _ = transport("backup", "1")
    assert result.returncode == 0, result.stdout + result.stderr

    for name in manual:
        assert (transport.backup_dir / name).read_bytes() == b"PGDMPmanual"
    ours = [n for n in _dumps(transport.backup_dir) if re.fullmatch(r"jarvis-\d{8}-\d{6}\.dump", n)]
    assert len(ours) == 1 and ours[0] > old[-1]
    assert "removed 3 old backup(s)" in result.stdout


def test_backup_tightens_hand_made_dumps_but_not_symlink_targets(transport, tmp_path: Path) -> None:
    """The outage-day dump (`pg_dump > jarvis-2026-09-25.dump`) was 644 in a
    755 dir, holding every memory and vault document."""
    transport.backup_dir.mkdir(mode=0o755)
    transport.backup_dir.chmod(0o755)
    manual = transport.backup_dir / "jarvis-2026-09-25.dump"
    manual.write_bytes(b"PGDMPmanual")
    manual.chmod(0o644)
    outside = tmp_path / "elsewhere.dump"
    outside.write_bytes(b"not ours")
    outside.chmod(0o644)
    (transport.backup_dir / "linked.dump").symlink_to(outside)
    notes = transport.backup_dir / "notes.txt"
    notes.write_text("keep my mode")
    notes.chmod(0o644)

    result, _ = transport("backup")
    assert result.returncode == 0, result.stdout + result.stderr
    assert _mode(transport.backup_dir) == 0o700
    assert _mode(manual) == 0o600
    assert manual.read_bytes() == b"PGDMPmanual"
    assert _mode(outside) == 0o644  # chmod never followed the symlink
    assert _mode(notes) == 0o644


def test_backup_honors_jarvis_compose_file(transport, tmp_path: Path) -> None:
    """Deployments started from the repo's docker/docker-compose.yml could only
    be backed up by pointing JARVIS_HOME elsewhere, which moved the backups."""
    (transport.jarvis_home / "docker-compose.yml").unlink()
    repo_compose = tmp_path / "repo" / "docker" / "docker-compose.yml"
    repo_compose.parent.mkdir(parents=True)
    repo_compose.write_text("services: {}\n")

    result, _ = transport("backup", extra_env={"JARVIS_COMPOSE_FILE": str(repo_compose)})
    assert result.returncode == 0, result.stdout + result.stderr
    assert len(_dumps(transport.backup_dir)) == 1  # still under $JARVIS_HOME/backups
    calls = transport.docker_log.read_text().splitlines()
    assert calls and all(f"compose -f {repo_compose} exec -T jarvis" in c for c in calls), calls


def test_failed_verification_keeps_nothing_and_rotates_nothing(transport) -> None:
    transport.backup_dir.mkdir(mode=0o700)
    old = [f"jarvis-2020010{d}-000000.dump" for d in range(1, 6)]
    for name in old:
        (transport.backup_dir / name).write_bytes(b"PGDMPold")

    result, _ = transport("backup", "2", extra_env={"FAKE_RESTORE_FAIL": "1"})
    assert result.returncode != 0
    assert "verification failed" in result.stdout
    assert _dumps(transport.backup_dir) == old
    assert not list(transport.backup_dir.glob("*.partial"))


def test_failed_dump_keeps_nothing(transport) -> None:
    result, _ = transport("backup", extra_env={"FAKE_DUMP_FAIL": "1"})
    assert result.returncode != 0
    assert "pg_dump failed" in result.stdout
    assert _dumps(transport.backup_dir) == []
    assert not list(transport.backup_dir.glob("*.partial"))


def test_hung_dump_is_killed_at_the_timeout(transport) -> None:
    result, elapsed = transport(
        "backup", extra_env={"FAKE_DUMP_HANG": "1", "JARVIS_BACKUP_TIMEOUT": "1"}
    )
    assert result.returncode != 0
    assert "timed out after 1s" in result.stdout
    assert elapsed < 8
    assert "Terminated" not in result.stderr
    assert _dumps(transport.backup_dir) == []
    assert not list(transport.backup_dir.glob("*.partial"))


def test_interrupted_backup_kills_the_dump_and_leaves_no_partial(transport, tmp_path: Path) -> None:
    """The dump runs in its own process group (so the timeout can kill all of
    it), which also shields it from Ctrl-C: the signal must be forwarded."""
    env = _base_env(tmp_path, transport.fake_bin)
    env.update(
        {
            "FAKE_DOCKER_LOG": str(transport.docker_log),
            "FAKE_DUMP_HANG": "1",
            "JARVIS_BACKUP_TIMEOUT": "60",
        }
    )
    proc = subprocess.Popen(
        ["/bin/bash", str(TRANSPORT), "backup"],
        env=env, stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    try:
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            if transport.docker_log.exists() and "pg_dump" in transport.docker_log.read_text():
                break
            time.sleep(0.05)
        assert list(transport.backup_dir.glob("*.partial"))
        proc.terminate()
        started = time.monotonic()
        proc.communicate(timeout=8)
        elapsed = time.monotonic() - started
    finally:
        if proc.poll() is None:
            proc.kill()
    assert proc.returncode != 0
    # communicate() returning quickly proves the hung `sleep 30` (holding
    # stderr) was killed along with the script.
    assert elapsed < 5
    assert not list(transport.backup_dir.glob("*.partial"))
    assert _dumps(transport.backup_dir) == []


@pytest.mark.parametrize("keep", ["0", "abc", "-1", "2x"])
def test_backup_rejects_an_invalid_keep_count(transport, keep: str) -> None:
    result, _ = transport("backup", keep)
    assert result.returncode != 0
    assert "Invalid keep count" in result.stdout
    assert not transport.docker_log.exists()


def test_backup_without_compose_file_fails_cleanly(transport) -> None:
    (transport.jarvis_home / "docker-compose.yml").unlink()
    result, _ = transport("backup")
    assert result.returncode != 0
    assert "No docker-compose.yml" in result.stdout
    assert not transport.docker_log.exists()


def _fake_curl(fake_bin: Path, exit_code: int, body: str = "") -> Path:
    log = fake_bin.parent / "curl.log"
    _executable(
        fake_bin / "curl",
        f"""#!/bin/bash
printf '%s\\n' "$*" >> "{log}"
printf '%s' '{body}'
exit {exit_code}
""",
    )
    return log


def test_status_reports_a_core_that_accepts_but_never_answers(transport) -> None:
    log = _fake_curl(transport.fake_bin, 28)
    result, _ = transport("status")
    assert result.returncode == 0, result.stderr
    assert "did not answer within 2s" in result.stdout
    for call in log.read_text().splitlines():
        assert "--max-time 2" in call and "--connect-timeout 1" in call


def test_status_surfaces_degraded_postgres_from_health(transport) -> None:
    body = json.dumps(
        {"status": "ok", "postgres": {"status": "recovering",
                                      "error": "the database system is in recovery mode"}}
    )
    _fake_curl(transport.fake_bin, 0, body)
    result, _ = transport("status")
    assert result.returncode == 0, result.stderr
    assert "Core server healthy" in result.stdout
    assert "Core database degraded: recovering (the database system is in recovery mode)" in result.stdout


def test_status_is_quiet_about_healthy_postgres(transport) -> None:
    _fake_curl(transport.fake_bin, 0, json.dumps({"status": "ok", "postgres": {"status": "ok"}}))
    result, _ = transport("status")
    assert result.returncode == 0, result.stderr
    assert "degraded" not in result.stdout


# ─── entrypoint.sh ──────────────────────────────────────────────────


@pytest.fixture
def entrypoint(tmp_path: Path):
    """Source the real entrypoint's functions with fakes first on PATH."""
    fake_bin = tmp_path / "fakebin"
    fake_bin.mkdir()
    log = tmp_path / "calls.log"
    # su postgres -c "<cmd>"  ->  run <cmd> as the current user
    _executable(
        fake_bin / "su",
        '#!/bin/bash\nshift\n[ "$1" = "-c" ] && shift\nexec /bin/bash -c "$1"\n',
    )
    _executable(fake_bin / "chown", "#!/bin/bash\nexit 0\n")
    _executable(
        fake_bin / "df",
        """#!/bin/bash
echo "Filesystem 1024-blocks Used Available Capacity Mounted on"
echo "/dev/fakevdb1 102400000 102000000 ${FAKE_DF_AVAIL_KB:-400000} 99% /var/lib/postgresql/data"
""",
    )
    _executable(
        fake_bin / "pg_ctl",
        f"""#!/bin/bash
printf 'pg_ctl %s ulimit_c=%s\\n' "$*" "$(ulimit -c)" >> "{log}"
exit ${{FAKE_PG_CTL_RC:-0}}
""",
    )
    pgdata = tmp_path / "pgdata"
    pgdata.mkdir()
    env = _base_env(tmp_path, fake_bin)
    env.update(
        {
            "JARVIS_ENTRYPOINT_LIB_ONLY": "1",
            "PGDATA": str(pgdata),
            "PG_WATCHDOG_INTERVAL": "0.05",
            "PG_WATCHDOG_MAX_FAILS": "3",
        }
    )

    def run(body: str, extra_env: dict[str, str] | None = None, timeout: float = 10):
        started = time.monotonic()
        result = _bash(f'source "{ENTRYPOINT}"\n{body}', dict(env, **(extra_env or {})), timeout)
        return result, time.monotonic() - started

    run.fake_bin = fake_bin
    run.log = log
    run.pgdata = pgdata
    return run


class _SilentServer:
    """Accepts TCP connections and never answers: a wedged event loop."""

    def __init__(self) -> None:
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.bind(("127.0.0.1", 0))
        self.sock.listen(16)
        self.sock.settimeout(0.2)
        self.port = self.sock.getsockname()[1]
        self.conns: list[socket.socket] = []
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _loop(self) -> None:
        while not self.stop.is_set():
            try:
                conn, _ = self.sock.accept()
                self.conns.append(conn)
            except OSError:
                continue

    def close(self) -> None:
        self.stop.set()
        self.thread.join(timeout=2)
        for conn in self.conns:
            conn.close()
        self.sock.close()


class _HealthHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802
        body = b'{"status":"ok"}'
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args) -> None:
        pass


def test_wait_for_health_gives_up_on_a_server_that_never_answers(entrypoint) -> None:
    server = _SilentServer()
    try:
        result, elapsed = entrypoint(
            f"rc=0; wait_for_health http://127.0.0.1:{server.port}/health wedged 3 || rc=$?\n"
            'echo "rc=$rc"'
        )
    finally:
        server.close()
    assert "rc=1" in result.stdout, result.stdout + result.stderr
    assert "wedged failed to start" in result.stdout
    assert elapsed < 8


def test_wait_for_health_succeeds_against_a_healthy_server(entrypoint) -> None:
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), _HealthHandler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        result, elapsed = entrypoint(
            f"wait_for_health http://127.0.0.1:{httpd.server_address[1]}/health core 5"
        )
    finally:
        httpd.shutdown()
        httpd.server_close()
    assert result.returncode == 0, result.stdout + result.stderr
    assert "core is ready" in result.stdout
    assert elapsed < 5


# A postgres process exists but refuses connections: crash recovery in a loop.
_PG_PROCESS_ALIVE = "postgres_running() { return 0; }\n"


def test_pg_watchdog_exits_nonzero_after_consecutive_failures(entrypoint) -> None:
    _executable(
        entrypoint.fake_bin / "pg_isready",
        f'#!/bin/bash\nprintf "pg_isready %s\\n" "$*" >> "{entrypoint.log}"\nexit 2\n',
    )
    result, elapsed = entrypoint(
        _PG_PROCESS_ALIVE
        + 'start_pg_watchdog\nrc=0; wait "$PG_WATCHDOG_PID" || rc=$?\necho "rc=$rc"'
    )
    assert "rc=1" in result.stdout, result.stdout + result.stderr
    assert "FATAL: PostgreSQL not accepting connections" in result.stderr
    assert "/dev/fakevdb1" in result.stderr  # df -h of PGDATA
    probes = [l for l in entrypoint.log.read_text().splitlines() if l.startswith("pg_isready")]
    assert len(probes) == 3
    # -U postgres: the default role (root) makes the server log an error per probe.
    assert all("-U postgres" in p and "-h 127.0.0.1" in p for p in probes)
    assert elapsed < 5


def test_pg_watchdog_resets_on_recovery_and_keeps_running(entrypoint) -> None:
    counter = entrypoint.pgdata.parent / "count"
    # Fails twice, succeeds once, forever: never three failures in a row.
    _executable(
        entrypoint.fake_bin / "pg_isready",
        f"""#!/bin/bash
n=$(( $(cat "{counter}" 2>/dev/null || echo 0) + 1 ))
echo "$n" > "{counter}"
[ $((n % 3)) -eq 0 ]
""",
    )
    # Wait (bounded, load-tolerant) for 7 probes: without the reset, the 3rd
    # failure lands on probe 4 and the watchdog would already have exited.
    result, _ = entrypoint(
        _PG_PROCESS_ALIVE
        + "start_pg_watchdog\n"
        "for _ in $(seq 1 100); do\n"
        f'    [ "$(cat "{counter}" 2>/dev/null || echo 0)" -ge 7 ] && break\n'
        "    sleep 0.05\n"
        "done\n"
        'if kill -0 "$PG_WATCHDOG_PID" 2>/dev/null; then echo ALIVE; fi\n'
        'kill "$PG_WATCHDOG_PID"'
    )
    assert "ALIVE" in result.stdout, result.stdout + result.stderr
    assert int(counter.read_text()) >= 7  # it really was probing


def test_pg_watchdog_exits_early_once_no_postgres_process_is_left(entrypoint) -> None:
    """A postmaster that exited (WAL PANIC + failed startup, kill -9) is never
    coming back in place: waiting the full MAX_FAILS checks kept the database
    down ~2 min after the disk had already been freed."""
    _executable(
        entrypoint.fake_bin / "pg_isready",
        f'#!/bin/bash\nprintf "pg_isready %s\\n" "$*" >> "{entrypoint.log}"\nexit 2\n',
    )
    result, elapsed = entrypoint(
        "postgres_running() { return 1; }\n"
        'start_pg_watchdog\nrc=0; wait "$PG_WATCHDOG_PID" || rc=$?\necho "rc=$rc"',
        extra_env={"PG_WATCHDOG_MAX_FAILS": "50", "PG_WATCHDOG_GONE_FAILS": "2"},
    )
    assert "rc=1" in result.stdout, result.stdout + result.stderr
    assert "FATAL: PostgreSQL is not running (no postgres process for 2 checks" in result.stderr
    assert "/dev/fakevdb1" in result.stderr  # df -h of PGDATA
    probes = [l for l in entrypoint.log.read_text().splitlines() if l.startswith("pg_isready")]
    assert len(probes) == 2
    assert elapsed < 5


def test_pg_watchdog_gone_count_resets_when_a_postgres_process_reappears(entrypoint) -> None:
    """Only CONSECUTIVE checks without any postgres process end it early; a
    process seen in between (still in recovery) falls back to MAX_FAILS."""
    counter = entrypoint.pgdata.parent / "count"
    _executable(
        entrypoint.fake_bin / "pg_isready",
        f'#!/bin/bash\nn=$(( $(cat "{counter}" 2>/dev/null || echo 0) + 1 ))\n'
        f'echo "$n" > "{counter}"\nexit 2\n',
    )
    # No process on odd probes, a process on even ones: never 2 gone in a row.
    result, _ = entrypoint(
        f'postgres_running() {{ [ $(( $(cat "{counter}") % 2 )) -eq 0 ]; }}\n'
        'start_pg_watchdog\nrc=0; wait "$PG_WATCHDOG_PID" || rc=$?\necho "rc=$rc"',
        extra_env={"PG_WATCHDOG_MAX_FAILS": "6", "PG_WATCHDOG_GONE_FAILS": "2"},
    )
    assert "rc=1" in result.stdout, result.stdout + result.stderr
    assert "FATAL: PostgreSQL not accepting connections for 6 checks" in result.stderr
    assert "is not running" not in result.stderr
    assert int(counter.read_text()) == 6


def test_stale_postmaster_pid_removed_only_when_dead_and_no_postgres(entrypoint) -> None:
    pidfile = entrypoint.pgdata / "postmaster.pid"
    no_pg = "postgres_running() { return 1; }\n"

    pidfile.write_text("99999999\n/var/lib/postgresql/data\n")
    result, _ = entrypoint(no_pg + "remove_stale_postmaster_pid")
    assert result.returncode == 0, result.stderr
    assert not pidfile.exists()
    assert "Removing stale postmaster.pid" in result.stdout

    # The recorded pid is alive (this very shell): keep it.
    result, _ = entrypoint(no_pg + 'echo "$$" > "$PGDATA/postmaster.pid"\nremove_stale_postmaster_pid')
    assert result.returncode == 0, result.stderr
    assert pidfile.exists()

    # A postgres process is running: never touch its lock file.
    pidfile.write_text("99999999\n")
    result, _ = entrypoint("postgres_running() { return 0; }\nremove_stale_postmaster_pid")
    assert result.returncode == 0, result.stderr
    assert pidfile.exists()


def test_failed_pg_start_prints_diagnostics_instead_of_dying_silently(entrypoint) -> None:
    """Under set -e a failing `pg_ctl start` used to exit before the error
    branch ran: 15h of restarts showed only 'Examine the log output.'"""
    (entrypoint.pgdata / "PG_VERSION").write_text("17\n")
    (entrypoint.pgdata / "core").write_text("")
    (entrypoint.pgdata / "postmaster.pid").write_text("99999999\n")

    result, _ = entrypoint(
        "postgres_running() { return 1; }\nstart_embedded_postgres\necho UNREACHABLE",
        extra_env={"FAKE_PG_CTL_RC": "1", "FAKE_DF_AVAIL_KB": "400000"},
    )
    assert result.returncode == 1
    assert "UNREACHABLE" not in result.stdout
    assert "[jarvis] ERROR: PostgreSQL failed to start" in result.stderr
    assert "/dev/fakevdb1" in result.stderr  # df -h of PGDATA
    assert "WARNING: only 390 MiB free" in result.stderr
    assert not (entrypoint.pgdata / "core").exists()
    assert not (entrypoint.pgdata / "postmaster.pid").exists()

    start = [l for l in entrypoint.log.read_text().splitlines() if l.startswith("pg_ctl start")]
    assert len(start) == 1
    # Server output goes to the container log, not a file on the full volume.
    assert " -l " not in start[0] and "postgresql.log" not in start[0]
    assert "-w -t 30" in start[0]
    assert start[0].endswith("ulimit_c=0")


def test_enough_free_space_does_not_warn(entrypoint) -> None:
    result, _ = entrypoint("warn_low_disk", extra_env={"FAKE_DF_AVAIL_KB": "52428800"})
    assert result.returncode == 0, result.stderr
    assert "WARNING" not in result.stderr


def test_cleanup_propagates_the_exit_code_and_still_stops_postgres(entrypoint) -> None:
    """cleanup() used to hard-code exit 0, and under set -e its `kill` of an
    already-dead child aborted it before `pg_ctl stop` ran."""
    result, elapsed = entrypoint(
        "PG_STARTED=true\nCORE_PID=99999999\nEXPLORER_PID=99999998\ncleanup 7"
    )
    assert result.returncode == 7, result.stdout + result.stderr
    assert "Shutdown complete." in result.stdout
    stops = [l for l in entrypoint.log.read_text().splitlines() if l.startswith("pg_ctl stop")]
    assert len(stops) == 1
    assert "-m fast -t 15" in stops[0]
    assert elapsed < 5


@pytest.mark.parametrize(
    ("url", "shown"),
    [
        ("postgresql://jarvis:s3cr'et@db.internal:5432/jarvis?sslmode=require", "db.internal:5432/jarvis"),
        ("postgresql://db.internal/jarvis?password=s3cr'et", "db.internal/jarvis"),
        ("host=db.internal password=s3cr'et", "(conninfo)"),
    ],
    ids=["userinfo", "query-password", "keyword-conninfo"],
)
def test_external_postgres_wait_never_exposes_the_password(entrypoint, url, shown) -> None:
    """It logged `${POSTGRES_URL%%@*}` (= scheme://user:password) and spliced
    the URL into the python -c source, where a quote broke it and the secret
    sat in the process list."""
    seen = entrypoint.pgdata.parent
    _executable(
        entrypoint.fake_bin / "python3",
        f'#!/bin/bash\nprintf \'%s\\n\' "$@" > "{seen}/py.argv"\n'
        f'printf \'%s\' "$POSTGRES_URL" > "{seen}/py.url"\nexit 0\n',
    )
    result, elapsed = entrypoint("wait_for_external_postgres", extra_env={"POSTGRES_URL": url})
    assert result.returncode == 0, result.stdout + result.stderr
    assert f"Using external PostgreSQL: {shown}\n" in result.stdout
    assert "PostgreSQL is ready" in result.stdout
    assert "s3cr" not in result.stdout + result.stderr
    assert "s3cr" not in (seen / "py.argv").read_text()
    assert (seen / "py.url").read_text() == url  # connects with the real URL
    assert elapsed < 5


def test_entrypoint_main_flow_contract() -> None:
    text = ENTRYPOINT.read_text()
    # The watchdog only makes sense for the embedded postmaster, once it is up.
    assert re.search(r"start_embedded_postgres\n\s+start_pg_watchdog\n", text)
    # A non-zero `wait -n` must reach cleanup with its code.
    assert "wait -n || EXIT_CODE=$?" in text
    assert 'cleanup "$EXIT_CODE"' in text
    assert "trap 'cleanup 0' SIGTERM SIGINT" in text
    # The explorer health wait is non-fatal.
    assert re.search(r'"memory-explorer" 30 \\\n\s+\|\| echo', text)
    # Keep in-place crash recovery; the watchdog is the supervisor.
    assert "restart_after_crash=off" not in text
