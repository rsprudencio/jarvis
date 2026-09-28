#!/bin/bash
# Jarvis AI Assistant launcher
# Auto-starts Docker container and launches Claude with Jarvis plugins.
# Install to a PATH directory (e.g. ~/.local/bin/jarvis) and chmod +x.
# Source: https://github.com/rsprudencio/jarvis
#
# Every external call is time-bounded, and Jarvis being down never stops Claude
# from starting: whatever the probes find, the script ends in `exec claude`.
# Diagnostics go to stderr.
#
# Overrides: JARVIS_HOME, JARVIS_CORE_URL (default http://localhost:8741),
# JARVIS_START_TIMEOUT (20s), JARVIS_COMPOSE_TIMEOUT (60s),
# JARVIS_DOCKER_TIMEOUT (5s), JARVIS_PLUGIN_CHECK_TIMEOUT (10s),
# JARVIS_SKIP_PLUGIN_CHECK=1.
set -u

JARVIS_HOME="${JARVIS_HOME:-${HOME:-}/.jarvis}"
compose_file="$JARVIS_HOME/docker-compose.yml"
core_url="${JARVIS_CORE_URL:-http://localhost:8741}"
core_url="${core_url%/}"

log() { printf '[jarvis] %s\n' "$*" >&2; }

# Non-negative integer setting, else the default (a stray value must not
# break the arithmetic below under set -u).
seconds_or() { case "$1" in ''|*[!0-9]*) echo "$2" ;; *) echo "$1" ;; esac; }
start_timeout=$(seconds_or "${JARVIS_START_TIMEOUT:-}" 20)
compose_timeout=$(seconds_or "${JARVIS_COMPOSE_TIMEOUT:-}" 60)
docker_timeout=$(seconds_or "${JARVIS_DOCKER_TIMEOUT:-}" 5)
plugin_check_timeout=$(seconds_or "${JARVIS_PLUGIN_CHECK_TIMEOUT:-}" 10)

# Ctrl-C. Inside $(...) the parent shell got the same SIGINT and finishes
# the line, so only the top level prints the newline.
on_interrupt() { [ "$BASH_SUBSHELL" -gt 0 ] || printf '\n' >&2; exit 130; }
trap on_interrupt INT

# Run a command with a hard deadline (macOS ships no coreutils `timeout`).
# The command gets its own process group and the WHOLE group is killed on
# expiry: `docker compose` is a CLI-plugin child of `docker`, so signalling only
# the parent would leave the child holding our pipe open. Returns 124 on
# timeout, else the command's status (128+n when a signal killed it). A
# signal-range status once the budget is spent counts as the timeout:
# `docker compose` traps our TERM and exits 130 (canceled), not 143. That
# group is outside the terminal's foreground group, so Ctrl-C is forwarded to
# it by hand before we exit 130. Stdin is /dev/null: nothing here may prompt.
with_timeout() {
    local secs="$1" start="$SECONDS" interrupted="" pid watchdog rc
    shift
    set -m                                  # async jobs get their own process group
    "$@" </dev/null &
    pid=$!
    { sleep "$secs"; kill -TERM -- "-$pid"; sleep 1; kill -KILL -- "-$pid"; } >/dev/null 2>&1 &
    watchdog=$!
    set +m
    trap 'interrupted=1; kill -INT -- "-$pid" 2>/dev/null' INT
    wait "$pid" 2>/dev/null
    rc=$?
    if [ -n "$interrupted" ]; then
        for _ in 1 2 3 4 5 6 7 8 9 10; do   # 1s to honor Ctrl-C, then force it
            kill -0 -- "-$pid" 2>/dev/null || break
            sleep 0.1
        done
        kill -KILL -- "-$pid" 2>/dev/null
    fi
    kill -- "-$watchdog" 2>/dev/null
    { wait "$pid"; wait "$watchdog"; } 2>/dev/null
    trap on_interrupt INT
    [ -n "$interrupted" ] && on_interrupt
    if [ "$rc" -gt 128 ] && [ $((SECONDS - start)) -ge "$secs" ]; then
        return 124
    fi
    return "$rc"
}

# One /health probe, never longer than ~2s. Sets health_body and probe_state:
#   ok        HTTP 200
#   wedged    TCP accepted but no reply in 2s (event loop busy / DB outage)
#   down      refused, or an empty reply / reset from Docker Desktop's port
#             proxy with nothing behind it (container stopped or restarting)
#   http_<n>  some other HTTP status
#   curl_<n>  any other curl failure
probe_core() {
    local out rc
    out=$(curl -s --connect-timeout 1 --max-time 2 -w '\n%{http_code}' "$core_url/health" 2>/dev/null)
    rc=$?
    health_body="${out%$'\n'*}"
    case "$rc" in
        0)       if [ "${out##*$'\n'}" = 200 ]; then probe_state=ok; else probe_state="http_${out##*$'\n'}"; fi ;;
        28)      probe_state=wedged ;;
        7|52|56) probe_state=down ;;
        *)       probe_state="curl_$rc" ;;
    esac
}

# Prints restarting | running | down; non-zero status when docker failed
# (124 = no answer within JARVIS_DOCKER_TIMEOUT).
container_state() {
    local states rc
    states=$(with_timeout "$docker_timeout" docker compose -f "$compose_file" ps -a --format '{{.State}}' 2>/dev/null)
    rc=$?
    [ "$rc" -eq 0 ] || return "$rc"
    case $'\n'"$states"$'\n' in
        *$'\n'restarting$'\n'*) echo restarting ;;
        *$'\n'running$'\n'*)    echo running ;;
        *)                      echo down ;;
    esac
}

# Poll until ok, wedged or crash-looping, or until JARVIS_START_TIMEOUT runs
# out (probe_state=timeout). Progress dots go to stderr.
wait_for_core() {
    local deadline=$((SECONDS + start_timeout))
    printf '[jarvis] Waiting for Jarvis core' >&2
    while [ "$SECONDS" -lt "$deadline" ]; do
        probe_core
        if [ "$probe_state" = ok ]; then
            printf ' ready.\n' >&2
            return
        fi
        [ "$probe_state" = down ] || break
        if [ "$(container_state)" = restarting ]; then
            probe_state=restarting
            break
        fi
        printf '.' >&2
        sleep 1
    done
    printf '\n' >&2
    [ "$probe_state" = down ] && probe_state=timeout
}

# /health reports the cached database status as {"postgres": {"status": ...}}
# (absent on older cores). Only a plain lowercase word is ever printed.
postgres_hint() {
    local pg
    [ "${#health_body}" -le 65536 ] || return 0
    pg=$(with_timeout 5 python3 -c '
import json, re, sys
try:
    status = json.loads(sys.argv[1]).get("postgres", {}).get("status")
except Exception:
    status = None
if isinstance(status, str) and re.fullmatch(r"[a-z_]{1,32}", status):
    print(status)
' "$health_body" 2>/dev/null)
    case "$pg" in
        ''|ok|unknown) ;;
        disk_full) log "Jarvis memory is degraded: postgres=disk_full. Free Docker disk space (docker system df)." ;;
        *) log "Jarvis memory is degraded: postgres=$pg. See: docker compose -f $compose_file logs --tail 200" ;;
    esac
}

if ! command -v claude >/dev/null 2>&1; then
    echo "Error: claude CLI not found on PATH." >&2
    exit 127
fi

# Verify core plugin is installed. Only a successful listing that lacks
# jarvis@ means "not installed"; a slow or failing CLI just warns.
if [ -z "${JARVIS_SKIP_PLUGIN_CHECK:-}" ]; then
    plugins_json=$(with_timeout "$plugin_check_timeout" claude plugin list --json 2>/dev/null)
    rc=$?
    if [ "$rc" -eq 0 ]; then
        with_timeout 5 python3 -c '
import json, sys
try:
    plugins = json.loads(sys.argv[1])
except ValueError:
    sys.exit(2)
if not isinstance(plugins, list):
    sys.exit(2)
found = any(isinstance(p, dict) and str(p.get("id", "")).startswith("jarvis@") for p in plugins)
sys.exit(0 if found else 3)
' "$plugins_json" 2>/dev/null
        rc=$?
    fi
    case "$rc" in
        0) ;;
        3)
            echo "Error: Jarvis core plugin not installed." >&2
            echo "Install with: claude plugin install jarvis@jarvis-plugins" >&2
            exit 1
            ;;
        124) log "Could not verify the Jarvis plugin install (timed out) - continuing." ;;
        *)   log "Could not verify the Jarvis plugin install (exit $rc) - continuing." ;;
    esac
fi

# Auto-start Docker container if compose file exists and core is down
if [ -f "$compose_file" ]; then
    probe_core
    if [ "$probe_state" = down ]; then
        if ! command -v docker >/dev/null 2>&1; then
            probe_state=no_docker
        else
            cstate=$(container_state)
            rc=$?
            if [ "$rc" -ne 0 ]; then
                probe_state="docker_$rc"
            elif [ "$cstate" = restarting ]; then
                probe_state=restarting
            else
                if [ "$cstate" != running ]; then
                    log "Starting Jarvis container..."
                    with_timeout "$compose_timeout" docker compose -f "$compose_file" up -d >&2
                    rc=$?
                    [ "$rc" -eq 0 ] || probe_state="compose_$rc"
                fi
                [ "$probe_state" = down ] && wait_for_core
            fi
        fi
    fi

    case "$probe_state" in
        ok) postgres_hint ;;
        wedged)
            log "Jarvis core is not responding (event loop busy / DB outage?) - memory tools may be unavailable."
            log "Check: docker compose -f $compose_file logs --tail 200"
            ;;
        restarting)
            log "Jarvis container is crash-looping - starting Claude without Jarvis memory."
            log "See why: docker compose -f $compose_file logs --tail 30   (disk full? docker system df)"
            ;;
        timeout)
            log "Jarvis core not ready after ${start_timeout}s - starting Claude without Jarvis memory."
            log "Check: docker compose -f $compose_file logs --tail 200"
            ;;
        no_docker)   log "docker not found - starting Claude without Jarvis memory." ;;
        docker_124)  log "Docker is not responding (no answer in ${docker_timeout}s) - starting Claude without Jarvis memory." ;;
        docker_*)    log "Docker is not reachable (is it running?) - starting Claude without Jarvis memory." ;;
        compose_124) log "docker compose up did not finish in ${compose_timeout}s - starting Claude without Jarvis memory." ;;
        compose_*)   log "docker compose up failed (exit ${probe_state#compose_}) - starting Claude without Jarvis memory." ;;
        http_*)      log "Jarvis core answered /health with HTTP ${probe_state#http_} - memory tools may be unavailable." ;;
        *)           log "Jarvis core health probe failed (curl exit ${probe_state#curl_}) - memory tools may be unavailable." ;;
    esac
fi

# Launch Claude — MCP servers inject instructions automatically
exec claude "$@"
