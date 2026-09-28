#!/bin/bash
# Jarvis MCP Service Manager
# Manages ChromaDB lifecycle, shows service status and backs up the database.
#
# Usage: jarvis-transport.sh <command>
#
# Commands:
#   status          Show service status
#   backup [N]      Dump the embedded PostgreSQL to ~/.jarvis/backups (keep newest N, default 7)
#                   JARVIS_COMPOSE_FILE selects another compose file (e.g. the
#                   repo's docker/docker-compose.yml); backups stay in $JARVIS_HOME
#   chroma-start    Start local ChromaDB server
#   chroma-stop     Stop local ChromaDB server
#   chroma-status   Check ChromaDB server status
set -e

JARVIS_HOME="${JARVIS_HOME:-$HOME/.jarvis}"
CONFIG_FILE="$JARVIS_HOME/config.json"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

ok()   { echo -e "  ${GREEN}✓${NC} $1"; }
fail() { echo -e "  ${RED}✗${NC} $1"; }
warn() { echo -e "  ${YELLOW}!${NC} $1"; }
info() { echo -e "  ${BLUE}→${NC} $1"; }

# ── Helpers ──

read_config_key() {
    local key="$1" default="$2"
    if [ ! -f "$CONFIG_FILE" ]; then
        echo "$default"
        return
    fi
    python3 -c "
import json, sys
with open(sys.argv[1]) as f:
    print(json.load(f).get(sys.argv[2], sys.argv[3]))
" "$CONFIG_FILE" "$key" "$default"
}

read_memory_key() {
    local key="$1" default="$2"
    if [ ! -f "$CONFIG_FILE" ]; then
        echo "$default"
        return
    fi
    python3 -c "
import json, sys
with open(sys.argv[1]) as f:
    print(json.load(f).get('memory', {}).get(sys.argv[2], sys.argv[3]))
" "$CONFIG_FILE" "$key" "$default"
}

# Bounded curl: a server that accepts TCP but never answers (a wedged event
# loop) costs at most ~2s. Exit 28 = connected but no reply in time.
probe() {
    curl -sf --connect-timeout 1 --max-time 2 "$@" 2>/dev/null
}

# run_with_timeout <seconds> <cmd>...: run cmd in its own process group and
# kill the whole group (docker -> compose plugin -> ...) once the deadline
# passes. Returns 124 on timeout, like coreutils timeout (absent on macOS).
# Its own group means Ctrl-C no longer reaches it, so INT/TERM are forwarded.
run_with_timeout() {
    local secs="$1" pid rc=0
    local deadline=$((SECONDS + secs))
    shift
    set -m
    "$@" <&0 &
    pid=$!
    set +m
    trap 'kill -TERM -- "-$pid" 2>/dev/null; exit 130' INT TERM
    while kill -0 "$pid" 2>/dev/null; do
        if [ "$SECONDS" -ge "$deadline" ]; then
            # stderr muted: it would otherwise carry bash's "Terminated" job notice
            {
                kill -TERM -- "-$pid" || kill -TERM "$pid" || true
                sleep 1
                kill -KILL -- "-$pid" || true
                wait "$pid" || true
            } 2>/dev/null
            trap - INT TERM
            return 124
        fi
        sleep 0.2
    done
    trap - INT TERM
    { wait "$pid"; } 2>/dev/null || rc=$?
    return "$rc"
}

# ── Commands ──

cmd_status() {
    echo ""
    echo -e "${BOLD}Jarvis Service Status${NC}"
    echo ""

    # MCP server health
    echo -e "${BOLD}MCP Servers${NC}"
    echo ""
    info "MCP Core:    http://localhost:8741/mcp"
    info "MCP Todoist: http://localhost:8742/mcp"

    local core_health rc=0 pg_status
    core_health=$(probe http://localhost:8741/health) || rc=$?
    if [ "$rc" -eq 0 ]; then
        ok "Core server healthy"
        pg_status=$(python3 -c "
import json, sys
pg = json.loads(sys.argv[1]).get('postgres') or {}
status = pg.get('status')
if status and status != 'ok':
    print(status + (' (' + pg['error'] + ')' if pg.get('error') else ''))
" "$core_health" 2>/dev/null || true)
        if [ -n "$pg_status" ]; then
            warn "Core database degraded: $pg_status"
        fi
    elif [ "$rc" -eq 28 ]; then
        fail "Core server accepts connections but did not answer within 2s (stalled; its database may be down)"
    else
        fail "Core server not reachable"
    fi
    if probe http://localhost:8742/health > /dev/null; then
        ok "Todoist server healthy"
    else
        warn "Todoist server not reachable (may not be configured)"
    fi

    # ChromaDB status
    echo ""
    echo -e "${BOLD}ChromaDB${NC}"
    echo ""
    local chroma_port
    chroma_port=$(read_memory_key "chroma_port" "8743")
    local chroma_pidfile="$JARVIS_HOME/state/chroma.pid"

    if [ -f "$chroma_pidfile" ] && kill -0 "$(cat "$chroma_pidfile")" 2>/dev/null; then
        ok "Local server running (PID $(cat "$chroma_pidfile"), port $chroma_port)"
    elif probe "http://127.0.0.1:${chroma_port}/api/v2/heartbeat" >/dev/null; then
        ok "Reachable on port $chroma_port"
    else
        fail "Not running on port $chroma_port"
    fi

    # Docker container status
    echo ""
    echo -e "${BOLD}Docker${NC}"
    echo ""
    local compose_file="$JARVIS_HOME/docker-compose.yml"
    if [ -f "$compose_file" ]; then
        if run_with_timeout 5 docker compose -f "$compose_file" ps --quiet < /dev/null 2>/dev/null | grep -q .; then
            ok "Container running"
        else
            fail "Container not running (or Docker not answering within 5s)"
        fi
    else
        info "No docker-compose.yml found"
    fi
    echo ""
}

# ── Backup ──

# backup [N]: pg_dump (custom format) of the embedded database into
# $JARVIS_HOME/backups/jarvis-YYYYMMDD-HHMMSS.dump (dir 700, file 600),
# verified with pg_restore --list, keeping the newest N dumps (default 7).
# Embedded PostgreSQL only: external POSTGRES_URL deployments have their own backups.
cmd_backup() {
    local keep="${1:-${JARVIS_BACKUP_KEEP:-7}}"
    local dump_timeout="${JARVIS_BACKUP_TIMEOUT:-600}"
    local compose_file="${JARVIS_COMPOSE_FILE:-$JARVIS_HOME/docker-compose.yml}"
    local backup_dir="$JARVIS_HOME/backups"
    local ts dest partial rc toc entries size

    case "$keep" in
        ''|*[!0-9]*|0) fail "Invalid keep count: '$keep' (expected a positive integer)"; return 1 ;;
    esac
    case "$dump_timeout" in
        ''|*[!0-9]*|0) fail "Invalid JARVIS_BACKUP_TIMEOUT: '$dump_timeout' (expected seconds)"; return 1 ;;
    esac
    if [ ! -f "$compose_file" ]; then
        fail "No docker-compose.yml found at $compose_file (set JARVIS_COMPOSE_FILE to use another)"
        return 1
    fi
    if ! command -v docker >/dev/null 2>&1; then
        fail "docker not found"
        return 1
    fi

    mkdir -p "$backup_dir"
    chmod 700 "$backup_dir"
    # Dumps hold every memory and vault document: tighten any made by hand
    # (pg_dump > file leaves them 644). Regular files only, never a symlink
    # target elsewhere.
    find "$backup_dir" -maxdepth 1 -type f -name '*.dump' ! -perm 600 -exec chmod 600 {} + 2>/dev/null || true
    ts=$(date +%Y%m%d-%H%M%S)
    dest="$backup_dir/jarvis-$ts.dump"
    if [ -e "$dest" ]; then
        fail "Backup already exists: $dest (retry in a second)"
        return 1
    fi
    # Written under a name the rotation glob ignores; renamed only once
    # verified, and removed if we exit early (failure, Ctrl-C, timeout).
    partial="$dest.partial"
    BACKUP_PARTIAL="$partial"
    trap '[ -n "$BACKUP_PARTIAL" ] && rm -f "$BACKUP_PARTIAL"' EXIT
    (umask 077 && : > "$partial")
    chmod 600 "$partial"

    info "Dumping embedded PostgreSQL (timeout ${dump_timeout}s)..."
    rc=0
    run_with_timeout "$dump_timeout" docker compose -f "$compose_file" exec -T jarvis \
        su postgres -c 'pg_dump -h 127.0.0.1 -Fc jarvis' < /dev/null > "$partial" || rc=$?
    if [ "$rc" -ne 0 ]; then
        rm -f "$partial"
        if [ "$rc" -eq 124 ]; then
            fail "pg_dump timed out after ${dump_timeout}s"
        else
            fail "pg_dump failed (exit $rc) — is the container running? ($0 status)"
        fi
        return 1
    fi
    if [ ! -s "$partial" ]; then
        rm -f "$partial"
        fail "pg_dump produced an empty archive"
        return 1
    fi

    # Verify with the container's pg_restore: it matches the pg_dump that wrote
    # the archive (a host pg_restore may be missing or older).
    rc=0
    toc=$(run_with_timeout 120 docker compose -f "$compose_file" exec -T jarvis \
        pg_restore --list < "$partial" 2>/dev/null) || rc=$?
    entries=$(printf '%s\n' "$toc" | grep -c '^[0-9]' || true)
    if [ "$rc" -ne 0 ] || [ "${entries:-0}" -eq 0 ]; then
        rm -f "$partial"
        fail "Backup verification failed (pg_restore --list exit $rc, ${entries:-0} entries); nothing kept"
        return 1
    fi

    mv -f "$partial" "$dest"
    BACKUP_PARTIAL=""
    size=$(wc -c < "$dest" | tr -d ' ')
    ok "Backup written: $dest ($size bytes, $entries archive entries)"

    # Rotate: names sort chronologically, so the oldest come first. Only our
    # own jarvis-YYYYMMDD-HHMMSS.dump names: a hand-made dump in the same
    # directory (e.g. jarvis-2026-09-25.dump) sorts first and was deleted.
    local dumps=() f i removed=0
    for f in "$backup_dir"/jarvis-[0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9]-[0-9][0-9][0-9][0-9][0-9][0-9].dump; do
        [ -f "$f" ] && dumps+=("$f")
    done
    i=0
    while [ $((${#dumps[@]} - i)) -gt "$keep" ]; do
        rm -f "${dumps[$i]}"
        removed=$((removed + 1))
        i=$((i + 1))
    done
    if [ "$removed" -gt 0 ]; then
        info "Rotated: removed $removed old backup(s), kept newest $keep"
    fi
}

# ── ChromaDB Lifecycle ──

cmd_chroma_start() {
    local data_path port pidfile logfile
    data_path=$(read_memory_key "chroma_data_path" "$HOME/.jarvis/db")
    data_path="${data_path/#\~/$HOME}"
    port=$(read_memory_key "chroma_port" "8743")
    pidfile="$JARVIS_HOME/state/chroma.pid"
    logfile="$JARVIS_HOME/logs/chroma.log"

    if [ -f "$pidfile" ] && kill -0 "$(cat "$pidfile")" 2>/dev/null; then
        ok "ChromaDB already running (PID $(cat "$pidfile"), port $port)"
        return 0
    fi

    # Verify chroma CLI is available
    if ! command -v chroma >/dev/null 2>&1; then
        fail "chroma CLI not found"
        info "Install: pip install chromadb"
        return 1
    fi

    mkdir -p "$data_path" "$(dirname "$pidfile")" "$(dirname "$logfile")"

    chroma run --host 127.0.0.1 --port "$port" --path "$data_path" \
        > "$logfile" 2>&1 &
    echo $! > "$pidfile"

    # Wait for health
    for i in $(seq 1 15); do
        if probe "http://127.0.0.1:${port}/api/v2/heartbeat" >/dev/null; then
            ok "ChromaDB started (PID $(cat "$pidfile"), port $port)"
            return 0
        fi
        sleep 1
    done

    fail "ChromaDB failed to start (check $logfile)"
    # Clean up pidfile if server didn't come up
    rm -f "$pidfile"
    return 1
}

cmd_chroma_stop() {
    local pidfile="$JARVIS_HOME/state/chroma.pid"

    if [ ! -f "$pidfile" ]; then
        info "ChromaDB is not running (no pidfile)"
        return 0
    fi

    local pid
    pid=$(cat "$pidfile")

    if kill -0 "$pid" 2>/dev/null; then
        kill "$pid" 2>/dev/null
        # Wait for shutdown
        local timeout=5
        while [ $timeout -gt 0 ] && kill -0 "$pid" 2>/dev/null; do
            sleep 1
            timeout=$((timeout - 1))
        done
        if kill -0 "$pid" 2>/dev/null; then
            kill -9 "$pid" 2>/dev/null || true
        fi
        ok "ChromaDB stopped (was PID $pid)"
    else
        info "ChromaDB process already gone (stale pidfile)"
    fi

    rm -f "$pidfile"
}

cmd_chroma_status() {
    local port pidfile
    port=$(read_memory_key "chroma_port" "8743")
    pidfile="$JARVIS_HOME/state/chroma.pid"

    echo ""
    echo -e "${BOLD}ChromaDB Status${NC}"
    echo ""

    if [ -f "$pidfile" ] && kill -0 "$(cat "$pidfile")" 2>/dev/null; then
        ok "Running (PID $(cat "$pidfile"))"
    else
        fail "Not running"
    fi

    if probe "http://127.0.0.1:${port}/api/v2/heartbeat" >/dev/null; then
        ok "Healthy on port $port"
    else
        fail "Not reachable on port $port"
    fi

    echo ""
}

# ── Main ──

case "${1:-}" in
    status)        cmd_status ;;
    backup)        cmd_backup "${2:-}" ;;
    chroma-start)  cmd_chroma_start ;;
    chroma-stop)   cmd_chroma_stop ;;
    chroma-status) cmd_chroma_status ;;
    -h|--help|help|"")
        echo ""
        echo -e "${BOLD}Jarvis MCP Service Manager${NC}"
        echo ""
        echo "Usage: jarvis-transport.sh <command>"
        echo ""
        echo "Commands:"
        echo "  status          Show service status (MCP, ChromaDB, Docker)"
        echo "  backup [N]      Dump the embedded PostgreSQL to $JARVIS_HOME/backups,"
        echo "                  verify it, keep the newest N dumps (default 7)"
        echo "  chroma-start    Start local ChromaDB server"
        echo "  chroma-stop     Stop local ChromaDB server"
        echo "  chroma-status   Check ChromaDB server status"
        echo ""
        echo "Examples:"
        echo "  jarvis-transport.sh status"
        echo "  jarvis-transport.sh backup"
        echo "  jarvis-transport.sh chroma-start"
        echo ""
        ;;
    *)
        fail "Unknown command: $1"
        echo "  Run 'jarvis-transport.sh --help' for usage"
        exit 1
        ;;
esac
