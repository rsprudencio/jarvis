#!/bin/bash
# Jarvis MCP Server - Docker Entrypoint
# Manages embedded PostgreSQL (pgvector), jarvis-core, jarvis-obsidian,
# and optionally jarvis-todoist.
#
# PostgreSQL is embedded by default for single-user deployments.
# Set POSTGRES_URL to use an external database (team/managed deployments).

set -e

CORE_PORT="${JARVIS_CORE_PORT:-8741}"
TODOIST_PORT="${JARVIS_TODOIST_PORT:-8742}"
OBSIDIAN_PORT="${JARVIS_OBSIDIAN_PORT:-8744}"
EXPLORER_PORT="${JARVIS_EXPLORER_PORT:-8750}"
# pgdata lives inside the container filesystem by default (not on the bind mount)
# because macOS VirtioFS doesn't support chown, which PostgreSQL requires.
# Use PGDATA env var to override (e.g., for a dedicated Docker volume).
PGDATA="${PGDATA:-/var/lib/postgresql/data}"
CORE_PID=""
TODOIST_PID=""
OBSIDIAN_PID=""
EXPLORER_PID=""
PG_WATCHDOG_PID=""
PG_STARTED=false
# Watchdog: exit (and let the restart policy restart the container) once
# PostgreSQL has refused connections for INTERVAL x MAX_FAILS seconds, or
# sooner (INTERVAL x GONE_FAILS) once no postgres process is left at all.
PG_WATCHDOG_INTERVAL="${PG_WATCHDOG_INTERVAL:-10}"
PG_WATCHDOG_MAX_FAILS="${PG_WATCHDOG_MAX_FAILS:-12}"
PG_WATCHDOG_GONE_FAILS="${PG_WATCHDOG_GONE_FAILS:-3}"
# Warn (never refuse to start) below this much free space on the PGDATA volume.
PG_MIN_FREE_KB=1048576

# --- Git configuration for mounted vault ---
if [ -d "/vault" ]; then
    git config --global safe.directory /vault
fi

# Windows host CRLF handling
if [ "${JARVIS_AUTOCRLF}" = "true" ]; then
    git config --global core.autocrlf true
fi

# --- TLS configuration ---
TLS_CERT="${JARVIS_TLS_CERT:-}"
TLS_KEY="${JARVIS_TLS_KEY:-}"
TLS_ENABLED=false

if [ -n "$TLS_CERT" ] && [ -n "$TLS_KEY" ]; then
    if [ ! -r "$TLS_CERT" ]; then
        echo "[jarvis] ERROR: TLS cert not readable: ${TLS_CERT}" >&2
        exit 1
    fi
    if [ ! -r "$TLS_KEY" ]; then
        echo "[jarvis] ERROR: TLS key not readable: ${TLS_KEY}" >&2
        exit 1
    fi
    TLS_ENABLED=true
    echo "[jarvis] TLS enabled"
elif [ -n "$TLS_CERT" ] || [ -n "$TLS_KEY" ]; then
    echo "[jarvis] ERROR: Both JARVIS_TLS_CERT and JARVIS_TLS_KEY must be set" >&2
    exit 1
fi

# --- Internal hook token ---
JARVIS_INTERNAL_TOKEN="${JARVIS_INTERNAL_TOKEN:-$(python3 -c 'import secrets; print(secrets.token_hex(32))')}"
export JARVIS_INTERNAL_TOKEN

# --- Graceful shutdown ---
# cleanup <exit-code>: the code is propagated so a crashed child (or the PG
# watchdog) exits the container non-zero.
cleanup() {
    local rc="${1:-0}"
    # A kill of an already-dead child must not abort cleanup under set -e
    # (that used to skip the PostgreSQL stop entirely).
    set +e
    echo "[jarvis] Shutting down..."
    [ -n "$PG_WATCHDOG_PID" ] && kill "$PG_WATCHDOG_PID" 2>/dev/null
    [ -n "$CORE_PID" ] && kill "$CORE_PID" 2>/dev/null
    [ -n "$TODOIST_PID" ] && kill "$TODOIST_PID" 2>/dev/null
    [ -n "$OBSIDIAN_PID" ] && kill "$OBSIDIAN_PID" 2>/dev/null
    [ -n "$EXPLORER_PID" ] && kill "$EXPLORER_PID" 2>/dev/null
    # Wait up to 10s for jarvis-core to drain in-flight requests
    local timeout=10
    while [ $timeout -gt 0 ] && [ -n "$CORE_PID" ] && kill -0 "$CORE_PID" 2>/dev/null; do
        sleep 1
        timeout=$((timeout - 1))
    done
    # Stop PostgreSQL after MCP servers are done. -t 15 keeps the total inside
    # compose's 30s stop_grace_period (pg_ctl's default wait is 60s).
    if [ "$PG_STARTED" = "true" ]; then
        echo "[jarvis] Stopping embedded PostgreSQL..."
        su postgres -c "pg_ctl stop -D '${PGDATA}' -m fast -t 15" 2>/dev/null || true
    fi
    echo "[jarvis] Shutdown complete."
    exit "$rc"
}
trap 'cleanup 0' SIGTERM SIGINT

# --- Check for Todoist token ---
has_todoist_token() {
    if [ -n "$TODOIST_API_TOKEN" ]; then
        return 0
    fi
    local config="${JARVIS_HOME:-/config}/config.json"
    if [ -f "$config" ]; then
        python3 -c "
import json, sys
with open('$config') as f:
    c = json.load(f)
token = c.get('todoist', {}).get('api_token', '')
sys.exit(0 if token else 1)
" 2>/dev/null && return 0
    fi
    return 1
}

# --- Wait for health check ---
HEALTH_SCHEME="http"
CURL_TLS_FLAGS=""
if [ "$TLS_ENABLED" = "true" ]; then
    HEALTH_SCHEME="https"
    CURL_TLS_FLAGS="-k"
fi

# wait_for_health <url> <name> [max-seconds]: a deadline, not a retry count.
# Every probe is bounded, so a server that accepts TCP but never answers (a
# wedged event loop) cannot stall startup past the deadline.
wait_for_health() {
    local url="$1"
    local name="$2"
    local max_wait="${3:-30}"
    local deadline=$((SECONDS + max_wait))

    while [ "$SECONDS" -lt "$deadline" ]; do
        if curl -sf --connect-timeout 1 --max-time 2 $CURL_TLS_FLAGS "${url}" > /dev/null 2>&1; then
            echo "[jarvis] ${name} is ready"
            return 0
        fi
        sleep 1
    done
    echo "[jarvis] ERROR: ${name} failed to start"
    return 1
}

# --- Embedded PostgreSQL ---
# pg_isready as the postgres role: the default (root) makes the server log
# 'role "root" does not exist' on every probe.
pg_ready() {
    pg_isready -h 127.0.0.1 -p 5432 -U postgres -d postgres -t 3 -q 2>/dev/null
}

# True when any process named postgres is alive. Scans /proc on Linux so it
# does not depend on procps; pgrep elsewhere (tests on macOS).
postgres_running() {
    local comm
    if [ -d /proc/1 ]; then
        for comm in /proc/[0-9]*/comm; do
            [ "$(cat "$comm" 2>/dev/null)" = "postgres" ] && return 0
        done
        return 1
    fi
    pgrep -x postgres >/dev/null 2>&1
}

# A postmaster.pid left by a SIGKILLed container. Removed only when no
# postgres process exists AND the pid recorded in it is not alive.
remove_stale_postmaster_pid() {
    local pidfile="${PGDATA}/postmaster.pid"
    local pid
    [ -f "$pidfile" ] || return 0
    postgres_running && return 0
    pid=$(head -n 1 "$pidfile" 2>/dev/null | tr -cd '0-9')
    if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
        return 0
    fi
    echo "[jarvis] Removing stale postmaster.pid (pid ${pid:-unknown} is not running)"
    rm -f "$pidfile"
}

# Warn only: a nearly-full volume can still serve reads, so never refuse to
# start on free space alone. pg_ctl failing is what stops the container.
warn_low_disk() {
    local avail_kb
    avail_kb=$(df -Pk "${PGDATA}" 2>/dev/null | awk 'NR==2 {print $4}')
    case "$avail_kb" in ''|*[!0-9]*) return 0 ;; esac
    if [ "$avail_kb" -lt "$PG_MIN_FREE_KB" ]; then
        echo "[jarvis] WARNING: only $((avail_kb / 1024)) MiB free on the PostgreSQL volume — free Docker disk space (docker system df)" >&2
        df -h "${PGDATA}" >&2 || true
    fi
}

# Supervise the daemonized postmaster (pg_ctl detaches it, so `wait -n` can't
# see it). A tracked job that exits once PG has been unavailable for
# INTERVAL x MAX_FAILS seconds, so a crash/recovery loop restarts the
# container instead of leaving it 'Up' with a dead database. In-place crash
# recovery (restart_after_crash=on) still handles transient backend crashes.
# Once no postgres process exists at all (postmaster exited: WAL PANIC then a
# failed startup, kill -9) nothing inside the container will bring it back,
# so it exits after INTERVAL x GONE_FAILS instead: freeing disk space then
# brings the database back in ~30s rather than ~2 min.
start_pg_watchdog() {
    (
        fails=0
        gone=0
        while sleep "$PG_WATCHDOG_INTERVAL"; do
            if pg_ready; then
                fails=0
                gone=0
            else
                fails=$((fails + 1))
                if postgres_running; then
                    gone=0
                else
                    gone=$((gone + 1))
                fi
            fi
            if [ "$gone" -ge "$PG_WATCHDOG_GONE_FAILS" ]; then
                echo "[jarvis] FATAL: PostgreSQL is not running (no postgres process for ${gone} checks, ${PG_WATCHDOG_INTERVAL}s apart; server exited — disk full?)" >&2
                df -h "${PGDATA}" >&2 || true
                exit 1
            fi
            if [ "$fails" -ge "$PG_WATCHDOG_MAX_FAILS" ]; then
                echo "[jarvis] FATAL: PostgreSQL not accepting connections for ${fails} checks, ${PG_WATCHDOG_INTERVAL}s apart (crash/recovery loop? disk full?)" >&2
                df -h "${PGDATA}" >&2 || true
                exit 1
            fi
        done
    ) &
    PG_WATCHDOG_PID=$!
}

start_embedded_postgres() {
    echo "[jarvis] Starting embedded PostgreSQL..."

    # A PANIC would otherwise write a core the size of shared memory into
    # PGDATA (the postmaster's cwd) — on the volume that is already full.
    ulimit -c 0
    [ -f "${PGDATA}/core" ] && rm -f "${PGDATA}/core"

    # Ensure data directory exists with correct ownership
    mkdir -p "${PGDATA}"
    chown -R postgres:postgres "${PGDATA}"

    # First-run: initialize database cluster
    if [ ! -f "${PGDATA}/PG_VERSION" ]; then
        echo "[jarvis] First run — initializing PostgreSQL data directory..."
        su postgres -c "initdb -D '${PGDATA}' --encoding=UTF8 --locale=C"

        # Write postgresql.conf (internal-only, tuned for single-user)
        cat > "${PGDATA}/postgresql.conf" <<PGCONF
listen_addresses = '127.0.0.1'
port = 5432
wal_level = logical
shared_buffers = 128MB
work_mem = 4MB
maintenance_work_mem = 64MB
max_connections = 20
max_replication_slots = 10
max_wal_senders = 10
logging_collector = off
log_destination = 'stderr'
PGCONF

        # Trust local connections only (PG is not exposed outside container)
        cat > "${PGDATA}/pg_hba.conf" <<PGHBA
# TYPE  DATABASE  USER  ADDRESS       METHOD
local   all       all                 trust
host    all       all   127.0.0.1/32  trust
PGHBA

        chown postgres:postgres "${PGDATA}/postgresql.conf" "${PGDATA}/pg_hba.conf"
    fi

    remove_stale_postmaster_pid
    warn_low_disk

    # Start PostgreSQL. No -l: server output goes to the container log (bounded
    # by the compose logging options) instead of a file inside PGDATA, which
    # stops being writable exactly when the volume fills. (-l /dev/stderr is
    # not an option: re-opening root's stderr pipe as postgres fails EACCES.)
    # Explicit `if !` because under set -e a failing pg_ctl used to exit before
    # any diagnostics ran.
    if ! su postgres -c "pg_ctl start -D '${PGDATA}' -w -t 30"; then
        echo "[jarvis] ERROR: PostgreSQL failed to start (server output above)" >&2
        df -h "${PGDATA}" >&2 || true
        exit 1
    fi
    PG_STARTED=true

    # Wait for pg_isready
    local i=0
    while [ $i -lt 30 ]; do
        if pg_ready; then
            echo "[jarvis] PostgreSQL is ready"
            break
        fi
        i=$((i + 1))
        sleep 1
    done

    if [ $i -eq 30 ]; then
        echo "[jarvis] ERROR: PostgreSQL failed to start within 30s (server output above)" >&2
        df -h "${PGDATA}" >&2 || true
        exit 1
    fi

    # Create database, jarvis role, and run init.sql (all idempotent)
    # Pipe init.sql via stdin so root reads the file (postgres user may lack /app access)
    su postgres -c "psql -h 127.0.0.1 -p 5432 -tc \"SELECT 1 FROM pg_database WHERE datname='jarvis'\" | grep -q 1" || \
        su postgres -c "createdb -h 127.0.0.1 -p 5432 jarvis"
    su postgres -c "psql -h 127.0.0.1 -p 5432 -d jarvis" < /app/init.sql

    # Create 'jarvis' role matching config.json default (postgresql://jarvis:jarvis@...)
    # so docker exec and external scripts work without POSTGRES_URL override
    su postgres -c "psql -h 127.0.0.1 -p 5432 -d jarvis" <<'ROLES'
DO $$ BEGIN
    IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'jarvis') THEN
        CREATE ROLE jarvis WITH LOGIN PASSWORD 'jarvis';
    END IF;
END $$;
GRANT ALL PRIVILEGES ON DATABASE jarvis TO jarvis;
GRANT ALL ON SCHEMA public TO jarvis;
GRANT ALL ON ALL TABLES IN SCHEMA public TO jarvis;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON TABLES TO jarvis;
ROLES

    echo "[jarvis] Embedded PostgreSQL initialized (database: jarvis)"
}

# Grant read access on schemas created by jarvis-core (local, obsidian).
# Must run AFTER jarvis-core is healthy, since schema.py creates these schemas at startup.
grant_schema_access() {
    echo "[jarvis] Granting jarvis role access to local/obsidian schemas..."
    su postgres -c "psql -h 127.0.0.1 -p 5432 -d jarvis" <<'GRANTS'
DO $$ BEGIN
    -- Only grant if schemas exist (created by jarvis-core schema.py)
    IF EXISTS (SELECT 1 FROM information_schema.schemata WHERE schema_name = 'local') THEN
        EXECUTE 'GRANT USAGE ON SCHEMA local TO jarvis';
        EXECUTE 'GRANT SELECT ON ALL TABLES IN SCHEMA local TO jarvis';
        EXECUTE 'ALTER DEFAULT PRIVILEGES IN SCHEMA local GRANT SELECT ON TABLES TO jarvis';
    END IF;
    IF EXISTS (SELECT 1 FROM information_schema.schemata WHERE schema_name = 'obsidian') THEN
        EXECUTE 'GRANT USAGE ON SCHEMA obsidian TO jarvis';
        EXECUTE 'GRANT SELECT ON ALL TABLES IN SCHEMA obsidian TO jarvis';
        EXECUTE 'ALTER DEFAULT PRIVILEGES IN SCHEMA obsidian GRANT SELECT ON TABLES TO jarvis';
    END IF;
END $$;
GRANTS
}

# --- Wait for external PostgreSQL ---
wait_for_external_postgres() {
    # Log the host part only: the userinfo and a ?password= parameter both
    # carry the secret (`${POSTGRES_URL%%@*}` used to print user:password).
    local pg_where="(conninfo)"
    case "$POSTGRES_URL" in
        *://*)
            pg_where="${POSTGRES_URL#*://}"
            pg_where="${pg_where##*@}"
            pg_where="${pg_where%%\?*}"
            ;;
    esac
    echo "[jarvis] Using external PostgreSQL: ${pg_where}"
    echo "[jarvis] Waiting for PostgreSQL..."
    local pg_ready=false
    for i in $(seq 1 30); do
        # Via the environment, not spliced into the source: a quote in the
        # URL broke the script, and argv/source are visible in the process list.
        if POSTGRES_URL="$POSTGRES_URL" python3 -c "
import os
import psycopg
try:
    conn = psycopg.connect(os.environ['POSTGRES_URL'], connect_timeout=2)
    conn.execute('SELECT 1')
    conn.close()
except Exception:
    raise SystemExit(1)
" 2>/dev/null; then
            pg_ready=true
            break
        fi
        sleep 1
    done
    if [ "$pg_ready" = true ]; then
        echo "[jarvis] PostgreSQL is ready"
    else
        echo "[jarvis] ERROR: PostgreSQL not reachable"
        exit 1
    fi
}

# Tests source this file for its functions only (tests/test_deploy_scripts.py).
if [ "${JARVIS_ENTRYPOINT_LIB_ONLY:-}" = "1" ]; then
    # shellcheck disable=SC2317  # exit is reached when executed, not sourced
    return 0 2>/dev/null || exit 0
fi

# --- Start PostgreSQL (embedded or wait for external) ---
if [ -z "${POSTGRES_URL}" ]; then
    start_embedded_postgres
    start_pg_watchdog
    export POSTGRES_URL="postgresql://postgres@127.0.0.1:5432/jarvis"
else
    wait_for_external_postgres
fi

# --- Build TLS args ---
tls_args=()
if [ "$TLS_ENABLED" = "true" ]; then
    tls_args+=(--ssl-certfile "$TLS_CERT" --ssl-keyfile "$TLS_KEY")
fi

# --- mTLS: client certificate verification ---
TLS_CA="${JARVIS_TLS_CA:-}"
if [ -n "$TLS_CA" ]; then
    if [ ! -r "$TLS_CA" ]; then
        echo "[jarvis] ERROR: TLS CA cert not readable: ${TLS_CA}" >&2
        exit 1
    fi
    if [ "$TLS_ENABLED" != "true" ]; then
        echo "[jarvis] ERROR: JARVIS_TLS_CA requires JARVIS_TLS_CERT and JARVIS_TLS_KEY" >&2
        exit 1
    fi
    # CERT_OPTIONAL (1): request client cert, verify if presented, but don't require.
    # This lets health check curl work without a client cert.
    tls_args+=(--ssl-ca-certs "$TLS_CA" --ssl-cert-reqs 1)
    echo "[jarvis] mTLS enabled (client certs verified against ${TLS_CA})"
fi

# --- Start jarvis-core ---
echo "[jarvis] Starting jarvis-core on port ${CORE_PORT}..."
cd /app/jarvis-core
uvicorn http_app:app \
    --host 0.0.0.0 \
    --port "${CORE_PORT}" \
    --log-level info \
    --no-access-log \
    "${tls_args[@]}" &
CORE_PID=$!

# --- Start jarvis-obsidian ---
echo "[jarvis] Starting jarvis-obsidian on port ${OBSIDIAN_PORT}..."
cd /app/jarvis-obsidian
uvicorn http_app:app \
    --host 0.0.0.0 \
    --port "${OBSIDIAN_PORT}" \
    --log-level info \
    --no-access-log \
    "${tls_args[@]}" &
OBSIDIAN_PID=$!

# Set URL for core's health check detection
export JARVIS_OBSIDIAN_URL="${HEALTH_SCHEME}://127.0.0.1:${OBSIDIAN_PORT}"

# --- Conditionally start jarvis-todoist ---
if has_todoist_token; then
    echo "[jarvis] Todoist token found, starting jarvis-todoist on port ${TODOIST_PORT}..."
    cd /app/jarvis-todoist
    uvicorn http_app:app \
        --host 0.0.0.0 \
        --port "${TODOIST_PORT}" \
        --log-level info \
        --no-access-log \
        "${tls_args[@]}" &
    TODOIST_PID=$!
else
    echo "[jarvis] No Todoist token found, skipping jarvis-todoist."
fi

# --- Wait for jarvis-core health (creates schemas on startup) ---
wait_for_health "${HEALTH_SCHEME}://localhost:${CORE_PORT}/health" "jarvis-core" 30

# --- Grant schema access (after jarvis-core creates local/obsidian schemas) ---
if [ "$PG_STARTED" = "true" ]; then
    grant_schema_access
fi

# --- Start memory-explorer (after jarvis-core is healthy, needs DB schemas) ---
echo "[jarvis] Starting memory-explorer on port ${EXPLORER_PORT}..."
cd /app/memory-explorer
uvicorn app:app \
    --host 0.0.0.0 \
    --port "${EXPLORER_PORT}" \
    --log-level warning \
    --no-access-log &
EXPLORER_PID=$!

wait_for_health "${HEALTH_SCHEME}://localhost:${OBSIDIAN_PORT}/health" "jarvis-obsidian" 30
# Non-fatal: the explorer is a read-only UI. Aborting here would restart-loop
# the whole container (MCP servers included) over a slow explorer startup.
wait_for_health "${HEALTH_SCHEME}://localhost:${EXPLORER_PORT}/health" "memory-explorer" 30 \
    || echo "[jarvis] WARNING: memory-explorer not ready yet; continuing without waiting" >&2

if [ -n "$TODOIST_PID" ]; then
    wait_for_health "${HEALTH_SCHEME}://localhost:${TODOIST_PORT}/health" "jarvis-todoist" 30
fi

echo "[jarvis] All services started successfully."

# --- Wait for any process (or the PG watchdog) to exit, then shutdown ---
# `|| EXIT_CODE=$?`: a bare non-zero `wait -n` under set -e would exit here and
# skip cleanup (and the PostgreSQL stop).
EXIT_CODE=0
wait -n || EXIT_CODE=$?
echo "[jarvis] A process exited with code ${EXIT_CODE}, shutting down..."
cleanup "$EXIT_CODE"
