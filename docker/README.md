# Jarvis Docker Deployment

Run the Jarvis MCP servers in a Docker container. No Python, uv, or ChromaDB compilation needed on your host machine.

## Quick Start

```bash
# 1. Pull the image
docker pull ghcr.io/rsprudencio/jarvis:latest

# 2. Create config
mkdir -p ~/.jarvis
cat > ~/.jarvis/config.json << 'EOF'
{"vault_path": "/vault", "vault_confirmed": true}
EOF

# 3. Start the container (upgrading an older Quick Start install? Read the
#    pgdata note below FIRST: this can replace the container and its data)
docker compose -f docker/docker-compose.yml up -d

# 4. Verify
curl --max-time 2 http://localhost:8741/health
# → {"status":"ok","server":"jarvis-core","version":"...",
#    "postgres":{"status":"ok","error":null,"checked_at":...,"free_bytes":...}}
```

> **Upgrading a Quick Start deployment from before the `pgdata` volume?**
> Earlier versions of `docker/docker-compose.yml` kept the embedded database in
> the container's writable layer. The first `up -d` with the current file
> recreates the container on an empty `pgdata` volume, and the old data is
> gone. Dump it **before** that `up -d`, then restore it into the new
> container:
>
> ```bash
> # with the OLD container still running
> docker compose -f docker/docker-compose.yml exec -T jarvis \
>   su postgres -c 'pg_dump -h 127.0.0.1 -Fc jarvis' > jarvis.dump
> docker compose -f docker/docker-compose.yml exec -T jarvis \
>   pg_restore --list < jarvis.dump > /dev/null && echo "dump OK"  # verify first
>
> docker compose -f docker/docker-compose.yml up -d              # new container, empty pgdata
> docker compose -f docker/docker-compose.yml exec -T jarvis su postgres -c \
>   'dropdb -h 127.0.0.1 --force jarvis && createdb -h 127.0.0.1 jarvis && pg_restore -h 127.0.0.1 -d jarvis' \
>   < jarvis.dump
> docker compose -f docker/docker-compose.yml restart             # re-applies roles and grants
> ```
>
> Installer-generated deployments (`~/.jarvis/docker-compose.yml`) already used
> the `pgdata` volume and are not affected.

`status` is `"ok"` whenever the server answers. Database health rides in
`postgres.status` (`ok` | `recovering` | `unreachable` | `disk_full` | `unknown`
before the first probe), refreshed in the background every 10s — `/health` itself
never touches the database.

Or use the installer: `bash install.sh` and choose **[2] Docker**.

## Architecture

```
Host Machine
├── Claude Code
│   ├── Plugin (skills, agents, MCP instructions) ← installed via marketplace
│   ├── MCP config → http://localhost:8741/mcp, http://localhost:8742/mcp
│   └── Hooks (prompt_search, extract_observation) → ChromaDB :8743
│
└── Docker Container
    ├── ChromaDB      (port 8743) — Semantic memory database
    ├── jarvis-core   (port 8741) — Vault ops, memory, git audit
    ├── jarvis-todoist (port 8742) — Todoist API (if token configured)
    └── Volumes:
        ├── /vault  ← your Obsidian/markdown vault
        └── /config ← ~/.jarvis/ (config + ChromaDB data)
```

The container runs 3 processes managed by `entrypoint.sh`. ChromaDB starts first; once its heartbeat is healthy, jarvis-core and jarvis-todoist launch. All ChromaDB access goes through HTTP — both MCP servers and host-side hooks connect to port 8743.

Both MCP servers use **Streamable HTTP** transport (MCP SDK). Claude Code connects via `"type": "http"` URL-based MCP config.

## Claude Code MCP Configuration

After starting the container, tell Claude Code to connect via HTTP:

```bash
claude mcp add --transport http jarvis-core http://localhost:8741/mcp
claude mcp add --transport http jarvis-todoist-api http://localhost:8742/mcp
```

Or add to your settings JSON manually:

```json
{
  "mcpServers": {
    "jarvis-core": { "type": "http", "url": "http://localhost:8741/mcp" },
    "jarvis-todoist-api": { "type": "http", "url": "http://localhost:8742/mcp" }
  }
}
```

## Volume Mounts

| Container Path | Host Path | Purpose |
|---|---|---|
| `/vault` | Your Obsidian vault | Markdown/Org files, journal entries |
| `/config` | `~/.jarvis/` | Config, ChromaDB database, state |

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `JARVIS_HOME` | `/config` | Config directory inside container |
| `JARVIS_VAULT_PATH` | `/vault` | Vault directory inside container |
| `TODOIST_API_TOKEN` | (empty) | Todoist API token; enables jarvis-todoist on port 8742 |
| `JARVIS_AUTOCRLF` | `false` | Set `true` for Windows hosts (git line ending conversion) |
| `JARVIS_CORE_PORT` | `8741` | Port for jarvis-core |
| `JARVIS_TODOIST_PORT` | `8742` | Port for jarvis-todoist |
| `CHROMA_PORT` | `8743` | Port for ChromaDB server |
| `CHROMA_HOST` | `127.0.0.1` | ChromaDB bind address (inside container) |
| `JARVIS_BIND_HOST` | `127.0.0.1` | Host address the MCP ports (8741/8742/8744) are published on. Loopback by default; set `0.0.0.0` **only** with `server.auth.enabled: true` in config.json. The explorer (8750) is always loopback. The installer-generated compose follows `server.auth.enabled` automatically. |

## Management

The installer creates `~/.jarvis/jarvis-docker.sh`:

```bash
~/.jarvis/jarvis-docker.sh status   # Container status
~/.jarvis/jarvis-docker.sh logs     # Follow logs
~/.jarvis/jarvis-docker.sh restart  # Restart services
~/.jarvis/jarvis-docker.sh update   # Pull latest image + restart
~/.jarvis/jarvis-docker.sh stop     # Stop container
```

## Windows Notes

Docker is the **recommended** installation method for Windows:

- No Python compilation issues (ChromaDB's native deps are pre-built in the image)
- No uv/uvx installation needed
- Set `JARVIS_AUTOCRLF=true` to handle CRLF line endings in vault files
- Use forward slashes in volume paths: `-v C:/Users/you/vault:/vault`

With Docker Desktop and WSL2:
```bash
docker compose -f ~/.jarvis/docker-compose.yml up -d
```

## Troubleshooting

### Port already in use

```bash
# Check what's using port 8741
lsof -i :8741  # macOS/Linux
netstat -ano | findstr :8741  # Windows

# Change ports in docker-compose.yml:
# ports:
#   - "9741:8741"
#   - "9742:8742"
```

### Container won't start

```bash
# --tail reads from the end; forward reads of a long json log can stop early
docker compose -f ~/.jarvis/docker-compose.yml logs --tail 200
```

Container logs are capped (`logging: driver local, max-size 10m, max-file 5`) in
the repo compose and the installer-generated one. Installer deployments from
before this change get the log cap and loopback-only ports after re-running
`install.sh`, which recreates the container; their database already lives in
the named `pgdata` volume and survives that (take a backup anyway). Deployments
started from `docker/docker-compose.yml` before it gained the `pgdata` volume
must dump their database first — see the upgrade note under Quick Start.

Common issues:
- Volume path doesn't exist on host
- Port conflict with another service
- Missing config.json (create minimal one — see Quick Start)

### Database outage (disk full / recovery mode)

If the Docker VM disk fills, PostgreSQL can crash-loop in recovery mode. What you
will see:

- `curl --max-time 2 localhost:8741/health` → `postgres.status` is `disk_full`,
  `recovering` or `unreachable`; the statusline shows a red `DB:` segment.
- Hook endpoints answer `503` + `Retry-After: 10` within ~2s; hook clients keep
  their payloads queued and back off (30s doubling to 30 min).
- The Memory Explorer answers `503 Database unavailable: <cause>` + `Retry-After: 10`:
  at once while its background probe (every 10s) finds the database unreachable,
  otherwise within ~5s (the pool timeout).
- The `jarvis` launcher always starts Claude and prints one line saying why
  memory is degraded. Its probes are bounded; overrides: `JARVIS_CORE_URL`
  (default `http://localhost:8741`), `JARVIS_START_TIMEOUT` (20s),
  `JARVIS_COMPOSE_TIMEOUT` (60s), `JARVIS_DOCKER_TIMEOUT` (5s),
  `JARVIS_PLUGIN_CHECK_TIMEOUT` (10s), `JARVIS_SKIP_PLUGIN_CHECK=1`.
- The entrypoint's watchdog exits the container after ~2 min of PostgreSQL not
  accepting connections, or ~30s after the server process has exited (nothing
  restarts it in place), so Docker's restart policy restarts it cleanly.
- Once space is free and no disk-full error has occurred for 60s, `/health`
  reports `ok` again (the statusline and launcher follow it).

Recovery: free Docker disk space (`docker system df`, prune unused images/build
cache, or enlarge the VM disk), then `docker compose -f ~/.jarvis/docker-compose.yml restart`.

Back up the embedded database regularly (custom-format `pg_dump`, verified, dir
700 / files 600, newest N kept):

```bash
plugins/jarvis/scripts/jarvis-transport.sh backup      # keep newest 7
plugins/jarvis/scripts/jarvis-transport.sh backup 14   # keep newest 14
# Started from the repo compose instead of ~/.jarvis/docker-compose.yml
# (same environment as for `up`, e.g. JARVIS_VAULT_PATH exported):
JARVIS_COMPOSE_FILE=docker/docker-compose.yml plugins/jarvis/scripts/jarvis-transport.sh backup
```

Backups go to `$JARVIS_HOME/backups` either way. Each run also tightens any
other `*.dump` in that directory (e.g. one taken by hand with `pg_dump > file`)
to mode 600.

### ChromaDB startup slow

First startup with an empty database takes ~5-10 seconds for ChromaDB initialization. Subsequent starts are faster. The healthcheck has a 20-second start period to accommodate ChromaDB + jarvis-core startup sequence.

### Hooks in Docker mode

Claude Code hooks (session-cleanup, prompt-search, stop-extract) run on the **host**, not inside the container. They connect directly to ChromaDB on port 8743 via `HttpClient` — no extra configuration needed as long as port 8743 is exposed.

Requirements for hooks with Docker:
1. **Python 3.10+ on host** with `chromadb` package installed (hooks import from plugin source)
2. ChromaDB port (8743) accessible from host (default in docker-compose.yml)

## Building Locally

```bash
# From repo root
docker build -f docker/Dockerfile -t jarvis-local .

# Run with local image
docker run -d --name jarvis \
  -p 8741:8741 -p 8742:8742 -p 8743:8743 \
  -v ~/my-vault:/vault \
  -v ~/.jarvis:/config \
  -e JARVIS_HOME=/config \
  -e JARVIS_VAULT_PATH=/vault \
  jarvis-local
```

## Upgrading

```bash
# Pull latest
docker pull ghcr.io/rsprudencio/jarvis:latest

# Restart with new image
docker compose -f ~/.jarvis/docker-compose.yml up -d
```

Or use the helper: `~/.jarvis/jarvis-docker.sh update`
