# Claude Development Guide - Jarvis Plugin

This file contains development instructions and conventions for Claude when working on the Jarvis plugin.

---

## Commit Message Guidelines

### Subject Line (First Line)
- **Imperative mood**: "Add feature" not "Added feature"
- **Start with verb**: Add, Fix, Update, Remove, Refactor
- **Keep under 72 characters**
- **Include scope if helpful**: "Fix jarvis-todoist-agent: remove non-existent tool"

### Common Prefixes

| Prefix | Use For |
|--------|---------|
| Add | New features, files, capabilities |
| Fix | Bug fixes |
| Update | Enhancements to existing features |
| Remove | Deletions |
| Refactor | Code restructuring (no behavior change) |

### Body (Optional but Recommended for Non-Trivial Changes)

- Blank line after subject
- Explain **WHAT** and **WHY**, not HOW
- For version bumps: Include "Version bump: X.Y.Z → A.B.C (patch/minor/major)"

### Combined Commits (Preferred)

Feature + version bump in one commit:

```
Add jarvis-explorer-agent and bump to v0.3.0

New Features:
- Vault-aware exploration agent for search
- Supports vault structure and access control

Version bump: 0.2.2 → 0.3.0
```

Version-only commit (rare - only for hotfix releases):

```
Bump version to 0.3.2

Hotfix release for critical production issue.
Version bump: 0.3.1 → 0.3.2
```

---

## Development Workflow

When making plugin changes that require a version bump:

1. **Make code changes** (agents, skills, MCP server, etc.)
2. **`/bump`** - Bump version and stage version files (plugin.json + CLAUDE.md)
3. **`git add <other-files>`** - Stage all other changed files
4. **`git commit -m "Your changes and bump to v0.X.Y"`** - Commit with proper message
5. **`git tag -a v0.X.Y -m "Version 0.X.Y: Description"`** - Tag the commit
6. **`git push && git push --tags`** - Push commits and tags to remote
7. **`/reinstall`** - Clear cache and reinstall plugin
8. **Restart Claude Code** - Required for plugin changes to take effect

**The flow:** changes → bump → commit → tag → push → reinstall → restart

### Pre-Commit Checklist

Before committing plugin changes:

- [ ] Version bumped? (use `/bump` if releasing)
- [ ] All modified files staged? (`git status`)
- [ ] Commit message follows guidelines?
- [ ] No sensitive files? (.env, credentials)

---

## Testing

### Test Tiers

| Command | What it runs | When to use |
|---------|-------------|-------------|
| `make test` | Unit tests only (~1300 tests, no DB required) | Quick validation during development |
| `make test-e2e` | E2E tests against real PostgreSQL (~22 tests) | After any SQL, schema, or JSONB change |
| `make test-all` | Both unit + e2e | Before committing MCP server changes |

### E2E Testing Convention

**Default to e2e tests for anything that touches SQL.** The InMemoryDB mock cannot catch SQL-level bugs (type casts, JSONB operators, trigger behavior, halfvec storage). When investigating or fixing a database-related issue:

1. **Write an e2e test first** in `plugins/jarvis/mcp-server/tests/e2e/` that reproduces the problem against real PostgreSQL
2. **Fix the bug** — the e2e test verifies the fix works with actual SQL execution
3. **Run `make test-all`** to confirm both unit and e2e tests pass

Do NOT manually run SQL queries against a test database to verify fixes. Write a proper test instead — it becomes a permanent regression guard.

### E2E Test Files

| File | Coverage |
|------|----------|
| `test_tier2_e2e.py` | Tier 2 lifecycle: store, read, increment, filter, sort, delete |
| `test_memory_e2e.py` | Memory write/read/overwrite with file + PG roundtrip |
| `test_search_e2e.py` | Vector similarity search, retrieval count batch increment |
| `test_cross_cutting_e2e.py` | Metadata fidelity, idempotency, JSONB merge, triggers, halfvec, schema, jarvis_meta |

### Running E2E Tests Manually

```bash
# Start PG, run tests, tear down (preferred — all-in-one)
make test-e2e

# Or manually:
cd plugins/jarvis/mcp-server
docker compose -f docker-compose.e2e.yml up -d --wait
E2E_POSTGRES_URL="postgresql://jarvis:jarvis@localhost:25432/jarvis?sslmode=disable" \
    uv run python -m pytest tests/e2e/ -v
docker compose -f docker-compose.e2e.yml down -v
```

E2E tests skip gracefully when `E2E_POSTGRES_URL` is absent — they never cause failures in `make test`.

---

## Version Bumping Workflow

**Use `/bump` skill to bump version and stage files for commit.**

### The Rule

1. Make code changes (agents, skills, MCP server, etc.)
2. **Use `/bump`** to:
   - Update version in `plugins/jarvis/.claude-plugin/plugin.json` (and optionally other plugin manifests)
   - Update CLAUDE.md version history (for minor/major)
   - Stage version files automatically
3. Stage other changed files: `git add <files>`
4. Commit changes with proper message
5. **Tag the version commit**: `git tag -a v0.X.Y -m "Version 0.X.Y: Description"`
6. Push: `git push && git push --tags`
7. Reinstall plugin with `/reinstall`

**DO NOT bump version without code changes.** Empty version bumps are not allowed.

**ALWAYS tag version bump commits.** This creates a permanent marker for each release and enables proper version tracking (`git tag --contains <commit>`).

### Semantic Versioning Rules

Use the following criteria to decide bump type:

#### Patch (0.2.x → 0.2.x+1)
Use for:
- Bug fixes
- Small changes
- Documentation updates
- Minor refactoring
- Tool configuration tweaks
- Agent instruction clarifications

#### Minor (0.2.x → 0.3.0)
Use for:
- New features
- New skills/agents/commands
- Workflow changes
- Non-breaking enhancements
- New MCP integrations

#### Major (0.x.x → 1.0.0)
Use for:
- Breaking changes
- Complete workflow redesigns
- **ALWAYS ask user before major bumps**

### Current Version
Check: `plugins/jarvis/.claude-plugin/plugin.json` (core plugin version)

---

## Git Tag Workflow

After committing a version bump:

### 1. Create Annotated Tag

```bash
git tag -a v0.X.Y -m "Version 0.X.Y: Brief description"
```

### 2. Tag Message Convention

Follow this format for tag messages:

```
Version 0.3.1: Fix jarvis-todoist-agent missing tool
Version 0.3.0: Add jarvis-explorer-agent
Version 0.2.2: Add CLAUDE.md development guide
```

### 3. Push Tags

```bash
# Push specific tag
git push origin v0.X.Y

# Or push all tags
git push --tags
```

### 4. Verify

```bash
# Check if HEAD is tagged
git tag --contains HEAD

# List recent tags
git tag -l --sort=-version:refname | head -5

# Show tag details
git show v0.X.Y
```

---

## Plugin Reinstall Workflow

When reinstalling during development (after code changes):

### Step 1: Bump Version
Edit `plugins/jarvis/.claude-plugin/plugin.json` (and other affected plugin manifests) and increment version according to rules above.

### Step 2: Clean Cache & Reinstall

**Preferred:** Use **`/reinstall`** skill - handles everything automatically.

Manual alternative:
```bash
rm -rf ~/.claude/plugins/cache/jarvis-plugins/jarvis/*
claude plugin marketplace update
claude plugin uninstall jarvis@jarvis-plugins
claude plugin install jarvis@jarvis-plugins
```

### Step 3: Restart Claude Code
**Required** - Plugin changes only apply after full restart (not just reload).

---

## When to Reinstall

Reinstall is required after modifying:

- **Agent definitions** - Files in `plugins/*/agents/*.md`
- **Skills** - Files in `plugins/*/skills/*/SKILL.md`
- **MCP server code** - Files in `plugins/jarvis/mcp-server/`
- **Plugin manifests** - `plugins/*/.claude-plugin/plugin.json`
- **MCP configuration** - `plugins/jarvis/.mcp.json`
- **System prompt modules** - `plugins/*/mcp-server/system_prompt.py`

---

## Troubleshooting Reinstalls

If plugin doesn't load after reinstall:

1. **Verify cache cleared**:
   ```bash
   ls ~/.claude/plugins/cache/jarvis-plugins/jarvis/
   # Should only show current version
   ```

2. **Check marketplace updated**:
   ```bash
   claude plugin marketplace list
   ```

3. **Verify uninstall completed**:
   ```bash
   claude plugin list
   # Should NOT show jarvis
   ```

4. **Check for errors** in Claude Code logs

5. **Full restart** - Quit and reopen Claude Code (not just reload)

6. **Verify git state**:
   ```bash
   git status
   # Ensure changes are committed
   ```

---

## Development Notes

### Plugin Architecture (Modular Marketplace)

The plugin is split into 5 independent plugins in a single marketplace:

- **`plugins/jarvis/`** - Core: MCP server, vault management, semantic memory, auto-extract
- **`plugins/jarvis-obsidian/`** - PKM: git audit trail, journal, exploration agents + skills
- **`plugins/jarvis-todoist/`** - Optional: Todoist agent + skills
- **`plugins/jarvis-strategic/`** - Optional: Strategic analysis skills
- **`plugins/jarvis-toolbelt/`** - Engineering: security review, TDD workflows
- **`lib/jarvis-common/`** - Shared Python package (config, paths, namespaces)

### Key Files

| File | Purpose |
|------|---------|
| `.claude-plugin/marketplace.json` | Marketplace manifest (all plugins) |
| `plugins/jarvis/.claude-plugin/plugin.json` | Core plugin manifest (version, name) |
| `plugins/jarvis/mcp-server/system_prompt.py` | Jarvis core identity and constraints (MCP instructions) |
| `plugins/jarvis/.mcp.json` | MCP server registration |
| `plugins/jarvis/mcp-server/` | Python MCP server (13 tools) |
| `plugins/jarvis/skills/*/SKILL.md` | Core skill workflows |
| `plugins/jarvis-obsidian/.claude-plugin/plugin.json` | Obsidian plugin manifest |
| `plugins/jarvis-obsidian/mcp-server/` | Obsidian MCP server (9 git tools) |
| `plugins/jarvis-obsidian/agents/*.md` | PKM agent definitions (journal, audit, explorer) |
| `plugins/jarvis-obsidian/skills/*/SKILL.md` | PKM skill workflows |
| `lib/jarvis-common/` | Shared config, paths, namespaces package |
| `plugins/jarvis-todoist/agents/*.md` | Todoist agent definition |
| `plugins/jarvis-todoist/skills/*/SKILL.md` | Todoist skill workflows |
| `plugins/jarvis-strategic/skills/*/SKILL.md` | Strategic skill workflows |
| `docker/Dockerfile` | Multi-stage Docker image build |
| `docker/entrypoint.sh` | Process manager for containerized servers |
| `docker/docker-compose.yml` | Compose template for dev/reference |
| `.github/workflows/docker-publish.yml` | CI: build & push to GHCR on tags |
| `.github/workflows/docker-test.yml` | CI: test Docker image on PRs |

### Docker Development

All three MCP servers have `http_app.py` alongside `server.py` — thin ASGI wrappers using `StreamableHTTPSessionManager` from MCP SDK. Key facts:

- **Transport:** Streamable HTTP with `json_response=True` (not SSE)
- **Architecture:** Raw ASGI app (no Starlette) to avoid `/mcp` → `/mcp/` 307 redirects
- **Ports:** jarvis-core on 8741, jarvis-todoist on 8742, jarvis-obsidian on 8744
- **Config:** `JARVIS_HOME` and `JARVIS_VAULT_PATH` env vars override config.json paths
- **Shared code:** `lib/jarvis-common/` provides config, paths, and namespace resolution used by core and obsidian

```bash
# Build locally
docker build -f docker/Dockerfile -t jarvis-local .

# Run integration tests (requires image built)
python3 -m pytest docker/tests/ -v

# Test manually
docker run -d -p 8741:8741 -v /tmp/vault:/vault -v ~/.jarvis:/config \
  -e JARVIS_HOME=/config -e JARVIS_VAULT_PATH=/vault jarvis-local
curl http://localhost:8741/health
```

### Version History

- **3.7.0** - Outage resilience + network lockdown, from the 2026-09-24/25 incident (Docker VM disk filled → embedded PG hit ENOSPC on a `retrieval_candidates` INSERT → ~15h checkpoint-PANIC/recovery-mode loop → restart loop). What made a DB outage into a whole-system hang: async hook handlers (`http_app.py` prompt-context/ack/auto-extract context/ingest) and MCP `call_tool` ran blocking DB code ON the event loop, and the pool had no timeout, so every checkout waited 30s — one ingest = 8 checkouts = 240s of total loop freeze (observed live: exact 240s gaps, one 480s chain), starving the DB-free `/health`; the launcher's bare `curl` then hung silently for 0–480s and the explorer returned a verbatim 30s `PoolTimeout` 500 for every search (empty query was irrelevant). **Upgrade notes (read first):** (1) MCP ports 8741/8742/8744 now publish on `127.0.0.1` by default — they were reachable unauthenticated from the LAN (`*:874x` via Colima's ssh forwarder, allowed by the macOS firewall; verified with `tools/list` via the LAN IP); opt back in only with `JARVIS_BIND_HOST=0.0.0.0` + `server.auth.enabled=true` (install.sh follows the config). (2) The repo `docker/docker-compose.yml` gains the named `pgdata` volume (the installer compose already had it): Quick Start deployments keep PGDATA in the container layer, so **dump before the next `up -d`** (docker/README.md has the verified dump/restore sequence). (3) Rebuild the image and re-run install.sh / `/reinstall` Step 3b: install.sh now re-copies the launcher when it changed (it used to copy once). Changes: **core** — dedicated bounded executors (hooks 4 workers, 2.0s deadline with contextvars; MCP tools separate, 60s) so `/health` stays <3ms p99 under outage load; fail-fast pool (`memory.pool_timeout_seconds` 10, `max_waiting` 32, `check`, connect_timeout 3, 1.5s hook-path checkouts, thread-safe lazy init) + circuit breaker (`DatabaseUnavailable`, half-open ~10s, trips on OperationalError/PoolTimeout/TooManyRequests, pool-busy ≠ down); ingest stops at the first DB failure and returns `retryable` → HTTP 503 + `Retry-After: 10` (previously the client counted it delivered → silent loss); prompt/auto-extract context short-circuit `degraded:true`; `/health` keeps `status: ok` and adds a cached `postgres` object (`ok|recovering|unreachable|disk_full|unknown`, free bytes, 10s background probe; disk_full clears on the first successful write); `DiskFull` logged CRITICAL (was DEBUG — the only trace of the first ENOSPC); DSN/password scrubbing in all DB error text; access log allowlists headers (no Authorization/Cookie/internal token), tool calls log argument keys only (core + obsidian); partial-body disconnect no longer spins the loop at 100% CPU; 1 MiB body cap (413, drained); generated IDs (`obs::`/`learning::`/`worklog::`) strictly increasing per process + insert-never-overwrite (concurrency otherwise lost 23/120 same-ms writes to the upsert) and a per-`ingest_event_id` advisory lock so an overlapping replay stores once; dedup skips rerank to stay inside the deadline; ONNX encode serialized; retention janitor actually runs on startup (was gated on 24h uptime), deletes in 200-row batches and logs counts; redundant `idx_retrieval_candidates_event` dropped. **Explorer** — pool timeout 5 + check, 503 `Database unavailable: <real cause>` (fast-fail from the cached probe verdict), 504 on QueryCanceled, lazy source re-discovery (an explorer started while PG was down no longer 400s until restart), `postgres` in `/health`, SPA errors via `esc()`. **Launcher** (`jarvis.sh`) — `set -u`, every external call bounded by a process-group timeout, curl exit codes classified (wedged → warn + exec immediately; crash-loop → hint + exec; Docker missing/down/hung → exec), a plugin-list timeout no longer reports "not installed", always ends in `exec claude`. **Entrypoint** — PG watchdog (exit after ~120s unavailable, ~30s once the postmaster is gone, so the restart policy engages), `pg_ctl` failure prints log tail + `df`, low-disk warning, `ulimit -c 0`, safe stale `postmaster.pid` removal, `pg_isready -U postgres`. **Deploy** — compose `logging: local` 10m×5, `jarvis-transport.sh backup [N]` (verified `pg_dump -Fc`, dir 700/file 600, rotation of its own timestamped dumps only, `JARVIS_COMPOSE_FILE`), bounded curls in install.sh/jarvis-transport.sh/Makefile. **Hook clients** — per-entry exponential backoff (30s→30min), 60s `core_degraded` marker, queue cap 500 + dedupe by `ingest_event_id`, 400 = permanent drop, 503/retryable = re-queue; statusline shows DB status. Verified in the real image (Python 3.12, bash 5, PG 17.11): smoke, a live disk-full rehearsal (heap + WAL PANIC, watchdog restart, recovery with no loss, kill -9 of backend/postmaster, stale pid), and an upgrade + rollback on a restored copy of the production DB (counts exact, migration 5ms). jarvis-obsidian 1.0.1 (tool-arg logging). 3,034 tests (2,488 core unit + 149 e2e + 124 explorer + 100 obsidian + 84 statusline + 72 todoist + 17 host-inference); shellcheck clean.
- **3.6.0** - LLM contextual summaries (out-of-band) + retrieval data-integrity hardening. **Measured win**: one Haiku-written situating sentence per document lifts the flagship failure case from BGE logit −8.16 to **+0.03** on the answer chunk (mechanical path/title prefix vs relational summary), clearing the −4.0 injection gate. Architecture: generation is an explicit out-of-band operation in `bin/generate_summaries.py` — the ONLY LLM entry point (bounded `--limit`/`--concurrency`/`--timeout`, hash-idempotent, coverage report); `index_vault`/`index_file`/vault writes and all read sites (both query rerank paths, shadow scorer, reindexer) only ever READ `obsidian.document_context`. Inline generation had made the feature a guaranteed no-op in the container (no SDK/CLI/key), applied the spend cap per 10-chunk flush, serialized concurrency, and blocked the MCP event loop on an untimed LLM call during every vault write. Cache coherence is owned by the write path: a chunked file whose cached `content_hash` no longer matches DELETEs its row (readers are hash-blind by design), including in mechanical mode so the documented `contextual_summaries.enabled=false` rollback cannot arm the cache to serve stale sentences later; a failed DELETE escalates to CRITICAL instead of logging success. Embedding-space identity now records the **measured** state — `none|mechanical|summary` plus recorded-only `partial-summary` with coverage counts — scored against the whole store rather than the files a run happened to touch (sensitive dirs and secret-flagged files are skipped before the force-delete, so run-local counters stamped mixed spaces as clean `summary`); `check_model_consistency` warns loudly (never fatally) on mismatch AND on partial coverage, naming the correct two-step remedy. Also: prompt-injection framing for document bodies (fenced untrusted block, field defanging, instruction-following screen), SDK-verified LLM availability (a bare env var previously produced 500 warnings and zero summaries), shadow-scorer summary-cache-drift skip, era-aware `simulate_policy`/label export so Phase-2 calibration cannot mix mechanical- and summary-era events, `anthropic` in the image + `ANTHROPIC_API_KEY` passthrough, bounded SDK timeout (was 10 min × 2 retries). Operator sequence after upgrade: `bin/generate_summaries.py` (host or `docker exec`), then one `jarvis_index_vault(force=true)`. 2,555 tests (2,167 core unit + 136 e2e + satellites).
- **3.5.0** - Retrieval observability + hybrid retrieval + contextual chunk embeddings + host-inference hardening. (1) **Retrieval telemetry**: every retrieval records an event + per-candidate score trail (raw cosine, BGE logit, blend, terminal reason, channel) into `local.retrieval_events/retrieval_candidates` with feedback labeling tables, a read-only policy simulator (`cosine-only|bge-only|coarse+bge|cosine-or-bge`, censoring-aware — NULL-logit candidates are excluded, not counted as rejections), shadow BGE scoring (off-event-loop via `asyncio.to_thread`, guarded against model/embedding/augmentation identity drift and tokenizer divergence — skips instead of silently mis-scoring), delivery acknowledgments from the hook, and a Memory Explorer Retrieval tab (funnel, histograms, sortable full-screen candidate view with on-demand body resolution — telemetry itself never stores bodies). (2) **Contextual chunk embeddings**: fragments are embedded AND reranked as `Document: <path> — <title> › <heading>\n\n` + chunk (config `memory.chunking.contextual_embeddings`, rollback switch); stored text stays byte-identical; augmentation state is part of the embedding-space identity in `local.meta` (mismatch = loud startup warning); applied consistently at index, both query rerank paths, shadow scorer, and `reindex_embeddings.py`. Requires one `jarvis_index_vault(force=true)` after upgrade. (3) **Hybrid lexical channel**: generated `tsvector` columns + GIN on `obsidian.documents`/`local.memories` (body capped at 200k chars), IDF-informative term selection (file-granularity df over the union corpus, one batched round-trip, lexemes sanitized `^[a-z0-9]+$`), per-term rarity-ordered candidate fill (an OR-query capped by IDF-blind ts_rank_cd floods out the rare term's matches — measured live), per-user isolation matching the ANN path, chunked-parent exclusion, true raw cosine computed per lexical row. (4) **Recall-additive BGE logit gate**: `semantic_context` injects iff `cosine ≥ threshold` OR (lexical-channel row AND `raw_bge_logit ≥ memory.context_enrichment.bge_logit_threshold` [−4.0]); cosine-passers hold strict priority at the rerank cap (reserved lexical seats ride the channel's rarity rank, not cosine), max_results cut, budget, and all three dedup stages — property-tested `enabled ⊇ disabled`; reranker-down degrades to exactly cosine-only behavior. Phase 2 (BGE as sole arbiter, cosine demoted to coarse filter) is deliberately deferred pending new-space calibration labels; simulator + shadow data exist to rehearse it. (5) **Deployment/migration hardening**: `ModelMismatchError` now aborts startup via ASGI `lifespan.startup.failed` (a bare raise under uvicorn `lifespan=auto` left a zombie server answering /health with no MCP), embedding warm-up failure serves DEGRADED instead of dying, legacy `/app/models/embedding` identity gets accurate reindex remediation (`--force-model-record` records the compared identity now), partial-store reindex cannot relabel `local.meta` while other stores hold old-space vectors (SHARE-lock guarded), `install.sh` provisions host inference on arm64 macOS (fetch + launchd + health gate; hard preflight elsewhere), compose gains `host-gateway`, `rerank_multi` enforces `max_latency_ms`, `query_vault` reranked ANN window restored to `min(100,total)` pre-dedup. Explorer fixes: SPA killed by Python-eaten `\'`/`\n` escapes in onclick handlers (node --check + source-scan regression tests), stateful filters, label rendering/confirmation. ~160 new tests (1,961 core unit + 118 e2e).
- **3.4.1** - Retune per-prompt injection threshold 0.876 → 0.85 (recall over precision): the 2026-07-17 calibration selected 0.876 as the lowest zero-false-positive point, but at that cutoff recall was only 0.26 — most prompts injected nothing, and to the user it read as "injection is entirely broken." 0.85 is the recall-favoring operating point from the same labeled run (recall ~0.58, ~16% off-topic leak, ~1.18 mean matches/prompt vs 0.29 at 0.876). This is a deliberate precision→recall trade, not a bugfix; the design's "proper" recall fix remains the planned reranker, not a looser cosine cutoff. Changed all shipping defaults (`defaults/config.json`, `config.py`, `query.py` param + docstring, `hook_endpoints.py` fallback) plus user-facing docs (`capabilities.json`, `jarvis-settings` SKILL presets, `bench/__main__.py` help). Left untouched: the calibration harness sweep list, `test_injection_calibration.py` (tests the selector algorithm on fixed synthetic data), and the historical `bench/results/20260717-injection-quality.md` record. NOTE: existing installs have 0.876 materialized in `~/.jarvis/config.json` and must lower `memory.context_enrichment.threshold` there too — a reinstall alone won't change an already-written config value. Two silent-failure notes surfaced while diagnosing: (a) `_write_telemetry` only fires when matches exist, so a 0-match run logs nothing; (b) with `debug: false` there is no trace at all. 1,803 core unit tests still green (default-assertion test updated to 0.85).
- **3.4.0** - Unified retrieval scoring (Layer 4 of the passage-ranking redesign, defect #6 fix): one formula for every schema — `score = similarity + 0.24·(effective_importance − 0.5)`, no clamp — replacing the two incommensurable formulas (vault's clamped `min(1.0, sim + boost + recency)` vs memories' `0.7·sim + 0.3·imp` blend that capped memories at 0.94 while 517 vault chunks pinned at 1.0; a perfect-match memory ranked #2534 of 2,864). Similarity is now TRUE raw cosine (`1 − pgvector_distance`, range −1..1), not the old `1 − d/2` compression. Vault recency term dropped (`updated_at` is the reindex timestamp — 100% of chunks carried the max boost after every reindex); fixed phantom retrieval boost (`last_retrieved_at` None no longer parsed as "just retrieved"). `semantic_context` gates on RAW cosine similarity, decoupled from the importance boost — default threshold 0.5 → 0.85 (raw-cosine band from the calibration work) plus a `max_results` injection cap (default 20); raw `similarity` exposed in query/context results and telemetry. Observation dedup gates on max similarity across a 5-candidate window (top-by-relevance is not nearest-by-similarity). Defect #8 fixed: `query_vault` honors the caller's `n_results` (was silently returning `top_k`=10 when reranked); non-reranking overfetch now uses `ranking.overfetch_factor`. Staleness penalty floor removed (scores are unclamped). Cross-encoder reranking disabled by default (ms-marco-MiniLM measured −0.055 nDCG@10 on ArguAna — net-negative vs the bi-encoder alone; re-enable only with a reranker that provably improves labeled nDCG). `memory.ranking` config simplified to `{importance_weight: 0.24, overfetch_factor}`. NOTE for pre-3.4 installs: the installer materialized old defaults into `~/.jarvis/config.json` — `memory.ranking.similarity_weight/importance_weight: 0.3` and `memory.reranking.enabled: true` must be updated there too (this machine's live config already is). Also: retrieval benchmark harness (`bench/`, `make bench PRESET=core|full|arg KIND=embed|rerank|both`, nDCG@10 decides — never STS) with geometry-calibration script (centering/whitening/ABTT all measured harmful) and reranker scorecards. New regression guards: e2e self-match (memory's own text ranks #1 vs engineered cosine-0.9 decoy), threshold-gate decoupling, dedup similarity-window, decay-disabled path, n_results-honored, cross-schema commensurability suite. 1,803 core unit + 107 e2e tests.
- **3.3.8** - Fix Obsidian search 500 in admin portal: `_search_sync` applied the soft-delete predicate `status != 'deleted'` to every source with `type: "local"` (which means "in the local connection pool", not "the `local` schema"), but `obsidian.documents` has no `status` column — every Obsidian search (text/semantic/metadata) failed with `column "status" does not exist` since v3.3.2. Added `has_status` flag per source, gating the predicate. Also: memory-explorer tests were never wired into `make test` and couldn't run at all (missing system libpq), which is why this shipped — added `psycopg[binary]` dev dep, wired explorer into `make test`, fixed core/obsidian test commands (bare `python3`, missing `dev` extra). 13 new search tests (1954 total: 1801 core + 99 obsidian + 54 explorer).
- **3.3.0** - Generic metadata filtering: any JSONB metadata key is now filterable in both query and list modes via the `filter` parameter (e.g. `project_dir`, `git_branch`, `session_id`, `workstream`). Known keys (type, importance, tags, directory) still route to dedicated columns; unknown keys fall back to parameterized `metadata->>%s = %s` (GIN-indexed). `filter` parameter wired through `content_list()` for list mode (was query-only). Updated tool schema description, mock cursor regex for parameterized JSONB keys. 17 new tests (1785 total).
- **3.2.0** - Schema-aware context enrichment: rename `source` → `id` in match dicts, add `schema` attribute to injected XML (local/obsidian/remote_*), 3-way budget split (local/vault/remote with overflow), default schemas changed from `local,obsidian` to `all` (remote memories now surfaced in per-prompt injection), budget_used reporting updated (`core` → `local` + `remote`), backward-compatible `id`/`source` fallback in XML formatter, telemetry keys updated. Also: sync jarvis.sh with live version (remove stale ensure_postgres), test fixes for v3.1.4 multi-schema changes (scope downgrade removal, schema registry discovery mocking, sync_pull 2-arg signature). 1847 total tests.
- **3.1.0** - Admin portal CRUD for routing rules and sync remotes: move routing engine + sync validation to `jarvis-common` (shared library, thin re-exports in MCP server), thread-safe mtime-based config cache with `os.fstat()` TOCTOU avoidance, atomic `config_writer.py` with `O_CREAT|O_EXCL` locking + password sentinel re-hydration + validation-before-write, new `app_admin.py` FastAPI router (14 endpoints: CRUD rules/remotes/project-groups, rule tester, connection tester, auth gate via `jarvis_common.auth`), expanded admin tab frontend (auth token input, inline CRUD forms, up/down rule reorder, collapsible rule tester with match visualization), name-based rule addressing (not index), 409 Conflict on delete-remote-with-deps, env-var password round-trip preservation. jarvis-common bumped to v1.2.0 (3 new modules). 52 new tests (21 config_writer + 31 admin API), 1794 total.
- **2.0.0** - MCP-native system prompt architecture + statusline + strategic taxonomy: replace runtime `system-prompt.md` file reading with `system_prompt.py` Python module imported at startup (both core and todoist), delete all satellite `system-prompt.md` files, instructions now injected via MCP `InitializeResult.instructions` (enables Claude Code Desktop — no shell wrapper needed), rewrite `jarvis.sh` launcher (remove `--append-system-prompt` pipeline, keep Docker auto-start only, 66→43 lines), rewrite `/jarvis` activation skill for MCP-native paradigm, new `statusline/statusline.py` for Claude Code (model coloring, git info, MCP servers, Jarvis health, cost, context %; cached, stdlib-only, 52 tests), statusline install via `install.sh` + `/jarvis-settings`, rename strategic memories (trajectory→goals, values→principles, focus-areas→priorities, patterns→insights), update all references across 4 skills + capabilities.json + todoist agent + memory_files.py, rewrite jarvis-todoist system prompt to compact behavioral rules with alert handling (DAR-reviewed), stale reference cleanup across CLAUDE.md/README/docker docs/jarvis-close/capabilities.json. jarvis-todoist bumped to v1.6.0, jarvis-strategic to v1.2.0. 1,480 total tests.
- **1.37.0** - URL-based config simplification: eliminate 3-mode transport system (`local`/`container`/`remote`), `.mcp.json` always HTTP, Docker is the only install method, remove `mcp_transport`/`mcp_remote_url` config keys and `get_mcp_transport()`/`get_mcp_remote_url()` getters, remove server early-exit pattern from both servers, simplify prompt_search.py hook (inline-only retrieval bumps), rewrite `jarvis.sh` launcher (175→66 lines, simple Docker auto-start), rewrite `jarvis-transport.sh` as service manager (503→210 lines, remove mode switching), simplify installer (single Docker code path), remove MCP Transport from `/jarvis-settings`, update capabilities.json/README/docker docs, delete `test_server_early_exit.py` and `TestMcpTransportConfig` (6 fewer tests, 1384+72 total). jarvis-todoist bumped to v1.5.4 (early-exit removal). Net ~470 lines removed across 18 files.
- **1.34.0** - Fail-closed transport safety + write lock fix: prevent mixed-writer SQLite corruption in container/remote mode by replacing fallthrough-to-local-write with durable disk queue + replay pipeline, `_post_tier2_write` retry with exponential backoff (3 attempts, 429/5xx retryable), `_enqueue_pending_tier2_write` atomic file persistence, `_drain_pending_tier2_writes` opportunistic replay on next invocation, `_build_ingest_event_id` SHA-256 idempotency keys, server-side `ingest_event_id` dedup in `tier2_write` under write lock, `_remove_vault_file` now wraps `_delete_existing_chunks` in `chroma_write_lock()` (was the only call site missing it), `_force_local_transport` autouse fixtures for test determinism, 8 new tests (1383 total)
- **1.28.0** - Worklog auto-journal: augment existing Haiku extraction call to capture intent-focused activity records alongside observations (zero extra API cost), new `worklog` Tier 2 content type with `worklog::` namespace, organic workstream discovery from ChromaDB (no registry file), Jaccard word-overlap dedup (0.7 threshold) within sessions, `normalize_worklog_response()` + `store_worklog()` + `discover_workstreams()` + `is_duplicate_worklog()` pipeline functions, `get_worklog_config()` with `memory.worklog` config section, `worklogs_promoted` path for promotion support, `/jarvis-worklog` skill for reviewing activity by date/workstream, `_HAIKU_MAX_TOKENS` 800→1000, updated capabilities.json/system-prompt.md/defaults/config.json, 44 new tests (1135 total)
- **1.27.0** - Budget-based per-prompt injection with vault references: replace per-item `max_content_length` truncation with total character budget (default 8000, split 50/50 between tier2 and vault), vault items shown as compact references (path + heading, ~120ch) instead of truncated content, tier2 items (observations, learnings) shown with full content, budget overflow from unused half to the other, `semantic_context()` signature simplified to `(query, threshold, budget)`, JSONL telemetry at `~/.jarvis/telemetry/prompt_search.jsonl` for ongoing threshold/budget analysis, config key consolidation (`budget_tier2`/`budget_vault` → single `budget`), updated across 8 files (query.py, prompt_search.py, config.py, defaults/config.json, capabilities.json, SKILL.md, tests, user config), 1073 tests passing
- **1.26.0** - Standalone jarvis executable + Docker auto-start: replace shell function injection (`jarvis.bash`/`jarvis.zsh`) with unified `jarvis.sh` standalone executable installed to PATH (`~/.local/bin/jarvis`), installer auto-cleans old `# Jarvis AI Assistant START/END` markers from RC files, `jarvis-transport.sh` auto-starts Docker container on `container` mode (compose up + 15s health check), auto-stops on `local` mode, remote health check on `remote` mode, updated SKILL.md/capabilities.json/README.md references (shell function → executable)
- **1.25.0** - MCP transport mode switching: `mcp_transport` config key (`local`/`container`/`remote`), `mcp_remote_url` for remote Docker hosts, server early-exit pattern (stdio servers `sys.exit(0)` when transport != local), `jarvis-transport.sh` standalone helper script (status/local/container/remote commands), `/jarvis-settings` transport menu option, `get_mcp_transport()`/`get_mcp_remote_url()` config getters, installer Docker flow uses transport helper, docs updated (docker/README.md switching section, capabilities.json, README.md). jarvis-todoist bumped to v1.5.1 (early-exit support). 2 new files, 10 modified, 6 new tests (1149 total across both plugins).
- **1.24.0** - Docker distribution: Streamable HTTP transport layer (`http_app.py` raw ASGI wrappers with `json_response=True`, no Starlette to avoid 307 redirects), multi-stage Dockerfile (Python 3.12 + uv deps + git/curl runtime), `entrypoint.sh` process manager (health checks, graceful shutdown, conditional Todoist), docker-compose.yml, `JARVIS_HOME`/`JARVIS_VAULT_PATH`/`TODOIST_API_TOKEN` env var overrides in config.py + paths.py + todoist_api.py, Docker option in `install.sh` (detection, compose generation, container management helper), GitHub Actions CI/CD (docker-publish on tags for multi-platform amd64+arm64, docker-test on PRs), `get_verified_vault_path()` bug fix (now uses `get_vault_path()` to respect env vars), 12 new files + 7 modified, 5 http_app tests + 7 Docker integration tests (1143 total across both plugins). jarvis-todoist bumped to v1.5.0.
- **1.23.0** - Configurable file format support (Markdown / Org-mode): new `format_support.py` central abstraction module, `file_format` config key (`"md"` or `"org"`), Org-mode parsers (`:PROPERTIES:` drawers, `*` headings, `#+BEGIN_SRC` blocks), `jarvis_get_format_reference` MCP tool for agents to load format templates at runtime, format-aware indexing/chunking/querying across 10 source files, installer + settings format selection, format reference files (`defaults/formats/`), 40+ new tests (1055 total)
- **1.22.1** - Fix Todoist tool name prefix (underscore → hyphen to match Claude Code plugin name resolution across 5 files); bump jarvis-todoist to v1.4.2 (1057 total tests)
- **1.22.0** - Native Todoist API: add dedicated MCP server to jarvis-todoist plugin via official `todoist-api-python` SDK (local stdio, no session drops), eliminate external HTTP MCP dependency (ai.todoist.net/mcp), 9 tools (find_tasks, find_tasks_by_date, add_tasks, complete_tasks, update_tasks, delete_object, user_info, find_projects, add_projects), SDK singleton + cached inbox resolution, tool name migration across 5 files (agent, 3 skills, system-prompt), `todoist.api_token` config key in defaults/config.json, 68 todoist tests + 989 core tests (1057 total). jarvis-todoist bumped to v1.4.2.
- **1.21.0** - Multi-turn session extraction: replace single-turn `pick_best_turn` with session-level pipeline — `filter_substantive_turns` (all qualifying turns), `extract_first_user_message` (conversation opener context), `compute_content_budget` (dynamic scaling from output token volume), `build_session_prompt` (proportional budget allocation across turns), `SESSION_EXTRACTION_PROMPT` (numbered turns + array response schema), `normalize_extraction_response` (backward-compatible new/legacy schema), Haiku `max_tokens` 300→800, `max_observations` config key (default 3, capped per extraction), main() rewrite for multi-observation loop with per-obs storage, 44 new tests (961 total)
- **1.20.0** - Per-session watermark tracking for auto-extract: replace global 120s cooldown with per-session line watermarks (`~/.jarvis/state/sessions/<id>.json`), forward multi-turn parser (`parse_all_turns`) + best-turn scorer (`pick_best_turn`), `read_transcript_from()` replaces `tail -N`, atomic watermark writes via tempfile+os.replace, SessionStart hook for stale watermark cleanup (>30 days), `max_transcript_lines` default 100→500, remove `cooldown_seconds` config key entirely, simplified `stop-extract.sh` (no temp files), 44 new tests (917 total)
- **1.19.0** - Config template SSoT + AI-first docs: `defaults/config.json` as single source of truth for all config keys (replaces duplicated heredoc in install.sh and inline JSON in SKILL.md), `capabilities.json` moved into plugin distribution for self-reference, README rewrite as human-friendly quickstart, install.sh reads config from template with Python substitution, stale `/jarvis-setup` references fixed in config.py error messages, system-prompt self-reference pointer to capabilities.json, pre-commit checklist updated with stale-ref check
- **1.18.0** - Project-aware auto-extract & promote: extract file paths from tool_use blocks in transcript turns, Haiku scope classification (project vs global), `relevant_files` + `scope` metadata in observations, `sort_by` parameter for tier2_list (importance_desc default, importance_asc, created_at_desc/asc, none), sort_by plumbed through retrieve API + server schema, project-aware promotion routing (nests under `<type>_promoted/<project_dir>/`), enriched promotion frontmatter (scope, project, files fields), updated /jarvis-promote SKILL.md (Project column, sorted browse, preview context), 30 new tests (873 total)
- **1.17.0** - Per-prompt semantic search: automatic vault memory injection via `UserPromptSubmit` hook, `semantic_context()` search function with threshold filtering + sensitive dir exclusion + no retrieval count increment, prompt filtering (skip trivial/short/commands/confirmations), XML-formatted context output, `get_per_prompt_config()` config getter, single-Python-process hook pipeline (~250-500ms), configurable threshold/max_results/content_length, system prompt + settings skill updates, 32 new tests (841 total)
- **1.16.0** - Markdown chunking + importance scoring + query expansion: hybrid heading/paragraph chunking for per-section embeddings, 0.0-1.0 importance scoring from content signals (type weight, concept patterns, recency decay, retrieval frequency), rule-based query expansion with synonym mappings and intent detection, chunk deduplication in query results, 3 new config getters (chunking/scoring/expansion), backward-compatible with unchunked documents, 96 new tests (806 total)
- **1.15.0** - Installer + settings redesign: rewrite `install.sh` as curl-pipe-bash installer with prereq validation (Python 3.10+, uv/uvx, Claude CLI), MCP server verification, full config write with all ~30 defaults visible; rename `/jarvis-setup` → `/jarvis-settings` as menu-driven re-runnable config manager; delete `/memory-index` skill (folded into install.sh + settings); update all references across 12 files (14 user-invocable skills)
- **1.14.0** - Unified Content API: consolidate 14 write/read/delete tools into 3 (`jarvis_store`, `jarvis_retrieve`, `jarvis_remove`) with namespace-based routing, retrieve-mutate-reindex closed loop, auto-index on vault writes, `topics`→`tags` taxonomy cleanup, `learning`/`decision` content types, observation project context enrichment (project_dir, git_branch), remove TYPE_* aliases (21 total tools)
- **1.13.0** - `/promote` skill for Tier 2 content management (browse/preview/promote/auto-promote), auto-extract configuration in setup wizard with progressive disclosure (3 presets + custom), system prompt updates for discoverability
- **1.12.0** - Stop hook redesign: PostToolUse → Stop hook for conversation-turn-level observation, transcript JSONL parsing, substance/cooldown thresholds, drop inline mode, debug logging support
- **1.11.0** - Multi-mode background extraction: smart fallback (API → CLI), `background-api` (Anthropic SDK, needs API key), `background-cli` (Claude CLI via OAuth), refactored extraction into `call_haiku_api`/`call_haiku_cli`/`_parse_haiku_text` helpers, 30s timeout for CLI, mode-aware prerequisites health check, 35 new tests (577 total)
- **1.10.0** - Auto-Extract: passive observation capture from tool calls into Tier 2 memory, PostToolUse hook with 3 modes (disabled/background/inline), filtering module with anti-recursion skip lists and SHA-256 dedup, Haiku-based extraction for background mode, inline systemMessage for session model extraction, user-configurable skip list overrides, 53 new tests (542 total)
- **1.9.0** - Two-Tier SSoT architecture: Tier 2 (ChromaDB-first) ephemeral content, 5 new MCP tools (tier2_write/read/list/delete, promote), 7 content types (observation, pattern, summary, code, relationship, hint, plan), smart promotion based on importance/retrieval/age, tier-aware query results, 3 new namespaces (rel::, hint::, plan::), 54 new tests (30 total tools)
- **1.8.0** - Configurable paths: centralized path resolution via `tools/paths.py` replacing all hardcoded vault paths, 2 new MCP tools (jarvis_resolve_path, jarvis_list_paths), template variable substitution ({YYYY}/{MM}/{WW}), sensitive path detection, 45 new tests (25 total tools)
- **1.7.0** - Remove Serena dependency: replace all Serena MCP references across 14 files in 3 plugins with native jarvis_memory_* tools, strategic memories now file-backed at .jarvis/strategic/, read-modify-write pattern replaces serena_edit_memory (jarvis-strategic 1.1.0, jarvis-todoist 1.3.0)
- **1.6.0** - Memory CRUD tools: 4 new file-backed memory tools (jarvis_memory_write/read/list/delete), secret detection scanner, rename jarvis_memory_read→jarvis_doc_read and jarvis_memory_stats→jarvis_collection_stats with detailed mode, recency boost in query scoring (23 total tools)
- **1.5.0** - Unified collection & namespaces: ChromaDB `jarvis` collection with namespaced IDs (vault:: prefix), enriched metadata schema (universal type/namespace/timestamps + vault_type), tools/namespaces.py module
- **1.4.0** - Chroma-MCP consolidation: absorb 3 chroma-mcp tools into jarvis-tools (jarvis_query, jarvis_memory_read, jarvis_memory_stats), remove chroma-mcp dependency, rename MCP server tools→core
- **1.3.0** - ChromaDB semantic memory: /recall, /memory-index, /memory-stats skills, vault-wide indexing, explorer semantic pre-search, config migration to ~/.jarvis/
- **1.2.0** - Scheduling: SCHEDULED mode, schedule management skill, session-start checks, 6-option inbox routing, focus check
- **1.1.0** - Shell integration in setup wizard, jarvis.zsh/jarvis.bash snippets
- **1.0.0** - Modular architecture: split into jarvis, jarvis-todoist, jarvis-strategic plugins
- **0.3.0** - jarvis-explorer-agent (vault-aware search), test framework v1.0, capitalization fixes
- **0.2.1** - MCP rename (jarvis-tools→tools), Todoist workflow simplification, inbox processing enhancements
- **0.2.0** - Initial comprehensive test coverage, audit agent refinements

---

## Quick Commands Reference

**Preferred: Use `/reinstall` skill** - handles cache clear, marketplace update, uninstall, and reinstall automatically.

```bash
# Manual reinstall (if /reinstall unavailable)
rm -rf ~/.claude/plugins/cache/jarvis-plugins/jarvis/* && \
claude plugin marketplace update && \
claude plugin uninstall jarvis@jarvis-plugins && \
claude plugin install jarvis@jarvis-plugins

# Check installed version
cat ~/.claude/plugins/cache/jarvis-plugins/jarvis/*/plugin.json | grep version

# View agent configuration
cat ~/.claude/plugins/cache/jarvis-plugins/jarvis/*/agents/jarvis-journal-agent.md | head -10
```
