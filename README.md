# Jarvis - Personal AI Assistant for Claude Code and Codex

A dual-harness plugin marketplace that adds **Jarvis** to Claude Code and local Codex — a context-aware personal assistant with a knowledge vault, semantic memory, and strategic awareness.

Jarvis manages a folder of Markdown or Org-mode files (your "vault") as a personal knowledge base. It journals your thoughts, tracks your goals, searches by meaning, and learns from your conversations — all with a git-audited trail.

---

## Quick Install

### Option 1: One-Line Installer (Recommended)

```bash
curl -fsSL https://raw.githubusercontent.com/rsprudencio/jarvis/refs/heads/master/install.sh | bash
```

This validates prerequisites, installs the plugin, configures your vault, and sets up the `jarvis` shell command — all in one step.

### Option 2: Manual Install

**Prerequisites:**
- [Claude Code CLI](https://claude.ai/code) (latest version)
- [Docker](https://docs.docker.com/get-docker/) with Compose

**Install:**

```bash
# Add the marketplace and install plugins
claude plugin marketplace add rsprudencio/jarvis
claude plugin install jarvis@jarvis-plugins

# Optional: Todoist integration
claude plugin install jarvis-todoist@jarvis-plugins

# Optional: Strategic analysis (orient, catch-up, patterns, summaries)
claude plugin install jarvis-strategic@jarvis-plugins

# Start the MCP server container
docker pull ghcr.io/rsprudencio/jarvis:latest
docker compose -f ~/.jarvis/docker-compose.yml up -d
```

**First-time config:**

```
/jarvis-settings
```

This walks you through vault path selection, auto-extract mode, and shell integration.

See [docker/README.md](docker/README.md) for full Docker setup guide.

### Codex Install (Local CLI, IDE, or Desktop App)

Jarvis uses local MCP servers and lifecycle hooks, so Codex support applies to
local Codex surfaces running on the same host as Jarvis. Add the marketplace
and install the core plugin:

```bash
codex plugin marketplace add rsprudencio/jarvis
codex plugin add jarvis@jarvis-plugins
codex plugin add jarvis-obsidian@jarvis-plugins
```

Install Todoist or Strategic in the same way, then restart the Codex host,
open `/hooks` in Codex CLI, and trust the Jarvis hooks. Start a new thread after
installation so Codex loads the new skills, MCP servers, and hook definitions.

See [Codex support](docs/CODEX.md) for the complete install, upgrade, trust, and
validation workflow.

### Verify Installation

After installing, start Claude Code and run:

```
/jarvis-settings
```

If the MCP server loaded correctly, you'll see the configuration menu. If not, run `check-requirements` from the repo root to diagnose.

---

## Getting Started

Once installed, launch Jarvis:

```bash
jarvis              # If you set up the shell command during install
# or
claude              # Then type /jarvis:jarvis to activate mid-session
```

### First Things to Try

| Command | What it does |
|---------|--------------|
| `jarvis, journal this: I decided to use PostgreSQL for the new project` | Creates a journal entry with vault links |
| `/jarvis-recall database decisions` | Searches your vault by meaning |
| `/jarvis-orient` | Strategic briefing — what to focus on today |
| `/jarvis-todoist` | Process your Todoist inbox with smart routing |
| `/jarvis-close` | End-of-session review checklist |

### How Jarvis Works

1. **You talk to Claude as usual** — Jarvis runs in the background
2. **Auto-extract** captures valuable insights from your conversations into ephemeral memory
3. **Journal entries** create permanent, linked notes in your vault
4. **Semantic search** finds related content by meaning, not just keywords
5. **Strategic context** keeps you aligned with your goals across sessions

---

## What's in the Box

### 5 Composable Plugins

| Plugin | What | Install |
|--------|------|---------|
| **jarvis** (core) | Vault management, semantic memory, auto-extract, pattern detection | Required |
| **jarvis-obsidian** | PKM: git audit trail, journal, vault exploration | Required |
| **jarvis-todoist** | Smart task routing, inbox capture, Todoist sync | Optional (needs [API token](https://app.todoist.com/app/settings/integrations/developer)) |
| **jarvis-strategic** | Orient briefings, catch-up, summaries, pattern analysis | Optional |
| **jarvis-toolbelt** | Engineering: security review, adversarial review, TDD | Optional |

### 14 Skills (Slash Commands)

**Core:**
| Skill | Description |
|-------|-------------|
| `/jarvis:jarvis` | Activate Jarvis identity mid-session |
| `/jarvis-settings` | View and update configuration |
| `/jarvis-journal` | Create journal entries with intelligent vault linking |
| `/jarvis-inbox` | Process and organize vault inbox items |
| `/jarvis-recall <query>` | Semantic search across your vault |
| `/jarvis-close` | Session closure checklist |
| `/jarvis-memory-stats` | Memory system health and statistics |
| `/jarvis-schedule` | Manage recurring Jarvis actions |
| `/jarvis-audit` | Git audit protocol reference |

**Todoist** (requires jarvis-todoist plugin):
| Skill | Description |
|-------|-------------|
| `/jarvis-todoist` | Sync Todoist inbox with smart routing |
| `/jarvis-todoist-setup` | Configure Todoist routing rules |

**Strategic** (requires jarvis-strategic plugin):
| Skill | Description |
|-------|-------------|
| `/jarvis-orient` | Strategic briefing for starting a session |
| `/jarvis-catchup` | Catch-up after time away |
| `/jarvis-summarize` | Weekly/monthly journal summaries |
| `/jarvis-patterns` | Behavioral pattern analysis |

### 3 Agents (Autonomous Sub-Processes)

Jarvis delegates work to specialized agents that run in isolated context windows:

- **Journal Agent** (Haiku) — Drafts entries with vault linking and YAML frontmatter
- **Audit Agent** (Haiku) — JARVIS protocol git commits, history queries, rollbacks
- **Explorer Agent** (Haiku) — Vault-wide search with regex, date filtering, semantic pre-search

### MCP Tools

12-tool Python MCP server (jarvis-core) providing vault filesystem, semantic memory, content API, and path configuration. A second 9-tool MCP server (jarvis-obsidian) provides git audit trail, commit history, and file operations. See [plugins/jarvis/capabilities.json](plugins/jarvis/capabilities.json) for the full reference.

---

## Memory System

Jarvis has a unified semantic memory powered by PostgreSQL + pgvector:

**Vault Documents (`obsidian.documents`)**
Your vault files (`.md` or `.org`) — git-tracked, visible in Obsidian/Emacs, searchable via `/jarvis-recall`. Indexed with vector embeddings for semantic search.

**Auto-Generated Content (`local.memories`)**
Observations, patterns, decisions, and worklogs captured from your conversations. Stored with importance scoring, retrieval tracking, and time-based decay.

**Auto-Extract** runs passively after each conversation turn, using Haiku to identify insights worth remembering. Set to `background` (recommended) or `disabled` in `/jarvis-settings`.

---

## Platform Support

Jarvis works on **macOS, Linux, Windows, and WSL** with automatic platform detection and tailored error messages.

### Supported Platforms

| Platform | Status | Notes |
|----------|--------|-------|
| **macOS** 13+ | ✅ Fully supported | Homebrew, Command Line Tools, or python.org |
| **Linux** (Ubuntu, Debian, RedHat, CentOS) | ✅ Fully supported | apt, yum, or python.org |
| **Windows** 11 | ✅ Supported | Git Bash, PowerShell, or WSL2 |
| **WSL2** | ✅ Fully supported | Native Linux tooling inside Windows |

### Windows-Specific Notes

**Python**: Windows typically uses `python` instead of `python3`. Jarvis automatically detects both.

**Install options:**
- **Microsoft Store**: Search for "Python 3.11" or "Python 3.12" (easiest, adds to PATH automatically)
- **python.org**: Download installer, check "Add Python to PATH" during install
- **WSL2**: Recommended for advanced users, gives you full Linux environment

**Git**: Install [Git for Windows](https://git-scm.com/download/win) which includes Git Bash - a Unix-like terminal for Windows.

**Shell integration**: The `jarvis` executable is installed to `~/.local/bin/` and works in any shell (Bash, Zsh, Git Bash, etc.).

### PATH Enrichment

Jarvis automatically checks common install locations even if they're not in your `PATH`:

**Unix (macOS/Linux):**
- `~/.local/bin` — pip user installs

**Windows:**
- `%LOCALAPPDATA%\Programs\Python` — Microsoft Store Python
- `%PROGRAMFILES%\Git\cmd` — Git for Windows

If a tool is installed but not found, Jarvis provides platform-specific installation instructions.

---

## Configuration

All configuration lives in `~/.jarvis/config.json`. The installer writes a full config with all ~30 keys visible so you can discover and tweak options.

Run `/jarvis-settings` anytime to update configuration through a guided menu.

Key config sections:
- **vault_path** — Where your vault files live
- **file_format** — `"md"` (Markdown, default) or `"org"` (Org-mode) for new files
- **memory.auto_extract** — Observation capture mode and thresholds
- **promotion** — When ephemeral content gets promoted to vault files
- **paths** — Vault directory layout (all customizable)

---

## Troubleshooting

### Prerequisites Check Failed

If `install.sh` or `check-requirements` reports missing prerequisites:

**"Python not found"**
- **macOS**: `brew install python@3.12` or download from [python.org](https://python.org)
- **Linux**: `sudo apt install python3` (Debian/Ubuntu) or `sudo yum install python3` (RedHat/CentOS)
- **Windows**: Install from [Microsoft Store](https://www.microsoft.com/store/productId/9NCVDN91XZQP) or [python.org](https://python.org/downloads/)

**"Docker not found"**
- Install Docker Desktop from [docs.docker.com/get-docker/](https://docs.docker.com/get-docker/)
- Ensure Docker Compose is included (it is in Docker Desktop)
- After install, restart your terminal

**"git not found"**
- **macOS**: `xcode-select --install` or `brew install git`
- **Linux**: `sudo apt install git` (Debian/Ubuntu) or `sudo yum install git` (RedHat/CentOS)
- **Windows**: Download [Git for Windows](https://git-scm.com/download/win)

**"Python version too old"**
- Jarvis requires Python 3.10 or later
- Check version: `python3 --version` (or `python --version` on Windows)
- Upgrade using your platform's package manager or python.org installer

### Plugin Not Loading

If `/jarvis:jarvis` or `/jarvis-settings` doesn't work:

1. **Verify plugin installed**: `claude plugin list` — should show `jarvis@jarvis-plugins`
2. **Check MCP server**: Look for "Jarvis MCP server started" in Claude Code logs
3. **Run diagnostics**: `./check-requirements` from repo root
4. **Reinstall**: Use the `/reinstall` skill or manually:
   ```bash
   rm -rf ~/.claude/plugins/cache/jarvis-plugins/jarvis/*
   claude plugin uninstall jarvis@jarvis-plugins
   claude plugin install jarvis@jarvis-plugins
   ```
5. **Full restart**: Quit and reopen Claude Code (not just reload)

### Auto-Extract Not Working

If observations aren't being captured:

1. **Check mode**: Run `/jarvis-settings` → Auto-Extract Configuration. Mode should be `background` (requires Anthropic API key; auto-falls back to Claude CLI).
2. **Check logs**: Look for "Auto-extract" in Claude Code debug logs
3. **Verify hooks**: `ls ~/.claude/plugins/cache/jarvis-plugins/jarvis/*/hooks/` should show `Stop.md`

### Memory/Semantic Search Issues

If `/recall` returns no results or indexing fails:

1. **Check container**: `docker compose -f ~/.jarvis/docker-compose.yml ps` — should show healthy
2. **Verify PG connection**: `curl --max-time 2 localhost:8741/health` — should show `"postgres": {"status": "ok"}`. `recovering`, `unreachable` or `disk_full` means the database is down (often a full Docker VM disk: `docker system df`); see [docker/README.md → Database outage](docker/README.md#database-outage-disk-full--recovery-mode). Memory tools and hooks fail fast (503) until it recovers; queued auto-extract payloads replay afterwards.
3. **Rebuild index**: Run `/jarvis-settings` → "Re-index vault"
4. **Check stats**: Run `/jarvis-memory-stats` to see document count
5. **Back up**: `plugins/jarvis/scripts/jarvis-transport.sh backup` dumps the embedded database to `~/.jarvis/backups` (newest 7 kept)

### Windows-Specific Issues

**"Command not found" in Git Bash**
- Tools installed via Microsoft Store may not be in Git Bash's PATH
- Either use PowerShell, or add to Git Bash PATH: `export PATH="/c/Users/YourName/AppData/Local/Microsoft/WindowsApps:$PATH"`

**WSL vs Native Windows**
- If using WSL: Install Linux versions of tools inside WSL (`sudo apt install python3 git`)
- If using native Windows: Install Windows versions of tools

**Permission errors**
- Git Bash may need "Run as Administrator" for some operations
- WSL requires no special permissions

### Still Having Issues?

- **Check logs**: Claude Code → View → Toggle Developer Tools → Console
- **Verbose diagnostics**: `./check-requirements --verbose`
- **Report bug**: [GitHub Issues](https://github.com/rsprudencio/jarvis/issues)

---

## Development

### Repository Structure

```
.
├── install.sh                    # Curl-pipe-bash installer
├── check-requirements            # Standalone prereq checker
├── plugins/
│   ├── jarvis/                   # Core plugin
│   │   ├── agents/               # Journal, audit, explorer agents
│   │   ├── skills/               # 9 core skills
│   │   ├── mcp-server/           # Python MCP server (12 tools)
│   │   ├── hooks/                # Auto-extract Stop hook
│   │   └── statusline/           # Claude Code statusline
│   ├── jarvis-obsidian/          # PKM: git audit, journal, exploration
│   ├── jarvis-todoist/           # Todoist extension
│   ├── jarvis-strategic/         # Strategic analysis extension
│   └── jarvis-toolbelt/          # Engineering: security, adversarial review
├── CLAUDE.md                     # Development guide
└── LICENSE                       # CC BY-NC 4.0
```

### Running Tests

```bash
make test        # unit tests (no DB required)
make test-e2e    # e2e tests against real PostgreSQL
make test-all    # both
```

Unit tests covering config, file ops, git operations, memory, protocol, and server registration.

### Documentation

- **[CLAUDE.md](CLAUDE.md)** — Development conventions and version history
- **[plugins/jarvis/capabilities.json](plugins/jarvis/capabilities.json)** — Full capability reference (AI-first)

---

## License

**CC BY-NC 4.0** (Creative Commons Attribution-NonCommercial 4.0 International)

See [LICENSE](LICENSE) for full legal text.

---

**v3.1.1** | [Issues](https://github.com/rsprudencio/jarvis/issues) | [Changelog](CLAUDE.md#version-history)
