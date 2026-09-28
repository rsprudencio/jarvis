#!/bin/bash
# Jarvis Plugin Installer for Claude Code and Codex
# curl -fsSL https://raw.githubusercontent.com/rsprudencio/jarvis/refs/heads/master/install.sh | bash
# Codex-only: curl .../install.sh | JARVIS_HARNESS=codex bash
set -e

JARVIS_HOME="${JARVIS_HOME:-$HOME/.jarvis}"
JARVIS_HARNESS="${JARVIS_HARNESS:-auto}"
MARKETPLACE_NAME="jarvis-plugins"
MARKETPLACE_REPO="https://github.com/rsprudencio/jarvis"

# ── Interactive input setup ──
# When piped (curl | bash), stdin is the script — we need /dev/tty for prompts
HAS_TTY=false
if (exec </dev/tty) 2>/dev/null; then
    exec 3</dev/tty
    HAS_TTY=true
else
    exec 3</dev/null
fi

# Read from terminal (fd 3), fall back to default if unavailable
ask() {
    local prompt="$1" varname="$2" default="$3"
    if [ "$HAS_TTY" = true ]; then
        read -r -p "$prompt" "$varname" <&3 || true
    fi
    eval "local val=\$$varname"
    if [ -z "$val" ]; then
        eval "$varname='$default'"
    fi
}

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

# One bounded health probe: a server that accepts TCP but never answers (a
# wedged event loop) costs at most ~2s instead of hanging the installer.
health_ok() {
    curl -sf --connect-timeout 1 --max-time 2 "$1" > /dev/null 2>&1
}

# wait_for_health <max-seconds> <url>...: true once every URL answers.
wait_for_health() {
    local deadline=$((SECONDS + $1)) url all_ok
    shift
    while [ "$SECONDS" -lt "$deadline" ]; do
        all_ok=true
        for url in "$@"; do
            health_ok "$url" || { all_ok=false; break; }
        done
        [ "$all_ok" = true ] && return 0
        sleep 1
    done
    return 1
}

# Prints "true" when config.json enables server.auth (same truthiness as
# jarvis_common.auth.get_auth_config), else "false".
read_auth_enabled() {
    local config="$1"
    if [ ! -f "$config" ]; then
        echo "false"
        return 0
    fi
    python3 -c "
import json, sys
with open(sys.argv[1]) as f:
    cfg = json.load(f)
server = cfg.get('server') if isinstance(cfg, dict) else None
auth = server.get('auth') if isinstance(server, dict) else None
print('true' if isinstance(auth, dict) and auth.get('enabled', False) else 'false')
" "$config" 2>/dev/null || echo "false"
}

# generate_compose_file <image> <vault-path> <jarvis-home> <auth-enabled>
# MCP ports publish on loopback only unless auth is enabled; the explorer is
# always loopback-only.
generate_compose_file() {
    local image="$1" vault_path="$2" jarvis_home="$3" auth_enabled="$4"
    local bind="127.0.0.1:"
    local ports_note="Loopback-only: enable server.auth in config.json and re-run the installer to expose on the network"
    if [ "$auth_enabled" = "true" ]; then
        bind=""
        ports_note="All interfaces: server.auth.enabled=true in config.json"
    fi
    cat << COMPOSEEOF
# Jarvis — single-container deployment with embedded PostgreSQL
services:
  jarvis:
    image: $image
    ports:
      # $ports_note
      - "${bind}8741:8741"
      - "${bind}8742:8742"
      - "${bind}8744:8744"
      - "127.0.0.1:\${JARVIS_EXPLORER_PORT:-8750}:8750"
    volumes:
      - "$vault_path:/vault"
      - "$jarvis_home:/config"
      - pgdata:/var/lib/postgresql/data
    extra_hosts:
      # No-op on Docker Desktop (macOS); required for host.docker.internal
      # to resolve on Linux engines.
      - "host.docker.internal:host-gateway"
    environment:
      - JARVIS_HOME=/config
      - JARVIS_VAULT_PATH=/vault
      - TODOIST_API_TOKEN=\${TODOIST_API_TOKEN:-}
      # Optional: lets you generate contextual document summaries in-container
      #   docker exec -w /app/jarvis-core <container> python bin/generate_summaries.py
      # No runtime path calls an LLM; leave it empty to generate on the host with
      # the OAuth 'claude' CLI instead.
      - ANTHROPIC_API_KEY=\${ANTHROPIC_API_KEY:-}
      - AURORA_PASSWORD=\${AURORA_PASSWORD:-}
    stop_grace_period: 30s
    restart: unless-stopped
    # Bounded, rotated container logs (json-file grows without limit on the
    # same VM disk as the database volume).
    logging:
      driver: local
      options:
        max-size: "10m"
        max-file: "5"
    healthcheck:
      test: ["CMD", "curl", "-sf", "http://localhost:8741/health"]
      interval: 30s
      timeout: 5s
      start_period: 30s
      retries: 3

volumes:
  pgdata:
COMPOSEEOF
}

# Prints the path of a previously installed Jarvis launcher, if any.
find_installed_launcher() {
    local candidate
    for candidate in "$(command -v jarvis 2>/dev/null || true)" "$HOME/.local/bin/jarvis" "/usr/local/bin/jarvis"; do
        case "$candidate" in /*) ;; *) continue ;; esac
        if [ -f "$candidate" ] && grep -q "Jarvis AI Assistant launcher" "$candidate" 2>/dev/null; then
            echo "$candidate"
            return 0
        fi
    done
    return 1
}

# install_managed_file <src> <dst> <label>: copy a shipped file (launcher,
# statusline) over its installed copy on EVERY install when the content
# differs, so fixes reach existing users. The previous copy is backed up to
# $JARVIS_HOME/backups/installed/ (dir 700, file 600); the swap is an atomic
# rename so a running copy is never read half-written. Symlinks (dev setups)
# are left alone.
install_managed_file() {
    local src="$1" dst="$2" label="$3"
    local backup_dir="$JARVIS_HOME/backups/installed"
    local backup tmp

    if [ -L "$dst" ]; then
        info "$label is a symlink, leaving it as-is: $dst"
        return 0
    fi
    if [ -f "$dst" ] && cmp -s "$src" "$dst"; then
        ok "$label up to date: $dst"
        return 0
    fi
    if [ -f "$dst" ]; then
        mkdir -p "$backup_dir" || return 1
        chmod 700 "$JARVIS_HOME/backups" "$backup_dir" || return 1
        backup="$backup_dir/$(basename "$dst").$(date +%Y%m%d-%H%M%S).bak"
        cp "$dst" "$backup" || return 1
        chmod 600 "$backup" || return 1
        info "Previous $label backed up: $backup"
    fi
    tmp="$dst.tmp.$$"
    if ! cp "$src" "$tmp" || ! chmod 755 "$tmp" || ! mv -f "$tmp" "$dst"; then
        rm -f "$tmp"
        return 1
    fi
    ok "Installed $label: $dst"
}

# Tests source this file for its helpers only (tests/test_deploy_scripts.py).
if [ "${JARVIS_INSTALL_LIB_ONLY:-}" = "1" ]; then
    # shellcheck disable=SC2317  # exit is reached when executed, not sourced
    return 0 2>/dev/null || exit 0
fi

if [ "$HAS_TTY" = false ]; then
    warn "No terminal detected — using all defaults (non-interactive mode)"
    echo ""
fi

echo -e "${CYAN}"
echo "  ╦╔═╗╦═╗╦  ╦╦╔═╗"
echo "  ║╠═╣╠╦╝╚╗╔╝║╚═╗"
echo " ╚╝╩ ╩╩╚═ ╚╝ ╩╚═╝"
echo -e "${NC}"
echo -e "  ${BOLD}AI Assistant Plugin Installer${NC}"
echo ""

# Keep the existing Claude-first behavior when both harnesses are installed.
if [ "$JARVIS_HARNESS" = "auto" ]; then
    if command -v claude >/dev/null 2>&1; then
        JARVIS_HARNESS="claude"
    elif command -v codex >/dev/null 2>&1; then
        JARVIS_HARNESS="codex"
    fi
fi

case "$JARVIS_HARNESS" in
    claude|codex) ;;
    *)
        fail "Neither Claude Code nor Codex CLI was found"
        echo ""
        echo -e "  Install one harness, or set ${BLUE}JARVIS_HARNESS=claude|codex${NC}."
        echo ""
        exit 1
        ;;
esac

if [ "$JARVIS_HARNESS" = "codex" ]; then
    PLUGIN_INSTALL_VERB="add"
else
    PLUGIN_INSTALL_VERB="install"
fi

AUTO_EXTRACT_MODE_OVERRIDE=""
if [ "$JARVIS_HARNESS" = "codex" ] \
    && [ -z "${ANTHROPIC_API_KEY:-}" ] \
    && ! command -v claude >/dev/null 2>&1; then
    AUTO_EXTRACT_MODE_OVERRIDE="disabled"
    warn "Passive auto-extraction will be disabled (no Anthropic key or Claude CLI)"
fi

plugin_cli() {
    "$JARVIS_HARNESS" plugin "$@"
}

plugin_install() {
    plugin_cli "$PLUGIN_INSTALL_VERB" "$1"
}

ok "Harness: $JARVIS_HARNESS"
echo ""

# ═══════════════════════════════════════════════
# 📦 Install Core Plugin
# ═══════════════════════════════════════════════

DOCKER_IMAGE="ghcr.io/rsprudencio/jarvis:latest"

# Verify the selected harness exists (silent check)
if ! command -v "$JARVIS_HARNESS" >/dev/null 2>&1; then
    fail "$JARVIS_HARNESS CLI not found"
    echo ""
    echo -e "  Set ${BLUE}JARVIS_HARNESS=claude|codex${NC} to select an installed harness."
    echo ""
    exit 1
fi

# Verify Python 3 exists (silent check)
if command -v python3 >/dev/null 2>&1; then
    PYTHON_CMD="python3"
else
    fail "Python 3 not found"
    echo ""
    echo -e "  ${BOLD}macOS:${NC} ${BLUE}brew install python@3.12${NC}"
    echo -e "  ${BOLD}Linux:${NC} ${BLUE}sudo apt install python3${NC} (Debian/Ubuntu)"
    echo -e "  Or download from: ${BLUE}https://python.org/downloads/${NC}"
    echo ""
    exit 1
fi

echo -e "${BOLD}📦 Install Core Plugin${NC}"
echo ""

# Add marketplace (may fail if already added — that's OK)
plugin_cli marketplace add rsprudencio/jarvis >/dev/null 2>&1 || true

# Verify marketplace is available
if ! plugin_cli marketplace list 2>/dev/null | grep -q "jarvis-plugins"; then
    fail "Could not add marketplace"
    echo -e "  Run manually: ${BLUE}$JARVIS_HARNESS plugin marketplace add rsprudencio/jarvis${NC}"
    exit 1
fi

# Codex snapshots Git marketplaces locally. Refresh an existing snapshot before
# installation; a fresh add is already current, so an upgrade failure is only a warning.
if [ "$JARVIS_HARNESS" = "codex" ]; then
    plugin_cli marketplace upgrade "$MARKETPLACE_NAME" >/dev/null 2>&1 \
        || warn "Could not refresh the Codex marketplace snapshot"
fi

# Install core plugin
echo -e "  Installing ${BLUE}jarvis@jarvis-plugins${NC}..."
plugin_install jarvis@jarvis-plugins >/dev/null 2>&1 || {
    fail "Could not install jarvis plugin"
    echo ""
    echo -e "  Try manually: ${BLUE}$JARVIS_HARNESS plugin $PLUGIN_INSTALL_VERB jarvis@jarvis-plugins${NC}"
    exit 1
}

echo -e "  Installing ${BLUE}jarvis-obsidian@jarvis-plugins${NC}..."
plugin_install jarvis-obsidian@jarvis-plugins >/dev/null 2>&1 || {
    fail "Could not install the required Jarvis vault plugin"
    exit 1
}

ok "Core and vault plugins installed"
echo ""

# ═══════════════════════════════════════════════
# ✅ Check Prerequisites
# ═══════════════════════════════════════════════

# ═══════════════════════════════════════════════
# ✅ Check Prerequisites
# ═══════════════════════════════════════════════

echo -e "${BOLD}✅ Check Prerequisites${NC}"
echo ""

# Docker is required
HAS_DOCKER=false
if command -v docker >/dev/null 2>&1; then
    if docker compose version >/dev/null 2>&1; then
        HAS_DOCKER=true
    fi
fi

if [ "$HAS_DOCKER" = false ]; then
    fail "Docker with Compose not found"
    echo ""
    echo -e "  Jarvis requires Docker to run the MCP server."
    echo -e "  Install: ${BLUE}https://docs.docker.com/get-docker/${NC}"
    echo ""
    exit 1
fi
ok "Docker with Compose found"
echo ""

# ═══════════════════════════════════════════════
# 🧩 Install Optional Extensions
# ═══════════════════════════════════════════════

echo -e "${BOLD}🧩 Install Optional Extensions${NC}"
echo ""
echo -e "  ${BOLD}Available extensions:${NC}"
echo -e "    ${CYAN}[1]${NC} jarvis-todoist   — Task management via Todoist"
echo -e "    ${CYAN}[2]${NC} jarvis-strategic — Strategic analysis & briefings"
if [ "$JARVIS_HARNESS" = "claude" ]; then
    echo -e "    ${CYAN}[3]${NC} jarvis-toolbelt  — Adversarial & security review agents"
    echo -e "    ${CYAN}[4]${NC} All"
    echo -e "    ${CYAN}[5]${NC} Skip"
    DEFAULT_EXT_CHOICE="5"
else
    echo -e "    ${CYAN}[3]${NC} All supported Codex extensions"
    echo -e "    ${CYAN}[4]${NC} Skip"
    DEFAULT_EXT_CHOICE="4"
fi
echo ""
ask "  Choice [$DEFAULT_EXT_CHOICE]: " EXT_CHOICE "$DEFAULT_EXT_CHOICE"

install_ext() {
    plugin_install "$1@$MARKETPLACE_NAME" 2>/dev/null && ok "$1 installed" || warn "$1 install failed"
}

if [ "$JARVIS_HARNESS" = "claude" ]; then
    case "$EXT_CHOICE" in
        1) install_ext jarvis-todoist ;;
        2) install_ext jarvis-strategic ;;
        3) install_ext jarvis-toolbelt ;;
        4)
            install_ext jarvis-todoist
            install_ext jarvis-strategic
            install_ext jarvis-toolbelt
            ;;
        *) info "Skipping extensions" ;;
    esac
else
    case "$EXT_CHOICE" in
        1) install_ext jarvis-todoist ;;
        2) install_ext jarvis-strategic ;;
        3)
            install_ext jarvis-todoist
            install_ext jarvis-strategic
            ;;
        *) info "Skipping extensions" ;;
    esac
fi

echo ""

# Resolve the installed core directory from the selected harness (post-extensions).
PLUGIN_DIR=$(plugin_cli list --json 2>/dev/null | $PYTHON_CMD -c "
import sys, json
data = json.load(sys.stdin)
plugins = data if isinstance(data, list) else data.get('installed', [])
for p in plugins:
    plugin_id = p.get('id', p.get('pluginId', ''))
    if plugin_id.startswith('jarvis@'):
        path = p.get('installPath') or p.get('source', {}).get('path')
        if path:
            print(path)
        break
" 2>/dev/null)

if [ -z "$PLUGIN_DIR" ] || [ ! -d "$PLUGIN_DIR" ]; then
    fail "Plugin directory not found"
    echo "    Run: $JARVIS_HARNESS plugin list --json"
    echo "    Try reinstalling: $JARVIS_HARNESS plugin $PLUGIN_INSTALL_VERB jarvis@$MARKETPLACE_NAME"
    exit 1
fi


# ═══════════════════════════════════════════════
# ⚙️  Configure Jarvis
# ═══════════════════════════════════════════════

echo -e "${BOLD}⚙️  Configure Jarvis${NC}"
echo ""

# Check for existing config
if [ -f "$JARVIS_HOME/config.json" ]; then
    EXISTING_VAULT=$(python3 -c "import json; print(json.load(open('$JARVIS_HOME/config.json')).get('vault_path', 'not set'))" 2>/dev/null || echo "not set")
    echo -e "  Existing config found (vault: ${CYAN}$EXISTING_VAULT${NC})"
    ask "  Reconfigure? [y/N]: " RECONFIG "N"
    if [ "$RECONFIG" != "y" ] && [ "$RECONFIG" != "Y" ]; then
        info "Keeping existing config\n"
        VAULT_PATH="$EXISTING_VAULT"
        SKIP_CONFIG=true
    fi
fi

if [ "$SKIP_CONFIG" != true ]; then
    echo "  Where should Jarvis store your knowledge vault?"
    echo -e "  ${BOLD}(A vault is a folder of markdown files — like an Obsidian vault)${NC}"
    echo ""

    # Detect common vault locations
    VAULT_OPTIONS=()
    VAULT_LABELS=()

    # Always offer starter vault
    VAULT_OPTIONS+=("$HOME/.jarvis/vault")
    VAULT_LABELS+=("~/.jarvis/vault/ (starter vault — good for trying Jarvis)")

    # Detect Obsidian vaults
    for candidate in "$HOME/Documents/Obsidian" "$HOME/Documents/obsidian" "$HOME/Obsidian" "$HOME/obsidian" "$HOME/vaults" "$HOME/notes"; do
        if [ -d "$candidate" ]; then
            VAULT_OPTIONS+=("$candidate")
            VAULT_LABELS+=("$candidate/ (detected)")
        fi
    done

    # Display options
    for i in "${!VAULT_LABELS[@]}"; do
        echo -e "    ${CYAN}[$((i+1))]${NC} ${VAULT_LABELS[$i]}"
    done
    CUSTOM_IDX=$((${#VAULT_OPTIONS[@]} + 1))
    echo -e "    ${CYAN}[$CUSTOM_IDX]${NC} Enter custom path"
    echo ""

    ask "  Choice [1]: " VAULT_CHOICE "1"

    # Check if VAULT_CHOICE is a valid number
    if [[ "$VAULT_CHOICE" =~ ^[0-9]+$ ]]; then
        # It's a number - use menu logic
        if [ "$VAULT_CHOICE" -eq "$CUSTOM_IDX" ]; then
            ask "  Enter vault path: " VAULT_PATH "$HOME/.jarvis/vault"
            # Expand ~ if present
            VAULT_PATH="${VAULT_PATH/#\~/$HOME}"
        elif [ "$VAULT_CHOICE" -ge 1 ] && [ "$VAULT_CHOICE" -le "${#VAULT_OPTIONS[@]}" ]; then
            VAULT_PATH="${VAULT_OPTIONS[$((VAULT_CHOICE-1))]}"
        else
            # Invalid number - use default
            VAULT_PATH="${VAULT_OPTIONS[0]}"
        fi
    else
        # User pasted a path directly - treat as custom input
        VAULT_PATH="$VAULT_CHOICE"
        # Expand ~ if present
        VAULT_PATH="${VAULT_PATH/#\~/$HOME}"
    fi

    ok "Vault: $VAULT_PATH"
    echo ""

    # File format selection
    echo -e "  ${BOLD}File Format${NC}"
    echo "  Choose the format for new vault files (existing files are always readable)."
    echo ""
    echo -e "    ${CYAN}[1]${NC} Markdown (.md) — Standard, works with Obsidian (recommended)"
    echo -e "    ${CYAN}[2]${NC} Org-mode (.org) — For Emacs/Org-mode users"
    echo ""
    ask "  Choice [1]: " FORMAT_CHOICE "1"

    case "$FORMAT_CHOICE" in
        2)
            FILE_FORMAT="org"
            ok "Format: Org-mode (.org)"
            ;;
        *)
            FILE_FORMAT="md"
            ok "Format: Markdown (.md)"
            ;;
    esac
    echo ""
fi

# The shipped launcher and statusline are Claude-specific. Codex loads Jarvis
# through its plugin manifest, so installing either would be misleading.
EXISTING_LAUNCHER=""
if [ "$JARVIS_HARNESS" = "claude" ]; then
    EXISTING_LAUNCHER=$(find_installed_launcher || true)
fi
if [ -n "$EXISTING_LAUNCHER" ]; then
    # Already opted in on an earlier install: refresh it rather than re-ask,
    # so launcher fixes reach existing installs.
    echo -e "  ${BOLD}Shell Integration${NC}"
    info "'jarvis' command already installed ($EXISTING_LAUNCHER); it will be refreshed if it changed"
    SHELL_SETUP="Y"
elif [ "$JARVIS_HARNESS" = "claude" ]; then
    echo -e "  ${BOLD}Shell Integration${NC}"
    echo "  The 'jarvis' command launches Claude with your Jarvis identity."
    echo -e "  ${YELLOW}⚠️  Highly recommended — this is the only way to make Claude fully impersonate Jarvis.${NC}"
    echo ""
    ask "  Install 'jarvis' command to your PATH? [Y/n]: " SHELL_SETUP "Y"
else
    SHELL_SETUP="N"
    info "Codex loads Jarvis directly; skipping the Claude launcher"
fi

echo ""

# ═══════════════════════════════════════════════
# 💾 Write Configuration
# ═══════════════════════════════════════════════

mkdir -p "$JARVIS_HOME"

if [ "$SKIP_CONFIG" != true ]; then
    echo -e "${BOLD}💾 Write Configuration${NC}"
    echo ""
    # Create vault directory
    mkdir -p "$VAULT_PATH"
    ok "Vault directory ready: $VAULT_PATH"

    # Write config from shipped template (SSoT) with user values substituted
    TIMESTAMP=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
    TEMPLATE="$PLUGIN_DIR/defaults/config.json"
    if [ -f "$TEMPLATE" ]; then
        $PYTHON_CMD -c "
import json, sys
with open(sys.argv[1]) as f:
    cfg = json.load(f)
cfg['vault_path'] = sys.argv[2]
cfg['vault_confirmed'] = True
cfg['configured_at'] = sys.argv[3]
cfg['file_format'] = sys.argv[4]
if sys.argv[5]:
    cfg.setdefault('memory', {}).setdefault('auto_extract', {})['mode'] = sys.argv[5]
json.dump(cfg, sys.stdout, indent=2)
" "$TEMPLATE" "$VAULT_PATH" "$TIMESTAMP" "${FILE_FORMAT:-md}" "$AUTO_EXTRACT_MODE_OVERRIDE" > "$JARVIS_HOME/config.json"
    else
        # Fallback: minimal config if template not found in plugin distribution
        cat > "$JARVIS_HOME/config.json" << FALLBACKEOF
{
  "vault_path": "$VAULT_PATH",
  "vault_confirmed": true,
  "configured_at": "$TIMESTAMP"
}
FALLBACKEOF
    fi
    ok "Config written: $JARVIS_HOME/config.json (all defaults visible)"
fi

# Shell integration (independent of config — always runs if user said Y)
if [ "$SHELL_SETUP" = "Y" ] || [ "$SHELL_SETUP" = "y" ]; then
    # Clean up old shell function injection if present
    for rc in "$HOME/.zshrc" "$HOME/.bashrc" "$HOME/.bash_profile"; do
        if grep -q "# Jarvis AI Assistant START" "$rc" 2>/dev/null; then
            sed -i.bak '/# Jarvis AI Assistant START/,/# Jarvis AI Assistant END/d' "$rc"
            rm -f "$rc.bak"
            info "Removed old shell function from $rc"
        fi
    done

    # Detect best install directory (an existing launcher stays where it is,
    # otherwise the stale copy would keep shadowing the new one on PATH)
    INSTALL_DIR=""
    if [ -n "$EXISTING_LAUNCHER" ]; then
        INSTALL_DIR="$(dirname "$EXISTING_LAUNCHER")"
    elif [ -d "$HOME/.local/bin" ] && echo "$PATH" | grep -q "$HOME/.local/bin"; then
        INSTALL_DIR="$HOME/.local/bin"
    elif [ -d "/usr/local/bin" ] && [ -w "/usr/local/bin" ]; then
        INSTALL_DIR="/usr/local/bin"
    else
        INSTALL_DIR="$HOME/.local/bin"
        mkdir -p "$INSTALL_DIR"
    fi

    # Install (or refresh) the jarvis executable
    SHELL_SCRIPT="$PLUGIN_DIR/shell/jarvis.sh"
    if [ -f "$SHELL_SCRIPT" ]; then
        install_managed_file "$SHELL_SCRIPT" "$INSTALL_DIR/jarvis" "launcher" \
            || warn "Could not write $INSTALL_DIR/jarvis — copy manually: cp \"$SHELL_SCRIPT\" \"$INSTALL_DIR/jarvis\""
    else
        warn "jarvis.sh not found at $SHELL_SCRIPT"
    fi

    # Check if directory is in PATH
    if ! command -v jarvis >/dev/null 2>&1; then
        warn "$INSTALL_DIR is not in your PATH"
        info "Add to your shell config: export PATH=\"$INSTALL_DIR:\$PATH\""
    fi
else
    info "Skipping shell integration"
    if [ "$JARVIS_HARNESS" = "claude" ]; then
        echo -e "  You can still activate Jarvis inside any Claude session by typing: ${BLUE}/jarvis:jarvis${NC}"
    fi
fi

echo ""

# ═══════════════════════════════════════════════
# 🐳 Docker Setup (if Docker method selected)
# ═══════════════════════════════════════════════

echo -e "${BOLD}🐳 Docker Setup${NC}"
echo ""

# Optional Todoist API token
echo -e "  ${BOLD}Todoist Integration (optional)${NC}"
echo "  If you use Todoist, enter your API token for task management."
echo -e "  Get one at: ${BLUE}https://app.todoist.com/app/settings/integrations/developer${NC}"
echo ""
ask "  Todoist API token (press Enter to skip): " TODOIST_TOKEN ""

if [ -n "$TODOIST_TOKEN" ]; then
    ok "Todoist token saved"
else
    info "Skipping Todoist (can add later in config)"
fi
echo ""

# ═══════════════════════════════════════════════
# 🧠 Native Host Inference (required)
# ═══════════════════════════════════════════════
# The Jarvis image contains no model runtime or weights. Embedding and
# reranking are served natively on this machine by llama.cpp LaunchAgents
# (ports 8751/8752); the container reaches them via host.docker.internal.

echo -e "  ${BOLD}Native Host Inference${NC}"
if health_ok http://127.0.0.1:8751/health && health_ok http://127.0.0.1:8752/health; then
    ok "Host inference services already running (8751 embedding, 8752 reranker)"
else
    if [ "$(uname -s)" != "Darwin" ] || [ "$(uname -m)" != "arm64" ]; then
        fail "This release requires Apple Silicon macOS for native host inference"
        echo -e "  The Docker image runs no models itself; it depends on host services"
        echo -e "  (llama.cpp + Metal via launchd) that are only supported on arm64 macOS."
        echo -e "  Linux/Intel hosts are not supported by this release."
        echo ""
        exit 1
    fi
    if ! command -v llama-server >/dev/null 2>&1; then
        fail "llama-server not found"
        echo -e "  Install llama.cpp first: ${BLUE}brew install llama.cpp${NC}, then re-run this installer."
        echo ""
        exit 1
    fi
    ok "llama-server: $(command -v llama-server)"

    HOST_INF_DIR="$JARVIS_HOME/host-inference"
    # JARVIS_REF pins the tooling to a branch/tag/SHA matching a pinned image.
    HOST_INF_RAW="https://raw.githubusercontent.com/rsprudencio/jarvis/${JARVIS_REF:-master}/host-inference"
    mkdir -p "$HOST_INF_DIR"
    for HOST_INF_FILE in host_models.py launchd.py configure.py models.json; do
        if ! curl -fsSL --connect-timeout 10 --max-time 300 "$HOST_INF_RAW/$HOST_INF_FILE" -o "$HOST_INF_DIR/$HOST_INF_FILE"; then
            fail "Could not download host-inference tooling ($HOST_INF_FILE)"
            exit 1
        fi
    done
    ok "Host inference tooling: $HOST_INF_DIR"

    info "Fetching pinned model weights (first run downloads ~1 GB)..."
    if ! $PYTHON_CMD "$HOST_INF_DIR/host_models.py" fetch; then
        fail "Model download failed — see output above"
        exit 1
    fi

    info "Installing launchd services (embedding :8751, reranker :8752)..."
    if ! $PYTHON_CMD "$HOST_INF_DIR/launchd.py" install; then
        fail "launchd install failed — see output above"
        exit 1
    fi

    HOST_INF_OK=false
    if wait_for_health 60 http://127.0.0.1:8751/health http://127.0.0.1:8752/health; then
        HOST_INF_OK=true
    fi
    if [ "$HOST_INF_OK" = true ]; then
        ok "Host inference services healthy"
    else
        fail "Host inference services did not become healthy"
        echo -e "  Logs: ${CYAN}$JARVIS_HOME/logs/model-host${NC}"
        echo -e "  Status: ${BLUE}$PYTHON_CMD $HOST_INF_DIR/launchd.py status${NC}"
        exit 1
    fi
fi
echo ""

# Pull Docker image
info "Pulling Docker image: $DOCKER_IMAGE"
if docker pull "$DOCKER_IMAGE" 2>&1 | tail -2; then
    ok "Docker image pulled"
else
    warn "Could not pull image from GHCR"
    info "You may need to build locally: docker build -f docker/Dockerfile -t jarvis-local ."
    echo ""
    ask "  Continue without Docker image? [y/N]: " CONTINUE_NO_IMAGE "N"
    if [ "$CONTINUE_NO_IMAGE" != "y" ] && [ "$CONTINUE_NO_IMAGE" != "Y" ]; then
        fail "Docker image required. Build locally or check your network."
        exit 1
    fi
    DOCKER_IMAGE="jarvis-local"
fi
echo ""

# Write docker-compose.yml for user
COMPOSE_FILE="$JARVIS_HOME/docker-compose.yml"
AUTH_ENABLED=$(read_auth_enabled "$JARVIS_HOME/config.json")
generate_compose_file "$DOCKER_IMAGE" "$VAULT_PATH" "$JARVIS_HOME" "$AUTH_ENABLED" > "$COMPOSE_FILE"
ok "Docker Compose file: $COMPOSE_FILE"
if [ "$AUTH_ENABLED" = "true" ]; then
    info "server.auth is enabled — MCP ports published on all interfaces"
else
    info "MCP ports bound to 127.0.0.1 (enable server.auth in config.json and re-run to expose them)"
fi

# Start the container
info "Starting Jarvis container..."
if docker compose -f "$COMPOSE_FILE" up -d 2>&1; then
    # Wait for health
    HEALTH_OK=false
    if wait_for_health 30 http://localhost:8741/health; then
        HEALTH_OK=true
    fi

    if [ "$HEALTH_OK" = true ]; then
        ok "Jarvis MCP server is running"
        HEALTH_RESP=$(curl -sf --connect-timeout 1 --max-time 2 http://localhost:8741/health 2>/dev/null || true)
        # Parse and display key health info
        PG_STATUS=$($PYTHON_CMD -c "
import json, sys
pg = json.loads(sys.argv[1]).get('postgres') or {}
status = pg.get('status', '?')
print(f'pg:{status}' + (f' ({pg[\"error\"]})' if status != 'ok' and pg.get('error') else ''))
" "$HEALTH_RESP" 2>/dev/null || echo "")
        if [ -n "$PG_STATUS" ]; then
            info "Health: $PG_STATUS"
        else
            info "$HEALTH_RESP"
        fi
    else
        warn "Container started but health check failed — check: docker compose -f $COMPOSE_FILE logs --tail 200"
    fi
else
    fail "Docker compose failed to start"
    echo "  Debug: docker compose -f $COMPOSE_FILE logs --tail 200"
fi
echo ""

# Write management helper script
HELPER_SCRIPT="$JARVIS_HOME/jarvis-docker.sh"
cat > "$HELPER_SCRIPT" << 'HELPEREOF'
#!/bin/bash
# Jarvis Docker management helper
COMPOSE_FILE="${JARVIS_HOME:-$HOME/.jarvis}/docker-compose.yml"

case "${1:-status}" in
    start)   docker compose -f "$COMPOSE_FILE" up -d ;;
    stop)    docker compose -f "$COMPOSE_FILE" down ;;
    restart) docker compose -f "$COMPOSE_FILE" restart ;;
    logs)    docker compose -f "$COMPOSE_FILE" logs -f --tail=50 ;;
    status)  docker compose -f "$COMPOSE_FILE" ps ;;
    update)
        docker compose -f "$COMPOSE_FILE" pull
        docker compose -f "$COMPOSE_FILE" up -d
        ;;
    *)
        echo "Usage: jarvis-docker.sh {start|stop|restart|logs|status|update}"
        exit 1
        ;;
esac
HELPEREOF
chmod +x "$HELPER_SCRIPT"
ok "Management helper: $HELPER_SCRIPT"

# Copy toolbelt scripts to ~/.jarvis/bin/ if toolbelt was installed
if [ "$JARVIS_HARNESS" = "claude" ] \
    && { [ "$EXT_CHOICE" = "3" ] || [ "$EXT_CHOICE" = "4" ]; }; then
    mkdir -p "$JARVIS_HOME/bin"
    # Claude caches <marketplace>/<plugin>/<version>.
    MARKETPLACE_DIR="$(dirname "$(dirname "$PLUGIN_DIR")")"
    DAR_SRC=$(find "$MARKETPLACE_DIR/jarvis-toolbelt" -path '*/bin/dar-review' -type f 2>/dev/null | head -1)
    if [ -n "$DAR_SRC" ] && [ -f "$DAR_SRC" ]; then
        cp "$DAR_SRC" "$JARVIS_HOME/bin/dar-review"
        chmod +x "$JARVIS_HOME/bin/dar-review"
        ok "DAR wrapper: $JARVIS_HOME/bin/dar-review"
    else
        warn "dar-review binary not found in toolbelt distribution"
    fi
fi
echo ""

# ═══════════════════════════════════════════════
# 📊 Statusline Setup (optional)
# ═══════════════════════════════════════════════

if [ "$JARVIS_HARNESS" = "codex" ]; then
    STATUSLINE_SETUP="N"
    info "Codex selected; skipping the Claude-specific statusline"
else
    echo -e "${BOLD}📊 Statusline${NC}"
    echo "  Jarvis includes a statusline showing model, MCP servers, cost, context, and server health."
    echo ""
    ask "  Install Jarvis statusline? [Y/n]: " STATUSLINE_SETUP "Y"
fi

if [ "$STATUSLINE_SETUP" = "Y" ] || [ "$STATUSLINE_SETUP" = "y" ]; then
    SL_SRC="$PLUGIN_DIR/statusline/statusline.py"
    SL_DST="$JARVIS_HOME/statusline.py"
    if [ -f "$SL_SRC" ]; then
        install_managed_file "$SL_SRC" "$SL_DST" "statusline" \
            || warn "Could not write $SL_DST — copy manually: cp \"$SL_SRC\" \"$SL_DST\""

        # Detect Claude config dir and merge statusLine into settings.json
        CLAUDE_CFG_DIR="${CLAUDE_CONFIG_DIR:-$HOME/.claude}"
        SETTINGS_FILE="$CLAUDE_CFG_DIR/settings.json"

        if [ -f "$SETTINGS_FILE" ]; then
            $PYTHON_CMD -c "
import json, sys
with open(sys.argv[1]) as f:
    settings = json.load(f)
settings['statusLine'] = {'type': 'command', 'command': sys.argv[2]}
with open(sys.argv[1], 'w') as f:
    json.dump(settings, f, indent=2)
    f.write('\n')
" "$SETTINGS_FILE" "$SL_DST"
            ok "settings.json updated with statusLine"
        else
            mkdir -p "$CLAUDE_CFG_DIR"
            echo "{\"statusLine\": {\"type\": \"command\", \"command\": \"$SL_DST\"}}" | $PYTHON_CMD -m json.tool > "$SETTINGS_FILE"
            ok "Created $SETTINGS_FILE with statusLine"
        fi
    else
        warn "statusline.py not found in plugin distribution"
    fi
else
    info "Skipping statusline"
fi
echo ""

# ═══════════════════════════════════════════════
# Complete!
# ═══════════════════════════════════════════════

echo -e "${BOLD}═══════════════════════════════════════════════${NC}"
echo -e "${BOLD}  Installation Complete!${NC}"
echo -e "${BOLD}═══════════════════════════════════════════════${NC}"
echo ""
echo -e "  Vault:       ${CYAN}$VAULT_PATH${NC}"
echo -e "  Compose:     ${CYAN}$JARVIS_HOME/docker-compose.yml${NC}"
echo -e "  MCP Core:    ${CYAN}http://localhost:8741/mcp${NC}"
echo -e "  MCP Todoist: ${CYAN}http://localhost:8742/mcp${NC}"
echo -e "  Admin:       ${CYAN}http://localhost:8750${NC}"

if [ "$SHELL_SETUP" = "Y" ] || [ "$SHELL_SETUP" = "y" ]; then
    echo -e "  Shell:       ${CYAN}jarvis${NC} installed to ${INSTALL_DIR:-PATH}"
fi
if [ "$STATUSLINE_SETUP" = "Y" ] || [ "$STATUSLINE_SETUP" = "y" ]; then
    echo -e "  Statusline:  ${CYAN}$JARVIS_HOME/statusline.py${NC}"
fi

echo -e "  Config:      ${CYAN}$JARVIS_HOME/config.json${NC}"
echo ""
echo -e "  ${BOLD}Quick Start:${NC}"
if [ "$JARVIS_HARNESS" = "codex" ]; then
    echo -e "    ${BLUE}\$ codex${NC}                      — Launch Codex with Jarvis loaded"
    echo -e "    ${BLUE}/hooks${NC}                        — Review and trust Jarvis hooks"
    echo -e "    Ask Codex to find relevant memories for your task."
else
    echo -e "    ${BLUE}\$ jarvis${NC}                     — Launch Jarvis"
    echo -e "    ${BLUE}\$ jarvis \"/jarvis-recall AI tools\"${NC}  — Search your vault"
    echo -e "    ${BLUE}/jarvis-settings${NC}              — Update configuration"
fi

echo ""
echo -e "  ${BOLD}Docker Management:${NC}"
echo -e "    ${BLUE}\$ $JARVIS_HOME/jarvis-docker.sh status${NC}   — Check container"
echo -e "    ${BLUE}\$ $JARVIS_HOME/jarvis-docker.sh logs${NC}     — View logs"
echo -e "    ${BLUE}\$ $JARVIS_HOME/jarvis-docker.sh restart${NC}  — Restart"
echo -e "    ${BLUE}\$ $JARVIS_HOME/jarvis-docker.sh update${NC}   — Pull & restart"

echo ""

if [ "$SHELL_SETUP" = "Y" ] || [ "$SHELL_SETUP" = "y" ]; then
    if ! command -v jarvis >/dev/null 2>&1; then
        echo -e "  ${YELLOW}Add to your shell config to use 'jarvis':${NC}"
        echo -e "    ${BLUE}export PATH=\"${INSTALL_DIR:-\$HOME/.local/bin}:\$PATH\"${NC}"
        echo ""
    fi
fi

if [ "$JARVIS_HARNESS" = "codex" ]; then
    echo -e "  ${BOLD}Restart Codex, trust the hooks in /hooks, then start a new thread.${NC}"
else
    echo -e "  ${BOLD}First time? Just run: jarvis${NC}"
fi
echo ""
