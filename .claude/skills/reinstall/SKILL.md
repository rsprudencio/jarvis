---
description: Reinstall Jarvis plugins with optional cache clear
---

# Jarvis Plugin Reinstall Workflow

You are helping the developer reinstall Jarvis plugins during active development.

## Modular Architecture

The plugin is split into 3 independent plugins in the `jarvis-plugins` marketplace:

| Plugin | Cache Path (relative to config dir) |
|--------|--------------------------------------|
| `jarvis` | `plugins/cache/jarvis-plugins/jarvis/` |
| `jarvis-todoist` | `plugins/cache/jarvis-plugins/jarvis-todoist/` |
| `jarvis-strategic` | `plugins/cache/jarvis-plugins/jarvis-strategic/` |

## Workflow

### Step 0: Detect Claude Config Directory

**CRITICAL**: Never assume `~/.claude/` - the config directory location varies by installation.

Run this detection script:

```bash
# Method 1: Check CLAUDE_CONFIG_DIR environment variable
if [ -n "$CLAUDE_CONFIG_DIR" ]; then
  DETECTED_DIR="$CLAUDE_CONFIG_DIR"
  echo "Found CLAUDE_CONFIG_DIR: $DETECTED_DIR"
# Method 2: Check current session's auto-memory location
elif [ -d "$(pwd)/.claude/projects" ]; then
  DETECTED_DIR="$(cd $(pwd)/.claude && pwd | sed 's|/projects/.*||')"
  echo "Detected from current session: $DETECTED_DIR"
# Method 3: Search common locations for active projects directory
elif [ -d ~/.claude-personal/projects ] && [ "$(ls -A ~/.claude-personal/projects 2>/dev/null)" ]; then
  DETECTED_DIR=~/.claude-personal
  echo "Found active config at: $DETECTED_DIR"
elif [ -d ~/.claude/projects ] && [ "$(ls -A ~/.claude/projects 2>/dev/null)" ]; then
  DETECTED_DIR=~/.claude
  echo "Found active config at: $DETECTED_DIR"
else
  DETECTED_DIR=~/.claude
  echo "No config detected, defaulting to: $DETECTED_DIR"
fi

# Verify it looks like a Claude config directory
if [ ! -d "$DETECTED_DIR/plugins" ]; then
  echo "WARNING: $DETECTED_DIR/plugins not found - may not be correct config directory"
  echo "Please verify manually with: ls -la ~/ | grep claude"
fi

echo ""
echo "Using Claude config directory: $DETECTED_DIR"
```

**Store the detected directory** for use in subsequent steps.

### Step 1: Clear Cache (Optional - only with 'clean' argument)

**Default behavior** (`/reinstall`): Skip cache deletion - faster reinstall, preserves old versions for rollback.

**Clean mode** (`/reinstall clean`): Delete all cached plugin versions.

```bash
# Only run this if user called '/reinstall clean'
if [ "$1" = "clean" ]; then
  echo "Cleaning plugin cache..."
  rm -rf "$DETECTED_DIR/plugins/cache/jarvis-plugins/"*
  echo "Cache cleared."
else
  echo "Skipping cache clear (use '/reinstall clean' to force clean)."
fi
```

### Step 2: Reinstall All Plugins

**CRITICAL**:
- All `claude plugin` commands MUST be prefixed with `CLAUDE_CONFIG_DIR=$DETECTED_DIR` to target the correct config directory.
- Must `unset CLAUDECODE` first — Claude Code sets this env var in child processes, and `claude plugin` commands refuse to run when it's set (nested session protection).

```bash
unset CLAUDECODE && \
CLAUDE_CONFIG_DIR="$DETECTED_DIR" claude plugin marketplace update && \
CLAUDE_CONFIG_DIR="$DETECTED_DIR" claude plugin uninstall jarvis@jarvis-plugins 2>/dev/null; \
CLAUDE_CONFIG_DIR="$DETECTED_DIR" claude plugin uninstall jarvis-todoist@jarvis-plugins 2>/dev/null; \
CLAUDE_CONFIG_DIR="$DETECTED_DIR" claude plugin uninstall jarvis-strategic@jarvis-plugins 2>/dev/null; \
CLAUDE_CONFIG_DIR="$DETECTED_DIR" claude plugin install jarvis@jarvis-plugins && \
CLAUDE_CONFIG_DIR="$DETECTED_DIR" claude plugin install jarvis-todoist@jarvis-plugins && \
CLAUDE_CONFIG_DIR="$DETECTED_DIR" claude plugin install jarvis-strategic@jarvis-plugins
```

### Step 3: Verify Installation

```bash
echo ""
echo "Installed plugins:"
unset CLAUDECODE && CLAUDE_CONFIG_DIR="$DETECTED_DIR" claude plugin list | grep -E "jarvis|todoist|strategic"
echo ""
echo "Active cache directory: $DETECTED_DIR/plugins/cache/jarvis-plugins/"
ls -la "$DETECTED_DIR/plugins/cache/jarvis-plugins/"
```

### Step 3b: Refresh the Host-Side Copies (launcher + statusline)

The plugin reinstall never touches files `install.sh` copied onto the host: the
`jarvis` launcher (`plugins/jarvis/shell/jarvis.sh`) and `~/.jarvis/statusline.py`
(`plugins/jarvis/statusline/statusline.py`). Stale copies keep old behavior (e.g.
an unbounded health probe that hangs the launcher during a DB outage). Re-copy
them when they differ, keeping a private backup. Symlinks (dev setups) are left
alone. Run from the repo root:

```bash
JARVIS_HOME="${JARVIS_HOME:-$HOME/.jarvis}"
refresh_copy() {  # refresh_copy <src> <installed copy>
  local src="$1" dst="$2" bdir="$JARVIS_HOME/backups/installed" backup tmp
  if [ ! -f "$dst" ] || [ -L "$dst" ]; then echo "skip: $dst (absent or symlink)"; return 0; fi
  if cmp -s "$src" "$dst"; then echo "up to date: $dst"; return 0; fi
  mkdir -p "$bdir" && chmod 700 "$JARVIS_HOME/backups" "$bdir" || return 1
  backup="$bdir/$(basename "$dst").$(date +%Y%m%d-%H%M%S).bak"
  cp "$dst" "$backup" && chmod 600 "$backup" || return 1
  tmp="$dst.tmp.$$"
  if cp "$src" "$tmp" && chmod 755 "$tmp" && mv -f "$tmp" "$dst"; then
    echo "refreshed: $dst (previous copy: $backup)"
  else
    rm -f "$tmp"; return 1
  fi
}
launcher=$(command -v jarvis 2>/dev/null)
if [ -n "$launcher" ] && grep -q "Jarvis AI Assistant launcher" "$launcher" 2>/dev/null; then
  refresh_copy plugins/jarvis/shell/jarvis.sh "$launcher"
fi
refresh_copy plugins/jarvis/statusline/statusline.py "$JARVIS_HOME/statusline.py"
```

Re-running `install.sh` does the same (it refreshes both copies on every run).

### Step 4: Remind User to Restart

**IMPORTANT:** Plugin changes only take effect after a full restart of Claude Code, not just a reload.

Inform the user:
```
Plugins reinstalled successfully

Config directory: [show detected directory]

RESTART REQUIRED
Plugin changes will only take effect after you restart Claude Code.

- macOS: Cmd+Q to quit, then reopen
- Linux/Windows: Close and reopen the application
```

---

## Usage

```
/reinstall           # Reinstall all 3 plugins (preserve cache)
/reinstall clean     # Reinstall all 3 plugins (delete cache first)
```

---

## Rationale

### Why preserve cache by default?
- **Faster**: Skips deletion of potentially large directories
- **Safer**: Old versions remain available for comparison/rollback
- **Cleaner**: Cache naturally gets pruned by Claude over time

### When to use clean mode?
- Testing plugin loading from scratch
- Debugging cache corruption
- Verifying packaging issues
- After major version changes

---

## Notes

- This skill is for development only (not included in the plugin distribution)
- Never skip the restart reminder - users often forget this step
- All 3 plugins are reinstalled together to ensure consistency
- Config directory detection runs fresh each time to handle multiple Claude installations
