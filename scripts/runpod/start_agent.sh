#!/bin/bash
# Run on the RunPod pod, inside the cloned repo (on a volume under /workspace).
# Installs Claude Code if it is missing (a pod restart wipes everything outside /workspace),
# then starts it with permission prompts skipped and points it at AGENT_TASK.md.
# On the first start after an install, Claude Code asks you to log in: do that yourself.
# The pod is a disposable container; skipping prompts is meant only for it.
set -euo pipefail
cd "$(dirname "$0")/../.."
export PATH="$HOME/.local/bin:$PATH"
if ! command -v claude >/dev/null 2>&1; then
  curl -fsSL https://claude.ai/install.sh | bash
  grep -q '.local/bin' ~/.bashrc 2>/dev/null || echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
fi
git checkout claude/project-thread-mnmiam
git pull --no-rebase
export IS_SANDBOX=1   # the pod runs as root; this tells Claude Code it is a sandbox container
exec claude --dangerously-skip-permissions \
  "Read scripts/runpod/AGENT_TASK.md in full and carry out the whole task it describes, phase by phase, without waiting for me between phases."
