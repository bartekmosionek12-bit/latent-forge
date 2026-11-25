#!/bin/bash
# Run on the RunPod pod, inside the cloned repo, AFTER you have logged in to Claude Code yourself
# (`claude`, then /login) and cloned the repo with your own GitHub credentials.
# Starts Claude Code with permission prompts skipped and points it at AGENT_TASK.md.
# The pod is a disposable container; skipping prompts is meant only for it.
set -euo pipefail
cd "$(dirname "$0")/../.."
git checkout claude/project-thread-mnmiam
export IS_SANDBOX=1   # the pod runs as root; this tells Claude Code it is a sandbox container
exec claude --dangerously-skip-permissions \
  "Read scripts/runpod/AGENT_TASK.md in full and carry out the whole task it describes, phase by phase, without waiting for me between phases."
