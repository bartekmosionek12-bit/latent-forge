#!/bin/bash
# One-time pod setup for topic-injection Phase 2 (arms A-D). Run from the repo root on the pod.
# Creates .venv with Python 3.13 (same as the CPU runs) and installs the pinned stack.
# No tokens or keys are needed: every model used is public and not gated.
set -euo pipefail
cd "$(dirname "$0")/../.."
pip install -q uv==0.9.5
uv venv -p 3.13 .venv
. .venv/bin/activate
uv pip install torch==2.15.0                      # CUDA build from PyPI
uv pip install -r requirements.txt
python - <<'PY'
import torch, transformers
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "-")
print("transformers", transformers.__version__)
assert torch.cuda.is_available(), "no GPU visible"
assert transformers.__version__ == "4.47.1"
PY
# pre-download the public models used by Phase 2 (no token needed)
python - <<'PY'
from huggingface_hub import snapshot_download
for m in ["Qwen/Qwen2.5-0.5B-Instruct", "protectai/deberta-v3-base-prompt-injection-v2",
          "deepset/deberta-v3-base-injection", "HuggingFaceTB/SmolLM2-360M",
          "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", "microsoft/Phi-3.5-mini-instruct"]:
    print(snapshot_download(m))
PY
echo "setup OK"
