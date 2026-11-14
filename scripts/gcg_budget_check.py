"""Analyse a long GCG run: full loss curve, rate of decrease per window, and whether the
best-so-far suffix produces the target at checkpoints. Reads the JSON written by
scripts/phase2_gcg_demo.py (which stores per-step losses and suffixes)."""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation import generate_greedy, target_loss  # noqa: E402
from models import ModelConfig, load_model  # noqa: E402

path = sys.argv[1] if len(sys.argv) > 1 else "runs/gcg_budget_check_500.json"
r = json.load(open(path))
h = r["loss_history"]
suffixes = r["extra"]["suffix_per_step"]
n = len(h)
run_min = [min(h[: i + 1]) for i in range(n)]

print(f"model {r['model']}, target {r['target']!r}, prompt {r['prompt']!r}, steps {n}")
print(f"config: {r['extra']['gcg_config']}\n")
print("loss curve (per-step best candidate loss | running min)")
for i in sorted({0} | set(range(24, n, 25))):
    print(f"  step {i + 1:4d}  loss {h[i]:.4f}  running-min {run_min[i]:.4f}")


def rate(a: int, b: int) -> float:
    """drop of the running minimum per 100 steps between steps a and b (1-based)."""
    return (run_min[a - 1] - run_min[b - 1]) / (b - a) * 100


print("\nrate of decrease of the running minimum (loss units per 100 steps)")
windows = [(1, 100), (100, 250), (250, 375), (375, n)]
for a, b in windows:
    print(f"  steps {a:3d}-{b:3d}: {run_min[a - 1]:.4f} -> {run_min[b - 1]:.4f}   rate {rate(a, b):.3f}")
last_improve = max(i for i in range(n) if h[i] == run_min[i] and (i == 0 or h[i] < run_min[i - 1])) + 1
print(f"  last step that set a new minimum: {last_improve}")

bundle = load_model(ModelConfig(name=r["model"]))
tgt_ids = bundle.tokenizer(r["target"], add_special_tokens=False)["input_ids"]
print(f"\nbest-so-far suffix at checkpoints (target = {len(tgt_ids)} tokens: "
      f"{[bundle.tokenizer.decode([t]) for t in tgt_ids]})")
for c in [c for c in (100, 250, 375, n) if c <= n]:
    k = min(range(c), key=lambda i: h[i])
    sfx = suffixes[k]
    loss, acc = target_loss(bundle, r["prompt"], sfx, r["target"])
    gen = generate_greedy(bundle, r["prompt"], sfx, max_new_tokens=len(tgt_ids) + 8)
    print(f"  @{c:3d} (best at step {k + 1}): loss {loss:.4f}  token acc {acc:.2f} "
          f"({round(acc * len(tgt_ids))}/{len(tgt_ids)})  exact {gen.startswith(r['target'])}\n"
          f"        generation {gen!r}\n        suffix     {sfx!r}")
