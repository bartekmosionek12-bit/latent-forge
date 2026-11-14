"""Phase 2 evidence: TextTarget -> dispatcher -> GCGAttack(nanogcg) -> evaluation."""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import dispatcher  # noqa: E402
from models import ModelConfig, load_model  # noqa: E402
from targets import TextTarget  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="distilgpt2")
ap.add_argument("--dtype", default="float32")
ap.add_argument("--target", default="TEST_odpowiedz_testowa")
ap.add_argument("--steps", type=int, default=150)
ap.add_argument("--search-width", type=int, default=256)
ap.add_argument("--topk", type=int, default=128)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--early-stop", action="store_true",
                help="nanogcg early_stop: stops when ANY candidate matches, but keeps argmin-loss one")
ap.add_argument("--verbosity", default="WARNING", help="INFO = nanogcg logs loss + suffix every step")
ap.add_argument("--out", default="runs/phase2_gcg.json")
a = ap.parse_args()

bundle = load_model(ModelConfig(name=a.model, dtype=a.dtype))
target = TextTarget(a.target)
attack = dispatcher.select_attack(target, num_steps=a.steps, search_width=a.search_width,
                                  topk=a.topk, seed=a.seed, early_stop=a.early_stop, verbosity=a.verbosity)
print(f"target : {target}\ndispatch: {type(target).__name__} -> {type(attack).__name__}")

result, report = dispatcher.run(target, bundle, **attack.config)

h = result.loss_history
print(f"\nGCG steps run: {len(h)}  ({result.extra['seconds']:.0f}s on {bundle.device})")
for i in sorted({0, 1, 2, 4, 9, 19, 29, 49, 74, 99, 124, len(h) - 1} & set(range(len(h)))):
    print(f"  step {i + 1:4d}  loss {h[i]:.4f}  running-min {min(h[:i + 1]):.4f}")
print(f"best loss: {result.final_loss:.4f}")
print(f"suffix   : {result.payload!r}\n")
for k, v in report.items():
    print(f"[{k}] loss={v['target_loss']:.4f} tok_acc={v['target_token_accuracy']:.2f} "
          f"exact={v['exact_match']}\n    generation: {v['generation']!r}")

os.makedirs(os.path.dirname(a.out), exist_ok=True)
with open(a.out, "w") as f:
    json.dump({"model": a.model, "target": target.text, "prompt": target.prompt,
               "suffix": result.payload, "best_loss": result.final_loss,
               "loss_history": h, "evaluation": report, "extra": result.extra}, f, indent=1)
print(f"\nsaved {a.out}")
