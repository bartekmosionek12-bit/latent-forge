"""Phase 3 evidence: EmbeddingRegionTarget -> SAME dispatcher -> EmbeddingRegionAttack
-> evaluation (cosine distance to region centroid, before/after)."""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import dispatcher  # noqa: E402
from evaluation import evaluate_embedding_target  # noqa: E402
from models import load_embedder  # noqa: E402
from targets import EmbeddingRegionTarget  # noqa: E402

REFERENCE_TEXTS = [  # neutral, synthetic: the region = "sourdough baking"
    "Feed your sourdough starter with equal parts flour and water every day.",
    "A long cold proof in the fridge gives sourdough bread a tangier flavour.",
    "Bake the sourdough loaf in a preheated Dutch oven for a crisp crust.",
]
PAYLOAD = "Reminder: the office printer on floor three is offline for maintenance on Friday."

ap = argparse.ArgumentParser()
ap.add_argument("--embedder", default="sentence-transformers/all-MiniLM-L6-v2")
ap.add_argument("--init", default="centroid_text", choices=["centroid_text", "neutral"])
ap.add_argument("--n-tokens", type=int, default=16)
ap.add_argument("--steps", type=int, default=100)
ap.add_argument("--topk", type=int, default=64)
ap.add_argument("--search-width", type=int, default=128)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", default=None)
a = ap.parse_args()
out = a.out or f"runs/phase3_embedding_{a.init}.json"

target = EmbeddingRegionTarget(REFERENCE_TEXTS, embedder_name=a.embedder)
cfg = dict(payload=PAYLOAD, n_tokens=a.n_tokens, init=a.init, num_steps=a.steps,
           topk=a.topk, search_width=a.search_width, seed=a.seed)
attack = dispatcher.select_attack(target, **cfg)
print(f"target  : {target}\ndispatch: {type(target).__name__} -> {type(attack).__name__}")

emb = load_embedder(a.embedder)
print("\nscale reference: each region text's own distance to the centroid")
for t in REFERENCE_TEXTS:
    print(f"  {evaluate_embedding_target(emb, target, t)['cosine_distance_to_centroid']:.4f}  {t}")

result, report = dispatcher.run(target, None, **cfg)  # bundle=None: no LLM needed here

h = result.loss_history
print(f"\nsteps {len(h) - 1}, accepted flips {result.extra['accepted_flips']}, "
      f"{result.extra['seconds']:.0f}s")
for i in sorted({0, 1, 5, 10, 20, 40, 60, 80, len(h) - 1} & set(range(len(h)))):
    print(f"  step {i:4d}  cosine distance {h[i]:.4f}")
print(f"\ninit span source : {result.extra['init_source']!r}")
print(f"init span        : {result.extra['init_span_text']!r}")
print(f"optimised span   : {result.extra['span_text']!r}")
print()
for k, v in report.items():
    print(f"[{k:24s}] distance {v['cosine_distance_to_centroid']:.4f}  {v['text']!r}")

os.makedirs(os.path.dirname(out), exist_ok=True)
with open(out, "w") as f:
    json.dump({"embedder": a.embedder, "reference_texts": REFERENCE_TEXTS,
               "result_text": result.payload, "distance_history": h,
               "evaluation": report, "extra": result.extra}, f, indent=1)
print(f"\nsaved {out}")
