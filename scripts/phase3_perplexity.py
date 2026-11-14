"""Phase 3 add-on: perplexity (distilgpt2) of payload / payload+centroid-text /
payload+optimised span, read from the saved Phase 3 runs. Measurement only."""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation import perplexity  # noqa: E402
from models import ModelConfig, load_model  # noqa: E402

lm_name = sys.argv[1] if len(sys.argv) > 1 else "distilgpt2"
lm = load_model(ModelConfig(name=lm_name))
rows = []
for run in ("centroid_text", "neutral"):
    ev = json.load(open(f"runs/phase3_embedding_{run}.json"))["evaluation"]
    for k in ("payload_only", "payload_plus_init_span", "after_attack"):
        if run == "neutral" and k == "payload_only":
            continue
        label = f"{run}:{k}"
        rows.append((label, ev[k]["cosine_distance_to_centroid"], perplexity(lm, ev[k]["text"]), ev[k]["text"]))

print(f"\n=== perplexity add-on (LM: {lm_name}, measurement only, attack unchanged) ===")
print(f"{'variant':40s} {'cos dist':>8s} {'PPL':>10s}")
for label, d, p, _ in rows:
    print(f"{label:40s} {d:8.4f} {p:10.1f}")
for label, _, _, t in rows:
    print(f"  {label}: {t!r}")
