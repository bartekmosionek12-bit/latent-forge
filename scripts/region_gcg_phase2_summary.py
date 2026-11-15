"""Side-by-side comparison of region-GCG Phase 1 vs Phase 2 variants (A band objective,
B blocked topic tokens, A+B) on the two independent measures."""
from __future__ import annotations

import json
import os

RUNS = [("P1  min (Phase 1)", "runs/region_gcg_phase1.json"),
        ("A   band", "runs/region_gcg_phase2_A_band.json"),
        ("B   min + block", "runs/region_gcg_phase2_B_block.json"),
        ("A+B band + block", "runs/region_gcg_phase2_AB_band_block.json"),
        ("S   min + semantic", "runs/region_gcg_phase3_S_min_sem.json"),
        ("SA  band + semantic", "runs/region_gcg_phase3_SA_band_sem.json")]

data = [(n, json.load(open(p))) for n, p in RUNS if os.path.exists(p)]
ctrl = data[0][1]["evaluation"]["variants"]
print("=== Region-targeted GCG, Phases 1-3: same prompt, seed 0, 200 steps, distilgpt2 layer 3 ===")
print("block: list = 990 subword tokens of 124 cooking words (Phase 2); semantic = list + 5675 tokens MiniLM finds")
print("closer to cooking than to all 7 control topics (Phase 3), 6319 in total\n")
print("(1) hidden cos dist (attack metric) | (2) MiniLM dist of continuation to topic (lower = more"
      " on-topic) + cooking-keyword hits/words\n")
ref = data[0][1]["evaluation"]["reference_scale"]
print("reference texts: hidden " + ", ".join(f"{x:.3f}" for x in ref["reference_last_token_hidden_cos_dist"])
      + " | MiniLM " + ", ".join(f"{x:.3f}" for x in ref["reference_minilm_cos_dist_to_topic"]))
print(f"\n{'run':22s} {'(1) hidden':>10s} {'(2) MiniLM':>10s} {'kw hits/words':>14s}")


def row(name, v):
    nw = len(v["continuation"].split())
    print(f"{name:22s} {v['hidden_cos_dist']:10.4f} {v['minilm_cos_dist_to_topic']:10.4f} "
          f"{len(v['keyword_hits']):>7d}/{nw:<6d}")


row("ctrl: no suffix", ctrl["no_suffix"])
row("ctrl: naive topical", ctrl["naive_topical"])
for n, d in data:
    row(n, d["evaluation"]["variants"]["optimised"])

print("\nfinal suffixes and continuations")
for n, d in data:
    v = d["evaluation"]["variants"]["optimised"]
    x = d["extra"]
    extra = ""
    if x.get("radius") is not None:
        extra += f" r={x['radius']:.4f}"
    if x.get("n_blocked_tokens"):
        extra += f" blocked={x['n_blocked_tokens']} left_in_suffix={x['blocked_tokens_in_suffix']}"
    print(f"[{n}]{extra}\n  suffix      : {v['suffix']!r}\n  continuation: {v['continuation']!r}\n"
          f"  keyword hits: {v['keyword_hits']}")

print("\ntrajectories: step -> (1) hidden / (2) MiniLM / kw hits")
for n, d in data:
    print(f"[{n}]")
    print("  " + "  ".join(f"{t['step']}:{t['hidden_cos_dist']:.3f}/{t['minilm_cos_dist_to_topic']:.2f}/"
                           f"{len(t['keyword_hits'])}" for t in d["trajectory"]))
