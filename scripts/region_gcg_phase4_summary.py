"""Phase 4 summary: region GCG + readability (AutoDAN-style), three measures per run.

(1) hidden cos distance to the centroid (distilgpt2 block 3, the region term)
(2) independent topicality of the continuation (MiniLM distance + cooking keywords)
(3) suffix perplexity under an INDEPENDENT model (Qwen2.5-0.5B-Instruct), suffix tokens
    conditioned on the prompt; distilgpt2 suffix CE shown too (circular: it is optimised).
Writes the table + example suffixes to stdout (tee'd to runs/region_gcg_phase4_autodan.log)."""
from __future__ import annotations

import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.perplexity import conditional_nll  # noqa: E402
from models import ModelConfig, load_model  # noqa: E402

RUNS = [("P1  (Phase 1, grad top-k, w=0)", "runs/region_gcg_phase1.json"),
        ("C0  (union proposals, w=0)", "runs/region_gcg_phase4_ctrl_w0.json"),
        ("R1  (union, w=0.01)", "runs/region_gcg_phase4_w0.01.json"),
        ("R3  (union, w=0.03)", "runs/region_gcg_phase4_w0.03.json"),
        ("R10 (union, w=0.1)", "runs/region_gcg_phase4_w0.1.json")]
INDEP = "Qwen/Qwen2.5-0.5B-Instruct"

runs = [(n, json.load(open(p))) for n, p in RUNS]
p1 = runs[0][1]
prompt, refs = p1["prompt"], p1["reference_texts"]
ref_d = p1["evaluation"]["reference_scale"]["reference_last_token_hidden_cos_dist"]
ref_m = p1["evaluation"]["reference_scale"]["reference_minilm_cos_dist_to_topic"]
lo, hi = min(ref_d), max(ref_d)

small = load_model(ModelConfig(name="distilgpt2"))
indep = load_model(ModelConfig(name=INDEP))
tok = small.tokenizer


def readability(sfx: str) -> dict:
    ids = tok(sfx, add_special_tokens=False)["input_ids"]
    return {"ce_distilgpt2": conditional_nll(small, prompt, sfx),
            "ppl_qwen": math.exp(conditional_nll(indep, prompt, sfx)),
            "distinct": len(set(ids)) / max(len(ids), 1)}


print(f"Phase 4: region-targeted GCG + readability (AutoDAN-style), distilgpt2 block {p1['layer']}")
print(f"prompt: {prompt!r} | 20 tokens, 200 steps, search width 256, top-k 128, seed 0, objective = min")
print("J = d + w * CE_suffix(distilgpt2, in context); proposals: P1 gradient top-128, others union "
      "(gradient top-64 + model next-token top-64)")
print(f"independent readability model: {INDEP} (suffix PPL given the prompt)")
print(f"reference texts: hidden dist {', '.join(f'{x:.3f}' for x in ref_d)} -> natural range [{lo:.3f}, {hi:.3f}]"
      f" | MiniLM {', '.join(f'{x:.3f}' for x in ref_m)}")

rows = []
base = p1["evaluation"]["variants"]
for name, key in (("baseline: filler x*20", "init_filler"), ("baseline: naive topical", "naive_topical"),
                  ("baseline: random_0", "random_0"), ("baseline: no suffix", "no_suffix")):
    v = base[key]
    rows.append((name, v, readability(v["suffix"]) if v["suffix"] else None, None))
for name, r in runs:
    v = r["evaluation"]["variants"]["optimised"]
    rows.append((name, v, readability(v["suffix"]), r))

print("\n[three measures] (1) hidden dist (lower = closer; natural range above) | (2) MiniLM dist "
      "(lower = more on-topic) + keyword fraction | (3) suffix PPL under Qwen-0.5B (lower = more readable)")
hdr = (f"  {'run':34s} {'(1) hidden':>10s} {'in range':>8s} {'(2) MiniLM':>10s} {'kw':>6s} "
       f"{'(3) PPL Qwen':>12s} {'CE dgpt2':>8s} {'distinct':>8s} {'swaps':>5s}")
print(hdr)
summary = []
for name, v, rd, r in rows:
    inr = "yes" if lo <= v["hidden_cos_dist"] <= hi else ("below" if v["hidden_cos_dist"] < lo else "above")
    ppl = f"{rd['ppl_qwen']:12.1f}" if rd else f"{'-':>12s}"
    ce = f"{rd['ce_distilgpt2']:8.2f}" if rd else f"{'-':>8s}"
    di = f"{rd['distinct']:8.2f}" if rd else f"{'-':>8s}"
    sw = f"{r['extra']['accepted_swaps']:5d}" if r else f"{'-':>5s}"
    print(f"  {name:34s} {v['hidden_cos_dist']:10.4f} {inr:>8s} {v['minilm_cos_dist_to_topic']:10.4f} "
          f"{v['keyword_frac']:6.3f} {ppl} {ce} {di} {sw}")
    summary.append({"run": name, "hidden": v["hidden_cos_dist"], "in_range": inr,
                    "minilm": v["minilm_cos_dist_to_topic"], "kw": v["keyword_frac"], **(rd or {}),
                    "suffix": v["suffix"], "continuation": v["continuation"]})

print("\n[trajectory] hidden dist / distilgpt2 suffix CE at checkpoints (from the runs' own histories)")
for name, r in runs[1:]:
    x = r["extra"]
    pts = [0, 10, 25, 50, 100, 150, 200]
    print(f"  {name:34s} " + "  ".join(f"{s}:{x['distance_history'][s]:.3f}/{x['ce_history'][s]:.2f}" for s in pts))
x = p1["extra"]
print(f"  {runs[0][0]:34s} " + "  ".join(f"{s}:{p1['loss_history'][s]:.3f}" for s in [0, 10, 25, 50, 100, 150, 200])
      + "  (no CE history in Phase 1)")

print("\n[examples] final suffix, continuation, keyword hits")
for name, v, rd, r in rows:
    if r is None and name != "baseline: naive topical":
        continue
    print(f"[{name}]\n  suffix      : {v['suffix']!r}\n  continuation: {v['continuation']!r}\n"
          f"  keyword hits: {v['keyword_hits']}")

print("\n[mid-run suffixes] step 50 / 100 per Phase 4 run")
for name, r in runs[1:]:
    for s in (50, 100):
        print(f"  {name:34s} step {s:3d}: {r['extra']['suffix_history'][s]!r}")

json.dump(summary, open("runs/region_gcg_phase4_autodan.json", "w"), indent=1)
print("\nsaved runs/region_gcg_phase4_autodan.json")
