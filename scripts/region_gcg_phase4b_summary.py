"""Phase 4b summary: Phase 4 (region GCG + readability) repeated with a FLUENT, off-topic
start suffix instead of the "x x x" filler; side-by-side with the Phase 4 runs.

Same three measures as Phase 4: (1) hidden distance + natural range, (2) MiniLM / cooking
keywords of the continuation, (3) suffix PPL under Qwen2.5-0.5B-Instruct (independent),
plus exploration: accepted swaps, step of the last accepted swap, distinct tokens."""
from __future__ import annotations

import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.perplexity import conditional_nll  # noqa: E402
from models import ModelConfig, load_model  # noqa: E402

W = ("0.01", "0.03", "0.1")
RUNS = ([("P1  x-init  w=0 (Phase 1)", "runs/region_gcg_phase1.json")]
        + [(f"4   x-init  w={w}", f"runs/region_gcg_phase4_w{w}.json") for w in W]
        + [(f"4b  fluent  w={w}", f"runs/region_gcg_phase4b_w{w}.json") for w in W])
INDEP = "Qwen/Qwen2.5-0.5B-Instruct"

runs = [(n, json.load(open(p))) for n, p in RUNS]
p1, fl = runs[0][1], runs[-1][1]
prompt = p1["prompt"]
ref_d = p1["evaluation"]["reference_scale"]["reference_last_token_hidden_cos_dist"]
lo, hi = min(ref_d), max(ref_d)
small = load_model(ModelConfig(name="distilgpt2"))
indep = load_model(ModelConfig(name=INDEP))
tok = small.tokenizer


def readability(sfx: str) -> dict:
    ids = tok(sfx, add_special_tokens=False)["input_ids"]
    return {"ce_distilgpt2": conditional_nll(small, prompt, sfx),
            "ppl_qwen": math.exp(conditional_nll(indep, prompt, sfx)),
            "distinct": len(set(ids)) / max(len(ids), 1)}


def last_change(r) -> int | None:
    sh = r["extra"].get("suffix_history")
    if not sh:
        return None
    return max([i for i in range(1, len(sh)) if sh[i] != sh[i - 1]], default=0)


init_text = fl["extra"]["init_suffix"]
print("Phase 4b: region-targeted GCG + readability, FLUENT off-topic start instead of 'x x x'")
print(f"prompt: {prompt!r} | distilgpt2 block {p1['layer']} | 20 tokens, 200 steps, width 256, top-k 128, "
      "seed 0, objective = min, union proposals, J = d + w * CE_suffix")
print(f"fluent start suffix: {init_text!r}")
print(f"independent readability model: {INDEP} | natural range of reference texts: [{lo:.3f}, {hi:.3f}]")

rows = [("start: filler x*20", p1["evaluation"]["variants"]["init_filler"], None),
        ("start: fluent sentence", fl["evaluation"]["variants"]["init_filler"], None),
        ("baseline: naive topical", p1["evaluation"]["variants"]["naive_topical"], None)]
rows += [(n, r["evaluation"]["variants"]["optimised"], r) for n, r in runs]

print("\n[three measures + exploration]")
print(f"  {'run':28s} {'(1) hidden':>10s} {'range':>6s} {'(2) MiniLM':>10s} {'kw':>6s} {'(3) PPL Qwen':>12s} "
      f"{'CE dgpt2':>8s} {'distinct':>8s} {'swaps':>5s} {'last swap @':>11s}")
summary = []
for name, v, r in rows:
    rd = readability(v["suffix"])
    d = v["hidden_cos_dist"]
    inr = "in" if lo <= d <= hi else ("below" if d < lo else "above")
    sw = f"{r['extra']['accepted_swaps']:5d}" if r else f"{'-':>5s}"
    lc = last_change(r) if r else None
    lcs = f"{lc:11d}" if lc is not None else f"{'-':>11s}"
    print(f"  {name:28s} {d:10.4f} {inr:>6s} {v['minilm_cos_dist_to_topic']:10.4f} {v['keyword_frac']:6.3f} "
          f"{rd['ppl_qwen']:12.1f} {rd['ce_distilgpt2']:8.2f} {rd['distinct']:8.2f} {sw} {lcs}")
    summary.append({"run": name, "hidden": d, "range": inr, "minilm": v["minilm_cos_dist_to_topic"],
                    "kw": v["keyword_frac"], **rd, "accepted_swaps": r["extra"]["accepted_swaps"] if r else None,
                    "last_swap_step": lc, "suffix": v["suffix"], "continuation": v["continuation"],
                    "keyword_hits": v["keyword_hits"]})

print("\n[trajectory] hidden dist / distilgpt2 suffix CE at checkpoints")
for name, r in runs[1:]:
    x = r["extra"]
    print(f"  {name:28s} " + "  ".join(f"{s}:{x['distance_history'][s]:.3f}/{x['ce_history'][s]:.2f}"
                                       for s in (0, 10, 25, 50, 100, 150, 200)))

print("\n[examples] final suffix / continuation / keyword hits (x-init vs fluent at the same weight)")
for name, v, r in rows:
    print(f"[{name}]\n  suffix      : {v['suffix']!r}\n  continuation: {v['continuation']!r}\n"
          f"  keyword hits: {v['keyword_hits']}")

print("\n[how much of the start survives] tokens of the final suffix identical to the start, same position")
for name, r in runs[1:]:
    a = tok(r["extra"]["init_suffix"], add_special_tokens=False)["input_ids"]
    b = tok(r["payload"] if "payload" in r else r["suffix"], add_special_tokens=False)["input_ids"]
    same = sum(int(x == y) for x, y in zip(a, b)) if len(a) == len(b) else None
    print(f"  {name:28s} {same}/{len(a)}")

print("\n[mid-run suffixes, fluent start]")
for name, r in runs[4:]:
    for s in (10, 50, 100):
        print(f"  {name:28s} step {s:3d}: {r['extra']['suffix_history'][s]!r}")

json.dump(summary, open("runs/region_gcg_phase4b_fluent_init.json", "w"), indent=1)
print("\nsaved runs/region_gcg_phase4b_fluent_init.json")
