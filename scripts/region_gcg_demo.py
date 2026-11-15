"""Region-targeted GCG, Phase 1 evidence on a toy model.

HiddenStateRegionTarget -> SAME dispatcher -> GCGRegionAttack -> evaluation with TWO
independent success measures (hidden-state distance = attack loss; topical score of
the continuation = separate MiniLM embedder + keyword list), plus controls and an
independent gradient check."""
from __future__ import annotations

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import dispatcher  # noqa: E402
from models import ModelConfig, load_model  # noqa: E402
from targets import HiddenStateRegionTarget  # noqa: E402

COOKING = [  # neutral topic
    "Chop the onions and garlic, then fry them in olive oil until golden.",
    "Simmer the tomato sauce for twenty minutes and season it with salt and basil.",
    "Roast the vegetables in a hot oven and serve them with rice.",
]

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="distilgpt2")
ap.add_argument("--dtype", default="float32")
ap.add_argument("--layer", type=int, default=3)
ap.add_argument("--prompt", default="Yesterday the city council met to discuss")
ap.add_argument("--n-tokens", type=int, default=20)
ap.add_argument("--steps", type=int, default=200)
ap.add_argument("--search-width", type=int, default=256)
ap.add_argument("--topk", type=int, default=128)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--objective", default="min", choices=["min", "band"])
ap.add_argument("--block-topic", action="store_true",
                help="forbid suffix tokens that are part of cooking words (reference words + keyword list)")
ap.add_argument("--out", default="runs/region_gcg_phase1.json")
a = ap.parse_args()

bundle = load_model(ModelConfig(name=a.model, dtype=a.dtype))
target = HiddenStateRegionTarget(COOKING, layer=a.layer, prompt=a.prompt)
STOP = {"the", "and", "then", "them", "in", "it", "with", "for", "until", "a", "hot", "twenty",
        "minutes", "serve"}
import re  # noqa: E402

from evaluation.topical import COOKING_KEYWORDS  # noqa: E402

BLOCK_WORDS = tuple(sorted(({w.lower() for r in COOKING for w in re.findall(r"[a-zA-Z]+", r)} - STOP)
                           | set(COOKING_KEYWORDS)))
cfg = dict(n_tokens=a.n_tokens, num_steps=a.steps, search_width=a.search_width,
           topk=a.topk, seed=a.seed, objective=a.objective,
           blocked_words=BLOCK_WORDS if a.block_topic else ())
attack = dispatcher.select_attack(target, **cfg)
print(f"target  : {target}\ndispatch: {type(target).__name__} -> {type(attack).__name__}")
print(f"objective: {a.objective} | block topic tokens: {a.block_topic}")

# ---- independent gradient verification (before the run) -----------------------
attack.setup(bundle)
sfx = attack.init_suffix()
ids_loss = attack.loss_ids(attack.full_ids(sfx)[None]).item()
emb = attack.W[sfx].detach().clone().requires_grad_(True)
emb_loss = attack.loss_from_suffix_embeds(emb)
g = torch.autograd.grad(emb_loss, emb)[0]
print("\n[gradient check]")
print(f"  ids path loss {ids_loss:.6f} vs one-hot/inputs_embeds path {emb_loss.item():.6f} "
      f"(|diff| {abs(ids_loss - emb_loss.item()):.2e})")
gen = torch.Generator().manual_seed(0)
eps, rel = 1e-2, []  # 1e-2: float32 hidden-state cast makes smaller eps rounding-dominated
for k in range(5):
    v = torch.randn(emb.shape, generator=gen)
    v = v / v.norm()
    with torch.no_grad():
        fd = (attack.loss_from_suffix_embeds(emb + eps * v) - attack.loss_from_suffix_embeds(emb - eps * v)) / (2 * eps)
    an = (g * v).sum()
    rel.append(abs(fd.item() - an.item()) / max(abs(an.item()), 1e-12))
    print(f"  dir {k}: finite-diff {fd.item():+.6f}  autograd {an.item():+.6f}  rel.err {rel[-1]:.2e}")
tg = attack.token_grad(sfx)
pos = torch.randint(0, sfx.numel(), (128,), generator=gen)
allowed_ids = attack.allowed.nonzero()[:, 0].cpu()
new = allowed_ids[torch.randint(0, len(allowed_ids), (128,), generator=gen)]
cands = sfx.repeat(128, 1)
cands[torch.arange(128), pos] = new
actual = attack.loss_ids(torch.stack([attack.full_ids(s) for s in cands])) - ids_loss
pred = tg[pos, new] - tg[pos, sfx[pos]]
rk = lambda x: x.argsort().argsort().float()  # noqa: E731
rho = torch.corrcoef(torch.stack([rk(pred.detach().cpu()), rk(actual.cpu())]))[0, 1].item()
print(f"  128 random single-token swaps: Spearman(first-order prediction, actual change) = {rho:.3f}")
# practical check: are swaps proposed by the gradient top-k better than random swaps?
tg2 = tg.clone()
tg2[:, ~attack.allowed] = float("inf")
top = (-tg2).topk(a.topk, dim=1).indices
p2 = torch.randint(0, sfx.numel(), (256,), generator=gen)
cg = sfx.repeat(256, 1)
cg[torch.arange(256), p2] = top[p2, torch.randint(0, a.topk, (256,), generator=gen)]
cr = sfx.repeat(256, 1)
cr[torch.arange(256), p2] = allowed_ids[torch.randint(0, len(allowed_ids), (256,), generator=gen)].to(sfx.device)
dg = attack.loss_ids(torch.stack([attack.full_ids(s) for s in cg])) - ids_loss
dr = attack.loss_ids(torch.stack([attack.full_ids(s) for s in cr])) - ids_loss
topk_vs_random = {"grad_mean": dg.mean().item(), "grad_improve": (dg < 0).float().mean().item(),
                  "rand_mean": dr.mean().item(), "rand_improve": (dr < 0).float().mean().item()}
print(f"  256 swaps from gradient top-{a.topk}: mean change {dg.mean():+.4f}, improve {(dg < 0).float().mean():.0%}"
      f" | 256 random swaps: mean change {dr.mean():+.4f}, improve {(dr < 0).float().mean():.0%}")

# ---- the run through the dispatcher -------------------------------------------
result, report = dispatcher.run(target, bundle, **cfg)
h = result.loss_history
x = result.extra
print(f"\n[optimisation] {len(h) - 1} steps, {x['accepted_swaps']} accepted swaps, {x['seconds']:.0f}s")
if x["radius"] is not None:
    print(f"  band objective |d - r|: reference distances "
          + ", ".join(f"{d:.4f}" for d in x["reference_distances"]) + f" -> r = midpoint = {x['radius']:.4f}")
if a.block_topic:
    print(f"  blocked {x['n_blocked_tokens']} topic tokens from {len(BLOCK_WORDS)} words; "
          f"blocked tokens left in final suffix: {x['blocked_tokens_in_suffix']}")
dh = x["distance_history"]
for i in sorted({0, 10, 25, 50, 100, 150, len(h) - 1} & set(range(len(h)))):
    print(f"  step {i:4d}  hidden cos dist {dh[i]:.4f}  objective {h[i]:.4f}")

ref = report["reference_scale"]
print("\n[scale] reference texts themselves: last-token hidden cos dist "
      + ", ".join(f"{x:.3f}" for x in ref["reference_last_token_hidden_cos_dist"])
      + " | MiniLM topic dist " + ", ".join(f"{x:.3f}" for x in ref["reference_minilm_cos_dist_to_topic"]))
print("\n[two measures] (1) hidden cos dist = attack loss; (2) topical score of the continuation"
      " = MiniLM distance to topic (lower = more on-topic) + cooking-keyword fraction")
print(f"  {'variant':14s} {'(1) hidden':>10s} {'(2) MiniLM':>10s} {'(2) kw':>7s}")
for k, v in report["variants"].items():
    print(f"  {k:14s} {v['hidden_cos_dist']:10.4f} {v['minilm_cos_dist_to_topic']:10.4f} {v['keyword_frac']:7.3f}")
print()
for k in ("no_suffix", "random_0", "naive_topical", "optimised"):
    v = report["variants"][k]
    print(f"[{k}]\n  suffix      : {v['suffix']!r}\n  continuation: {v['continuation']!r}\n"
          f"  keyword hits: {v['keyword_hits']}")

# ---- both measures ALONG the trajectory (where do they diverge?) ---------------
from evaluation.hidden_region import continuation, last_token_distance  # noqa: E402
from evaluation.topical import COOKING_KEYWORDS, topical_scores  # noqa: E402

c = target.centroid(bundle)
traj = []
print("\n[trajectory] both measures at checkpoints of the same run")
print(f"  {'step':>4s} {'(1) hidden':>10s} {'(2) MiniLM':>10s} {'(2) kw':>7s} {'words':>5s}  continuation")
for st in sorted({0, 1, 2, 3, 5, 10, 25, 50, 100, 150, len(h) - 1} & set(range(len(h)))):
    sx = result.extra["suffix_history"][st]
    text = target.prompt + sx
    cont = continuation(bundle, text)
    ts = topical_scores(cont, COOKING, target.eval_embedder_name, COOKING_KEYWORDS)
    nw = len(cont.split())
    traj.append({"step": st, "suffix": sx, "hidden_cos_dist": last_token_distance(bundle, text, a.layer, c),
                 "continuation": cont, "n_words": nw, **ts})
    t_ = traj[-1]
    print(f"  {st:4d} {t_['hidden_cos_dist']:10.4f} {ts['minilm_cos_dist_to_topic']:10.4f} "
          f"{ts['keyword_frac']:7.3f} {nw:5d}  {cont[:70]!r}")

os.makedirs(os.path.dirname(a.out), exist_ok=True)
with open(a.out, "w") as f:
    json.dump({"model": a.model, "layer": a.layer, "prompt": a.prompt, "reference_texts": COOKING,
               "loss_history": h, "suffix": result.payload, "extra": result.extra,
               "gradient_check": {"ids_vs_embeds_abs_diff": abs(ids_loss - emb_loss.item()),
                                  "fd_eps": eps, "fd_rel_err": rel, "swap_spearman": rho,
                                  "topk_vs_random": topk_vs_random},
               "evaluation": report, "trajectory": traj}, f, indent=1)
print(f"\nsaved {a.out}")
