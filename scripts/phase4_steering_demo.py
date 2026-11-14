"""Phase 4 evidence: ActivationDirectionTarget -> SAME dispatcher ->
ActivationSteeringAttack -> evaluation (projection of outputs on the direction)."""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import dispatcher  # noqa: E402
from evaluation.activation import residual_activations  # noqa: E402
from models import ModelConfig, load_model  # noqa: E402
from targets import ActivationDirectionTarget  # noqa: E402

# neutral contrastive pairs: same content, formal (positive) vs informal (negative)
FORMAL = [
    "Dear Sir or Madam, I am writing to inform you that the meeting has been rescheduled.",
    "We would be grateful if you could kindly confirm your attendance at your earliest convenience.",
    "Please find attached the requested documents for your review and consideration.",
    "I would like to express my sincere gratitude for your assistance in this matter.",
    "The committee has reviewed your application and will respond in due course.",
    "Kindly be advised that the office will remain closed on Monday.",
    "It is with great pleasure that we announce the opening of our new branch.",
    "Should you require any further information, please do not hesitate to contact us.",
]
INFORMAL = [
    "hey guys, so the meeting got moved lol",
    "can u let me know if ur coming? thx",
    "here's the stuff u asked for, check it out",
    "omg thanks sooo much for helping me out!!",
    "yeah they looked at ur thing, they'll get back to u sometime",
    "heads up, office is closed monday btw",
    "yay we finally opened the new place!! so pumped",
    "if u need anything else just hit me up",
]
TEST_PROMPTS = [
    "Regarding the schedule for next week,",
    "I wanted to tell you about the trip.",
    "About the new coffee machine in the kitchen:",
]

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="distilgpt2")
ap.add_argument("--dtype", default="float32")
ap.add_argument("--layer", type=int, default=3)
ap.add_argument("--alphas", default="0,2,4,8")
ap.add_argument("--max-new-tokens", type=int, default=30)
ap.add_argument("--out", default="runs/phase4_steering.json")
a = ap.parse_args()
alphas = [float(x) for x in a.alphas.split(",")]

bundle = load_model(ModelConfig(name=a.model, dtype=a.dtype))
target = ActivationDirectionTarget(FORMAL, INFORMAL, layer=a.layer)
cfg = dict(alphas=alphas, test_prompts=TEST_PROMPTS, max_new_tokens=a.max_new_tokens)
attack = dispatcher.select_attack(target, **cfg)
print(f"target  : {target}\ndispatch: {type(target).__name__} -> {type(attack).__name__}")

result, report = dispatcher.run(target, bundle, **cfg)
x = result.extra
typical = residual_activations(bundle, TEST_PROMPTS[0], a.layer)[1:].norm(dim=-1).mean().item()
print(f"\nlayer {a.layer}: |direction| = {x['direction_norm']:.2f}, "
      f"typical |h_t| on a test prompt = {typical:.2f}  ({x['seconds']:.0f}s)")
ref = report["reference"]
print(f"projection scale: formal prompts {ref['positive_prompts_mean_proj']:.2f}, "
      f"informal prompts {ref['negative_prompts_mean_proj']:.2f}")
print("(alpha > 0 pushes towards FORMAL; alpha = 0 is the unsteered baseline)\n")
for alpha in alphas:
    r = report["by_alpha"][alpha]
    print(f"=== alpha {alpha:+.1f}  mean projection of output (unsteered re-read): "
          f"{r['mean_projection_of_output']:.2f}")
    for p, g in r["generations"].items():
        print(f"  {p!r}\n    -> {g!r}")

os.makedirs(os.path.dirname(a.out), exist_ok=True)
with open(a.out, "w") as f:
    json.dump({"model": a.model, "layer": a.layer, "positive": FORMAL, "negative": INFORMAL,
               "direction_norm": x["direction_norm"], "typical_resid_norm": typical,
               "evaluation": {**report, "by_alpha": {str(k): v for k, v in report["by_alpha"].items()}},
               "direction": result.payload.tolist()}, f, indent=1)
print(f"\nsaved {a.out}")
