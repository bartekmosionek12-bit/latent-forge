"""Phase 5: one end-to-end run of all three attack paths through ONE dispatcher.

Three goal types, one call site: for each Target, dispatcher.run(target, bundle)
picks the Attack, runs it and evaluates it with the matching metric. The attacked
LLM is a parameter (--model); the embedder is a property of the EmbeddingRegionTarget.
Configs reproduce the Phase 2-4 evidence runs (same seeds, same settings).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import dispatcher  # noqa: E402
from evaluation import perplexity  # noqa: E402
from models import ModelConfig, load_model  # noqa: E402
from targets import ActivationDirectionTarget, EmbeddingRegionTarget, TextTarget  # noqa: E402

# ---- neutral, synthetic test data -------------------------------------------
SOURDOUGH = [
    "Feed your sourdough starter with equal parts flour and water every day.",
    "A long cold proof in the fridge gives sourdough bread a tangier flavour.",
    "Bake the sourdough loaf in a preheated Dutch oven for a crisp crust.",
]
PAYLOAD = "Reminder: the office printer on floor three is offline for maintenance on Friday."
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
STEER_PROMPTS = ["Regarding the schedule for next week,", "I wanted to tell you about the trip.",
                 "About the new coffee machine in the kitchen:"]
# same topics as user turns, for instruct models with a chat template
STEER_PROMPTS_CHAT = ["Write two sentences about the schedule for next week.",
                      "Tell me briefly about a trip you took.",
                      "Write a short note about the new coffee machine in the kitchen."]

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="distilgpt2")
ap.add_argument("--dtype", default="float32")
ap.add_argument("--gcg-steps", type=int, default=250)
ap.add_argument("--steer-layer", type=int, default=3)
ap.add_argument("--alphas", default="0,2,4,8")
ap.add_argument("--steer-max-new-tokens", type=int, default=30)
ap.add_argument("--out", default="runs/e2e_all_paths.json")
a = ap.parse_args()

bundle = load_model(ModelConfig(name=a.model, dtype=a.dtype))
from evaluation.activation import has_real_chat_template  # noqa: E402

CHAT = has_real_chat_template(bundle.tokenizer)  # checked BEFORE nanogcg may install its fallback
steer_prompts = STEER_PROMPTS_CHAT if CHAT else STEER_PROMPTS
print(f"model {a.model} ({a.dtype}), real chat template: {CHAT}")

# goal -> attack-specific knobs. The dispatcher picks the attack; these are only
# its hyper-parameters (each attack validates its own option names).
jobs = [
    (TextTarget("TEST_odpowiedz_testowa"),
     dict(num_steps=a.gcg_steps, search_width=256, topk=128, seed=0, verbosity="WARNING")),
    (EmbeddingRegionTarget(SOURDOUGH, embedder_name="sentence-transformers/all-MiniLM-L6-v2"),
     dict(payload=PAYLOAD, n_tokens=16, init="centroid_text", num_steps=200, seed=0)),
    (ActivationDirectionTarget(FORMAL, INFORMAL, layer=a.steer_layer),
     dict(alphas=[float(x) for x in a.alphas.split(",")], test_prompts=steer_prompts,
          max_new_tokens=a.steer_max_new_tokens, chat_template=CHAT)),
]

summary = {"model": a.model, "paths": []}
for target, cfg in jobs:
    t0 = time.time()
    result, report = dispatcher.run(target, bundle, **cfg)
    print(f"\n{'=' * 78}\n{type(target).__name__} -> dispatcher -> {result.attack} "
          f"({time.time() - t0:.0f}s)\n  goal: {target.describe()}")
    entry = {"target": type(target).__name__, "attack": result.attack}

    if isinstance(target, TextTarget):
        h = result.loss_history
        marks = sorted({i for i in (0, 9, 49, 99, len(h) - 1) if i < len(h)})
        print("  GCG loss: " + " -> ".join(f"step{i + 1} {h[i]:.3f}" for i in marks)
              + f"; best {result.final_loss:.4f}")
        print(f"  suffix: {result.payload!r}")
        for k, v in report.items():
            print(f"  [{k}] loss {v['target_loss']:.4f}, token acc {v['target_token_accuracy']:.2f}, "
                  f"exact {v['exact_match']}\n      -> {v['generation']!r}")
        entry.update(best_loss=result.final_loss, suffix=result.payload, evaluation=report)

    elif isinstance(target, EmbeddingRegionTarget):
        print(f"  optimised span: {result.extra['span_text']!r}")
        for k, v in report.items():
            ppl = perplexity(bundle, v["text"])
            v["perplexity_" + a.model] = ppl
            print(f"  [{k:24s}] cos dist {v['cosine_distance_to_centroid']:.4f}  PPL {ppl:8.1f}")
        entry.update(evaluation=report)

    elif isinstance(target, ActivationDirectionTarget):
        ref = report["reference"]
        print(f"  chat template: {result.extra['chat_template']}; "
              f"|direction| {result.extra['direction_norm']:.2f}; projection scale: formal "
              f"{ref['positive_prompts_mean_proj']:.2f}, informal {ref['negative_prompts_mean_proj']:.2f}")
        for alpha, r in report["by_alpha"].items():
            g = r["generations"][steer_prompts[0]]
            print(f"  alpha {alpha:+.0f}: projection {r['mean_projection_of_output']:6.2f}  "
                  f"{steer_prompts[0]!r} -> {g[:90]!r}")
        for alpha, r in report["by_alpha"].items():
            for p, g in r["generations"].items():
                print(f"    [alpha {alpha:+.0f}] {p!r}\n        -> {g!r}")
        entry.update(evaluation={**report, "by_alpha": {str(k): v for k, v in report["by_alpha"].items()}})

    summary["paths"].append(entry)

os.makedirs(os.path.dirname(a.out), exist_ok=True)
with open(a.out, "w") as f:
    json.dump(summary, f, indent=1)
print(f"\nsaved {a.out}")
