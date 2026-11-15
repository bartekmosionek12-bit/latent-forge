# Migration: three attacks to Qwen2.5-7B-Instruct on RunPod (design note)

Status: plan only. Nothing has run on a GPU or on the 7B model yet.
The region-GCG line (Phases 0-4b) is closed and is not part of this migration.

## Scope

The three paths that already work through one dispatcher (README "Transfer check" on
Qwen2.5-0.5B-Instruct, CPU):

| path | uses the attacked LLM for | what moves to 7B |
|---|---|---|
| `GCGAttack` (nanoGCG) | gradients + candidate losses | everything; this is the expensive path |
| `ActivationSteeringAttack` | direction extraction, steered generation, re-read projection | layer and alpha grid must be re-tuned |
| `EmbeddingRegionAttack` | only the perplexity (detectability) measure | attack unchanged (MiniLM); only PPL is re-measured under 7B |

## Code changes needed before the first GPU run (small, found while reading)

1. `scripts/gcg_budget_check.py` reloads the model as `ModelConfig(name=...)`, i.e. float32.
   7B in fp32 is ~30 GB and does not fit a 24 GB card. It must take the dtype (store it
   in the phase-2 JSON and read it back, or add `--dtype`).
2. `scripts/phase2_gcg_demo.py` has no `--batch-size`. nanoGCG halves the batch on OOM
   by itself, but every retry costs a full forward; an explicit value avoids that.
3. Nothing else is expected to change in code (README "Switching the model" lists what
   was checked: layout, decoder layers, dtype cast in the hook, embedder independence).

## Stages (each one stops after its evidence)

**Stage 0: smoke test and measurements, no attack (~15-20 min GPU).**
One script, one log (`runs/qwen7b_stage0_smoke.log`):
- GPU name, VRAM, CUDA/torch versions; `pytest -q` (toy models, sanity of the install).
- Load 7B; peak memory after load; 28 decoder layers, d_model 3584.
- `split_prompt` layout with the real tokenizer (expected
  `<|im_start|>user\nReply with a short answer:` ... `<|im_end|>\n<|im_start|>assistant\n`).
- **dtype evidence (open question 3):** the same ~20 prompts in fp16 and bf16: NaN/inf
  in logits or hidden states, max |activation| per layer vs the fp16 limit (65504), and
  next-token argmax agreement. On a 48 GB card also an fp32 reference, so agreement is
  measured against fp32 and not only between the two half formats.
- Residual-stream norm per layer on the steering prompts (input to the stage-1 alpha grid).
- **GCG timing:** 5 steps at search width 256, seconds per step and peak memory, with
  and without an explicit batch size. This turns the GCG budget into a known cost.

**Stage 1: activation steering on 7B.**
Layers 10 / 14 / 18 of 28; alpha chosen so the injected vector is 0.1-1.0x the measured
residual norm at that layer (on both earlier models the optimum sat at 0.3-0.6x and output
broke down near 1x). Chat mode `auto` (already on for instruct models).
Controls: alpha 0 baseline, formal/informal references re-read by the unsteered model,
and a random direction with the same norm (separates "this direction" from "any push").
Measure: projection under the unsteered model (as before). An independent second measure
is worth adding here (e.g. MiniLM distance to the formal vs informal references); that
would be a short note of its own before implementing.

**Stage 2: GCG on 7B.**
Same target and prompt as the 0.5B budget check (`TEST_odpowiedz_testowa`, seed 0,
width 256, `early_stop` off), analysed with `gcg_budget_check.py` at checkpoints by
generation/exact match, not by loss alone (0.5B showed loss and success disagree).
Controls: generation with no suffix and with a random suffix of the same length.
Budget shape is open question 1 (below).

**Stage 3: embedding-region path + end-to-end.**
`e2e_all_paths.py` with the chosen layer/alpha/steps, so all three paths go through one
dispatcher on 7B; README gets a third column next to distilgpt2 and Qwen-0.5B.

## Open questions (asked when they matter)

1. **GCG budget on 7B** (needed at stage 2): one long run (500+ steps) or several seeds?
   0.5B reached exact match only at step 379 of 500, with step-like progress.
2. **Chat template for steering prompts** (stage 1): already implemented and on by default
   (`chat_template="auto"`, checked on 0.5B). Plan: keep it; base-style prompts would be
   an extra comparison, not the default.
3. **fp16 vs native bf16** (stage 0): answered by measurement in stage 0, not by guess.

## What is needed on the RunPod side

- A pod with >= 24 GB VRAM. A5000 / RTX 4090 (24 GB) is enough for fp16/bf16 7B with GCG
  at reduced batch. A 48 GB card (A6000 / L40S) avoids OOM batch halving and allows the
  fp32 reference in stage 0.
- PyTorch CUDA 12.x template, volume >= 50 GB (model ~15 GB, HF cache, venv with torch).
- No HF token needed: Qwen2.5-7B-Instruct is not gated.
- A way to run commands and bring logs back (see the reply in the project thread).
