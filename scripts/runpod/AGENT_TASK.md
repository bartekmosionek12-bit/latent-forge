# Task for Claude Code on the RunPod pod: topic injection, Phases 2 (GPU part), 3, 4a, 5

You are running on a disposable RunPod GPU pod, started by Bartek (the repo owner), with
permission prompts skipped. Work only inside this repository. Read this whole file and the
README section "Topic injection via activation-targeted GCG" (all its subsections) before
doing anything. Reports for Bartek are in Polish; code, README and commit messages in English.

Bartek asked (2026-11-25) for Phases 2, 3, 4a and 5 to run back to back without waiting for
him between them. That lifts his "stop after every phase" rule for these four phases only.
Phase 4b (closed API model) and Phase 6 (tool / publication) are NOT part of this task.

## Plan change 2026-11-26 (Bartek): stop after Phase 2

Phase 2 gave SEPARATE = 0 for every arm. Bartek decided: finish the Phase 2 seeds of the chosen
arm and `measure`, commit and push, update `runs/topic_phase2/GPU_REPORT.md`, then STOP. Do NOT start
Phases 3, 4a or 5 (`agent_chain.sh` must not get past "Phase 2 done"). The sections on those phases
below are kept for the record only.

## Hard rules (from Bartek, not negotiable)

1. No API keys, tokens or secrets in the repo, in logs, or in your output. Never print, copy
   or look for credentials (`~/.git-credentials`, env vars, `gh auth token`, ...). No key and
   no HF token is needed. Use only public, NOT gated models (no Llama, no Gemma, no Prompt
   Guard). If a download asks for a token, pick another model and say so in the report.
2. Neutral topics only: cooking (main), gardening (control). Fictional products with the
   TEST_ prefix only. No refusal-suppression term, refusal direction or anything like it,
   not even as a baseline. No topics the model refuses. No attempts to extract data or the
   system prompt. Local models on this pod only; no requests to any third-party app or API.
3. Do not change what was pre-registered or decided: the measures in the README section
   "Pre-registered measures" (with the D2 plan change), the Phase 2 arms, budgets, targets,
   thresholds. New phases follow the specs below; if something there is unclear or does not
   work as written, choose the most conservative reading, write the choice into the README
   as a plan change made before the data, and continue.
4. Pinned versions; do not upgrade transformers (4.47.1) or torch. New code paths go
   through the dispatcher (new Target + Attack registered there), with unit tests. Run
   `pytest -q -m "not slow"` before every commit that touches code; no regressions.
5. Log every optimisation step to `.log` + `.jsonl`; an interrupted run must keep its data
   and resume.
6. Git: commit only on branch `claude/project-thread-mnmiam`. Never force-push or rewrite
   history. If a push is rejected, `git pull --no-rebase` and push again. Use `git add -f`
   for `runs/` (it is gitignored). End every commit message with the line
   `Co-Authored-By: Claude <noreply@anthropic.com>`. Push at least every 30 min of running.
7. Long commands: start them with `nohup ... > file 2>&1 &` and check the output every few
   minutes; never hold one tool call for hours.
8. Budget: stop the whole task (report and end) if total GPU wall time passes **20 h**, or if
   any single phase is estimated (from a 5-step timing) at more than **6 h**.
9. If something fails in a way you cannot fix without changing the design, STOP, push what
   you have and write the report.

## Phase 2, GPU part (arms A-D)

Everything is implemented in `scripts/runpod/phase2_gpu.sh` and `scripts/topic_phase2.py`.
The CPU part (baselines, arm E, ablation E_all) and `runs/topic_phase2/calibration.json` are
committed: do not regenerate them.

1. `git checkout claude/project-thread-mnmiam && git pull --no-rebase`.
2. If `.venv` is missing: `bash scripts/runpod/setup.sh` (must end with `cuda True` and
   `setup OK`; if no GPU is visible, stop).
3. `bash scripts/runpod/phase2_gpu.sh smoke 2>&1 | tee runs/topic_phase2/gpu_smoke.txt`.
   Estimate = 3 x (arm A hours) + (arm D hours) + 0.5 h. Apply rule 8.
4. `bash scripts/runpod/phase2_gpu.sh run` (background), then `measure`.
5. Best arm, rule fixed before the data: among A-D, the arm with the highest mean SEPARATE
   whose mean preservation is >= 0.8; ties or none passing -> highest SEPARATE overall, and
   say so. Then `bash scripts/runpod/phase2_gpu.sh seeds <best>` and `measure` again.
6. Answer the five Phase 2 questions in a new README subsection "Phase 2 results", each
   separately, with numbers from `runs/topic_phase2_table.log`, sample suffixes of every arm,
   and the lexical audit / manual review of arm A (is it a broken or foreign-language
   explicit instruction? say it plainly). If no arm passes both detectors and the PPL filters
   with more separate sentences than the naive sentence, write exactly that.
7. Commit, push, write `runs/topic_phase2/GPU_REPORT.md` (Polish, short).

## Phase 3: universality

- One suffix optimised jointly over the 4 training pages (PHASE2_TRAIN), evaluated on those
  4 and on the 4 held-out pages (the other 4 in PRODUCT_PAGES), never seen in optimisation.
- Arm: the Phase 2 best arm, same budget per step, 300 steps; each step samples targets
  across the 4 training pages (k=4 in total, rotating pages). Implement as a new Target
  (e.g. `UniversalPageInjectionTarget`, a list of page/question/product/continuations) and
  Attack registered in the dispatcher, reusing PageSuffixGCGAttack internals; add tests that
  the joint loss equals the mean of per-page losses.
- Control topic: repeat with gardening (new gardening continuations "{A_q} Tip: {gardening
  tip}", 24 tips, same format; gardening already has TOPIC_REFERENCES). Measure SEPARATE for
  cooking and for gardening on both suffixes (a cooking suffix should not inject gardening,
  and vice versa).
- Baselines on all 8 pages: clean, naive sentence, explicit instruction. Same measures,
  generation and table format as Phase 2. Seeds 0 and 1.
- Outputs under `runs/topic_phase3/`, README subsection "Phase 3 results", report
  `runs/topic_phase3/REPORT.md` (Polish).

## Phase 4a: transfer to another open model, locally

- Take the Phase 3 universal suffixes (cooking and gardening, both seeds) as TEXT and paste
  them unchanged into the same page prompts of two other open, non-gated models:
  `Qwen/Qwen2.5-1.5B-Instruct` (same tokenizer) and `HuggingFaceTB/SmolLM2-1.7B-Instruct`
  (different family). No re-optimisation.
- Same baselines, measures and table for each model; note refusals if any (the 1.5B refused
  2 of 8 questions in Phase 1b: report those pages separately, do not drop them silently).
- Outputs under `runs/topic_phase4a/`, README subsection "Phase 4a results", report
  `runs/topic_phase4a/REPORT.md` (Polish).

## Phase 5: Pareto front

- Loss L = l1 * KL_task + l2 * L_topic + l3 * L_fluency on Qwen2.5-0.5B, per training page:
  - KL_task: mean over the clean greedy answer's tokens (teacher-forced) of
    KL(p_clean || p_with_suffix);
  - L_topic: the Phase 2 continuation-set loss;
  - L_fluency: the arm C fluency term.
- At least 5 weight points: l2 = 1, l3 = 0.2, l1 in {0, 0.1, 0.3, 1, 3}; 200 steps, seed 0,
  4 training pages. New option in the dispatcher path with a unit test of the KL term.
- Measure with the Phase 2 measures; plot SEPARATE (y) against preservation (x), one point
  per weight setting, plus clean / naive / explicit / Phase 2 best arm as reference points.
  Save `runs/topic_phase5/pareto.png` and the data as jsonl.
- README subsection "Phase 5 results", report `runs/topic_phase5/REPORT.md` (Polish).

## End

Write `runs/AGENT_SUMMARY.md` (Polish): what ran, wall times, any plan changes and fixes,
one paragraph per phase with the main numbers, and what is left (Phase 4b needs Bartek's
consent and his own key; Phase 6 is shown to him before any publication). Commit, push, stop.
